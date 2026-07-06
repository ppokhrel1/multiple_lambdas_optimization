from multi_resolution import *

import copy
import sys

# ==========================================
# 4. Utilities & Training
# ==========================================

def evaluate_standalone(model, loader, device):
    """
    Evaluates a single model (Expert or Combiner) on a dataset.
    Returns: Average MSE
    """
    model.eval()
    total_mse = 0
    steps = 0
    with torch.no_grad():
        for bx, by in loader:
            pred = model(bx)
            total_mse += F.mse_loss(pred, by).item()
            steps += 1
    return total_mse / steps



def train_combiner(combiner, train_loader, val_loader, test_loader, budget, device, name="Model"):
    print(f"\n>>> Training {name} <<<")
    start_time = time.time()

    # 1. Parameter Groups
    model_params = [p for n, p in combiner.named_parameters() if ('lam' not in n) and ('router' not in n)]
    router_params = [p for n, p in combiner.named_parameters() if ('lam' not in n) and ('router' in n)]
    dual_params = [p for n, p in combiner.named_parameters() if 'lam' in n]

    ETA_THETA = 1e-5
    ETA_LAMBDA = 1e-7  # two-time-scale (Assumption 4): source-weight router updates ~100x slower than theta
    opt_primal = optim.Adam([
        {'params': model_params,  'lr': ETA_THETA},
        {'params': router_params, 'lr': ETA_LAMBDA},
    ], weight_decay=1e-7)
    sched_primal = optim.lr_scheduler.StepLR(opt_primal, step_size=150, gamma=0.5)  # diminishing-step schedule
    opt_dual = optim.Adam(dual_params, lr=1e-4, maximize=True) if dual_params else None

    loss_fn = PINNLoss(mse_weight=10.0, physics_weight=1e-3).to(device)
    history = {'train_mse': [], 'val_mse': [], 'test_mse': [], 'cost': []}

    for epoch in range(500):
        if epoch > 0: sched_primal.step()  # per-epoch LR decay
        # --- PHASE 1: TRAINING ---
        combiner.train()
        train_loss, train_mse, train_cost, tr_sum = 0, 0, 0, 0

        for bx, by in train_loader:
            opt_primal.zero_grad()
            if opt_dual: opt_dual.zero_grad()

            pred = combiner(bx)
            # Use the specific training_step logic defined in your classes
            loss, weights, cost = combiner.training_step(bx, pred, by, loss_fn, budget)
            mse = F.mse_loss(pred, by)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(combiner.parameters(), 1.0)
            opt_primal.step()
            if opt_dual: opt_dual.step()

            train_loss += loss.item()
            train_mse += mse.item()
            tr_sum += weights.sum(dim=-1).mean().item()
            train_cost += cost#.item()

        num_batches = len(train_loader)
        avg_sum = tr_sum / num_batches
        avg_cost = train_cost / num_batches
        if hasattr(combiner, 'update_dual'):
            if isinstance(combiner, (ADMMCombiner, ImpAugLagrangianCombiner)):
                combiner.update_dual(avg_sum, avg_cost, budget)
            else:
                # Basic Lagrangian update for budget
                with torch.no_grad():
                    #combiner.lam_sum.add_(0.1 * (avg_sum - 1.0))
                    combiner.lam_budget.add_(0.1 * max(0, avg_cost - budget))
        # --- PHASE 2: VALIDATION & WEIGHT MONITORING ---
        combiner.eval()
        val_loss, val_mse = 0, 0
        current_weights = None

        with torch.no_grad():
            for bx, by in val_loader:
                pred = combiner(bx)
                v_loss, _, _ = combiner.training_step(bx, pred, by, loss_fn, budget)
                val_loss += v_loss.item()
                val_mse += F.mse_loss(pred, by).item()

                # Capture weights from the last batch for logging
                if current_weights is None:
                    current_weights = combiner.get_weights(bx)

        # --- PHASE 3: TESTING ---
        ts_loss, ts_mse = 0, 0
        with torch.no_grad():
            for tx, ty in test_loader:
                tx, ty = tx.to(device), ty.to(device)
                t_pred = combiner(tx)
                t_loss, _, _ = combiner.training_step(tx, t_pred, ty, loss_fn, budget)
                ts_loss += t_loss.item()
                ts_mse += F.mse_loss(t_pred, ty).item()

        num_batches = len(train_loader)
        # Average Metrics for Logging
        avg_tr_loss, avg_tr_mse = train_loss/num_batches, train_mse/num_batches
        avg_val_loss, avg_val_mse = val_loss/len(val_loader), val_mse/len(val_loader)
        avg_ts_loss, avg_ts_mse = ts_loss/len(test_loader), ts_mse/len(test_loader)


        # Calculate Averages
        num_tr, num_val = len(train_loader), len(val_loader)
        avg_tr_loss, avg_tr_mse = train_loss/num_tr, train_mse/num_tr
        avg_val_loss, avg_val_mse = val_loss/num_val, val_mse/num_val
        avg_cost = train_cost/num_tr


        # Log results
        history['train_mse'].append(avg_tr_mse)
        history['val_mse'].append(avg_val_mse)
        history['test_mse'].append(avg_ts_mse)
        history['cost'].append(avg_cost)

        # Weighted Mean of first 5 experts for visualization
        w_dist = current_weights.mean(dim=0) if current_weights.dim() > 1 else current_weights
        w_str = ", ".join([f"{w}" for w in w_dist[:5]])

        print(f"Ep {epoch:02d} | Loss: [Tr {avg_tr_loss}, Val {avg_val_loss}, Ts {avg_ts_loss}]  "
              f"| MSE: [Tr {avg_tr_mse}, Val {avg_val_mse}, Ts {avg_ts_mse}] "
              f"| Cost: {avg_cost} | Top-5 W: [{w_str}]")

    end_time = time.time()
    elapsed_time = end_time - start_time

    return history, elapsed_time



def train_experts(device):
    res = 128
    expert_configs = [
        {'name': 'FiniteDiff', 'model': FiniteDifferenceExpert(hidden_dim=32), 'cost': 1.0},
        {'name': 'CNN',       'model': Expert1D(hidden_channels=32),           'cost': 3.0},
        {'name': 'FNO',       'model': FNO1D(modes=16, width=32),              'cost': 4.0},
        {'name': 'DeepONet',  'model': DeepONet(),                             'cost': 2.0}
    ]

    experts = []
    # Dictionary to store pre-training metrics for each architecture
    pretrain_history = {
        conf['name']: {
            'train_loss': [], 'train_mse': [], 
            'val_loss': [], 'val_mse': []
        } for conf in expert_configs
    }

    print("\n=== Pre-training Heterogeneous Experts (Res 128) ===")

    # Setup shared data
    train_x, train_y = generate_data(num_samples=100, nx=res, device=device, freq_range=(1,4))
    val_x, val_y = generate_data(num_samples=30, nx=res, device=device, freq_range=(1,4))
    
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32)

    for conf in expert_configs:
        name = conf['name']
        print(f"\n>>> Training {name} Expert (Cost {conf['cost']})...")
        model = conf['model'].to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = PINNLoss(mse_weight=1.0, physics_weight=1e-3).to(device)

        for epoch in range(20):
            # --- Training ---
            model.train()
            epoch_tr_loss, epoch_tr_mse = 0, 0
            for bx, by in train_loader:
                optimizer.zero_grad()
                pred = model(bx)
                if pred.dim() == 3: pred = pred.squeeze(1)

                loss = loss_fn(pred, by)
                mse = F.mse_loss(pred, by)
                
                loss.backward()
                optimizer.step()
                
                epoch_tr_loss += loss.item()
                epoch_tr_mse += mse.item()

            # --- Validation ---
            model.eval()
            epoch_vl_loss, epoch_vl_mse = 0, 0
            with torch.no_grad():
                for v_bx, v_by in val_loader:
                    v_pred = model(v_bx)
                    if v_pred.dim() == 3: v_pred = v_pred.squeeze(1)
                    
                    epoch_vl_loss += loss_fn(v_pred, v_by).item()
                    epoch_vl_mse += F.mse_loss(v_pred, v_by).item()

            # Log Averages
            avg_tr_loss = epoch_tr_loss / len(train_loader)
            avg_tr_mse = epoch_tr_mse / len(train_loader)
            avg_vl_loss = epoch_vl_loss / len(val_loader)
            avg_vl_mse = epoch_vl_mse / len(val_loader)

            pretrain_history[name]['train_loss'].append(avg_tr_loss)
            pretrain_history[name]['train_mse'].append(avg_tr_mse)
            pretrain_history[name]['val_loss'].append(avg_vl_loss)
            pretrain_history[name]['val_mse'].append(avg_vl_mse)

            if (epoch + 1) % 5 == 0:
                print(f"  Ep {epoch+1:2}: TrMSE {avg_tr_mse:.6f} | VlMSE {avg_vl_mse:.6f} | VlLoss {avg_vl_loss:.6f}")

        experts.append(MultiResExpertWrapper(model, res, conf['cost']))

    return experts, [a['name'] for a in expert_configs], pretrain_history

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # 1. Train Heterogeneous Experts at Fixed Scale
    experts, expert_names, _ = train_experts(device)
    for e in experts:
        e.eval()
        #for p in e.parameters(): p.requires_grad = False


    print("\nGenerating Multi-Scale Generalization Datasets (All Res 128)...")
    # 2. Train: Standard Frequencies (1-3)
    train_x, train_y = generate_data(num_samples=200, nx=128, device=device, freq_range=(1, 3))
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)

    # 3. Validation: Standard Frequencies (1-3)
    val_x, val_y = generate_data(num_samples=50, nx=128, device=device, freq_range=(1, 3))
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32)

    # 4. Test: Higher Frequencies (3-6) -> Out-of-Distribution / Harder
    test_x, test_y = generate_data(num_samples=50, nx=128, device=device, freq_range=(3, 6))
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=32)

    # Budget targets a mix of Cheap (FD/CNN) and Expensive (FNO) methods
    if len(sys.argv) > 1:
        try:
            BUDGET = float(sys.argv[1])
            print(f"Budget set from command line argument: {BUDGET}")
        except ValueError:
            print(f"Invalid budget argument '{sys.argv[1]}'. Defaulting to 2.0")
            BUDGET = 2.0
    else:
        print("No budget argument provided. Defaulting to 2.0")
        BUDGET = 2.0
    results = {}

    print("\n>>> Evaluating Baselines <<<")

    # 1. Single Experts
    for i, expert in enumerate(experts):
        start_time = time.time()
        mse = evaluate_standalone(expert, test_loader, device)
        name = expert_names[i]
        end_time = time.time()
        elapsed_time = end_time - start_time
        results[f'Base_{name}'] = {'test_mse': [mse], 'cost': [expert.cost], 'time': elapsed_time}
        print(f"  {name}: MSE {mse}, Cost {expert.cost}")

    #all_experts = [copy.deepcopy(experts)] * 5
    # --- 1. Softmax ---
    model = SoftmaxCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "Softmax")
    results['Softmax'] = {'history': history, 'time': elapsed}

    # --- 2. Lagrangian ---
    experts, expert_names, _ = train_experts(device)
    model = LagrangianCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "Lagrangian")
    results['Lagrangian'] = {'history': history, 'time': elapsed}


    experts, expert_names, _ = train_experts(device)
    model = ImpAugLagrangianCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "ImpLagrangian")
    results['ImpLagrangian'] = {'history': history, 'time': elapsed}

    # --- 3. Aug Lagrangian ---
    experts, expert_names, _ = train_experts(device)
    model = AugLagrangianCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "AugLag (Routing)")
    results['AugLag_Routing'] = {'history': history, 'time': elapsed}

    # --- 4. ADMM ---
    experts, expert_names, _ = train_experts(device)
    admm_model = ADMMCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(admm_model, train_loader, val_loader, test_loader, BUDGET, device, "ADMM (Routing)")
    results['ADMM_Routing'] = {'history': history, 'time': elapsed}

    # --- Final Summary ---


    print("\n=== Final Test Summary (Method Selection) ===")
    print(f"{'Method':<20} | {'Test MSE':<10} | {'Final Cost':<10} | {'Time (s)':<10}")
    print("-" * 60)
    for k, v in results.items():
        hist = v.get('history', v)
        t = v['time']
        mse = hist['test_mse'][-1]
        cost = hist['cost'][-1]
        print(f"{k:<20} | {mse:.6f}   | {cost:.4f}     | {t:.2f}")
if __name__ == "__main__":
    main()


