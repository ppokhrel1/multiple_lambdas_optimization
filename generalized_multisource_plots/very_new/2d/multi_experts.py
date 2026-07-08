from multi_resolution import *
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 4. Utilities & Training (2D Heterogeneous + Baselines)
# ==========================================

def evaluate_standalone(model, loader, device):
    """
    Evaluates a single model (Expert or Combiner) on a dataset.
    Returns: Average MSE
    """
    model.eval()
    loss_fn = PINNLoss2D(mse_weight=10.0, physics_weight=1e-3).to(device)

    total_mse = 0
    steps = 0
    with torch.no_grad():
        for bx, by in loader:
            pred = model(bx)
            total_mse += F.mse_loss(pred, by).item()
            steps += 1
    return total_mse / steps

def evaluate_uniform_baseline(experts, loader, device):
    """
    Evaluates a simple average ensemble (1/N weight for all experts).
    """
    loss_fn = PINNLoss2D(mse_weight=10.0, physics_weight=1e-3).to(device)
    total_mse = 0
    steps = 0
    # Calculate fixed cost: Average of all expert costs
    avg_cost = sum([e.cost for e in experts]) / len(experts)

    with torch.no_grad():
        for bx, by in loader:
            # Stack predictions: (Batch, H, W, N_Experts)
            preds = torch.stack([e(bx) for e in experts], dim=-1)
            # Uniform Average: Mean across last dim
            ensemble_pred = preds.mean(dim=-1)
            total_mse += F.mse_loss(ensemble_pred, by).item()
            steps += 1
    return total_mse / steps, avg_cost

def train_combiner(combiner, train_loader, val_loader, test_loader, budget, device, name="Model"):
    print(f"\n>>> Training {name} <<<")
    start_time = time.time()

    # 1. Parameter Groups
    model_params = [p for n, p in combiner.named_parameters() if ('lam' not in n) and ('router' not in n)]
    router_params = [p for n, p in combiner.named_parameters() if ('lam' not in n) and ('router' in n)]
    dual_params = [p for n, p in combiner.named_parameters() if 'lam' in n ]

    ETA_THETA = 1e-4
    ETA_LAMBDA = 1e-6  # two-time-scale (Assumption 4): source-weight router updates ~100x slower than theta
    opt_primal = optim.Adam([
        {'params': model_params,  'lr': ETA_THETA},
        {'params': router_params, 'lr': ETA_LAMBDA},
    ], weight_decay=1e-7)
    sched_primal = optim.lr_scheduler.StepLR(opt_primal, step_size=150, gamma=0.5)  # diminishing-step schedule
    opt_dual = optim.Adam(dual_params, lr=1e-3, maximize=True) if dual_params else None

    loss_fn = PINNLoss2D(mse_weight=10.0, physics_weight=1e-3, dealias=True).to(device)
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
                    combiner.lam_sum.add_(0.1 * (avg_sum - 1.0))
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

def train_heterogeneous_experts_2d(device):
    res = 64
    steps = 20

    expert_configs = [
        {'name': 'FiniteDiff', 'model': FiniteDifferenceExpert2D(hidden_dim=32), 'cost': 1.0},
        {'name': 'CNN',       'model': Expert2D(hidden_channels=32),             'cost': 3.0},
        {'name': 'FNO',       'model': FNO2D(modes=12, width=32),                'cost': 4.0},
        {'name': 'DeepONet',  'model': DeepONet2D(),                             'cost': 2.0}
    ]

    experts = []
    expert_names = []
    # Dictionary to store detailed metrics for each architecture
    pretrain_history = {
        conf['name']: {
            'train_loss': [], 'train_mse': [],
            'val_loss': [], 'val_mse': []
        } for conf in expert_configs
    }

    print("\n=== Pre-training Heterogeneous 2D Experts ===")

    # 1. Setup shared data for pre-training (Res 64)
    train_x, train_y = generate_2d_data(num_samples=100, nx=res, steps=steps, device=device)
    val_x, val_y     = generate_2d_data(num_samples=30, nx=res, steps=steps, device=device)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=8, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_x, val_y), batch_size=8)

    for conf in expert_configs:
        name = conf['name']
        print(f"\n>>> Training {name} Expert (Cost {conf['cost']})...")
        model = conf['model'].to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        # PINNLoss2D often uses a high weight for MSE to ensure convergence
        loss_fn = PINNLoss2D(mse_weight=10.0, physics_weight=1e-3, dealias=True).to(device)

        epochs = 20
        for epoch in range(epochs):
            # --- Training Phase ---
            model.train()
            epoch_tr_loss, epoch_tr_mse = 0, 0
            for bx, by in train_loader:
                optimizer.zero_grad()
                pred = model(bx)
                if pred.dim() == 4: pred = pred.squeeze(1)

                loss = loss_fn(pred, by)
                mse = F.mse_loss(pred, by)

                loss.backward()
                optimizer.step()

                epoch_tr_loss += loss.item()
                epoch_tr_mse += mse.item()

            # --- Validation Phase ---
            model.eval()
            epoch_vl_loss, epoch_vl_mse = 0, 0
            with torch.no_grad():
                for v_bx, v_by in val_loader:
                    v_pred = model(v_bx)
                    if v_pred.dim() == 4: v_pred = v_pred.squeeze(1)

                    epoch_vl_loss += loss_fn(v_pred, v_by).item()
                    epoch_vl_mse += F.mse_loss(v_pred, v_by).item()

            # Record and Print
            avg_tr_loss = epoch_tr_loss / len(train_loader)
            avg_tr_mse  = epoch_tr_mse / len(train_loader)
            avg_vl_loss = epoch_vl_loss / len(val_loader)
            avg_vl_mse  = epoch_vl_mse / len(val_loader)

            pretrain_history[name]['train_loss'].append(avg_tr_loss)
            pretrain_history[name]['train_mse'].append(avg_tr_mse)
            pretrain_history[name]['val_loss'].append(avg_vl_loss)
            pretrain_history[name]['val_mse'].append(avg_vl_mse)

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                print(f"  Ep {epoch+1:2} | TrMSE: {avg_tr_mse:.5f} TrLoss: {avg_tr_loss} | VlMSE: {avg_vl_mse:.5f} | VlLoss: {avg_vl_loss:.5f}")

        experts.append(MultiResExpertWrapper2D(model, res, conf['cost']))
        expert_names.append(name)

    return experts, expert_names, pretrain_history

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running 2D Heterogeneous Pipeline on {device}")

    # 1. Train Experts
    experts, expert_names, _ = train_heterogeneous_experts_2d(device)
    for e in experts:
        e.eval()
        #for p in e.parameters(): p.requires_grad = False

    print("\nGenerating Datasets...")
    train_x, train_y = generate_2d_data(num_samples=500, nx=128, steps=20, device=device)
    val_x, val_y     = generate_2d_data(num_samples=100, nx=128, steps=20, device=device)
    test_x, test_y   = generate_2d_data(num_samples=100, nx=128, steps=20, device=device)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_x, val_y), batch_size=32)
    test_loader  = DataLoader(TensorDataset(test_x, test_y), batch_size=32)

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

    # === A. Run Baselines ===
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

    # 2. Uniform Average (1/3 FD + 1/3 CNN + 1/3 FNO)
    uni_mse, uni_cost = evaluate_uniform_baseline(experts, test_loader, device)
    results['Base_Uniform'] = {'test_mse': [uni_mse], 'cost': [uni_cost]}
    print(f"  Uniform Ensemble: MSE {uni_mse}, Cost {uni_cost}")

    # === B. Run Combiners ===
    expert_config = [copy.deepcopy(experts)]*5
    # Softmax
    model = SoftmaxCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "Softmax")
    results['Softmax'] = {'history': history, 'time': elapsed}
    # Lagrangian (Global)
    experts, expert_names, _ = train_heterogeneous_experts_2d(device)
    model = LagrangianCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "Lagrangian")
    results['Lagrangian'] = {'history': history, 'time': elapsed}

    # AugLag (Routing)
    experts, expert_names, _ = train_heterogeneous_experts_2d(device)
    model = AugLagrangianCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "AugLag (Routing)")
    results['AugLag_Routing'] = {'history': history, 'time': elapsed}

    experts, expert_names, _ = train_heterogeneous_experts_2d(device)
    model = ImpAugLagrangianCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "ImpLagrangian")
    results['ImpAugLag_Routing'] = {'history': history, 'time': elapsed}

    # ADMM (Routing)
    experts, expert_names, _ = train_heterogeneous_experts_2d(device)
    admm_model = ADMMCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(admm_model, train_loader, val_loader, test_loader, BUDGET, device, "ADMM (Routing)")
    results['ADMM_Routing'] = {'history': history, 'time': elapsed}

    # === Final Summary ===
    print(f"\n=== Final Test Summary (Budget: {BUDGET}) ===")
    print(f"{'Method':<20} | {'Test MSE':<10} | {'Cost':<10} | {'Status'}")

    # Sort keys to show baselines first, then models
    print(f"{'Method':<20} | {'Test MSE':<10} | {'Final Cost':<10} | {'Time (s)':<10}")
    print("-" * 60)
    for k, v in results.items():
        print(k, v)
        #hist = v.get('history', v)
        #t = v['time']
        #mse = hist['test_mse'][-1]
        #cost = hist['cost'][-1]
        #print(f"{k:<20} | {mse:.6f}   | {cost:.4f}     | {t:.2f}")


if __name__ == "__main__":
    main()



