from multi_resolution import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

class DARouter(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        # Outputs both the weights for the experts AND the Kalman Gain
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        # Head for expert weights (alphas)
        self.weight_head = nn.Linear(64, num_experts)
        # Head for Kalman Gain (K) - squashed between 0 and 1
        self.gain_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        features = self.net(x.unsqueeze(1))
        weights = self.weight_head(features)
        gain = self.gain_head(features)
        return weights, gain

class BaseCombiner(nn.Module):
    def __init__(self, experts, device, use_routing=False):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.register_buffer('costs', torch.tensor([e.cost for e in experts], device=device))
        self.use_routing = use_routing
        self.device = device

        if self.use_routing:
            # The router output is now interpreted as log-variance
            self.router = DARouter(len(experts)).to(device)
        else:
            # theta represents log-variance for fixed weighting
            self.theta = nn.Parameter(torch.zeros(len(experts)))

    def get_logits(self, x):
        return self.router(x)[0] if self.use_routing else self.theta.unsqueeze(0).expand(x.size(0), -1)

    def get_weights(self, x):
        logits, K = self.router(x)
        weights = F.softmax(logits, dim=-1) # (Batch, num_experts)
        return weights

    def get_cost(self, weights):
        return torch.matmul(weights, self.costs).mean()

    def get_scores(self, x):
        return self.get_logits(x)

    def forward(self, x_clean):
        # --- TRUE ANALYSIS LOOP (DATA ASSIMILATION) ---
        # 1. Get background states from all experts
        u_backgrounds = [expert.model(expert.apply_noise(x_clean)) for expert in self.experts]
    
        # 2. Derive weights
        router_input = torch.stack(u_backgrounds, dim=0).mean(dim=0)
        
        # 3. Derive weights from NOISY data
        logits, K = self.router(router_input)
        #w = F.softmax(logits, dim=-1) # (Batch, num_experts)
        w = self.get_weights(router_input)
        # 3. Compute weighted background state
        u_background = torch.zeros_like(u_backgrounds[0])
        for i in range(len(self.experts)):
            # Reshape weight from (Batch) to (Batch, 1, 1) to match (Batch, H, W)
            w_i = w[:, i].view(-1, 1, 1) 
            u_background += w_i * u_backgrounds[i]

        # --- ANALYSIS CORRECTION ---
        analysis_correction = torch.zeros_like(u_background)
        for i, expert in enumerate(self.experts):
            z_obs = expert.apply_noise(x_clean) # Observation from source i
            innovation = z_obs - u_background

            # Reshape weight for broadcasting here too
            w_i = w[:, i].view(-1, 1, 1)
            analysis_correction += w_i * innovation

        u_analysis = u_background + K.view(-1, 1, 1) * analysis_correction
        return u_analysis


# Use the existing Logic Classes (Softmax, Lagrangian, ADMM)
# but ensure they inherit from BaseCombiner2D.

# The Logic classes (Softmax, Lagrangian, ADMM) remain structurally identical
# because they operate on scalar Loss/Cost values.
# We just inherit from the new BaseCombiner.

class SoftmaxCombiner(BaseCombiner):
    def training_step(self, x, pred, y, criterion, budget):
        loss = criterion(pred, y)
        weights = self.get_weights(x)
        current_cost = self.get_cost(weights)
        # Softmax doesn't use the budget to penalize, just tracks it
        return loss, weights, current_cost.item()

class LagrangianCombiner(BaseCombiner):
    def __init__(self, experts, device, use_routing=False):
        super().__init__(experts, device, use_routing)
        # Separate multipliers for the Sum-to-1 constraint and the Budget constraint
        #self.lam_sum = nn.Parameter(torch.tensor(0.0, device=device))
        self.lam_budget = nn.Parameter(torch.tensor(0.0, device=device))

    def get_weights(self, x):
        # HARD CONSTRAINT: Always project to simplex
        return project_to_simplex(self.get_logits(x))

    def training_step(self, x, pred, y, criterion, budget):
        weights = self.get_weights(x)
        mse_loss = criterion(pred, y)

        # Constraint 1: Sum of weights must be 1
        sum_violation = weights.sum(dim=-1) - 1.0

        # Constraint 2: Cost must be <= budget
        current_cost = self.get_cost(weights)
        budget_violation = F.relu(current_cost - budget) # Only penalize if over budget

        total_loss = mse_loss + \
                    self.lam_budget * budget_violation
                    #  self.lam_sum * sum_violation.mean()


        return total_loss, weights, current_cost.item()

class AugLagrangianCombiner(BaseCombiner):
    def __init__(self, experts, device, use_routing=False):
        super().__init__(experts, device, use_routing)
        # Dual variables for Sum and Budget
        #self.register_buffer('lam_sum', torch.tensor(0.0, device=device))
        self.register_buffer('lam_budget', torch.tensor(0.0, device=device))
        self.register_buffer('rho', torch.tensor(1.0, device=device))

    def get_weights(self, x):
        # We use project_to_simplex for hard constraint satisfaction
        return project_to_simplex(self.get_logits(x))

    def training_step(self, x, pred, y, criterion, budget):
        weights = self.get_weights(x)
        mse_loss = criterion(pred, y)

        # Violations
        sum_viol = weights.sum(dim=-1) - 1.0
        current_cost = self.get_cost(weights)
        budget_viol = F.relu(current_cost - budget) # Inequality: Cost <= Budget

        # L = MSE + λc + (ρ/2)c²
        penalty = (self.lam_budget * budget_viol + (self.rho / 2) * (budget_viol**2))
        #(self.lam_sum * sum_viol.mean() + (self.rho / 2) * (sum_viol**2).mean()) + \


        return mse_loss + penalty, weights, current_cost.item()

    def update_dual(self, avg_sum, avg_cost, budget):
        with torch.no_grad():
            # Standard Dual Ascent
            #self.lam_sum.add_(self.rho * (avg_sum - 1.0))
            self.lam_budget.add_(self.rho * F.relu(torch.tensor(avg_cost - budget)))
            # Clamp lambda to prevent exploding gradients
            #self.lam_sum.clamp_(-10.0, 10.0)
            self.lam_budget.clamp_(0.0, 10.0)


class ImpAugLagrangianCombiner(AugLagrangianCombiner):
    def __init__(self, experts, device, use_routing=False):
        super().__init__(experts, device, use_routing)
        self.prev_viol = float('inf')

    def update_dual(self, avg_sum, avg_cost, budget):
        with torch.no_grad():
            sum_v = avg_sum - 1.0
            bud_v = max(0, avg_cost - budget)

            # 1. Update Duals
            #self.lam_sum.add_(self.rho * sum_v)
            self.lam_budget.add_(self.rho * bud_v)

            # 2. Adaptive Rho: Increase penalty if progress is slow
            current_total_viol = abs(sum_v) + bud_v
            if current_total_viol > 0.01 and current_total_viol > self.prev_viol * 0.95:
                self.rho.mul_(1.2) # Increase pressure
                self.rho.clamp_(max=50.0)

            self.prev_viol = current_total_viol

class ADMMCombiner(BaseCombiner):
    def __init__(self, experts, device, use_routing=False):
        super().__init__(experts, device, use_routing)
        #self.register_buffer('u_sum', torch.tensor(0.0, device=device))
        self.register_buffer('u_budget', torch.tensor(0.0, device=device))
        self.register_buffer('rho', torch.tensor(2.0, device=device))

    def training_step(self, x, pred, y, criterion, budget):
        weights = self.get_weights(x)
        mse_loss = criterion(pred, y)

        avg_sum = weights.sum(dim=-1).mean()
        current_cost = self.get_cost(weights)

        # ADMM Residue for Simplex
        #sum_penalty = (self.rho / 2) * (avg_sum - 1.0 + self.u_sum)**2

        # ADMM Residue for Budget (Inequality projection)
        z_budget = torch.min(current_cost + self.u_budget, torch.tensor(budget, device=self.device))
        budget_penalty = (self.rho / 2) * (current_cost - z_budget + self.u_budget)**2

        return mse_loss  + budget_penalty, weights, current_cost.item()

    def update_dual(self, avg_sum, avg_cost, budget):
        with torch.no_grad():
            #self.u_sum.add_(avg_sum - 1.0)
            z_budget = torch.min(avg_cost + self.u_budget, torch.tensor(budget, device=self.device))
            self.u_budget.add_(avg_cost - z_budget)





class NoisySourceWrapper2D(nn.Module):
    def __init__(self, shared_model, iv_noise_std, meas_noise_std, cost, device):
        super().__init__()
        self.model = shared_model
        self.iv_noise_std = iv_noise_std
        self.meas_noise_std = meas_noise_std
        self.cost = cost
        self.device = device

    def apply_noise(self, x_clean):
        # x_clean: (Batch, H, W)

        # 1. Smooth 2D Correlated Noise (simulating sensor array artifacts)
        if self.iv_noise_std > 0:
            noise = torch.randn(x_clean.shape, device=self.device)
            noise_ft = torch.fft.rfft2(noise)

            # Low-pass filter (keep only low-frequency corners)
            modes = 8
            mask = torch.zeros_like(noise_ft)
            mask[:, :modes, :modes] = 1.0
            mask[:, -modes:, :modes] = 1.0

            noise_ft = noise_ft * mask
            iv_noise = torch.fft.irfft2(noise_ft, s=x_clean.shape[-2:])

            if iv_noise.std() > 1e-9:
                iv_noise = iv_noise / iv_noise.std()
            x_noisy = x_clean + iv_noise * self.iv_noise_std
        else:
            x_noisy = x_clean

        # 2. Add White Gaussian Noise
        if self.meas_noise_std > 0:
            x_noisy = x_noisy + torch.randn_like(x_clean) * self.meas_noise_std

        return x_noisy

    def forward(self, x):
        return self.model(x)

def train_combiner(combiner, train_loader, val_loader, test_loader, budget, device, name="Model"):
    print(f"\n>>> Training {name} (Coordinate Descent) <<<")
    start_time = time.time()

    # 1. Parameter Segmentation
    model_params = [p for n, p in combiner.named_parameters() if ('lam' not in n) and ('router' not in n)]
    router_params = [p for n, p in combiner.named_parameters() if ('lam' not in n) and ('router' in n)]
    #router_params = list(combiner.router.parameters()) if combiner.use_routing else [combiner.theta]
    dual_params = [p for n, p in combiner.named_parameters() if 'lam' in n]

    # 2. Separate Optimizers
    ETA_THETA = 1e-5
    ETA_LAMBDA = 1e-7  # two-time-scale (Assumption 4): source-weight router updates ~100x slower than theta
    opt_model = optim.Adam([
        {'params': model_params,  'lr': ETA_THETA},
        {'params': router_params, 'lr': ETA_LAMBDA},
    ], weight_decay=1e-7)  # curb weight growth that inflates spectral derivatives
    sched_model = optim.lr_scheduler.StepLR(opt_model, step_size=150, gamma=0.5)  # diminishing-step schedule
    #opt_router = optim.Adam(router_params, lr=1e-5)
    opt_dual = optim.Adam(dual_params, lr=1e-2, maximize=True) if dual_params else None
    GRAD_CLIP = 1.0  # cap update norm so high-freq content can't run away in the physics residual

    loss_fn = PINNLoss2D(mse_weight=10.0, physics_weight=1e-3, dealias=True).to(device)

    history = {
        'train_mse': [], 'val_mse': [], 'test_mse': [],
        'train_loss': [], 'val_loss': [], 'test_loss': [],
        'cost': [], 'weights': []
    }

    for epoch in range(500):
        if epoch > 0: sched_model.step()  # per-epoch LR decay
        # --- TRAINING PHASE (Coordinate Descent) ---
        combiner.train()
        tr_mse_acc, tr_loss_acc, tr_cost_acc = 0, 0, 0
        epoch_weights = torch.zeros(len(combiner.experts), device=device)

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)

            # STEP 1: Optimize Model (Freeze Router)
            # for p in router_params: p.requires_grad = False
            # for p in model_params: p.requires_grad = True
            opt_model.zero_grad()
            pred_r = combiner(bx)
            loss_m, weights_r, cost_r = combiner.training_step(bx, combiner(bx), by, loss_fn, budget)
            loss_m.backward()
            torch.nn.utils.clip_grad_norm_(model_params + router_params, GRAD_CLIP)
            opt_model.step()

            # STEP 2: Optimize Router (Freeze Model)
            # for p in model_params: p.requires_grad = False
            # for p in router_params: p.requires_grad = True
            # opt_router.zero_grad()
            # pred_r = combiner(bx)
            # loss_r, weights_r, cost_r = combiner.training_step(bx, pred_r, by, loss_fn, budget)
            # loss_r.backward()
            # opt_router.step()

            # STEP 3: Optimize Dual (If applicable)
            if opt_dual:
                opt_dual.zero_grad()
                loss_d, _, _ = combiner.training_step(bx, combiner(bx), by, loss_fn, budget)
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(dual_params, GRAD_CLIP)
                opt_dual.step()

            with torch.no_grad():
                tr_mse_acc += F.mse_loss(pred_r, by).item()
                tr_loss_acc += loss_m.item()
                tr_cost_acc += cost_r
                epoch_weights += weights_r.mean(dim=0) if weights_r.dim() == 2 else weights_r

        num_batches = len(train_loader)
        avg_cost = tr_cost_acc / num_batches
        avg_sum = (epoch_weights.sum() / num_batches).item()

        if hasattr(combiner, 'update_dual'):
            if isinstance(combiner, (ADMMCombiner, ImpAugLagrangianCombiner)):
                combiner.update_dual(avg_sum, avg_cost, budget)
            else:
                # Basic Lagrangian update for budget
                with torch.no_grad():
                    #combiner.lam_sum.add_(0.1 * (avg_sum - 1.0))
                    combiner.lam_budget.add_(0.1 * max(0, avg_cost - budget))
        # --- EVALUATION PHASE (Val & Test) ---
        combiner.eval()
        val_mse_acc, val_loss_acc = 0, 0
        test_mse_acc, test_loss_acc = 0, 0

        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_pred = combiner(vx)
                v_loss, _, _ = combiner.training_step(vx, v_pred, vy, loss_fn, budget)
                val_mse_acc += F.mse_loss(v_pred, vy).item()
                val_loss_acc += v_loss.item()

            for tx, ty in test_loader:
                tx, ty = tx.to(device), ty.to(device)
                t_pred = combiner(tx)
                t_loss, _, _ = combiner.training_step(tx, t_pred, ty, loss_fn, budget)
                test_mse_acc += F.mse_loss(t_pred, ty).item()
                test_loss_acc += t_loss.item()

        # --- AVERAGING & LOGGING ---
        avg_tr_mse, avg_tr_loss = tr_mse_acc / len(train_loader), tr_loss_acc / len(train_loader)
        avg_val_mse, avg_val_loss = val_mse_acc / len(val_loader), val_loss_acc / len(val_loader)
        avg_ts_mse, avg_ts_loss = test_mse_acc / len(test_loader), test_loss_acc / len(test_loader)
        avg_cost = tr_cost_acc / len(train_loader)
        avg_weights = (epoch_weights / len(train_loader)).cpu().numpy()



        lam_val = combiner.lam.item() if hasattr(combiner, 'lam') else 0.0
        weight_str = " | ".join([f"S{i}:{w}" for i, w in enumerate(avg_weights)])

        print(f"Ep {epoch:03d} | MSE (Tr/Val/Ts): {avg_tr_mse}/{avg_val_mse}/{avg_ts_mse} |"
              f"       | Loss(Tr/Val/Ts): {avg_tr_loss}/{avg_val_loss}/{avg_ts_loss} |"
              f"       | Cost: {avg_cost} | λ: {lam_val} | Weights: [{weight_str}]")

        # Store History
        history['train_mse'].append(avg_tr_mse); history['val_mse'].append(avg_val_mse); history['test_mse'].append(avg_ts_mse)
        history['train_loss'].append(avg_tr_loss); history['val_loss'].append(avg_val_loss); history['test_loss'].append(avg_ts_loss)
        history['cost'].append(avg_cost); history['weights'].append(avg_weights)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("elapsed time: ", elapsed_time)

    return history, elapsed_time


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running 2D Multi-Source Optimization on {device}")

    # 1. Initialize Shared 2D FNO Backbone
    # Ensure FNO2D is defined earlier in your notebook
    shared_fno = FNO2D(modes=12, width=32).to(device)

    # 2. Define 4 Sources (Varying noise levels and costs)
    sources_config = [
        (0.01, 0.01, 2.0), # Clean / Expensive
        (0.05, 0.05, 1.5), # Moderate
        (0.15, 0.15, 1.0), # Noisy
        (0.30, 0.30, 0.5)  # Very Noisy / Cheap
    ]

    experts = [NoisySourceWrapper2D(shared_fno, iv, meas, c, device)
               for iv, meas, c in sources_config]

    # 3. Generate 2D Clean Data (Ground Truth Only)
    # Using your existing Navier-Stokes generator
    print("Generating 2D Clean Data...")
    train_x, train_y = generate_2d_data(num_samples=200, nx=128, steps=20, device=device)
    val_x, val_y = generate_2d_data(num_samples=50, nx=128, steps=20, device=device)
    test_x, test_y = generate_2d_data(num_samples=50, nx=128, steps=20, device=device)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=32)

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
    # --- Run Strategies ---
    # Softmax Baseline
    shared_fno = FNO2D(modes=12, width=32).to(device)
    experts = [NoisySourceWrapper2D(shared_fno, iv, meas, c, device)
               for iv, meas, c in sources_config]
    model = SoftmaxCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "Softmax")
    results['Softmax'] = {'history': history, 'time': elapsed}

    shared_fno = FNO2D(modes=12, width=32).to(device)
    experts = [NoisySourceWrapper2D(shared_fno, iv, meas, c, device)
               for iv, meas, c in sources_config]
    admm_model = ADMMCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(admm_model, train_loader, val_loader, test_loader, BUDGET, device, "ADMM (Routing)")
    results['ADMM_Routing'] = {'history': history, 'time': elapsed}

    shared_fno = FNO2D(modes=12, width=32).to(device)
    experts = [NoisySourceWrapper2D(shared_fno, iv, meas, c, device)
               for iv, meas, c in sources_config]
    model = ImpAugLagrangianCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "ImpLagrangian")
    results['ImpLagrangian'] = {'history': history, 'time': elapsed}

    shared_fno = FNO2D(modes=12, width=32).to(device)
    experts = [NoisySourceWrapper2D(shared_fno, iv, meas, c, device)
               for iv, meas, c in sources_config]
    # --- 2. Lagrangian ---
    model = LagrangianCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "Lagrangian")
    results['Lagrangian'] = {'history': history, 'time': elapsed}


    shared_fno = FNO2D(modes=12, width=32).to(device)
    experts = [NoisySourceWrapper2D(shared_fno, iv, meas, c, device)
               for iv, meas, c in sources_config]
    # --- 3. Aug Lagrangian ---
    model = AugLagrangianCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "AugLag (Routing)")
    results['AugLag_Routing'] = {'history': history, 'time': elapsed}

    # --- Final Summary ---
    print("\n=== Final Results (2D Source Selection) ===")
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



