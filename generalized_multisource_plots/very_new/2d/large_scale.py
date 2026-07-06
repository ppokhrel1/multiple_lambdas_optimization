from multi_resolution import *

import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 5. Noisy Sources (2D Adaptation)
# ==========================================

class NoisySourceWrapper2D(nn.Module):
    """
    Wraps a shared 2D model (FNO2D) but corrupts the input with specific
    2D noise profiles. Simulates fetching data from 2D sensor arrays
    of varying quality.
    """
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

    def get_correlated_noise(self, shape):
        # shape: (Batch, H, W)
        # Generate random white noise
        noise = torch.randn(shape, device=self.device)

        # Go to spectral space: (Batch, H, W//2 + 1)
        noise_ft = torch.fft.rfft2(noise)

        # Low-pass filter: Keep only low modes
        modes = 4 # Fewer modes = smoother/larger error blobs

        # Create a mask to zero out high frequencies
        mask = torch.zeros_like(noise_ft)

        # Keep low freq in X (top-left) and Y
        mask[:, :modes, :modes] = 1.0
        # Keep low freq in X (negative/bottom-left for periodicity) and Y
        mask[:, -modes:, :modes] = 1.0

        noise_ft = noise_ft * mask

        # Back to real space
        return torch.fft.irfft2(noise_ft, s=(shape[-2], shape[-1]))

    def forward(self, x_clean):
        # x_clean: [Batch, H, W] Ground Truth State

        # 1. Add Initial Value (IV) / Model Error Noise (Correlated/Smooth)
        if self.iv_noise_std > 0:
            iv_noise = self.get_correlated_noise(x_clean.shape)
            # Normalize to unit variance then scale
            if iv_noise.std() > 1e-9:
                iv_noise = iv_noise / iv_noise.std()
            x_noisy = x_clean + iv_noise * self.iv_noise_std
        else:
            x_noisy = x_clean

        # 2. Add Measurement Noise (White/Gaussian/Pixel-wise)
        if self.meas_noise_std > 0:
            meas_noise = torch.randn_like(x_clean) * self.meas_noise_std
            x_noisy = x_noisy + meas_noise

        # Pass noisy 2D data to the shared FNO
        return self.model(x_noisy)


class TopKExpertSelector:
    def __init__(self, top_k):
        self.top_k = top_k

    def __call__(self, scores):
        k = min(self.top_k, scores.size(-1))
        values, indices = torch.topk(scores, k, dim=-1)
        weights = torch.softmax(values, dim=-1)
        return indices, weights

class BaseCombiner2D(nn.Module):
    def __init__(self, experts, device, use_routing=False, top_k=10):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        # Ensure costs are float for matmul/multiplication
        self.register_buffer('costs', torch.tensor([e.cost for e in experts], device=device).float())
        self.use_routing = use_routing
        self.device = device
        self.top_k = top_k
        self.selector = TopKExpertSelector(top_k)

        if self.use_routing:
            # Example Router: You can replace this with a 2D FNO Router if preferred
            self.router = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, len(experts))
            ).to(device)
        else:
            self.theta = nn.Parameter(torch.zeros(len(experts), device=device))

    def get_logits(self, x):
        if self.use_routing:
            # Standardize input to 4D for Conv2d: [Batch, 1, H, W]
            if x.dim() == 3:
                x = x.unsqueeze(1)
            return self.router(x)
        else:
            return self.theta.unsqueeze(0).expand(x.size(0), -1)

    def get_scores(self, x):
        return self.get_logits(x)

    def get_cost(self, top_k_weights, indices):
        """
        Calculates cost based ONLY on selected 10 experts.
        """
        # top_k_weights: [Batch, 10]
        # indices: [Batch, 10]
        active_costs = self.costs[indices] # [Batch, 10]
        return (top_k_weights * active_costs).sum(dim=-1).mean()

    def get_topk_weights(self, x):
        """
        Helper to get Top-K indices and their normalized (sum-to-1) weights.
        """
        logits = self.get_logits(x) # [Batch, 100] (Float)

        # values = the actual logits, indices = the source IDs
        values, indices = torch.topk(logits, self.top_k, dim=-1)

        # Goal 2: Apply Sum-to-1 constraint ONLY on the top 10 float logits
        top_k_weights = torch.softmax(values, dim=-1)

        return indices, top_k_weights

    def forward(self, x):
        # Goal 1 & 2: Get the 10 sources and their normalized weights
        indices, top_k_weights = self.get_topk_weights(x)

        output = torch.zeros_like(x)
        view_shape = [-1] + [1] * (x.dim() - 1)

        # Dynamic Execution
        for k in range(self.top_k):
            expert_ids = indices[:, k]
            unique_ids = torch.unique(expert_ids)

            for expert_id in unique_ids:
                mask = (expert_ids == expert_id)
                batch_idx = torch.where(mask)[0]

                # Run expert
                expert_out = self.experts[expert_id](x[batch_idx])

                # Apply weight
                w = top_k_weights[batch_idx, k].view(*view_shape)
                output[batch_idx] += expert_out * w

        return output, top_k_weights, indices

    def get_weights(self, x):
        logits = self.get_logits(x)
        return F.softmax(logits, dim=-1)



# ==========================================
# 2. Updated Combiner Subclasses
# ==========================================
class SoftmaxCombiner(BaseCombiner2D):
    def training_step(self, x, y, criterion, budget=None):
      pred, top_k_weights, indices = self.forward(x)
      loss = criterion(pred, y)
      weights = self.get_weights(x)
      # Reduce across the batch to get a single scalar
      current_cost = self.get_cost(top_k_weights, indices)
      sum_constraint = torch.abs(weights.sum(dim=-1) - 1.0).mean()
      return loss, weights, current_cost.item()

class LagrangianCombiner(BaseCombiner2D):
    def __init__(self, experts, device, use_routing=False):
        super().__init__(experts, device, use_routing)
        # Separate multipliers for the Sum-to-1 constraint and the Budget constraint
        self.lam_sum = nn.Parameter(torch.tensor(0.0, device=device))
        self.lam_budget = nn.Parameter(torch.tensor(0.0, device=device))

    def get_weights(self, x):
        # HARD CONSTRAINT: Always project to simplex
        return project_to_simplex(self.get_logits(x))

    def training_step(self, x, y, criterion, budget):
        # Forward pass returns Top-K weights (sum-to-1) and their indices
        pred, top_k_weights, indices = self.forward(x)

        # 1. Primary Objective (Task Accuracy)
        mse_loss = criterion(pred, y)

        # 2. Constraint: Cost <= Budget
        current_cost = self.get_cost(top_k_weights, indices)
        # Inequality constraint: g(x) = current_cost - budget <= 0
        budget_violation = current_cost - budget

        # 3. Lagrangian: L = MSE + λ * violation
        # Note: We use relu on violation if we only penalize being OVER budget
        total_loss = mse_loss + self.lam_budget * F.relu(budget_violation)

        return total_loss, top_k_weights, current_cost.item()

class AugLagrangianCombiner(BaseCombiner2D):
    def __init__(self, experts, device, use_routing=False):
        super().__init__(experts, device, use_routing)
        self.register_buffer('lam_sum', torch.tensor(0.0, device=device))
        self.register_buffer('lam_budget', torch.tensor(0.0, device=device))
        self.register_buffer('rho', torch.tensor(1.0, device=device))


    def get_weights(self, x):
        # We use project_to_simplex for hard constraint satisfaction
        return project_to_simplex(self.get_logits(x))

    def training_step(self, x, y, criterion, budget):
        pred, top_k_weights, indices = self.forward(x)

        mse_loss = criterion(pred, y)
        current_cost = self.get_cost(top_k_weights, indices)

        # Inequality violation
        violation = F.relu(current_cost - budget)

        # L = MSE + λ*v + (ρ/2)*v²
        penalty = (self.lam_budget * violation) + (self.rho / 2) * (violation**2)

        total_loss = mse_loss + penalty

        return total_loss, top_k_weights, current_cost.item()

    def update_dual(self, avg_sum, avg_cost, budget):
        with torch.no_grad():
            # Standard Dual Ascent
            self.lam_sum.add_(self.rho * (avg_sum - 1.0))
            self.lam_budget.add_(self.rho * F.relu(torch.tensor(avg_cost - budget)))
            # Clamp lambda to prevent exploding gradients
            self.lam_sum.clamp_(-10.0, 10.0)
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
            self.lam_sum.add_(self.rho * sum_v)
            self.lam_budget.add_(self.rho * bud_v)

            # 2. Adaptive Rho: Increase penalty if progress is slow
            current_total_viol = abs(sum_v) + bud_v
            if current_total_viol > 0.01 and current_total_viol > self.prev_viol * 0.95:
                self.rho.mul_(1.2) # Increase pressure
                self.rho.clamp_(max=50.0)

            self.prev_viol = current_total_viol

class ADMMCombiner(BaseCombiner2D):
    def __init__(self, experts, device, use_routing=False, top_k=10):
        super().__init__(experts, device, use_routing, top_k)
        # We only need one dual variable for the Budget inequality
        self.register_buffer('u_budget', torch.tensor(0.0, device=device))
        # Start with a smaller rho for stability
        self.register_buffer('rho', torch.tensor(0.5, device=device))

    def get_weights(self, x):
        # HARD CONSTRAINT: Always project to simplex
        return project_to_simplex(self.get_logits(x))

    def training_step(self, x, y, criterion, budget):
        pred, top_k_weights, indices = self.forward(x)

        mse_loss = criterion(pred, y)
        current_cost = self.get_cost(top_k_weights, indices)

        # ADMM slack variable projection: z = min(cost + u, budget)
        # This finds the closest point in the feasible set [0, budget]
        z_budget = torch.min(current_cost + self.u_budget, torch.tensor(budget, device=self.device))

        # Penalty: (ρ/2) * ||cost - z + u||²
        # This drives the cost toward the feasible projection z
        admm_penalty = (self.rho / 2) * (current_cost - z_budget + self.u_budget)**2

        total_loss = mse_loss + admm_penalty

        return total_loss, top_k_weights, current_cost.item()

    def update_dual(self, avg_sum, avg_cost, budget):
        with torch.no_grad():
            # Update only the budget dual variable
            z_budget = torch.min(avg_cost + self.u_budget, torch.tensor(budget, device=self.device))
            self.u_budget.add_(avg_cost - z_budget)

            # Gradually increase rho to tighten the constraint, but cap it
            self.rho.mul_(1.05).clamp_(max=10.0)

def train_combiner(combiner, train_loader, val_loader, test_loader, budget, device, name="Model"):
    print(f"\n>>> Training {name} (Top-{combiner.top_k} 2D) | Budget: {budget} <<<")
    start_time = time.time()

    # Separate parameters for optimizer handling
    model_params = [p for n, p in combiner.named_parameters() if ('lam' not in n) and ('router' not in n)]
    router_params = [p for n, p in combiner.named_parameters() if ('lam' not in n) and ('router' in n)]
    dual_params = [p for n, p in combiner.named_parameters() if 'lam' in n ]

    ETA_THETA = 1e-5
    ETA_LAMBDA = 1e-7  # two-time-scale (Assumption 4): source-weight router updates ~100x slower than theta
    opt_primal = optim.Adam([
        {'params': model_params,  'lr': ETA_THETA},
        {'params': router_params, 'lr': ETA_LAMBDA},
    ], weight_decay=1e-7)
    sched_primal = optim.lr_scheduler.StepLR(opt_primal, step_size=150, gamma=0.5)  # diminishing-step schedule
    # Dual parameters (like lambda in Lagrangian) use maximize=True for gradient ascent
    opt_dual = optim.Adam(dual_params, lr=1e-2, maximize=True) if dual_params else None

    loss_fn = PINNLoss2D(mse_weight=10.0, physics_weight=1e-3).to(device)

    history = {
        'train_loss': [], 'val_loss': [], 'test_loss': [],
        'train_mse': [], 'val_mse': [], 'test_mse': [],
        'cost': []
    }

    for epoch in range(500):
        if epoch > 0: sched_primal.step()  # per-epoch LR decay
        # --- TRAINING ---
        combiner.train()
        tr_loss, tr_mse, tr_cost, tr_sum = 0, 0, 0, 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt_primal.zero_grad()
            if opt_dual: opt_dual.zero_grad()

            pred, _, _ = combiner(bx)
            mse_val = F.mse_loss(pred, by)

            # Pass budget to the training step logic
            loss, weights, current_batch_cost = combiner.training_step(bx, by, loss_fn, budget)

            loss.backward()
            opt_primal.step()
            if opt_dual: opt_dual.step()

            tr_loss += loss.item()
            tr_mse += mse_val.item()
            tr_cost += current_batch_cost
            tr_sum += weights.sum(dim=-1).mean().item()


        # Calculate epoch averages for Dual Updates
        num_batches = len(train_loader)
        avg_cost = tr_cost / num_batches
        avg_sum = tr_sum / num_batches

        # --- DUAL UPDATES (POST-EPOCH) ---
        if hasattr(combiner, 'update_dual'):
            if isinstance(combiner, (ADMMCombiner, ImpAugLagrangianCombiner)):
                combiner.update_dual(avg_sum, avg_cost, budget)
            else:
                # Basic Lagrangian update for budget
                with torch.no_grad():
                    combiner.lam_sum.add_(0.1 * (avg_sum - 1.0))
                    combiner.lam_budget.add_(0.1 * max(0, avg_cost - budget))

        # --- VALIDATION ---
        combiner.eval()
        val_loss, val_mse = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred, _, _ = combiner(bx)
                v_loss, _, _ = combiner.training_step(bx, by, loss_fn, budget)
                val_loss += v_loss.item()
                val_mse += F.mse_loss(pred, by).item()

        # --- TEST ---
        ts_loss, ts_mse = 0, 0
        with torch.no_grad():
            for tx, ty in test_loader:
                tx, ty = tx.to(device), ty.to(device)
                t_pred, _, _ = combiner(tx)
                t_loss, _, _ = combiner.training_step(tx, ty, loss_fn, budget)
                ts_loss += t_loss.item()
                ts_mse += F.mse_loss(t_pred, ty).item()

        # Average Metrics for Logging
        avg_tr_loss, avg_tr_mse = tr_loss/num_batches, tr_mse/num_batches
        avg_val_loss, avg_val_mse = val_loss/len(val_loader), val_mse/len(val_loader)
        avg_ts_loss, avg_ts_mse = ts_loss/len(test_loader), ts_mse/len(test_loader)

        # --- LOGGING SOURCE USAGE ---
        with torch.no_grad():
            sample_vx, _ = next(iter(val_loader))
            sample_vx = sample_vx.to(device)
            scores = combiner.get_scores(sample_vx)
            indices, weights = combiner.selector(scores)
            full_weights = combiner.get_weights(sample_vx).mean(dim=1)

            if combiner.use_routing:
                flat_indices = indices.flatten()
                flat_weights = weights.flatten()
                unique_ids = torch.unique(flat_indices)
                id_contributions = []
                for uid in unique_ids:
                    mask = (flat_indices == uid)
                    total_w = flat_weights[mask].sum() / sample_vx.size(0)
                    id_contributions.append((uid.item(), total_w.item()))
                id_contributions.sort(key=lambda x: x[1], reverse=True)
                top_usage = id_contributions[:10]
                usage_str = ", ".join([f"S{idx}:{w}" for idx, w in top_usage])
            else:
                g_idx, g_w = indices[0], weights[0]
                usage_str = ", ".join([f"S{g_idx[i].item()}:{g_w[i].item()}" for i in range(min(10, len(g_idx)))])

        print(f"Ep {epoch:3d} | LOSS [Tr:{avg_tr_loss} Val: {avg_val_loss} Ts:{avg_ts_loss}] | "
              f"MSE [Tr:{avg_tr_mse} Val: {avg_val_mse} Ts:{avg_ts_mse}] | "
              f"Cost: {avg_cost}/{budget} | Top Usage: {usage_str} Full Weight Distribution (First 10): {full_weights[:10].cpu().numpy()} |")

        # Update History
        history['train_mse'].append(avg_tr_mse); history['test_mse'].append(avg_ts_mse)
        history['train_loss'].append(avg_tr_loss); history['test_loss'].append(avg_ts_loss)
        history['cost'].append(avg_cost)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("time taken: ", elapsed_time)

    return history, elapsed_time


def initialize_experts_2d(device):
    """
    Initializes 4 Data Source Experts sharing ONE FNO2D model.
    """
    print("\n=== Initializing Shared FNO2D and Data Sources ===")

    # 1. The Shared Backbone (2D)
    # Using the FNO2D class defined previously
    shared_fno = FNO2D(modes=12, width=32).to(device)

    # 2. Define 4 Sources
    num_sources = 100

    # Generate 100 sources:
    # IV/Meas Noise scales up from 0.01 to 0.50
    # Cost scales down from 4.0 to 0.1
    sources_config = [
        (
            0.01 + (i * (0.50 - 0.01) / (num_sources - 1)), # IV Noise
            0.01 + (i * (0.30 - 0.01) / (num_sources - 1)), # Meas Noise
            max(0.1, 2.0 - (i * 1.9 / (num_sources - 1)))                         # Exponential Cost decay
        )
        for i in range(num_sources)
    ]

    experts = []
    for i, (iv, meas, cost) in enumerate(sources_config):
        print(f"Source {i+1}: IV_Noise={iv}, Meas_Noise={meas}, Cost={cost}")
        # Note: We pass the SAME model instance to all wrappers
        expert = NoisySourceWrapper2D(shared_fno, iv, meas, cost, device)
        experts.append(expert)

    return experts

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running 2D Source Selection on {device}")

    # 1. Initialize Shared Model + Wrappers
    experts = initialize_experts_2d(device)

    # 2. Generate Data (Clean Ground Truth) 64x64
    # The Experts will add 2D noise on the fly
    print("\nGenerating Ground Truth 2D Datasets...")
    train_x, train_y = generate_2d_data(num_samples=200, nx=64, steps=20, device=device)
    val_x, val_y   = generate_2d_data(num_samples=20, nx=64, steps=20, device=device)
    test_x, test_y   = generate_2d_data(num_samples=20, nx=64, steps=20, device=device)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
    # Reusing test set for val/test to save time in example
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
    #experts_config = [copy.deepcopy(experts)] * 5
    # --- 1. Softmax (Baseline - ignores budget) ---

    # --- 4. ADMM ---
    experts = initialize_experts_2d(device)
    admm_model = ADMMCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(admm_model, train_loader, val_loader, test_loader, BUDGET, device, "ADMM (Routing)")
    results['ADMM_Routing'] = {'history': history, 'time': elapsed}
    # --- 3. Aug Lagrangian ---
    experts = initialize_experts_2d(device)
    model = AugLagrangianCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "AugLag (Routing)")
    results['AugLag_Routing'] = {'history': history, 'time': elapsed}

    experts = initialize_experts_2d(device)
    model = SoftmaxCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "Softmax")
    results['Softmax'] = {'history': history, 'time': elapsed}

    # --- 2. Lagrangian ---
    experts = initialize_experts_2d(device)
    model = LagrangianCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "Lagrangian")
    results['Lagrangian'] = {'history': history, 'time': elapsed}

    experts = initialize_experts_2d(device)
    model = ImpAugLagrangianCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "ImpLagrangian")
    results['ImpLagrangian'] = {'history': history, 'time': elapsed}

    # --- Final Summary ---
    labels = list(results.keys())
    #mse_vals = [results[k]['test_mse'][-1] for k in labels]
    #cost_vals = [results[k]['cost'][-1] for k in labels]

    print("\n=== Final Test Summary (Data Source Selection) ===")
    print(f"{'Method':<20} | {'Test MSE':<10} | {'Final Cost':<10} | {'Time (s)':<10}")
    print("-" * 60)
    for k, v in results.items():
        print(k, v)
        #hist = v.get('history', v)
        #t = v['time']
        #mse = 0 #hist['test_mse'][-1]
        #cost = 0# hist['cost'][-1]
        #print(f"{k:<20} | {mse:.6f}   | {cost:.4f}     | {t:.2f}")


if __name__ == "__main__":
    main()

