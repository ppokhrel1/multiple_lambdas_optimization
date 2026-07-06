from multi_resolution import *

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import numpy as np

# Your existing NoisySourceWrapper remains unchanged
class NoisySourceWrapper(nn.Module):
    def __init__(self, shared_model, iv_noise_std, meas_noise_std, cost, device):
        super().__init__()
        self.model = shared_model
        self.iv_noise_std = iv_noise_std
        self.meas_noise_std = meas_noise_std
        self.cost = cost
        self.device = device

    def apply_noise(self, x_clean):
        # 1. Smooth Correlated Noise (IV/Drift)
        if self.iv_noise_std > 0:
            noise = torch.randn(x_clean.shape, device=self.device)
            noise_ft = torch.fft.rfft(noise, dim=-1)
            modes = 8
            mask = torch.zeros_like(noise_ft)
            mask[:, :modes] = 1.0
            noise_ft = noise_ft * mask
            iv_noise = torch.fft.irfft(noise_ft, n=x_clean.shape[-1])
            if iv_noise.std() > 1e-9:
                iv_noise = iv_noise / iv_noise.std()
            x_noisy = x_clean + iv_noise * self.iv_noise_std
        else:
            x_noisy = x_clean

        # 2. White Gaussian Noise (Measurement)
        if self.meas_noise_std > 0:
            x_noisy = x_noisy + torch.randn_like(x_clean) * self.meas_noise_std
        return x_noisy

    def forward(self, x):
        # Apply the noise to the input BEFORE the model processes it
        noisy_x = self.apply_noise(x)
        return self.model(noisy_x)


# ==========================================
# 5. Top-K Combiner System (NEW)
# ==========================================

class TopKExpertSelector:
    """Efficiently selects and weights top-k experts"""
    def __init__(self, top_k):
        self.top_k = top_k

    def __call__(self, scores):
        """
        Args:
            scores: [num_experts] or [batch_size, num_experts]
        Returns:
            indices: [top_k] or [batch_size, top_k]
            weights: [top_k] or [batch_size, top_k] (normalized)
        """
        k = min(self.top_k, scores.size(-1))
        values, indices = torch.topk(scores, k, dim=-1)
        weights = torch.softmax(values, dim=-1)
        return indices, weights

class BaseTopKCombiner(nn.Module):
    def __init__(self, experts, device, use_routing=False, top_k=10):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.register_buffer('costs', torch.tensor([e.cost for e in experts], device=device).float())
        self.use_routing = use_routing
        self.device = device
        self.top_k = top_k
        self.selector = TopKExpertSelector(top_k)

        if self.use_routing:
            self.router = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(16, len(experts))
            ).to(device)
            self.router = FNORouter1D(len(experts)).to(device)
        else:
            self.theta = nn.Parameter(torch.zeros(len(experts), device=device))


    def get_logits(self, x):
        if self.use_routing:
            # Routing networks like FNORouter1D already return [Batch, Num_Experts]
            logits = self.router(x)
            # If the router accidentally returns [Batch, 1, Num_Experts], squeeze it
            # if logits.dim() == 3 and logits.size(1) == 1:
            #     logits = logits.squeeze(1)
            return logits
        else:
            # Static theta is [Num_Experts], needs to become [Batch, Num_Experts]
            return self.theta.unsqueeze(0).expand(x.size(0), -1)

    def get_scores(self, x):
        # We return the raw logits here because the Selector handles the Softmax
        return self.get_logits(x)

    def get_weights(self, x):
        """Returns the normalized weights for the top-k experts."""
        logits = self.get_logits(x)
        return F.softmax(logits)
        # indices, weights = self.selector(logits)

        # # Create a sparse [Batch, Num_Experts] tensor where rows sum to 1.0
        # full_weights = torch.zeros(logits.shape[0], len(self.experts), device=self.device)
        # full_weights.scatter_(1, indices, weights)
        # return full_weights

    def get_cost(self, weights_or_logits):
        # If passed logits, convert to weights first
        # if weights_or_logits.shape[-1] == len(self.experts) and torch.is_floating_point(weights_or_logits):
        #     # Check if it looks like logits (doesn't sum to 1)
        #     if not torch.allclose(weights_or_logits.sum(dim=-1), torch.tensor(1.0, device=self.device), atol=1e-3):
        #         weights = self.get_weights(weights_or_logits) # This is actually 'x' in this context
        #     else:
        #         weights = weights_or_logits

        return (weights_or_logits * self.costs).sum(dim=-1).mean()

    def forward(self, x):
        # 1. Get the Top-K indices and their RE-NORMALIZED weights
        logits = self.get_logits(x)
        indices, top_k_weights = self.selector(logits)

        expert_input = x #.squeeze(1) if (self.use_routing and x.dim() == 3) else x
        output = torch.zeros_like(expert_input)

        # 2. Iterate through the Top-K slots
        for k in range(indices.shape[1]):
            expert_ids = indices[:, k]
            unique_ids = torch.unique(expert_ids)
            for expert_id in unique_ids:
                mask = (expert_ids == expert_id)
                batch_idx = torch.where(mask)[0]

                expert_out = self.experts[expert_id](expert_input[batch_idx])

                # CRITICAL: Use top_k_weights (which sum to 1) instead of the full softmax
                w = top_k_weights[batch_idx, k].view(-1, 1)
                output[batch_idx] += expert_out * w
        return output, top_k_weights

# --- Fixed Subclasses ---
class SoftmaxCombiner(BaseTopKCombiner):
    def training_step(self, x, pred, y, criterion, budget=None):
      loss = criterion(pred, y)
      weights = self.get_weights(x)
      # Reduce across the batch to get a single scalar
      current_cost = self.get_cost(weights)
      sum_constraint = torch.abs(weights.sum(dim=-1) - 1.0).mean()
      return loss, weights, current_cost.item()

class LagrangianCombiner(BaseTopKCombiner):
    def __init__(self, experts, device, use_routing=False):
        super().__init__(experts, device, use_routing)
        # Separate multipliers for the Sum-to-1 constraint and the Budget constraint
        self.lam_sum = nn.Parameter(torch.tensor(0.0, device=device))
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
        scores = self.get_scores(x)
        current_cost = self.get_cost(weights)
        budget_violation = F.relu(current_cost - budget) # Only penalize if over budget

        total_loss = mse_loss + \
                    self.lam_budget * budget_violation
                    #  self.lam_sum * sum_violation.mean()


        return total_loss, weights, current_cost.item()

class AugLagrangianCombiner(BaseTopKCombiner):
    def __init__(self, experts, device, use_routing=False):
        super().__init__(experts, device, use_routing)
        self.register_buffer('lam_sum', torch.tensor(0.0, device=device))
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
            self.lam_sum.add_(self.rho * (avg_sum - 1.0))
            self.lam_budget.add_(self.rho * F.relu(torch.tensor(avg_cost - budget)))
            # Clamp lambda to prevent exploding gradients
            self.lam_sum.clamp_(-10.0, 10.0)
            self.lam_budget.clamp_(0.0, 10.0)

class ImpAugLagrangianCombiner(AugLagrangianCombiner):
    def __init__(self, experts, device, use_routing=False,):
        # Explicitly pass top_k to the parent BaseTopKCombiner
        super().__init__(experts, device, use_routing)
        self.prev_viol = float('inf')

    def update_dual(self, avg_sum, avg_cost, budget):
        with torch.no_grad():
            sum_v = avg_sum - 1.0
            bud_v = max(0, avg_cost - budget)

            self.lam_sum.add_(self.rho * sum_v)
            self.lam_budget.add_(self.rho * bud_v)

            current_total_viol = abs(sum_v) + bud_v
            if current_total_viol > 0.01 and current_total_viol > self.prev_viol * 0.95:
                self.rho.mul_(1.2)
                self.rho.clamp_(max=50.0)
            self.prev_viol = current_total_viol

class ADMMCombiner(BaseTopKCombiner):
    def __init__(self, experts, device, use_routing=False, top_k=10):
        super().__init__(experts, device, use_routing, top_k)
        # We only need one dual variable for the Budget inequality
        self.register_buffer('u_budget', torch.tensor(0.0, device=device))
        # Start with a smaller rho for stability
        self.register_buffer('rho', torch.tensor(0.5, device=device))

    def get_weights(self, x):
        # HARD CONSTRAINT: Always project to simplex
        return project_to_simplex(self.get_logits(x))

    def training_step(self, x, pred, y, criterion, budget):
        weights = self.get_weights(x)
        mse_loss = criterion(pred, y)
        current_cost = self.get_cost(weights)

        # ADMM Residue for Budget (Inequality projection)
        # Logic: z = min(cost + u, budget). Penalty is (cost - z + u)^2
        # This only penalizes if cost + u > budget
        z_budget = torch.min(current_cost + self.u_budget, torch.tensor(budget, device=self.device))

        # We use a squared penalty on the difference between current cost and projected safe cost
        budget_penalty = (self.rho / 2) * (current_cost - z_budget + self.u_budget)**2

        return mse_loss + budget_penalty, weights, current_cost.item()

    def update_dual(self, avg_sum, avg_cost, budget):
        with torch.no_grad():
            # Update only the budget dual variable
            z_budget = torch.min(avg_cost + self.u_budget, torch.tensor(budget, device=self.device))
            self.u_budget.add_(avg_cost - z_budget)

            # Gradually increase rho to tighten the constraint, but cap it
            self.rho.mul_(1.05).clamp_(max=10.0)

# ==========================================
# 6. Updated Training Function
# ==========================================
def train_combiner(combiner, train_loader, val_loader, test_loader, budget, device, name="Model"):
    print(f"\n>>> Training {name} (Top-{combiner.top_k} 1D) | Budget: {budget} <<<")
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
    opt_dual = optim.Adam(dual_params, lr=1e-3, maximize=True) if dual_params else None
    loss_fn = PINNLoss(mse_weight=10.0, physics_weight=1e-3).to(device)

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
        epoch_weights = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt_primal.zero_grad()
            if opt_dual: opt_dual.zero_grad()

            pred, weights = combiner(bx)
            mse_val = F.mse_loss(pred, by)
            #print(weights.mean(dim=1))
            # Pass budget to the training step logic
            loss, _, current_batch_cost = combiner.training_step(bx, pred, by, loss_fn, budget)


            loss.backward()
            opt_primal.step()
            if opt_dual: opt_dual.step()

            tr_loss += loss.item()
            tr_mse += mse_val.item()
            tr_cost += current_batch_cost
            epoch_weights += weights.mean().item()


        # Calculate epoch averages for Dual Updates
        num_batches = len(train_loader)
        avg_cost = tr_cost / num_batches
        #print(epoch_weights)
        avg_sum = epoch_weights / num_batches

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
                pred, top_k_weights = combiner(bx)
                v_loss, _, _ = combiner.training_step(bx, pred, by, loss_fn, budget)
                val_loss += v_loss.item()
                val_mse += F.mse_loss(pred, by).item()

        # --- TEST ---
        ts_loss, ts_mse = 0, 0
        with torch.no_grad():
            for tx, ty in test_loader:
                tx, ty = tx.to(device), ty.to(device)
                t_pred, top_k_weights = combiner(tx)
                t_loss, _, _ = combiner.training_step(tx, t_pred, ty, loss_fn, budget)
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
    return history, elapsed_time


# ==========================================
# 7. Updated Initialization & Main
# ==========================================

def initialize_experts(device, top_k=10):
    """Initialize experts with usage-aware logging"""
    print(f"\n=== Initializing Shared FNO and {100} Data Sources (Top-{top_k} will be used) ===")

    shared_fno = FNO1D(modes=16, width=32).to(device)

    num_sources = 100
    sources_config = [
        (
            0.01 + (i * (0.50 - 0.01) / (num_sources - 1)),
            0.01 + (i * (0.30 - 0.01) / (num_sources - 1)),
            max(0.1, 2.0 - (i * 1.9 / (num_sources - 1)))
        )
        for i in range(num_sources)
    ]

    experts = []
    for i, (iv, meas, cost) in enumerate(sources_config):
        #if i < 5 or i % 20 == 0:  # Print first 5 and every 20th
        print(f"Source {i+1:03d}: IV_Noise={iv}, Meas_Noise={meas}, Cost={cost}")
        expert = NoisySourceWrapper(shared_fno, iv, meas, cost, device)
        experts.append(expert)

    return experts


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")



    # Generate data
    print("\nGenerating Ground Truth Datasets...")
    train_x, train_y = generate_data(num_samples=200, nx=64, device=device)
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)

    val_x, val_y = generate_data(num_samples=20, nx=64, device=device)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32)

    test_x, test_y = generate_data(num_samples=20, nx=64, device=device)
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

    TOP_K = 10  # **KEY PARAMETER**: Use only top-10 experts per forward pass

    results = {}
    print(f"Generating data...")
    #expert_config = [copy.deepcopy(experts)]*5
    # Train all methods with top-k selection
    experts = initialize_experts(device)
    model = ADMMCombiner(experts, device, use_routing=True, ).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "ADMM (Routing)")
    results['ADMM_TopK_Routing'] = {'history': history, 'time': elapsed}

    experts = initialize_experts(device)
    model = LagrangianCombiner(experts, device, use_routing=True,).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "Lagrangian")
    results['Lagrangian_TopK'] = {'history': history, 'time': elapsed}

    experts = initialize_experts(device)
    model = SoftmaxCombiner(experts, device, use_routing=True,).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "Softmax")
    results['Softmax_TopK'] = {'history': history, 'time': elapsed}

    experts = initialize_experts(device)
    model = ImpAugLagrangianCombiner(experts, device, use_routing=True,).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "ImpLagrangian")
    results['ImpLagrangian_TopK'] = {'history': history, 'time': elapsed}

    experts = initialize_experts(device)
    model = AugLagrangianCombiner(experts, device, use_routing=True, ).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "AugLag (Routing)")
    results['AugLag_TopK_Routing'] = {'history': history, 'time': elapsed}

    # Summary
    print("\n=== Final Test Summary (Top-K Selection) ===")
    print(f"{'Method':<20} | {'Test MSE':<10} | {'Final Cost':<10} | {'Time (s)':<10}")
    print("-" * 60)
    for k, v in results.items():
        hist = v['history']
        t = v['time']
        mse = hist['test_mse'][-1]
        cost = hist['cost'][-1]
        print(f"{k:<20} | {mse:.6f}   | {cost:.4f}     | {t:.2f}")


if __name__ == "__main__":
    main()


