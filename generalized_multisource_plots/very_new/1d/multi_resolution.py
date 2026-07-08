import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader, TensorDataset
import random
import copy
import sys
import time


random.seed(9)

def project_to_simplex(v, z=1.0):
    """
    Projects a vector v onto the simplex: sum(w) = z, w >= 0.
    """
    n_features = v.shape[-1]
    v_sorted, _ = torch.sort(v, dim=-1, descending=True)
    v_cumsum = torch.cumsum(v_sorted, dim=-1)

    # Indices for the formula
    indices = torch.arange(1, n_features + 1, device=v.device).float()

    # Find the rho (threshold)
    rho_mask = v_sorted + (z - v_cumsum) / indices > 0
    # Find the last index where the condition is true
    rho = torch.sum(rho_mask, dim=-1, keepdim=True) - 1

    # Get the max value of the thresholding
    theta = (v_cumsum.gather(-1, rho.long()) - z) / (rho.float() + 1)
    return torch.clamp(v - theta, min=0)


class FNORouter1D(nn.Module):
    def __init__(self, num_experts, modes=16, width=32):
        super().__init__()
        self.modes = modes
        self.width = width
        self.fc0 = nn.Linear(1, width)

        # Spectral layers from your FNO1D logic
        self.conv0 = SpectralConv1d(width, width, modes)
        self.w0 = nn.Conv1d(width, width, 1)

        # Output head: Map spectral features to expert weights
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, num_experts)

    def forward(self, x):
        # x shape: [Batch, Length]
        if x.dim() == 3:
            x = x.squeeze(1) # Remove channel if present -> [Batch, Length]
        x = x.unsqueeze(-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        # Spectral pass
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        # Global Pooling to get a single vector per batch item
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        x = F.gelu(self.fc1(x))
        logits = self.fc2(x)
        return logits # [Batch, num_experts]


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 1. Physics: 1D Burgers' Equation
# ==========================================
import torch
import math

class Burgers1D:
    def __init__(self, nx=128, t_max=1.0, nu=0.01, device='cpu'):
        self.nx = nx
        self.t_max = t_max
        self.nu = nu
        self.device = device
        self.x = torch.linspace(0, 2*math.pi, nx, device=device)[:-1]
        self.dx = self.x[1] - self.x[0]
        self.k = torch.fft.fftfreq(nx-1, d=1/(nx-1), device=device) * 2 * math.pi * (1/ (2*math.pi)) * (nx-1)
        self.k = self.k.to(device)
        # Dealiasing filter (2/3 rule)
        self.dealias_filter = (torch.abs(self.k) < (nx-1) * (2/3) / 2).float()

    def solve_trajectory(self, u0, steps=100):
        dt = self.t_max / steps
        u = u0.clone()
        trajectory = [u.clone()]

        k_sq = self.k ** 2
        ik = 1j * self.k

        # Crank-Nicolson factors
        factor_lhs = 1 + 0.5 * dt * self.nu * k_sq
        factor_rhs = 1 - 0.5 * dt * self.nu * k_sq

        for _ in range(steps):
            # Dealiased FFT
            u_hat = torch.fft.fft(u) * self.dealias_filter

            # Compute derivative with dealiasing
            ux = torch.fft.ifft(ik * u_hat).real

            # Nonlinear term with proper handling
            nonlinear = 0.5 * (u ** 2)  # Use u^2/2 to avoid aliasing in conservative form
            nonlinear_hat = torch.fft.fft(nonlinear) * self.dealias_filter

            # Update in Fourier space
            u_hat_new = (u_hat * factor_rhs - dt * nonlinear_hat) / factor_lhs

            # Inverse transform with numerical safety
            u = torch.fft.ifft(u_hat_new).real

            # Stability clamping to prevent blow-up (very rare but possible)
            u = torch.clamp(u, min=-10.0, max=10.0)

            trajectory.append(u.clone())

        return torch.stack(trajectory)


def generate_data(num_samples=100, nx=128, steps=20, device='cpu', freq_range=(1, 8)):
    """
    Generates challenging Burgers equation data with complex multi-scale interactions.
    Features:
    - Multi-frequency wave packets with random amplitudes/phases
    - Shock-forming initial conditions
    - Variable viscosity (higher Reynolds numbers)
    - Random perturbations and sharp gradients
    - Multiple generation modes for diversity
    """
    if num_samples > 10:
        print(f"Generating {num_samples} challenging trajectories at res {nx} with enhanced dynamics...")

    data_x, data_y = [], []

    for i in range(num_samples):
        # Randomize parameters per sample for diversity
        t_max = 0.3 + 0.4 * torch.rand(1, device=device).item()  # [0.3, 0.7]
        nu = 0.001 + 0.004 * torch.rand(1, device=device).item()  # [0.001, 0.005] (lower viscosity = more turbulent)

        physics = Burgers1D(nx=nx, t_max=t_max, nu=nu, device=device)

        # Multiple complex generation modes
        mode = torch.randint(0, 4, (1,)).item()  # 4 different modes

        if mode == 0:
            # Mode 0: Multi-frequency wave packet with random phases/amplitudes
            n_waves = torch.randint(2, 5, (1,)).item()
            u0 = torch.zeros_like(physics.x)
            for j in range(n_waves):
                freq = torch.randint(freq_range[0], freq_range[1] + 2, (1,)).item()
                amp = 0.3 + 0.7 * torch.rand(1, device=device).item()
                phase = 2 * math.pi * torch.rand(1, device=device).item()
                offset = 0.5 * math.pi * torch.rand(1, device=device).item()
                u0 += amp * torch.sin(freq * physics.x + phase) * torch.exp(-0.2 * (physics.x - offset)**2)

        elif mode == 1:
            # Mode 1: Shock-forming initial condition (N-wave)
            x_center = physics.x[torch.randint(0, len(physics.x), (1,)).item()]
            steepness = 3.0 + 2.0 * torch.rand(1, device=device).item()
            u0 = torch.tanh(steepness * (physics.x - x_center))
            # Add perturbations
            u0 += 0.1 * torch.randn_like(physics.x) * torch.sin(5 * physics.x)

        elif mode == 2:
            # Mode 2: Turbulent-like random Fourier series
            n_modes = min(nx // 4, 20)
            u0 = torch.zeros_like(physics.x)
            k_vec = torch.arange(1, n_modes+1, device=device).float()
            amplitudes = torch.randn_like(k_vec) / (k_vec ** (1.0 + 0.5 * torch.rand(1).item()))  # Power law spectrum
            phases = 2 * math.pi * torch.rand(n_modes, device=device)
            for k, amp, phase in zip(k_vec, amplitudes, phases):
                u0 += amp * torch.sin(k * physics.x + phase)
            # Normalize
            u0 = u0 / (torch.std(u0) + 1e-6) * (0.5 + 0.5 * torch.rand(1, device=device).item())

        else:
            # Mode 3: Interacting soliton-like structures
            u0 = torch.zeros_like(physics.x)
            n_peaks = torch.randint(2, 4, (1,)).item()
            for j in range(n_peaks):
                center = 0.5 * math.pi * (j + 1) + torch.rand(1, device=device).item()
                width = 0.1 + 0.2 * torch.rand(1, device=device).item()
                amp = 0.5 + 0.5 * torch.rand(1, device=device).item()
                u0 += amp * torch.exp(-((physics.x - center) / width) ** 2)
            # Add oscillatory background
            u0 += 0.2 * torch.sin(3 * physics.x + torch.rand(1, device=device).item())

        # Add deterministic perturbation for sharp gradients
        if i % 3 == 0:
            # Add high-frequency noise component
            noise_freq = freq_range[1] + torch.randint(1, 4, (1,)).item()
            u0 += 0.05 * torch.sin(noise_freq * physics.x + 2 * math.pi * torch.rand(1, device=device).item())

        # Final safety normalization (prevent overly large initial conditions)
        u0 = u0 / (torch.max(torch.abs(u0)) + 1e-6) * (0.8 + 0.4 * torch.rand(1, device=device).item())

        # Solve trajectory
        traj = physics.solve_trajectory(u0, steps=steps)

        # Additional safety check for NaNs/Infs
        if not torch.isfinite(traj).all():
            print(f"Warning: Non-finite values detected in sample {i}, regenerating...")
            # Fallback to simpler initial condition
            freq = torch.randint(freq_range[0], freq_range[1], (1,)).item()
            u0 = torch.sin(freq * physics.x)
            traj = physics.solve_trajectory(u0, steps=steps)

        data_x.append(traj[:-1])
        data_y.append(traj[1:])

    return torch.cat(data_x), torch.cat(data_y)


# ==========================================
# 2. Expert Models (CNN, FNO, DeepONet, FD)
# ==========================================

# --- A. Standard CNN (Baseline Expert) ---
class Expert1D(nn.Module):
    def __init__(self, hidden_channels=32, resolution=32, base_res=32):
        super().__init__()
        dilation = max(1, resolution // base_res)
        self.net = nn.Sequential(
            # FIX: Added dilation=dilation argument here
            nn.Conv1d(1, hidden_channels, 5, padding=2*dilation, dilation=dilation, padding_mode='circular'),
            nn.Tanh(),
            nn.Conv1d(hidden_channels, hidden_channels, 5, padding=2, padding_mode='circular'),
            nn.Tanh(),
            nn.Conv1d(hidden_channels, 1, 5, padding=2, padding_mode='circular')
        )

    def forward(self, x):
        return x.unsqueeze(1) + self.net(x.unsqueeze(1))

# --- B. Finite Difference Expert ---
class FiniteDifferenceExpert(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.register_buffer('k_ux', torch.tensor([[[-0.5, 0.0, 0.5]]]))
        self.register_buffer('k_uxx', torch.tensor([[[1.0, -2.0, 1.0]]]))

        self.mlp = nn.Sequential(
            nn.Conv1d(3, hidden_dim, 1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, 1, 1)
        )

    def forward(self, x):
        u = x.unsqueeze(1)
        u_padded = F.pad(u, (1, 1), mode='circular')
        ux = F.conv1d(u_padded, self.k_ux)
        uxx = F.conv1d(u_padded, self.k_uxx)
        features = torch.cat([u, ux, uxx], dim=1)
        dt_pred = self.mlp(features)
        return u + dt_pred

# --- C. Scale Invariant Fourier Neural Network (FNO) ---
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat))

    def complex_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.complex_mul1d(x_ft[:, :, :self.modes], self.weights)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1D(nn.Module):
    def __init__(self, modes=16, width=32):
        super().__init__()
        self.modes = modes
        self.width = width
        self.fc0 = nn.Linear(1, width)
        self.conv0 = SpectralConv1d(width, width, modes)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = x.unsqueeze(-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.squeeze(-1) + grid.squeeze(-1)

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx

# --- D. DeepONet ---
class DeepONet(nn.Module):
    def __init__(self, num_sensors=128, branch_dim=64, trunk_dim=64):
        super().__init__()
        self.num_sensors = num_sensors
        self.branch = nn.Sequential(
            nn.Linear(num_sensors, 128),
            nn.Tanh(),
            nn.Linear(128, branch_dim)
        )
        self.trunk = nn.Sequential(
            nn.Conv1d(1, 64, 1),
            nn.Tanh(),
            nn.Conv1d(64, trunk_dim, 1)
        )
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, u):
        if u.shape[1] != self.num_sensors:
             u_in = F.interpolate(u.unsqueeze(1), size=self.num_sensors, mode='linear').squeeze(1)
        else:
             u_in = u
        B = self.branch(u_in)
        batch, n = u.shape
        x_grid = torch.linspace(-1, 1, n, device=u.device).view(1, 1, n).repeat(batch, 1, 1)
        T = self.trunk(x_grid)
        B = B.unsqueeze(-1)
        out = (B * T).sum(dim=1) + self.bias
        return u + out


import torch.nn.functional as F

class PINNLoss(nn.Module):
    def __init__(self, mse_weight=10.0, physics_weight=1.0, nu=0.01, beta=1.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.physics_weight = physics_weight
        self.nu = nu
        self.beta = beta

        # Finite difference kernels for spatial derivatives
        self.register_buffer('k_ux', torch.tensor([[[-0.5, 0.0, 0.5]]]))
        self.register_buffer('k_uxx', torch.tensor([[[1.0, -2.0, 1.0]]]))

    def forward(self, pred, target):
        # 1. Standard MSE Loss
        mse = F.mse_loss(pred, target)

        # 2. Physics Loss (Residual of u_t + beta*u_x - nu*u_xx = 0)
        # Assuming pred is the state at t+dt and 'target' is the ground truth
        # Here we calculate the spatial residual of the prediction
        u = pred.unsqueeze(1) if pred.dim() == 2 else pred
        dx = 1.0 / u.shape[-1]

        u_padded = F.pad(u, (1, 1), mode='circular')
        ux = F.conv1d(u_padded, self.k_ux) / dx
        uxx = F.conv1d(u_padded, self.k_uxx) / (dx**2)

        # Example: Stationary or implicit residual
        # For a simple transport-diffusion:
        residual = self.beta * ux - self.nu * uxx
        physics_loss = torch.mean(residual**2)

        return (self.mse_weight * mse) + (self.physics_weight * physics_loss)



# --- Wrapper to handle Multi-Resolution Inputs ---
class MultiResExpertWrapper(nn.Module):
    def __init__(self, model, native_res, cost):
        super().__init__()
        self.model = model
        self.native_res = native_res
        self.cost = cost

    def forward(self, x):
        target_res = x.shape[-1]
        if target_res != self.native_res:
            x_in = F.interpolate(x.unsqueeze(1), size=self.native_res, mode='linear', align_corners=False)
        else:
            x_in = x.unsqueeze(1)
        out_native = self.model(x_in.squeeze(1))
        if out_native.dim() == 2:
            out_native = out_native.unsqueeze(1)
        if target_res != self.native_res:
            out_res = F.interpolate(out_native, size=target_res, mode='linear', align_corners=False).squeeze(1)
        else:
            out_res = out_native.squeeze(1)
        return out_res

class Router(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, num_experts)
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1))

# ==========================================
# 3. Combiners
# ==========================================


class BaseCombiner(nn.Module):
    def __init__(self, experts, device, use_routing=False):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.use_routing = use_routing
        self.device = device
        self.register_buffer('costs', torch.tensor([e.cost for e in experts], device=device))

        if self.use_routing:
            self.router = Router(len(experts)).to(device)
            # Learnable parameters for the router
            self.router_params = list(self.router.parameters())
        else:
            # Learnable weights (logits)
            self.theta = nn.Parameter(torch.zeros(len(experts)))

    def get_logits(self, x):
        if self.use_routing:
            return self.router(x)
        else:
            return self.theta.unsqueeze(0).expand(x.shape[0], -1)
    def get_weights(self, x):
      logits = self.get_logits(x)
      return F.softmax(logits, dim=-1)

    def get_cost(self, weights):
        # weights: [Batch, Experts]
        # self.costs: [Experts]
        return torch.matmul(weights, self.costs).mean()

    def forward(self, x):
        logits = self.get_logits(x)
        weights = F.softmax(logits, dim=-1)

        # Get expert outputs
        expert_outputs = [expert(x) for expert in self.experts]
        stacked_outputs = torch.stack(expert_outputs, dim=-1)  # [B, N, E]

        # Weighted combination
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
        weights_expanded = weights.unsqueeze(1)  # [B, 1, E]
        combined = (stacked_outputs * weights_expanded).sum(dim=-1)

        return combined #, weights

# 1. Softmax (Baseline - no constraint enforcement needed)
class SoftmaxCombiner(BaseCombiner):
    def training_step(self, x, pred, y, criterion, budget=None):
      loss = criterion(pred, y)
      weights = self.get_weights(x)
      current_cost = self.get_cost(weights)
      # Reduce across the batch to get a single scalar
      sum_constraint = torch.abs(weights.sum(dim=-1) - 1.0).mean()
      return loss, weights, current_cost.item()

class LagrangianCombiner(BaseCombiner):
    def __init__(self, experts, device, use_routing=False):
        super().__init__(experts, device, use_routing)
        # Multipliers for Sum and Budget
        self.lam_sum = nn.Parameter(torch.tensor(0.0, device=device))
        self.lam_budget = nn.Parameter(torch.tensor(0.0, device=device))

    def training_step(self, x, pred, y, criterion, budget):
        weights = project_to_simplex(self.get_logits(x))
        mse_loss = criterion(pred, y)

        sum_viol = weights.sum(dim=-1) - 1.0
        current_cost = self.get_cost(weights)
        budget_viol = F.relu(current_cost - budget) # Inequality
        #print(budget_viol, self.lam_budget)
        total_loss = mse_loss + self.lam_budget * budget_viol #self.lam_sum * sum_viol.mean() +
        return total_loss, weights, current_cost.item()

class AugLagrangianCombiner(BaseCombiner):
    def __init__(self, experts, device, use_routing=False):
        super().__init__(experts, device, use_routing)
        self.register_buffer('lam_sum', torch.tensor(0.0, device=device))
        self.register_buffer('lam_budget', torch.tensor(0.0, device=device))
        self.register_buffer('rho', torch.tensor(1.0, device=device))

    def training_step(self, x, pred, y, criterion, budget):
        weights = project_to_simplex(self.get_logits(x))
        mse_loss = criterion(pred, y)

        sum_viol = weights.sum(dim=-1) - 1.0
        current_cost = self.get_cost(weights)
        budget_viol = F.relu(current_cost - budget)

        penalty = (self.lam_budget * budget_viol + (self.rho/2) * (budget_viol**2))
        #(self.lam_sum * sum_viol.mean() + (self.rho/2) * (sum_viol**2).mean()) + \


        return mse_loss + penalty, weights, current_cost.item()

    def update_dual(self, avg_sum, avg_cost, budget):
        with torch.no_grad():
            # Update Sum (Equality)
            self.lam_sum.add_(self.rho * (avg_sum - 1.0))
            
            # Update Budget (Inequality) - Allow relaxation!
            # Note: Do not use .add_ with max(0), calculate the step and clamp the result
            new_lam_budget = self.lam_budget + self.rho * (avg_cost - budget)
            self.lam_budget.copy_(torch.clamp(new_lam_budget, min=0.0))

class ADMMCombiner(BaseCombiner):
    def __init__(self, experts, device, use_routing=False):
        super().__init__(experts, device, use_routing)
        self.register_buffer('u_sum', torch.tensor(0.0, device=device))
        self.register_buffer('u_budget', torch.tensor(0.0, device=device))
        self.register_buffer('rho', torch.tensor(2.0, device=device))

    def training_step(self, x, pred, y, criterion, budget):
        weights = project_to_simplex(self.get_logits(x))
        mse_loss = criterion(pred, y)

        avg_sum = weights.sum(dim=-1).mean()
        current_cost = self.get_cost(weights)

        # Simplex Equality Penalty
        sum_penalty = (self.rho / 2) * (avg_sum - 1.0 + self.u_sum)**2

        # Budget Inequality Projection
        z_budget = torch.min(current_cost + self.u_budget, torch.tensor(budget, device=self.device))
        budget_penalty = (self.rho / 2) * (current_cost - z_budget + self.u_budget)**2

        return mse_loss + budget_penalty, weights, current_cost.item()

    def update_dual(self, avg_sum, avg_cost, budget):
        with torch.no_grad():
            self.u_sum.add_(avg_sum - 1.0)
            z_budget = torch.min(avg_cost + self.u_budget, torch.tensor(budget, device=self.device))
            self.u_budget.add_(avg_cost - z_budget)

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




# ==========================================
# 4. Utilities & Training
# ==========================================

def evaluate_standalone(model, loader, device):
    model.eval()
    total_mse = 0
    steps = 0
    with torch.no_grad():
        for bx, by in loader:
            # Handle models that return tuples (Combiners) and those that don't (Experts)
            out = model(bx)
            pred = out[0] if isinstance(out, tuple) else out

            total_mse += F.mse_loss(pred, by).item()
            steps += 1
    return total_mse / steps

def train_combiner(combiner, train_loader, val_loader, test_loader, budget, device, name="Model"):
    print(f"\n>>> Training {name} <<<")
    start_time = time.time()

    # 1. Separate Parameters for Two Optimizers (Primal and Dual)
    model_params, router_params, dual_params = [], [], []
    for pname, p in combiner.named_parameters():
        if 'lam' in pname:
            dual_params.append(p)
        elif 'router' in pname:
            router_params.append(p)
        else:
            model_params.append(p)

    # 2. Initialize Optimizers
    # Primal: Adam (Minimizes Loss)
    ETA_THETA = 1e-4   # aligned with manuscript Sec 5.2 (eta_theta=1e-3); 1e-5 left model+router under-trained
    ETA_LAMBDA = 1e-6  # == ETA_THETA**2 (Assumption 4); 1e-7 froze the router near uniform -> routing lost to baselines
    opt_primal = optim.Adam([
        {'params': model_params,  'lr': ETA_THETA},
        {'params': router_params, 'lr': ETA_LAMBDA},
    ], weight_decay=1e-7)
    sched_primal = optim.lr_scheduler.StepLR(opt_primal, step_size=150, gamma=0.5)  # diminishing-step schedule

    # Dual: SGD with maximize=True (Maximizes Lagrangian w.r.t Duals)
    opt_dual = None
    if dual_params:
        try:
             opt_dual = optim.Adam(dual_params, lr=1e-2, weight_decay=1e-8, maximize=True)
        except TypeError:
             opt_dual = optim.Adam(dual_params, lr=-1e-2, weight_decay=1e-8, )

    loss_fn = PINNLoss(mse_weight=10.0, physics_weight=1e-3).to(device)

    # Track stats
    history = {
        'train_loss': [], 'val_loss': [], 'test_loss': [],
        'train_mse': [], 'val_mse': [], 'test_mse': [],
        'cost': []
    }

    # Run for 50 epochs (enough to see convergence)
    for epoch in range(500):
        if epoch > 0: sched_primal.step()  # per-epoch LR decay
        # --- TRAIN ---
        combiner.train()
        epoch_train_loss = 0
        epoch_train_mse = 0
        epoch_cost = 0
        tr_sum = 0
        for bx, by in train_loader:
            # Zero Grads
            opt_primal.zero_grad()
            if opt_dual: opt_dual.zero_grad()

            # Forward & Loss
            pred = combiner(bx)
            mse = F.mse_loss(pred, by) # Raw MSE
            loss, weights, cost = combiner.training_step(bx, pred, by, loss_fn, budget) # Augmented/Lagrangian Loss

            # Backward
            loss.backward()

            # Primal Step (Descent)
            torch.nn.utils.clip_grad_norm_(combiner.parameters(), 1.0)
            opt_primal.step()

            # Dual Step (Ascent) - For LagrangianCombiner
            if opt_dual:
                opt_dual.step()
                #with torch.no_grad():
                #    for p in dual_params: p.clamp_(min=0.0)

            epoch_train_loss += loss.item()
            epoch_train_mse += mse.item()
            tr_sum += weights.sum(dim=-1).mean().item()
            epoch_cost += cost

        num_batches = len(train_loader)
        avg_sum = tr_sum / num_batches
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_mse = epoch_train_mse / len(train_loader)
        avg_cost = epoch_cost / len(train_loader)
        if hasattr(combiner, 'update_dual'):
            if isinstance(combiner, (ADMMCombiner, ImpAugLagrangianCombiner)):
                combiner.update_dual(avg_sum, avg_cost, budget)
            else:
                # Basic Lagrangian update for budget
                with torch.no_grad():
                    #combiner.lam_sum.add_(0.1 * (avg_sum - 1.0))
                    combiner.lam_budget.add_(0.1 * max(0, avg_cost - budget))

        # --- VALIDATION ---
        combiner.eval()
        val_loss = 0
        val_mse = 0

        # Track Expert Weights on Validation Set
        total_w = torch.zeros(len(combiner.experts), device=device)
        total_samples = 0

        with torch.no_grad():
            for bx, by in val_loader:
                pred = combiner(bx)
                v_loss, _, _ = combiner.training_step(bx, pred, by, loss_fn, budget)
                val_loss += v_loss.item()
                val_mse += loss_fn(pred, by).item()

                # Capture weights
                w = combiner.get_weights(bx) # [Batch, Experts] or [Experts]
                if w.dim() == 1:
                    total_w += w * bx.size(0)
                else:
                    total_w += w.sum(dim=0)
                total_samples += bx.size(0)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mse = val_mse / len(val_loader)

        # Compute Average Weights
        avg_weights = total_w / total_samples
        w_str = "[" + ", ".join([f"{v}" for v in avg_weights.cpu().numpy()]) + "]"

        # --- TEST ---
        test_loss = 0
        test_mse = 0
        with torch.no_grad():
            for bx, by in test_loader:
                pred = combiner(bx)
                t_loss, _, _ = combiner.training_step(bx, pred, by, loss_fn, budget)
                test_loss += t_loss.item()
                test_mse += loss_fn(pred, by).item()
        avg_test_loss = test_loss / len(test_loader)
        avg_test_mse = test_mse / len(test_loader)


        # Log History
        history['train_loss'].append(avg_train_loss)
        history['train_mse'].append(avg_train_mse)
        history['val_loss'].append(avg_val_loss)
        history['val_mse'].append(avg_val_mse)
        history['test_loss'].append(avg_test_loss)
        history['test_mse'].append(avg_test_mse)
        history['cost'].append(avg_cost)

        # Determine Lambda/Dual Value for Printing
        lam_str = ""
        if hasattr(combiner, 'lam'):
            lam_str = f" | Lam {combiner.lam.item()}"
        elif hasattr(combiner, 'u'):
            lam_str = f" | Dual U {combiner.u.item()}"

        # Print with Weights
        print(f"Ep {epoch}: TrMSE {avg_train_mse} | ValMSE {avg_val_mse} | TestMSE {avg_test_mse} | Cost {avg_cost}{lam_str} | W {w_str} | TrLoss {avg_train_loss} | ValLoss {avg_val_loss} | TestLoss {avg_test_loss}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    return history, elapsed_time


def train_experts(device):
    resolutions = [32, 64, 128]
    costs = [1.0, 2.0, 4.0]
    experts = []

    # Dictionary to store expanded pre-training metrics
    pretrain_history = {
        res: {
            'train_loss': [], 'train_mse': [],
            'val_loss': [], 'val_mse': []
        } for res in resolutions
    }

    print("Pre-training Experts...")
    for res, cost in zip(resolutions, costs):
        # 1. Generate Data for this specific resolution
        train_x, train_y = generate_data(num_samples=100, nx=res, device=device)
        val_x, val_y = generate_data(num_samples=30, nx=res, device=device)

        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=16, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=16)

        model = Expert1D(hidden_channels=32, resolution=res, base_res=32).to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        loss_fn = PINNLoss(mse_weight=1.0, physics_weight=1e-3).to(device)

        # 2. Training Loop
        epochs = 30
        for epoch in range(epochs):
            model.train()
            epoch_tr_loss = 0
            epoch_tr_mse = 0

            for bx, by in train_loader:
                optimizer.zero_grad()
                pred = model(bx).squeeze(1)

                # Full PINN Loss (Physics + MSE)
                loss = loss_fn(pred, by)
                # Raw MSE for tracking
                mse = F.mse_loss(pred, by)

                loss.backward()
                optimizer.step()

                epoch_tr_loss += loss.item()
                epoch_tr_mse += mse.item()

            # 3. Validation
            model.eval()
            epoch_val_loss = 0
            epoch_val_mse = 0

            with torch.no_grad():
                for v_bx, v_by in val_loader:
                    v_pred = model(v_bx).squeeze(1)

                    v_loss = loss_fn(v_pred, v_by)
                    v_mse = F.mse_loss(v_pred, v_by)

                    epoch_val_loss += v_loss.item()
                    epoch_val_mse += v_mse.item()

            # Average Metrics
            avg_tr_loss = epoch_tr_loss / len(train_loader)
            avg_tr_mse = epoch_tr_mse / len(train_loader)
            avg_vl_loss = epoch_val_loss / len(val_loader)
            avg_vl_mse = epoch_val_mse / len(val_loader)

            # Record metrics
            pretrain_history[res]['train_loss'].append(avg_tr_loss)
            pretrain_history[res]['train_mse'].append(avg_tr_mse)
            pretrain_history[res]['val_loss'].append(avg_vl_loss)
            pretrain_history[res]['val_mse'].append(avg_vl_mse)

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"[Res {res:3}] Ep {epoch:2}: "
                      f"TrLoss: {avg_tr_loss:.5f} | TrMSE: {avg_tr_mse:.5f} | "
                      f"ValLoss: {avg_vl_loss:.5f} | ValMSE: {avg_vl_mse:.5f}")

        experts.append(MultiResExpertWrapper(model, res, cost))

    return experts, [str(a) for a in resolutions], pretrain_history


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    experts, expert_names, _ = train_experts(device)
    for e in experts:
        e.eval()
        #for p in e.parameters(): p.requires_grad = False


    print("Generating Datasets...")
    # 1. Train: Standard Frequencies (1-3)
    train_x, train_y = generate_data(num_samples=500, nx=128, device=device, freq_range=(1, 3))
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)

    # 2. Validation: Same as Train (1-3)
    val_x, val_y = generate_data(num_samples=200, nx=128, device=device, freq_range=(1, 3))
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32)

    # 3. Test: Higher Frequencies (3-6) -> Out-of-Distribution / Harder
    test_x, test_y = generate_data(num_samples=200, nx=128, device=device, freq_range=(2, 4))
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



    print("\n=== Final Test Summary (High Freq Data) ===")
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

