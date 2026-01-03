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

    def solve_trajectory(self, u0, steps=100):
        dt = self.t_max / steps
        u = u0.clone()
        trajectory = [u.clone()]

        k_sq = self.k ** 2
        ik = 1j * self.k
        factor_lhs = 1 + 0.5 * dt * self.nu * k_sq
        factor_rhs = 1 - 0.5 * dt * self.nu * k_sq

        for _ in range(steps):
            u_hat = torch.fft.fft(u)
            ux = torch.fft.ifft(ik * u_hat).real
            nonlinear = u * ux
            nonlinear_hat = torch.fft.fft(nonlinear)
            u_hat_new = (u_hat * factor_rhs - dt * nonlinear_hat) / factor_lhs
            u = torch.fft.ifft(u_hat_new).real
            trajectory.append(u.clone())

        return torch.stack(trajectory)

def generate_data(num_samples=100, nx=128, steps=50, device='cpu', freq_range=(1, 4)):
    """
    Generates data with specific frequency range.
    freq_range: Tuple (min_freq, max_freq_exclusive)
    """
    if num_samples > 10:
        print(f"Generating {num_samples} trajectories at res {nx} with freq {freq_range}...")

    physics = Burgers1D(nx=nx, t_max=0.5, nu=0.01, device=device)
    data_x, data_y = [], []
    for _ in range(num_samples):
        a = torch.rand(1, device=device)
        offset = torch.rand(1, device=device)
        freq = torch.randint(freq_range[0], freq_range[1], (1,)).item()
        u0 = -torch.sin(physics.x + offset) + 0.5 * torch.sin(freq * physics.x + a)
        traj = physics.solve_trajectory(u0, steps=steps)
        data_x.append(traj[:-1])
        data_y.append(traj[1:])
    return torch.cat(data_x), torch.cat(data_y)




# ==========================================
# 2. Expert Models (CNN, FNO, DeepONet, FD)
# ==========================================

# --- A. Standard CNN (Baseline Expert) ---
class Expert1D(nn.Module):
    def __init__(self, hidden_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_channels, 5, padding=2, padding_mode='circular'),
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
    def __init__(self, experts, budget, device, use_routing=False):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.register_buffer('costs', torch.tensor([e.cost for e in experts], device=device))
        self.budget = budget
        self.use_routing = use_routing
        self.device = device

        if self.use_routing:
            self.router = Router(len(experts)).to(device)
        else:
            self.theta = nn.Parameter(torch.zeros(len(experts)))

    def get_weights(self, x):
        if self.use_routing:
            logits = self.router(x)
            return F.softmax(logits, dim=-1)
        else:
            return F.softmax(self.theta, dim=0)

    def get_cost(self, weights):
        if weights.dim() == 2:
            batch_costs = torch.matmul(weights, self.costs)
            return batch_costs.mean()
        else:
            return torch.dot(weights, self.costs)

    def forward(self, x):
        w = self.get_weights(x)
        outputs = torch.stack([e(x) for e in self.experts], dim=-1)
        if w.dim() == 1:
            w_exp = w.view(1, 1, -1)
        else:
            w_exp = w.unsqueeze(1)
        return (outputs * w_exp).sum(dim=-1)

class SoftmaxCombiner(BaseCombiner):
    def training_step(self, x, pred, y, criterion):
        loss = criterion(pred, y)
        w = self.get_weights(x)
        cost = self.get_cost(w)
        return loss, cost.item(), 0.0

class LagrangianCombiner(BaseCombiner):
    """
    Standard Lagrangian: Primal-Dual Optimization.
    Dual variable `lam` is a Parameter, optimized via gradient ascent (using a 2nd optimizer).
    """
    def __init__(self, experts, budget, device, use_routing=False):
        super().__init__(experts, budget, device, use_routing)
        # Using Parameter to allow optimizer to update it
        self.lam = nn.Parameter(torch.tensor(0.0, device=device))
        self.loss_scale = 10.0

    def training_step(self, x, pred, y, criterion):
        mse = criterion(pred, y)
        w = self.get_weights(x)
        cost = self.get_cost(w)
        viol = F.relu((cost - self.budget) / self.budget)

        # Loss for Primal Optimizer: minimize (MSE + lambda * viol)
        # Gradient for Dual Optimizer (ascent): maximize (lambda * viol) -> which is same as minimizing (-lambda * viol)
        # However, standard practice is to use one loss and have dual optimizer maximize it.
        loss = (mse * self.loss_scale) + (self.lam * viol)
        return loss, cost.item(), viol.item()

class AugLagrangianCombiner(BaseCombiner):
    def __init__(self, experts, budget, device, use_routing=False):
        super().__init__(experts, budget, device, use_routing)
        self.register_buffer('lam', torch.tensor(0.0, device=device))
        self.register_buffer('rho', torch.tensor(1.0, device=device)) # Start with small penalty
        self.loss_scale = 10.0

        # Track previous violation to decide if we need to increase rho
        self.prev_viol = float('inf')

    def training_step(self, x, pred, y, criterion):
        mse = criterion(pred, y)
        w = self.get_weights(x)
        cost = self.get_cost(w)
        viol = F.relu((cost - self.budget) / self.budget)

        # Augmented Lagrangian Formula:
        # L = MSE + lambda * viol + (rho / 2) * viol^2
        lag_term = self.lam * viol
        penalty_term = (self.rho / 2) * (viol ** 2)

        total_loss = (mse * self.loss_scale) + lag_term + penalty_term
        return total_loss, cost.item(), viol.item()

    def update_dual(self, viol):
        with torch.no_grad():
            # 1. Update Lambda (Dual Ascent)
            self.lam.add_(self.rho * viol)
            self.lam.clamp_(min=0.0, max=100.0)

            # 2. Update Rho (Adaptive Penalty)
            # If violation is not decreasing (e.g., > 90% of previous), increase penalty
            if viol > 1e-3 and viol > 0.9 * self.prev_viol:
                self.rho.mul_(1.5)  # Multiply rho by 1.5
                self.rho.clamp_(max=100.0) # Cap it to prevent numerical explosion

            self.prev_viol = viol

class ImpAugLagrangianCombiner(AugLagrangianCombiner):
    def __init__(self, experts, budget, device, use_routing=False):
        super().__init__(experts, budget, device, use_routing)
        self.prev_viol = float('inf')

    def update_dual(self, viol):
        with torch.no_grad():
            self.lam.add_(self.rho * viol)
            self.lam.clamp_(min=0.0, max=100.0)
            if viol > 0.01 and viol > self.prev_viol * 0.9:
                self.rho.mul_(1.5)
                self.rho.clamp_(max=100.0)
            self.prev_viol = viol

class ADMMCombiner(BaseCombiner):
    def __init__(self, experts, budget, device, use_routing=False):
        super().__init__(experts, budget, device, use_routing)
        self.register_buffer('u', torch.tensor(0.0, device=device))
        self.register_buffer('z', torch.tensor(budget, device=device))
        self.register_buffer('rho', torch.tensor(2.0, device=device))
        self.loss_scale = 10.0

    def training_step(self, x, pred, y, criterion):
        mse = criterion(pred, y)
        w = self.get_weights(x)
        cost = self.get_cost(w)
        scaled_cost = cost / self.budget
        scaled_z = self.z / self.budget
        residual = scaled_cost - scaled_z + self.u
        penalty = (self.rho / 2) * (residual ** 2)
        total_loss = (mse * self.loss_scale) + penalty
        viol = F.relu((cost - self.budget)/self.budget)
        return total_loss, cost.item(), viol.item()

    def update_dual(self, current_cost):
        with torch.no_grad():
            cost_tensor = torch.tensor(current_cost, device=self.device)
            scaled_cost = cost_tensor / self.budget
            z_star = scaled_cost + self.u
            one_tensor = torch.tensor(1.0, device=self.device)
            new_z_norm = torch.min(z_star, one_tensor)
            self.z.copy_(new_z_norm * self.budget)
            dual_residual = scaled_cost - new_z_norm
            self.u.add_(dual_residual)
            self.u.clamp_(-10.0, 10.0)

# ==========================================
# 4. Utilities & Training
# ==========================================
def train_combiner(combiner, train_loader, val_loader, test_loader, budget, device, name="Model"):
    print(f"\n>>> Training {name} <<<")

    # 1. Separate Parameters for Two Optimizers (Primal and Dual)
    primal_params = []
    dual_params = []
    for pname, p in combiner.named_parameters():
        if 'lam' in pname:
            dual_params.append(p)
        else:
            primal_params.append(p)

    # 2. Initialize Optimizers
    # Primal: Adam (Minimizes Loss)
    opt_primal = optim.Adam(primal_params, lr=1e-5, weight_decay=1e-7)

    # Dual: SGD with maximize=True (Maximizes Lagrangian w.r.t Duals)
    opt_dual = None
    if dual_params:
        try:
            opt_dual = optim.Adam(primal_params, lr=1e-6, weight_decay=1e-8, maximize=True)
        except TypeError:
            opt_dual = optim.Adam(primal_params, lr=-1e-6, weight_decay=1e-8,) # Negative LR for ascent

    loss_fn = PINNLoss(mse_weight=1.0, physics_weight=1e-3).to(device)

    # Track stats
    history = {
        'train_loss': [], 'val_loss': [], 'test_loss': [],
        'train_mse': [], 'val_mse': [], 'test_mse': [],
        'cost': []
    }

    # Run for 50 epochs (enough to see convergence)
    for epoch in range(200):
        # --- TRAIN ---
        combiner.train()
        epoch_train_loss = 0
        epoch_train_mse = 0
        epoch_cost = 0

        for bx, by in train_loader:
            # Zero Grads
            opt_primal.zero_grad()
            if opt_dual: opt_dual.zero_grad()

            # Forward & Loss
            pred = combiner(bx)
            mse = loss_fn(pred, by) # Raw MSE
            loss, cost, viol = combiner.training_step(bx, pred, by, loss_fn) # Augmented/Lagrangian Loss

            # Backward
            loss.backward()

            # Primal Step (Descent)
            torch.nn.utils.clip_grad_norm_(combiner.parameters(), 1.0)
            opt_primal.step()

            # Dual Step (Ascent) - For LagrangianCombiner
            if opt_dual:
                opt_dual.step()
                with torch.no_grad():
                    for p in dual_params: p.clamp_(min=0.0)

            epoch_train_loss += loss.item()
            epoch_train_mse += mse.item()
            epoch_cost += cost

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_mse = epoch_train_mse / len(train_loader)
        avg_cost = epoch_cost / len(train_loader)

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
                v_loss, _, _ = combiner.training_step(bx, pred, by, loss_fn)
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
        w_str = "[" + ", ".join([f"{v:.2f}" for v in avg_weights.cpu().numpy()]) + "]"

        # --- TEST ---
        test_loss = 0
        test_mse = 0
        with torch.no_grad():
            for bx, by in test_loader:
                pred = combiner(bx)
                t_loss, _, _ = combiner.training_step(bx, pred, by, loss_fn)
                test_loss += t_loss.item()
                test_mse += loss_fn(pred, by).item()
        avg_test_loss = test_loss / len(test_loader)
        avg_test_mse = test_mse / len(test_loader)

        # Manual Dual Update for ALM / ADMM (Buffer-based)
        if hasattr(combiner, 'update_dual') and not opt_dual:
            if isinstance(combiner, ADMMCombiner):
                combiner.update_dual(avg_cost)
            else:
                avg_viol = max(0, (avg_cost - budget)/budget)
                combiner.update_dual(avg_viol)

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
            lam_str = f" | Lam {combiner.lam.item():.4f}"
        elif hasattr(combiner, 'u'):
            lam_str = f" | Dual U {combiner.u.item():.4f}"

        # Print with Weights
        print(f"Ep {epoch}: TrMSE {avg_train_mse:.5f} | ValMSE {avg_val_mse:.5f} | TestMSE {avg_test_mse:.5f} | Cost {avg_cost:.2f}{lam_str} | W {w_str} | TrLoss {avg_train_loss:.4f} | ValLoss {avg_val_loss:.4f} | TestLoss {avg_test_loss:.4f}")

    return history

def train_experts(device):
    resolutions = [32, 64, 128]
    costs = [1.0, 2.0, 4.0]
    experts = []
    print("Pre-training Experts...")
    for res, cost in zip(resolutions, costs):
        train_x, train_y = generate_data(num_samples=100, nx=res, device=device)
        dataset = TensorDataset(train_x, train_y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        # Use baseline Expert1D (CNN) for consistency
        model = Expert1D().to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        loss_fn = PINNLoss(mse_weight=1.0, physics_weight=1e-3).to(device)

        for _ in range(5):
            for bx, by in loader:
                optimizer.zero_grad()
                pred = model(bx)
                loss = loss_fn(pred.squeeze(1), by)
                loss.backward()
                optimizer.step()
        experts.append(MultiResExpertWrapper(model, res, cost))
    return experts


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    experts = train_experts(device)
    for e in experts:
        e.eval()
        for p in e.parameters(): p.requires_grad = False

    print("Generating Datasets...")
    # 1. Train: Standard Frequencies (1-3)
    train_x, train_y = generate_data(num_samples=5000, nx=128, device=device, freq_range=(1, 3))
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)

    # 2. Validation: Same as Train (1-3)
    val_x, val_y = generate_data(num_samples=1000, nx=128, device=device, freq_range=(1, 3))
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32)

    # 3. Test: Higher Frequencies (3-6) -> Out-of-Distribution / Harder
    test_x, test_y = generate_data(num_samples=1000, nx=128, device=device, freq_range=(3, 6))
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=32)

    BUDGET = 2.5
    results = {}

    # --- 1. Softmax ---
    model = SoftmaxCombiner(experts, BUDGET, device, use_routing=False).to(device)
    results['Softmax'] = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "Softmax")

    # --- 2. Lagrangian ---
    model = LagrangianCombiner(experts, BUDGET, device, use_routing=False).to(device)
    results['Lagrangian'] = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "Lagrangian")

    # --- 3. Aug Lagrangian ---
    model = AugLagrangianCombiner(experts, BUDGET, device, use_routing=True).to(device)
    results['AugLag_Routing'] = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "AugLag (Routing)")

    # --- 4. ADMM ---
    admm_model = ADMMCombiner(experts, BUDGET, device, use_routing=True).to(device)
    results['ADMM_Routing'] = train_combiner(admm_model, train_loader, val_loader, test_loader, BUDGET, device, "ADMM (Routing)")

    # --- Final Summary ---
    labels = list(results.keys())

    mse_vals = []
    cost_vals = []
    for k in labels:
        res = results[k]
        mse_vals.append(res['test_mse'][-1])
        cost_vals.append(res['cost'][-1])

    print("\n=== Final Test Summary (High Freq Data) ===")
    print(f"{'Method':<20} | {'Test MSE':<10} | {'Cost':<10} | {'Status'}")
    for k, m, c in zip(labels, mse_vals, cost_vals):
        stat = "OK" if c <= BUDGET + 0.1 else "VIOLATION"
        print(f"{k:<20} | {m:.5f} | {c:.2f}     | {stat}")

if __name__ == "__main__":
    main()



# ==========================================
# 4. Utilities & Training
# ==========================================

def train_combiner(combiner, train_loader, val_loader, test_loader, budget, device, name="Model"):
    print(f"\n>>> Training {name} <<<")

    # 1. Separate Parameters for Two Optimizers (Primal and Dual)
    primal_params = []
    dual_params = []
    for pname, p in combiner.named_parameters():
        if 'lam' in pname:
            dual_params.append(p)
        else:
            primal_params.append(p)

    # 2. Initialize Optimizers
    # Primal: Adam (Minimizes Loss)
    opt_primal = optim.Adam(primal_params, lr=1e-5, weight_decay=1e-6)

    # Dual: SGD with maximize=True (Maximizes Lagrangian w.r.t Duals)
    opt_dual = None
    if dual_params:
        try:
            opt_dual = optim.Adam(primal_params, lr=1e-6, weight_decay=1e-8, maximize=True)
        except TypeError:
            opt_dual = optim.Adam(primal_params, lr=-1e-6, weight_decay=1e-8) # Negative LR for ascent

    loss_fn = PINNLoss(mse_weight=1.0, physics_weight=1e-3).to(device)

    # Track stats
    history = {
        'train_loss': [], 'val_loss': [], 'test_loss': [],
        'train_mse': [], 'val_mse': [], 'test_mse': [],
        'cost': []
    }

    # Run for 50 epochs (enough to see convergence)
    for epoch in range(200):
        # --- TRAIN ---
        combiner.train()
        epoch_train_loss = 0
        epoch_train_mse = 0
        epoch_cost = 0

        for bx, by in train_loader:
            # Zero Grads
            opt_primal.zero_grad()
            if opt_dual: opt_dual.zero_grad()

            # Forward & Loss
            pred = combiner(bx)
            mse = loss_fn(pred, by) # Raw MSE
            loss, cost, viol = combiner.training_step(bx, pred, by, loss_fn) # Augmented/Lagrangian Loss

            # Backward
            loss.backward()

            # Primal Step (Descent)
            torch.nn.utils.clip_grad_norm_(combiner.parameters(), 1.0)
            opt_primal.step()

            # Dual Step (Ascent) - For LagrangianCombiner
            if opt_dual:
                opt_dual.step()
                with torch.no_grad():
                    for p in dual_params: p.clamp_(min=0.0)

            epoch_train_loss += loss.item()
            epoch_train_mse += mse.item()
            epoch_cost += cost

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_mse = epoch_train_mse / len(train_loader)
        avg_cost = epoch_cost / len(train_loader)

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
                v_loss, _, _ = combiner.training_step(bx, pred, by, loss_fn)
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
        w_str = "[" + ", ".join([f"{v:.2f}" for v in avg_weights.cpu().numpy()]) + "]"

        # --- TEST ---
        test_loss = 0
        test_mse = 0
        with torch.no_grad():
            for bx, by in test_loader:
                pred = combiner(bx)
                t_loss, _, _ = combiner.training_step(bx, pred, by, loss_fn)
                test_loss += t_loss.item()
                test_mse += loss_fn(pred, by).item()
        avg_test_loss = test_loss / len(test_loader)
        avg_test_mse = test_mse / len(test_loader)

        # Manual Dual Update for ALM / ADMM (Buffer-based)
        if hasattr(combiner, 'update_dual') and not opt_dual:
            if isinstance(combiner, ADMMCombiner):
                combiner.update_dual(avg_cost)
            else:
                avg_viol = max(0, (avg_cost - budget)/budget)
                combiner.update_dual(avg_viol)

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
            lam_str = f" | Lam {combiner.lam.item():.4f}"
        elif hasattr(combiner, 'u'):
            lam_str = f" | Dual U {combiner.u.item():.4f}"

        # Print with Weights
        print(f"Ep {epoch}: TrMSE {avg_train_mse:.5f} | ValMSE {avg_val_mse:.5f} | TestMSE {avg_test_mse:.5f} | Cost {avg_cost:.2f}{lam_str} | W {w_str} | TrLoss {avg_train_loss:.4f} | ValLoss {avg_val_loss:.4f} | TestLoss {avg_test_loss:.4f}")

    return history

def train_experts(device):
    """
    Trains 3 experts at Resolution 128 using different METHODS:
    1. Finite Difference (Cost 1.0)
    2. CNN (Cost 2.0)
    3. FNO (Cost 4.0)
    """
    res = 128
    # Define Expert Configurations: Method, Model, Cost
    expert_configs = [
        {'name': 'FiniteDiff', 'model': FiniteDifferenceExpert(hidden_dim=32), 'cost': 1.0},
        {'name': 'CNN',       'model': Expert1D(hidden_channels=32),           'cost': 2.0},
        {'name': 'FNO',       'model': FNO1D(modes=16, width=32),              'cost': 4.0}
    ]

    experts = []
    print("\n=== Pre-training Heterogeneous Experts (Res 128) ===")

    # Shared Training Data for Experts (Standard Freqs)
    train_x, train_y = generate_data(num_samples=100, nx=res, device=device, freq_range=(1,4))
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for conf in expert_configs:
        print(f"Training {conf['name']} Expert (Cost {conf['cost']})...")
        model = conf['model'].to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = PINNLoss(mse_weight=1.0, physics_weight=1e-3).to(device)

        for epoch in range(15): # Moderate pre-training
            total_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                pred = model(bx)
                # Ensure output shape matches target [B, N]
                if pred.dim() == 3: pred = pred.squeeze(1)

                loss = loss_fn(pred, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch+1) % 5 == 0:
                print(f"  Ep {epoch+1}: Loss {total_loss/len(loader):.5f}")

        experts.append(MultiResExpertWrapper(model, res, conf['cost']))

    return experts

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # 1. Train Heterogeneous Experts at Fixed Scale
    experts = train_experts(device)
    for e in experts:
        e.eval()
        for p in e.parameters(): p.requires_grad = False

    print("\nGenerating Multi-Scale Generalization Datasets (All Res 128)...")
    # 2. Train: Standard Frequencies (1-3)
    train_x, train_y = generate_data(num_samples=5000, nx=128, device=device, freq_range=(1, 3))
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)

    # 3. Validation: Standard Frequencies (1-3)
    val_x, val_y = generate_data(num_samples=1000, nx=128, device=device, freq_range=(1, 3))
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32)

    # 4. Test: Higher Frequencies (3-6) -> Out-of-Distribution / Harder
    test_x, test_y = generate_data(num_samples=1000, nx=128, device=device, freq_range=(3, 6))
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=32)

    # Budget targets a mix of Cheap (FD/CNN) and Expensive (FNO) methods
    BUDGET = 2.5
    results = {}

    # --- 1. Softmax ---
    model = SoftmaxCombiner(experts, BUDGET, device, use_routing=False).to(device)
    results['Softmax'] = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "Softmax")

    # --- 2. Lagrangian ---
    model = LagrangianCombiner(experts, BUDGET, device, use_routing=False).to(device)
    results['Lagrangian'] = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "Lagrangian")

    # --- 3. Aug Lagrangian ---
    model = AugLagrangianCombiner(experts, BUDGET, device, use_routing=True).to(device)
    results['AugLag_Routing'] = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "AugLag (Routing)")

    # --- 4. ADMM ---
    admm_model = ADMMCombiner(experts, BUDGET, device, use_routing=True).to(device)
    results['ADMM_Routing'] = train_combiner(admm_model, train_loader, val_loader, test_loader, BUDGET, device, "ADMM (Routing)")

    # --- Final Summary ---
    labels = list(results.keys())

    mse_vals = []
    cost_vals = []
    for k in labels:
        res = results[k]
        mse_vals.append(res['test_mse'][-1])
        cost_vals.append(res['cost'][-1])

    print("\n=== Final Test Summary (Method Selection) ===")
    print(f"{'Method':<20} | {'Test MSE':<10} | {'Cost':<10} | {'Status'}")
    for k, m, c in zip(labels, mse_vals, cost_vals):
        stat = "OK" if c <= BUDGET + 0.1 else "VIOLATION"
        print(f"{k:<20} | {m:.5f} | {c:.2f}     | {stat}")

if __name__ == "__main__":
    main()


class NoisySourceWrapper(nn.Module):
    """
    Wraps a shared model but corrupts the input with specific noise profiles.
    Simulates fetching data from sensors of varying quality.
    """
    def __init__(self, shared_model, iv_noise_std, meas_noise_std, cost, device):
        super().__init__()
        self.model = shared_model
        self.iv_noise_std = iv_noise_std
        self.meas_noise_std = meas_noise_std
        self.cost = cost
        self.device = device

    def get_correlated_noise(self, shape):
        # Generate smooth spatial noise (simulating IV errors or drift)
        noise = torch.randn(shape, device=self.device)
        # Low-pass filter via FFT to make it smooth/correlated
        noise_ft = torch.fft.rfft(noise, dim=-1)
        # Keep only low freq
        modes = 8
        noise_ft[:, modes:] = 0
        return torch.fft.irfft(noise_ft, n=shape[-1])

    def forward(self, x_clean):
        # x_clean: [Batch, N] Ground Truth State

        # 1. Add Initial Value (IV) / Model Error Noise (Correlated)
        if self.iv_noise_std > 0:
            iv_noise = self.get_correlated_noise(x_clean.shape)
            # Normalize to unit variance then scale
            iv_noise = iv_noise / (iv_noise.std() + 1e-8)
            x_noisy = x_clean + iv_noise * self.iv_noise_std
        else:
            x_noisy = x_clean

        # 2. Add Measurement Noise (White/Gaussian)
        if self.meas_noise_std > 0:
            meas_noise = torch.randn_like(x_clean) * self.meas_noise_std
            x_noisy = x_noisy + meas_noise

        # Pass noisy data to the shared FNO
        # The FNO learns to denoise implicitly by minimizing loss against clean target
        return self.model(x_noisy)


# ==========================================
# 5. Utilities & Training
# ==========================================

def train_combiner(combiner, train_loader, val_loader, test_loader, budget, device, name="Model"):
    print(f"\n>>> Training {name} <<<")

    # 1. Separate Parameters
    primal_params = []
    dual_params = []
    for pname, p in combiner.named_parameters():
        if 'lam' in pname:
            dual_params.append(p)
        else:
            primal_params.append(p)

    # 2. Optimizers
    opt_primal = optim.Adam(primal_params, lr=1e-5, weight_decay=1e-6)

    opt_dual = None
    if dual_params:
        try:
            opt_dual = optim.Adam(primal_params, lr=1e-6, weight_decay=1e-8, maximize=True)
        except TypeError:
            opt_dual = optim.Adam(primal_params, lr=-1e-6, weight_decay=1e-8)

    loss_fn = PINNLoss(mse_weight=1.0, physics_weight=1e-3).to(device)

    history = {
        'train_loss': [], 'val_loss': [], 'test_loss': [],
        'train_mse': [], 'val_mse': [], 'test_mse': [],
        'cost': []
    }

    for epoch in range(200):
        # --- TRAIN ---
        combiner.train()
        epoch_train_loss = 0
        epoch_train_mse = 0
        epoch_cost = 0

        for bx, by in train_loader:
            opt_primal.zero_grad()
            if opt_dual: opt_dual.zero_grad()

            pred = combiner(bx)
            mse = loss_fn(pred, by)
            loss, cost, viol = combiner.training_step(bx, pred, by, loss_fn)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(combiner.parameters(), 1.0)
            opt_primal.step()

            if opt_dual:
                opt_dual.step()
                with torch.no_grad():
                    for p in dual_params: p.clamp_(min=0.0)

            epoch_train_loss += loss.item()
            epoch_train_mse += mse.item()
            epoch_cost += cost

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_mse = epoch_train_mse / len(train_loader)
        avg_cost = epoch_cost / len(train_loader)

        # --- VALIDATION ---
        combiner.eval()
        val_loss = 0
        val_mse = 0
        total_w = torch.zeros(len(combiner.experts), device=device)
        total_samples = 0

        with torch.no_grad():
            for bx, by in val_loader:
                pred = combiner(bx)
                v_loss, _, _ = combiner.training_step(bx, pred, by, loss_fn)
                val_loss += v_loss.item()
                val_mse += loss_fn(pred, by).item()

                w = combiner.get_weights(bx)
                if w.dim() == 1:
                    total_w += w * bx.size(0)
                else:
                    total_w += w.sum(dim=0)
                total_samples += bx.size(0)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mse = val_mse / len(val_loader)

        # Compute Average Weights
        avg_weights = total_w / total_samples
        w_str = "[" + ", ".join([f"{v:.2f}" for v in avg_weights.cpu().numpy()]) + "]"

        # --- TEST ---
        test_loss = 0
        test_mse = 0
        with torch.no_grad():
            for bx, by in test_loader:
                pred = combiner(bx)
                t_loss, _, _ = combiner.training_step(bx, pred, by, loss_fn)
                test_loss += t_loss.item()
                test_mse += loss_fn(pred, by).item()
        avg_test_loss = test_loss / len(test_loader)
        avg_test_mse = test_mse / len(test_loader)

        # Dual Updates (ALM/ADMM)
        if hasattr(combiner, 'update_dual') and not opt_dual:
            if isinstance(combiner, ADMMCombiner):
                combiner.update_dual(avg_cost)
            else:
                avg_viol = max(0, (avg_cost - budget)/budget)
                combiner.update_dual(avg_viol)

        history['train_loss'].append(avg_train_loss)
        history['train_mse'].append(avg_train_mse)
        history['val_loss'].append(avg_val_loss)
        history['val_mse'].append(avg_val_mse)
        history['test_loss'].append(avg_test_loss)
        history['test_mse'].append(avg_test_mse)
        history['cost'].append(avg_cost)

        lam_str = ""
        if hasattr(combiner, 'lam'):
            lam_str = f" | Lam {combiner.lam.item():.4f}"
        elif hasattr(combiner, 'u'):
            lam_str = f" | Dual U {combiner.u.item():.4f}"

        print(f"Ep {epoch}: TrMSE {avg_train_mse:.5f} | ValMSE {avg_val_mse:.5f} | TestMSE {avg_test_mse:.5f} | Cost {avg_cost:.2f}{lam_str} | W {w_str} | TrLoss {avg_train_loss:.4f} | ValLoss {avg_val_loss:.4f} | TestLoss {avg_test_loss:.4f}")

    return history

def initialize_experts(device):
    """
    Initializes 4 Data Source Experts sharing ONE FNO model.
    Sources vary by noise type and cost.
    """
    print("\n=== Initializing Shared FNO and Data Sources ===")

    # 1. The Shared Backbone
    shared_fno = FNO1D(modes=16, width=32).to(device)

    # 2. Define 4 Sources with different noise profiles
    # Source 1: High Quality (Low Noise) -> High Cost
    # Source 4: Low Quality (High Noise) -> Low Cost

    sources_config = [
        # (IV Noise, Meas Noise, Cost)
        (0.01, 0.01, 4.0), # Source 1: Very Clean
        (0.05, 0.05, 2.0), # Source 2: Moderate
        (0.10, 0.10, 1.0), # Source 3: Noisy
        (0.20, 0.20, 0.5)  # Source 4: Very Noisy
    ]

    experts = []
    for i, (iv, meas, cost) in enumerate(sources_config):
        print(f"Source {i+1}: IV_Noise={iv}, Meas_Noise={meas}, Cost={cost}")
        # Note: We pass the SAME model instance to all wrappers
        expert = NoisySourceWrapper(shared_fno, iv, meas, cost, device)
        experts.append(expert)

    return experts

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # 1. Initialize Shared Model + Wrappers
    experts = initialize_experts(device)

    # 2. Generate Data (Clean Ground Truth)
    # The Experts will add noise on the fly during training
    print("\nGenerating Ground Truth Datasets...")
    train_x, train_y = generate_data(num_samples=5000, nx=128, device=device)
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)

    val_x, val_y = generate_data(num_samples=1000, nx=128, device=device)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32)

    test_x, test_y = generate_data(num_samples=1000, nx=128, device=device)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=32)

    # Budget targets mixing.
    # Costs: [4.0, 2.0, 1.0, 0.5]
    # Budget 2.5 implies we can use Source 2 + Source 4, or mix Source 1.
    BUDGET = 2.5
    results = {}

    # --- 1. Softmax (Baseline - ignores budget) ---
    model = SoftmaxCombiner(experts, BUDGET, device, use_routing=False).to(device)
    results['Softmax'] = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "Softmax")

    # --- 2. Lagrangian ---
    model = LagrangianCombiner(experts, BUDGET, device, use_routing=False).to(device)
    results['Lagrangian'] = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "Lagrangian")

    # --- 3. Aug Lagrangian ---
    model = AugLagrangianCombiner(experts, BUDGET, device, use_routing=True).to(device)
    results['AugLag_Routing'] = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "AugLag (Routing)")

    # --- 4. ADMM ---
    admm_model = ADMMCombiner(experts, BUDGET, device, use_routing=True).to(device)
    results['ADMM_Routing'] = train_combiner(admm_model, train_loader, val_loader, test_loader, BUDGET, device, "ADMM (Routing)")

    # --- Final Summary ---
    labels = list(results.keys())
    mse_vals = [results[k]['test_mse'][-1] for k in labels]
    cost_vals = [results[k]['cost'][-1] for k in labels]

    print("\n=== Final Test Summary (Data Source Selection) ===")
    print(f"{'Method':<20} | {'Test MSE':<10} | {'Cost':<10} | {'Status'}")
    for k, m, c in zip(labels, mse_vals, cost_vals):
        stat = "OK" if c <= BUDGET + 0.1 else "VIOLATION"
        print(f"{k:<20} | {m:.5f} | {c:.2f}     | {stat}")

if __name__ == "__main__":
    main()