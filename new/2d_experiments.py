import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader, TensorDataset


class NavierStokes2D:
    def __init__(self, nx=64, ny=64, t_max=1.0, nu=1e-3, device='cpu'):
        self.nx = nx
        self.ny = ny
        self.t_max = t_max
        self.nu = nu
        self.device = device

        # 1. Define Grid (Periodic, excluding last point)
        # We assume Lx = Ly = 2*pi
        self.x = torch.linspace(0, 2*math.pi, nx + 1, device=device)[:-1]
        self.y = torch.linspace(0, 2*math.pi, ny + 1, device=device)[:-1]
        self.X, self.Y = torch.meshgrid(self.x, self.y, indexing='ij')

        # 2. Define Wavenumbers for 2D FFT
        # kx needs shape (nx, 1), ky needs shape (1, ny) for broadcasting
        kx = torch.fft.fftfreq(nx, d=1/nx, device=device) * nx # Integer modes
        ky = torch.fft.fftfreq(ny, d=1/ny, device=device) * ny
        self.kx = kx.reshape(-1, 1)
        self.ky = ky.reshape(1, -1)

        # Laplacian in Fourier space: -(kx^2 + ky^2)
        self.k_sq = self.kx**2 + self.ky**2

        # Avoid division by zero at (0,0) mode
        self.k_sq[0, 0] = 1.0
        self.inv_k_sq = 1.0 / self.k_sq
        self.inv_k_sq[0, 0] = 0.0 # Set mean flow (k=0) to 0

        # Precompute imaginary k for derivatives
        self.ikx = 1j * self.kx
        self.iky = 1j * self.ky

    def solve_trajectory(self, w0, steps=100):
        """
        w0: Initial vorticity field of shape (nx, ny)
        """
        dt = self.t_max / steps
        w = w0.clone()
        trajectory = [w.clone()]

        # Crank-Nicolson factors for diffusion
        factor_lhs = 1 + 0.5 * dt * self.nu * self.k_sq
        factor_rhs = 1 - 0.5 * dt * self.nu * self.k_sq

        for _ in range(steps):
            # 1. To Spectral Space
            w_hat = torch.fft.fft2(w)

            # 2. Solve Poisson for Streamfunction: -Laplacian(psi) = w
            psi_hat = w_hat * self.inv_k_sq

            # 3. Compute Velocity (u, v) in Spectral Space
            # u = d(psi)/dy, v = -d(psi)/dx
            u_hat = self.iky * psi_hat
            v_hat = -self.ikx * psi_hat

            # 4. Compute Gradient of Vorticity in Spectral Space
            wx_hat = self.ikx * w_hat
            wy_hat = self.iky * w_hat

            # 5. Compute Non-linear Term (Advection) in Real Space
            # (u * w_x + v * w_y)
            u = torch.fft.ifft2(u_hat).real
            v = torch.fft.ifft2(v_hat).real
            wx = torch.fft.ifft2(wx_hat).real
            wy = torch.fft.ifft2(wy_hat).real

            advection = u * wx + v * wy
            advection_hat = torch.fft.fft2(advection)

            # 6. Time Step (Crank-Nicolson for Diffusion, Euler for Advection)
            # (1 + dt/2 * nu * k^2) * w_hat_new = (1 - dt/2 * nu * k^2) * w_hat - dt * advection_hat
            w_hat_new = (w_hat * factor_rhs - dt * advection_hat) / factor_lhs

            # 7. Back to Real Space
            w = torch.fft.ifft2(w_hat_new).real
            trajectory.append(w.clone())

        return torch.stack(trajectory)

def generate_2d_data(num_samples=10, nx=64, steps=50, device='cpu'):
    """
    Generates 2D Vorticity data.
    """
    print(f"Generating {num_samples} 2D trajectories at res {nx}x{nx}...")

    # Initialize Physics
    physics = NavierStokes2D(nx=nx, ny=nx, t_max=1.0, nu=1e-3, device=device)

    data_x, data_y = [], []

    for i in range(num_samples):
        # Generate random initial conditions (Gaussian Random Field approximation)
        # Using a sum of random Fourier modes
        w0 = torch.zeros((nx, nx), device=device)

        # Add random low-frequency modes
        num_modes = 10
        for _ in range(num_modes):
            kx_mode = torch.randint(-4, 4, (1,)).item()
            ky_mode = torch.randint(-4, 4, (1,)).item()
            phase = torch.rand(1).item() * 2 * math.pi
            amp = torch.randn(1).item()

            # w0 += A * sin(kx*X + ky*Y + phase)
            w0 += amp * torch.sin(kx_mode * physics.X + ky_mode * physics.Y + phase)

        # Normalize
        w0 = w0 / w0.std()

        # Solve
        traj = physics.solve_trajectory(w0, steps=steps)

        # Input (t) -> Target (t+1)
        data_x.append(traj[:-1])
        data_y.append(traj[1:])

        if (i+1) % 5 == 0:
            print(f"  Sample {i+1}/{num_samples} done.")

    return torch.cat(data_x), torch.cat(data_y)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# 2. Expert Models (2D Versions)
# ==========================================

# --- A. Standard CNN 2D (Baseline Expert) ---
class Expert2D(nn.Module):
    def __init__(self, hidden_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            # Input: (Batch, 1, H, W) -> Output: (Batch, Hidden, H, W)
            nn.Conv2d(1, hidden_channels, 5, padding=2, padding_mode='circular'),
            nn.Tanh(),
            nn.Conv2d(hidden_channels, hidden_channels, 5, padding=2, padding_mode='circular'),
            nn.Tanh(),
            nn.Conv2d(hidden_channels, 1, 5, padding=2, padding_mode='circular')
        )

    def forward(self, x):
        # x shape: (Batch, H, W)
        # Add channel dim: (Batch, 1, H, W)
        u = x.unsqueeze(1)
        # Residual connection: u(t+1) = u(t) + dt * Model(u)
        return x + self.net(u).squeeze(1)


# --- B. Finite Difference Expert 2D ---
class FiniteDifferenceExpert2D(nn.Module):
    """
    Extracts physical derivatives (gradients and laplacian)
    and passes them through a pixel-wise MLP (1x1 Conv).
    """
    def __init__(self, hidden_dim=32):
        super().__init__()

        # Define kernels for 2D Finite Difference
        # Sobel or central difference for gradients
        k_x = torch.tensor([[[[-0.5, 0.0, 0.5]]]]) # shape (1,1,1,3)
        k_y = torch.tensor([[[[-0.5], [0.0], [0.5]]]]) # shape (1,1,3,1)

        # 5-point Stencil for Laplacian
        k_lap = torch.tensor([[[[0.0, 1.0, 0.0],
                                [1.0, -4.0, 1.0],
                                [0.0, 1.0, 0.0]]]])

        self.register_buffer('k_x', k_x)
        self.register_buffer('k_y', k_y)
        self.register_buffer('k_lap', k_lap)

        # Input features: [u, u_x, u_y, laplacian] = 4 channels
        self.mlp = nn.Sequential(
            nn.Conv2d(4, hidden_dim, 1), # 1x1 Conv acts as MLP per pixel
            nn.Tanh(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.Tanh(),
            nn.Conv2d(hidden_dim, 1, 1)
        )

    def forward(self, x):
        # x shape: (Batch, H, W)
        u = x.unsqueeze(1)

        # Pad for periodic convolution
        # Pad left/right (for x) and top/bottom (for y)
        u_pad_x = F.pad(u, (1, 1, 0, 0), mode='circular')
        u_pad_y = F.pad(u, (0, 0, 1, 1), mode='circular')
        u_pad_lap = F.pad(u, (1, 1, 1, 1), mode='circular') # Pad all sides

        # Compute derivatives
        ux = F.conv2d(u_pad_x, self.k_x)
        uy = F.conv2d(u_pad_y, self.k_y)
        ulap = F.conv2d(u_pad_lap, self.k_lap)

        # Concatenate features
        features = torch.cat([u, ux, uy, ulap], dim=1)

        dt_pred = self.mlp(features)
        return x + dt_pred.squeeze(1)


# --- C. Fourier Neural Operator 2D (FNO) ---
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Modes to keep in X
        self.modes2 = modes2 # Modes to keep in Y

        self.scale = (1 / (in_channels * out_channels))
        # Weights for the corners of the Fourier matrix
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def complex_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute 2D FFT
        # Output shape: (batch, c, x, y//2 + 1)
        x_ft = torch.fft.rfft2(x)

        # Initialize output
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)

        # Multiply relevant modes
        # Corner 1 (Top-Left)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.complex_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)

        # Corner 2 (Bottom-Left) - Handling periodicity in X
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.complex_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Inverse 2D FFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2D(nn.Module):
    def __init__(self, modes=12, width=32):
        super().__init__()
        self.modes = modes
        self.width = width
        self.padding = 2 # Pad to avoid aliasing if needed, or remove for pure periodic

        # Project input (vorticity + 2 grid coords) to high dim
        self.fc0 = nn.Linear(3, width)

        self.conv0 = SpectralConv2d(width, width, modes, modes)
        self.conv1 = SpectralConv2d(width, width, modes, modes)
        self.conv2 = SpectralConv2d(width, width, modes, modes)

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x shape: (Batch, H, W)
        grid = self.get_grid(x.shape, x.device) # (Batch, H, W, 2)
        x = x.unsqueeze(-1) # (Batch, H, W, 1)

        # Concatenate grid
        x = torch.cat((x, grid), dim=-1) # (Batch, H, W, 3)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2) # (Batch, Width, H, W)

        # FNO Layers
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        # Projection back
        x = x.permute(0, 2, 3, 1) # (Batch, H, W, Width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x.squeeze(-1) # Output (Batch, H, W)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)


# --- D. DeepONet 2D ---
class DeepONet2D(nn.Module):
    def __init__(self, nx=64, ny=64, branch_dim=64, trunk_dim=64):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.num_sensors = nx * ny # Flattened input size

        # Branch Net: Takes full vorticity field u flattened
        self.branch = nn.Sequential(
            nn.Linear(self.num_sensors, 256),
            nn.Tanh(),
            nn.Linear(256, branch_dim)
        )

        # Trunk Net: Takes coordinates (x, y)
        # We implement this as a Conv2d(2 channels -> features)
        # acting on a coordinate grid
        self.trunk = nn.Sequential(
            nn.Conv2d(2, 64, 1), # Input channels = 2 (x and y)
            nn.Tanh(),
            nn.Conv2d(64, trunk_dim, 1)
        )
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, u):
        # u shape: (Batch, H, W)
        batch, h, w = u.shape

        # 1. Branch: Process input function u
        u_flat = u.view(batch, -1)
        B = self.branch(u_flat) # (Batch, branch_dim)

        # 2. Trunk: Process grid coordinates
        # Create meshgrid (Batch, 2, H, W)
        x_grid = torch.linspace(-1, 1, w, device=u.device)
        y_grid = torch.linspace(-1, 1, h, device=u.device)
        X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
        grid = torch.stack([X, Y], dim=0).unsqueeze(0).repeat(batch, 1, 1, 1)

        T = self.trunk(grid) # (Batch, trunk_dim, H, W)

        # 3. Dot Product (B * T)
        # Reshape B for broadcasting: (Batch, branch_dim, 1, 1)
        B = B.view(batch, -1, 1, 1)

        # Output = Sum(B * T) + bias
        out = (B * T).sum(dim=1) + self.bias

        return u + out

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 3. Combiners & Router (2D Adapted)
# ==========================================

# --- Wrapper to handle Multi-Resolution 2D Inputs ---
class MultiResExpertWrapper2D(nn.Module):
    def __init__(self, model, native_res, cost):
        super().__init__()
        self.model = model
        self.native_res = native_res # Assuming square grid (native_res, native_res)
        self.cost = cost

    def forward(self, x):
        # x shape: (Batch, H, W)
        target_h, target_w = x.shape[-2], x.shape[-1]

        # Downsample/Upsample Input if needed
        if (target_h != self.native_res) or (target_w != self.native_res):
            # Input to interpolate must be (Batch, Channel, H, W)
            x_in = F.interpolate(x.unsqueeze(1), size=(self.native_res, self.native_res),
                                 mode='bilinear', align_corners=False)
        else:
            x_in = x.unsqueeze(1)

        # Forward pass through Expert (expects (Batch, H, W))
        out_native = self.model(x_in.squeeze(1))

        # Ensure output is (Batch, 1, H, W) for interpolation
        if out_native.dim() == 3:
            out_native = out_native.unsqueeze(1)

        # Restore original resolution
        if (target_h != self.native_res) or (target_w != self.native_res):
            out_res = F.interpolate(out_native, size=(target_h, target_w),
                                    mode='bilinear', align_corners=False).squeeze(1)
        else:
            out_res = out_native.squeeze(1)

        return out_res # (Batch, H, W)

class Router2D(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.net = nn.Sequential(
            # Input: (Batch, 1, H, W)
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), # Global Average Pooling -> (Batch, 32, 1, 1)
            nn.Flatten(),
            nn.Linear(32, num_experts)
        )

    def forward(self, x):
        # x shape: (Batch, H, W) -> unsqueeze to (Batch, 1, H, W)
        return self.net(x.unsqueeze(1))

class BaseCombiner(nn.Module):
    def __init__(self, experts, budget, device, use_routing=False):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.register_buffer('costs', torch.tensor([e.cost for e in experts], device=device))
        self.budget = budget
        self.use_routing = use_routing
        self.device = device

        if self.use_routing:
            self.router = Router2D(len(experts)).to(device)
        else:
            self.theta = nn.Parameter(torch.zeros(len(experts)))

    def get_weights(self, x):
        if self.use_routing:
            logits = self.router(x)
            return F.softmax(logits, dim=-1) # (Batch, Experts)
        else:
            return F.softmax(self.theta, dim=0) # (Experts)

    def get_cost(self, weights):
        if weights.dim() == 2:
            # Routing: weighted average per sample, then mean over batch
            batch_costs = torch.matmul(weights, self.costs)
            return batch_costs.mean()
        else:
            # Global: dot product
            return torch.dot(weights, self.costs)

    def forward(self, x):
        # x: (Batch, H, W)
        w = self.get_weights(x)

        # Stack expert outputs: (Batch, H, W, Experts)
        # Note: All experts return (Batch, H, W) thanks to wrapper
        outputs = torch.stack([e(x) for e in self.experts], dim=-1)

        # Broadcast weights for 2D Grid
        if w.dim() == 1:
            # Global weights: (Experts) -> (1, 1, 1, Experts)
            w_exp = w.view(1, 1, 1, -1)
        else:
            # Routing weights: (Batch, Experts) -> (Batch, 1, 1, Experts)
            w_exp = w.view(x.shape[0], 1, 1, -1)

        # Weighted Sum
        return (outputs * w_exp).sum(dim=-1)

# The Logic classes (Softmax, Lagrangian, ADMM) remain structurally identical
# because they operate on scalar Loss/Cost values.
# We just inherit from the new BaseCombiner.

class SoftmaxCombiner(BaseCombiner):
    def training_step(self, x, pred, y, criterion):
        loss = criterion(pred, y)
        w = self.get_weights(x)
        cost = self.get_cost(w)
        return loss, cost.item(), 0.0

class LagrangianCombiner(BaseCombiner):
    def __init__(self, experts, budget, device, use_routing=False):
        super().__init__(experts, budget, device, use_routing)
        self.lam = nn.Parameter(torch.tensor(0.0, device=device))
        self.loss_scale = 10.0

    def training_step(self, x, pred, y, criterion):
        mse = criterion(pred, y)
        w = self.get_weights(x)
        cost = self.get_cost(w)
        viol = F.relu((cost - self.budget) / self.budget)
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

        # ADMM Residuals
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
            # Projection onto z <= Budget
            new_z_norm = torch.min(z_star, one_tensor)
            self.z.copy_(new_z_norm * self.budget)
            # Dual update
            dual_residual = scaled_cost - new_z_norm
            self.u.add_(dual_residual)
            self.u.clamp_(-10.0, 10.0)


class PINNLoss2D(nn.Module):
    def __init__(self, mse_weight=10.0, physics_weight=1.0, nu=0.01, beta=(1.0, 1.0)):
        super().__init__()
        self.mse_weight = mse_weight
        self.physics_weight = physics_weight
        self.nu = nu
        self.beta_x, self.beta_y = beta
        
        # 2D Central Difference Kernels
        # x-derivative: [[-0.5, 0, 0.5]]
        # y-derivative: [[-0.5], [0], [0.5]]
        self.register_buffer('k_x', torch.tensor([[[[-0.5, 0.0, 0.5]]]]))
        self.register_buffer('k_y', torch.tensor([[[[-0.5], [0.0], [0.5]]]]))
        
        # 5-point stencil for 2D Laplacian
        k_lap = torch.tensor([[[[0.0,  1.0, 0.0],
                                [1.0, -4.0, 1.0],
                                [0.0,  1.0, 0.0]]]])
        self.register_buffer('k_lap', k_lap)

    def forward(self, pred, target):
        # 1. Data Loss (MSE)
        mse = F.mse_loss(pred, target)
        
        # 2. Physics Loss
        # Ensure input is (Batch, 1, H, W)
        u = pred.unsqueeze(1) if pred.dim() == 3 else pred
        h, w = u.shape[-2], u.shape[-1]
        dx, dy = 1.0/w, 1.0/h
        
        # Periodic Padding for boundary consistency
        u_pad_x = F.pad(u, (1, 1, 0, 0), mode='circular')
        u_pad_y = F.pad(u, (0, 0, 1, 1), mode='circular')
        u_pad_lap = F.pad(u, (1, 1, 1, 1), mode='circular')
        
        # Compute Derivatives
        ux = F.conv2d(u_pad_x, self.k_x) / dx
        uy = F.conv2d(u_pad_y, self.k_y) / dy
        ulap = F.conv2d(u_pad_lap, self.k_lap) / (dx * dy)
        
        # Residual of the 2D Advection-Diffusion Equation
        # We assume the model predicts the spatial state; 
        # for a steady-state or implicit residual:
        residual = (self.beta_x * ux + self.beta_y * uy) - (self.nu * ulap)
        physics_loss = torch.mean(residual**2)
        
        return (self.mse_weight * mse) + (self.physics_weight * physics_loss)
        


# ==========================================
# 4. Utilities & Training (2D Adapted)
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
             opt_dual = optim.Adam(primal_params, lr=-1e-6, weight_decay=1e-8, )

    loss_fn = PINNLoss2D(mse_weight=10.0, physics_weight=0.5).to(device)

    # Track stats
    history = {
        'train_loss': [], 'val_loss': [], 'test_loss': [],
        'train_mse': [], 'val_mse': [], 'test_mse': [],
        'cost': []
    }

    # Run for 50 epochs (enough to see convergence)
    for epoch in range(50):
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

def train_experts_2d(device):
    # Reduced resolutions/samples for speed in this example
    resolutions = [32, 64, 128]
    costs = [1.0, 2.0, 4.0]
    experts = []

    print("Pre-training 2D Experts...")
    for res, cost in zip(resolutions, costs):
        print(f"  Training Expert (Res {res}, Cost {cost})...")
        # Ensure Expert2D and generate_2d_data are defined (from previous steps)
        train_x, train_y = generate_2d_data(num_samples=20, nx=res, steps=20, device=device)
        dataset = TensorDataset(train_x, train_y)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        model = Expert2D().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = PINNLoss2D(mse_weight=10.0, physics_weight=0.5).to(device)

        for _ in range(5): # Quick pre-train
            for bx, by in loader:
                optimizer.zero_grad()
                pred = model(bx)
                loss = loss_fn(pred, by)
                loss.backward()
                optimizer.step()

        experts.append(MultiResExpertWrapper2D(model, res, cost))
    return experts

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running 2D Pipeline on {device}")

    # 1. Train Experts
    experts = train_experts_2d(device)
    for e in experts:
        e.eval()
        for p in e.parameters(): p.requires_grad = False

    results = {}
    # 2. Generate Dataset (High Res Target)
    print("Generating High-Res (64x64) Datasets...")
    # Train/Val/Test
    train_x, train_y = generate_2d_data(num_samples=5000, nx=64, steps=20, device=device)
    test_x, test_y = generate_2d_data(num_samples=1000, nx=64, steps=20, device=device)

    val_x, val_y = generate_2d_data(num_samples=1000, nx=128, steps=20, device=device,)


    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=8, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=8)

    BUDGET = 2.5

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
    loss_fn = PINNLoss2D(mse_weight=10.0, physics_weight=0.5).to(device)

    total_mse = 0
    steps = 0
    with torch.no_grad():
        for bx, by in loader:
            pred = model(bx)
            total_mse += loss_fn(pred, by).item()
            steps += 1
    return total_mse / steps

def evaluate_uniform_baseline(experts, loader, device):
    """
    Evaluates a simple average ensemble (1/N weight for all experts).
    """
    loss_fn = PINNLoss2D(mse_weight=10.0, physics_weight=0.5).to(device)
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
            total_mse += loss_fn(ensemble_pred, by).item()
            steps += 1
    return total_mse / steps, avg_cost

def train_combiner(combiner, train_loader, val_loader, test_loader, budget, device, name="Model"):
    print(f"\n>>> Training {name} <<<")

    # 1. Separate Parameters
    primal_params = [p for n, p in combiner.named_parameters() if 'lam' not in n]
    dual_params = [p for n, p in combiner.named_parameters() if 'lam' in n]

    # 2. Optimizers
    opt_primal = optim.Adam(primal_params, lr=1e-5, weight_decay=1e-7)
    opt_dual = None
    if dual_params:
        try:
             opt_dual = optim.Adam(primal_params, lr=1e-6, weight_decay=1e-8, maximize=True)
        except TypeError:
             opt_dual = optim.Adam(primal_params, lr=-1e-6, weight_decay=1e-8, )

    loss_fn = PINNLoss2D(mse_weight=10.0, physics_weight=0.5).to(device)

    history = {'test_mse': [], 'cost': []} # Simplified history for summary

    for epoch in range(30):
        # --- TRAIN ---
        combiner.train()
        epoch_cost = 0

        for bx, by in train_loader:
            opt_primal.zero_grad()
            if opt_dual: opt_dual.zero_grad()

            pred = combiner(bx)
            _, cost, _ = combiner.training_step(bx, pred, by, loss_fn)

            # Re-calculate loss for backward to ensure graph connectivity
            mse = loss_fn(pred, by)
            if isinstance(combiner, AugLagrangianCombiner):
                w = combiner.get_weights(bx)
                viol = F.relu((combiner.get_cost(w) - combiner.budget) / combiner.budget)
                loss = (mse * combiner.loss_scale) + (combiner.lam * viol) + (combiner.rho/2 * viol**2)
            elif isinstance(combiner, ADMMCombiner):
                w = combiner.get_weights(bx)
                scaled_cost = combiner.get_cost(w) / combiner.budget
                scaled_z = combiner.z / combiner.budget
                residual = scaled_cost - scaled_z + combiner.u
                loss = (mse * combiner.loss_scale) + (combiner.rho/2 * residual**2)
            elif isinstance(combiner, LagrangianCombiner):
                 w = combiner.get_weights(bx)
                 viol = F.relu((combiner.get_cost(w) - combiner.budget) / combiner.budget)
                 loss = (mse * combiner.loss_scale) + (combiner.lam * viol)
            else: # Softmax
                 loss = mse

            loss.backward()
            torch.nn.utils.clip_grad_norm_(combiner.parameters(), 1.0)
            opt_primal.step()

            if opt_dual:
                opt_dual.step()
                with torch.no_grad():
                    for p in dual_params: p.clamp_(min=0.0)

            epoch_cost += cost

        avg_cost = epoch_cost / len(train_loader)

        # --- TEST ---
        combiner.eval()
        avg_test_mse = evaluate_standalone(combiner, test_loader, device)

        # Dual Updates
        if hasattr(combiner, 'update_dual') and not opt_dual:
            if isinstance(combiner, ADMMCombiner):
                combiner.update_dual(avg_cost)
            else:
                avg_viol = max(0, (avg_cost - budget)/budget)
                combiner.update_dual(avg_viol)

        history['test_mse'].append(avg_test_mse)
        history['cost'].append(avg_cost)

        if epoch % 5 == 0:
            print(f"Ep {epoch}: TestMSE {avg_test_mse:.5f} | Cost {avg_cost:.2f}")

    return history

def train_heterogeneous_experts_2d(device):
    """
    Trains 3 heterogeneous experts at Resolution 64
    """
    res = 64
    steps = 20

    # Assuming Expert2D, FiniteDifferenceExpert2D, FNO2D classes exist
    expert_configs = [
        {'name': 'FiniteDiff', 'model': FiniteDifferenceExpert2D(hidden_dim=32), 'cost': 1.0},
        {'name': 'CNN',       'model': Expert2D(hidden_channels=32),             'cost': 2.0},
        {'name': 'FNO',       'model': FNO2D(modes=12, width=32),                'cost': 4.0}
    ]

    experts = []
    expert_names = []
    print("\n=== Pre-training Heterogeneous 2D Experts ===")

    # Small training set for demonstration speed
    train_x, train_y = generate_2d_data(num_samples=30, nx=res, steps=steps, device=device)
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    for conf in expert_configs:
        print(f"Training {conf['name']} Expert (Cost {conf['cost']})...")
        model = conf['model'].to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = PINNLoss2D(mse_weight=10.0, physics_weight=0.5).to(device)

        for epoch in range(10):
            for bx, by in loader:
                optimizer.zero_grad()
                pred = model(bx)
                if pred.dim() == 4: pred = pred.squeeze(1)
                loss = loss_fn(pred, by)
                loss.backward()
                optimizer.step()

        experts.append(MultiResExpertWrapper2D(model, res, conf['cost']))
        expert_names.append(conf['name'])

    return experts, expert_names

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running 2D Heterogeneous Pipeline on {device}")

    # 1. Train Experts
    experts, expert_names = train_heterogeneous_experts_2d(device)
    for e in experts:
        e.eval()
        for p in e.parameters(): p.requires_grad = False

    print("\nGenerating Datasets...")
    train_x, train_y = generate_2d_data(num_samples=40, nx=64, steps=20, device=device)
    val_x, val_y     = generate_2d_data(num_samples=10, nx=64, steps=20, device=device)
    test_x, test_y   = generate_2d_data(num_samples=10, nx=64, steps=20, device=device)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=8, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_x, val_y), batch_size=8)
    test_loader  = DataLoader(TensorDataset(test_x, test_y), batch_size=8)

    BUDGET = 2.5
    results = {}

    # === A. Run Baselines ===
    print("\n>>> Evaluating Baselines <<<")

    # 1. Single Experts
    for i, expert in enumerate(experts):
        mse = evaluate_standalone(expert, test_loader, device)
        name = expert_names[i]
        results[f'Base_{name}'] = {'test_mse': [mse], 'cost': [expert.cost]}
        print(f"  {name}: MSE {mse:.5f}, Cost {expert.cost:.2f}")

    # 2. Uniform Average (1/3 FD + 1/3 CNN + 1/3 FNO)
    uni_mse, uni_cost = evaluate_uniform_baseline(experts, test_loader, device)
    results['Base_Uniform'] = {'test_mse': [uni_mse], 'cost': [uni_cost]}
    print(f"  Uniform Ensemble: MSE {uni_mse:.5f}, Cost {uni_cost:.2f}")

    # === B. Run Combiners ===

    # Softmax
    model = SoftmaxCombiner(experts, BUDGET, device, use_routing=False).to(device)
    results['Softmax'] = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "Softmax")

    # Lagrangian (Global)
    model = LagrangianCombiner(experts, BUDGET, device, use_routing=False).to(device)
    results['Lagrangian'] = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "Lagrangian")

    # AugLag (Routing)
    model = AugLagrangianCombiner(experts, BUDGET, device, use_routing=True).to(device)
    results['AugLag_Routing'] = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "AugLag (Routing)")

    # ADMM (Routing)
    admm_model = ADMMCombiner(experts, BUDGET, device, use_routing=True).to(device)
    results['ADMM_Routing'] = train_combiner(admm_model, train_loader, val_loader, test_loader, BUDGET, device, "ADMM (Routing)")

    # === Final Summary ===
    print(f"\n=== Final Test Summary (Budget: {BUDGET}) ===")
    print(f"{'Method':<20} | {'Test MSE':<10} | {'Cost':<10} | {'Status'}")

    # Sort keys to show baselines first, then models
    keys = sorted(results.keys())

    for k in keys:
        mse = results[k]['test_mse'][-1]
        cost = results[k]['cost'][-1]
        stat = "OK" if cost <= BUDGET + 0.1 else "VIOLATION"

        # Mark Baselines differently
        if "Base_" in k:
            stat = "REF"

        print(f"{k:<20} | {mse:.5f} | {cost:.2f}     | {stat}")

if __name__ == "__main__":
    main()


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


# ==========================================
# 6. Utilities & Training (2D)
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
    opt_primal = optim.Adam(primal_params, lr=1e-5, weight_decay=1e-7)

    opt_dual = None
    if dual_params:
        try:
             opt_dual = optim.Adam(primal_params, lr=1e-6, weight_decay=1e-8, maximize=True)
        except TypeError:
             opt_dual = optim.Adam(primal_params, lr=-1e-6, weight_decay=1e-8,)

    loss_fn = PINNLoss2D(mse_weight=10.0, physics_weight=0.5).to(device)

    history = {
        'train_loss': [], 'val_loss': [], 'test_loss': [],
        'cost': [], "test_mse":[],
    }

    # Reduced epochs for 2D computation
    for epoch in range(20):
        # --- TRAIN ---
        combiner.train()
        epoch_train_loss = 0
        epoch_cost = 0

        for bx, by in train_loader:
            opt_primal.zero_grad()
            if opt_dual: opt_dual.zero_grad()

            pred = combiner(bx)
            # bx has shape (Batch, H, W), pred has shape (Batch, H, W)

            _, cost, _ = combiner.training_step(bx, pred, by, loss_fn)
            # Recalculate loss for backward (training_step returns floats often, need graph)
            # NOTE: Ideally training_step returns the tensor loss for backward.
            # Re-calling it or modifying it to return tensor is best.
            # Assuming standard implementation here:
            mse = loss_fn(pred, by)

            # Re-implementing logic inline for clarity if class method returns floats
            # (Use the logic from your Combiner class)
            if isinstance(combiner, AugLagrangianCombiner):
                w = combiner.get_weights(bx)
                curr_cost = combiner.get_cost(w)
                viol = F.relu((curr_cost - combiner.budget) / combiner.budget)
                loss = (mse * combiner.loss_scale) + (combiner.lam * viol) + (combiner.rho/2 * viol**2)
            elif isinstance(combiner, ADMMCombiner):
                 # Simplified ADMM inline for brevity
                 w = combiner.get_weights(bx)
                 curr_cost = combiner.get_cost(w)
                 scaled_cost = curr_cost / combiner.budget
                 scaled_z = combiner.z / combiner.budget
                 residual = scaled_cost - scaled_z + combiner.u
                 loss = (mse * combiner.loss_scale) + (combiner.rho/2 * residual**2)
            elif isinstance(combiner, SoftmaxCombiner):
                 loss = mse
            else: # Lagrangian
                 w = combiner.get_weights(bx)
                 curr_cost = combiner.get_cost(w)
                 viol = F.relu((curr_cost - combiner.budget) / combiner.budget)
                 loss = (mse * combiner.loss_scale) + (combiner.lam * viol)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(combiner.parameters(), 1.0)
            opt_primal.step()

            if opt_dual:
                opt_dual.step()
                with torch.no_grad():
                    for p in dual_params: p.clamp_(min=0.0)

            epoch_train_loss += loss.item()
            epoch_cost += cost

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_cost = epoch_cost / len(train_loader)

        # --- TEST ---
        combiner.eval()
        test_mse = 0
        with torch.no_grad():
            for bx, by in test_loader:
                pred = combiner(bx)
                test_mse += loss_fn(pred, by).item()
        avg_test_mse = test_mse / len(test_loader)

        # Dual Updates
        if hasattr(combiner, 'update_dual') and not opt_dual:
            if isinstance(combiner, ADMMCombiner):
                combiner.update_dual(avg_cost)
            else:
                avg_viol = max(0, (avg_cost - budget)/budget)
                combiner.update_dual(avg_viol)

        history['test_mse'].append(avg_test_mse)
        history['cost'].append(avg_cost)

        print(f"Ep {epoch}: Loss {avg_train_loss:.4f} | TestMSE {avg_test_mse:.5f} | Cost {avg_cost:.2f}")

    return history

def initialize_experts_2d(device):
    """
    Initializes 4 Data Source Experts sharing ONE FNO2D model.
    """
    print("\n=== Initializing Shared FNO2D and Data Sources ===")

    # 1. The Shared Backbone (2D)
    # Using the FNO2D class defined previously
    shared_fno = FNO2D(modes=12, width=32).to(device)

    # 2. Define 4 Sources
    sources_config = [
        # (IV Noise, Meas Noise, Cost)
        (0.01, 0.01, 4.0), # Source 1: High Quality
        (0.05, 0.05, 2.0), # Source 2: Moderate
        (0.10, 0.10, 1.0), # Source 3: Noisy
        (0.20, 0.20, 0.5)  # Source 4: Very Noisy
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
    train_x, train_y = generate_2d_data(num_samples=40, nx=64, steps=20, device=device)
    test_x, test_y   = generate_2d_data(num_samples=10, nx=64, steps=20, device=device)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=8, shuffle=True)
    # Reusing test set for val/test to save time in example
    val_loader   = DataLoader(TensorDataset(test_x, test_y), batch_size=8)
    test_loader  = DataLoader(TensorDataset(test_x, test_y), batch_size=8)

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