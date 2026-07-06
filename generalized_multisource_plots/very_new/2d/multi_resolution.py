import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader, TensorDataset
import sys
import time

import sympy.printing

import random
import copy


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


import torch
import math

class NavierStokes2D:
    def __init__(self, nx=64, ny=64, t_max=1.0, nu=1e-3, device='cpu'):
        self.nx = nx
        self.ny = ny
        self.t_max = t_max
        self.nu = nu
        self.device = device

        # Define Grid (Periodic, excluding last point)
        self.x = torch.linspace(0, 2*math.pi, nx + 1, device=device)[:-1]
        self.y = torch.linspace(0, 2*math.pi, ny + 1, device=device)[:-1]
        self.X, self.Y = torch.meshgrid(self.x, self.y, indexing='ij')

        # Define Wavenumbers for 2D FFT
        kx = torch.fft.fftfreq(nx, d=1/nx, device=device) * nx
        ky = torch.fft.fftfreq(ny, d=1/ny, device=device) * ny
        self.kx = kx.reshape(-1, 1)
        self.ky = ky.reshape(1, -1)

        # Laplacian in Fourier space: -(kx^2 + ky^2)
        self.k_sq = self.kx**2 + self.ky**2
        self.k_sq[0, 0] = 1.0
        self.inv_k_sq = 1.0 / self.k_sq
        self.inv_k_sq[0, 0] = 0.0

        # Precompute imaginary k for derivatives
        self.ikx = 1j * self.kx
        self.iky = 1j * self.ky

        # Higher-order dealiasing (3/2 rule for more aggressive filtering)
        k_max_x = nx * (2/3) / 2
        k_max_y = ny * (2/3) / 2
        self.dealias_filter = (torch.abs(self.kx) <= k_max_x).float() * (torch.abs(self.ky) <= k_max_y).float()
        self.dealias_filter = self.dealias_filter.to(device)

        # Forcing wavenumber band for sustained turbulence
        self.forcing_k_min = max(1, nx // 8)
        self.forcing_k_max = max(2, nx // 4)

    def solve_trajectory(self, w0, steps=100, forcing=None):
        """
        w0: Initial vorticity field of shape (nx, ny)
        forcing: Optional time-dependent forcing function
        """
        dt = self.t_max / steps
        w = w0.clone()
        trajectory = [w.clone()]

        # Crank-Nicolson factors for diffusion
        factor_lhs = 1 + 0.5 * dt * self.nu * self.k_sq
        factor_rhs = 1 - 0.5 * dt * self.nu * self.k_sq

        # RK4 coefficients for more accurate time stepping
        def compute_rhs(w_hat_current):
            """Compute right-hand side of vorticity equation"""
            # Apply dealiasing
            w_hat_filtered = w_hat_current * self.dealias_filter

            # Solve for streamfunction
            psi_hat = w_hat_filtered * self.inv_k_sq

            # Compute velocities
            u_hat = self.iky * psi_hat
            v_hat = -self.ikx * psi_hat

            # Compute vorticity gradients
            wx_hat = self.ikx * w_hat_filtered
            wy_hat = self.iky * w_hat_filtered

            # Transform to real space
            u = torch.fft.ifft2(u_hat).real
            v = torch.fft.ifft2(v_hat).real
            wx = torch.fft.ifft2(wx_hat).real
            wy = torch.fft.ifft2(wy_hat).real

            # Compute advection
            advection = u * wx + v * wy
            advection_hat = torch.fft.fft2(advection) * self.dealias_filter

            return -advection_hat  # Return negative for time derivative

        for step in range(steps):
            # RK4 integration for better accuracy
            w_hat = torch.fft.fft2(w) * self.dealias_filter

            k1 = compute_rhs(w_hat)
            k2 = compute_rhs(w_hat + 0.5 * dt * k1)
            k3 = compute_rhs(w_hat + 0.5 * dt * k2)
            k4 = compute_rhs(w_hat + dt * k3)

            # Combine using RK4 formula
            w_hat_new = w_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            # Apply diffusion using Crank-Nicolson
            w_hat_new = (w_hat_new * factor_rhs) / factor_lhs

            # Back to real space with aggressive clamping
            w = torch.fft.ifft2(w_hat_new).real
            w = torch.clamp(w, min=-30.0, max=30.0)  # Stricter clamping for high Re

            # Optional external forcing (for sustained turbulence)
            if forcing is not None:
                w += dt * forcing(step, self.X, self.Y)

            trajectory.append(w.clone())

        return torch.stack(trajectory)

    def _apply_periodic_distance(self, dx, dy):
        """Apply periodic boundary conditions to distance calculations"""
        dx.copy_(torch.where(dx > math.pi, dx - 2*math.pi, dx))
        dx.copy_(torch.where(dx < -math.pi, dx + 2*math.pi, dx))
        dy.copy_(torch.where(dy > math.pi, dy - 2*math.pi, dy))
        dy.copy_(torch.where(dy < -math.pi, dy + 2*math.pi, dy))


def generate_2d_data(num_samples=10, nx=64, steps=50, device='cpu'):
    """
    Generates highly complex 2D Navier-Stokes vorticity data with matched 1D Reynolds numbers.
    Features:
    - Same viscosity range as 1D Burgers: [0.001, 0.005] for comparable Re
    - Extreme initial conditions: intense vortex collisions, sharp gradients
    - Multi-scale forcing for sustained turbulence
    - Enhanced numerical stability with RK4 + dealiasing
    - Extreme vorticity magnitudes and rapid dynamics
    """
    if num_samples > 5:
        print(f"Generating {num_samples} extremely challenging 2D NS trajectories at res {nx}x{nx}...")
    ny = nx
    data_x, data_y = [], []

    for i in range(num_samples):
        # MATCH 1D BURGERS REYNOLDS NUMBER RANGE: [0.001, 0.005]
        t_max = 0.2 + 0.6 * torch.rand(1, device=device).item()  # Shorter times for more dynamics
        nu = 0.001 + 0.004 * torch.rand(1, device=device).item()  # SAME AS 1D BURGERS: [0.001, 0.005]

        physics = NavierStokes2D(nx=nx, ny=nx, t_max=t_max, nu=nu, device=device)

        # Enhanced forcing function for sustained turbulence
        def forcing(step, X, Y):
            """Time-dependent random forcing in intermediate wavenumbers"""
            if step % 5 != 0:  # Apply every 5 steps
                return torch.zeros_like(X)

            # Random Fourier modes in forcing band
            f_hat = torch.zeros((nx, nx), dtype=torch.complex64, device=device)

            n_forcing_modes = 3
            for _ in range(n_forcing_modes):
                kx_f = torch.randint(physics.forcing_k_min, physics.forcing_k_max, (1,)).item()
                ky_f = torch.randint(physics.forcing_k_min, physics.forcing_k_max, (1,)).item()

                if torch.rand(1).item() > 0.5:
                    kx_f = -kx_f
                if torch.rand(1).item() > 0.5:
                    ky_f = -ky_f

                phase = 2 * math.pi * torch.rand(1, device=device).item()
                amp = 0.5 + 1.0 * torch.rand(1, device=device).item()

                if 0 <= kx_f < nx and 0 <= ky_f < ny:
                    # FIX: Use torch.polar for complex exponential
                    f_hat[kx_f, ky_f] = torch.polar(torch.tensor(amp, device=device), torch.tensor(phase, device=device))

            # Transform to real space
            f = torch.fft.ifft2(f_hat).real

            # Localized forcing
            mask = torch.exp(-((X - math.pi)**2 + (Y - math.pi)**2) / (0.5**2))

            return 5.0 * f * mask  # Amplified forcing

        # EXTREME generation modes with sharp gradients and collisions
        mode = torch.randint(0, 5, (1,)).item()  # 5 modes for more diversity

        if mode == 0:
            # Mode 0: Intense turbulence with power-law + extreme events
            w0 = torch.zeros((nx, nx), device=device)
            n_modes = min(nx // 2, 40)  # More modes

            # Broadband spectrum with rare extreme amplitudes
            for _ in range(n_modes):
                kx_mode = torch.randint(-nx//3, nx//3, (1,)).item()
                ky_mode = torch.randint(-ny//3, ny//3, (1,)).item()

                if kx_mode == 0 and ky_mode == 0:
                    continue

                k_mag = math.sqrt(kx_mode**2 + ky_mode**2)

                # Heavy-tailed amplitude distribution (rare extreme events)
                if torch.rand(1).item() < 0.1:  # 10% chance of extreme amplitude
                    amp_scale = 3.0
                else:
                    amp_scale = 1.0

                amp = amp_scale * torch.randn(1, device=device).item() / (k_mag ** (5/6) + 0.3)
                phase = 2 * math.pi * torch.rand(1, device=device).item()

                # Use both sin and cos for complex phase relationships
                w0 += amp * (torch.cos(kx_mode * physics.X + ky_mode * physics.Y + phase) +
                           0.5 * torch.sin(kx_mode * physics.X - ky_mode * physics.Y + phase))

            # Normalize to extreme magnitude
            w0 = w0 / (torch.std(w0) + 1e-6) * (2.0 + 2.0 * torch.rand(1, device=device).item())

        elif mode == 1:
            # Mode 1: EXTREME vortex dipole collisions
            w0 = torch.zeros((nx, nx), device=device)

            # Create strong dipole pair moving toward each other
            y_center1 = math.pi - 0.8
            y_center2 = math.pi + 0.8
            x_center = math.pi + 0.3 * (torch.rand(1, device=device).item() - 0.5)

            # First vortex (positive)
            dx1 = physics.X - x_center
            dy1 = physics.Y - y_center1
            physics._apply_periodic_distance(dx1, dy1)
            r_sq1 = dx1**2 + dy1**2
            vortex1 = 8.0 * torch.exp(-r_sq1 / (0.15**2))

            # Second vortex (negative, stronger)
            dx2 = physics.X - x_center
            dy2 = physics.Y - y_center2
            physics._apply_periodic_distance(dx2, dy2)
            r_sq2 = dx2**2 + dy2**2
            vortex2 = -10.0 * torch.exp(-r_sq2 / (0.12**2))

            w0 = vortex1 + vortex2

            # Add turbulent background
            bg = 0.3 * torch.randn_like(w0) * torch.sin(4 * physics.X) * torch.cos(4 * physics.Y)
            w0 += bg

        elif mode == 2:
            # Mode 2: Sharp shear layer with high perturbation amplitude
            w0 = torch.zeros((nx, nx), device=device)

            shear_strength = 5.0 + 5.0 * torch.rand(1, device=device).item()
            layer_thickness = 0.05 + 0.1 * torch.rand(1, device=device).item()
            y_center = math.pi + 0.8 * (torch.rand(1, device=device).item() - 0.5)

            # Very sharp tanh profile
            w0 = shear_strength * torch.tanh((physics.Y - y_center) / layer_thickness)

            # High-amplitude perturbations
            n_pert = 4
            for j in range(n_pert):
                kx_pert = 2 + torch.randint(0, 6, (1,)).item()
                amp_pert = 1.0 + 1.5 * torch.rand(1, device=device).item()
                phase_pert = 2 * math.pi * torch.rand(1, device=device).item()

                w0 += amp_pert * torch.sin(kx_pert * physics.X + phase_pert) * \
                      torch.exp(-((physics.Y - y_center) / layer_thickness)**4)

            # Small-scale chaotic noise
            noise_freq = 12
            noise = 0.5 * torch.randn_like(w0) * torch.sin(noise_freq * physics.X + 2*physics.Y)
            w0 += noise

        elif mode == 3:
            # Mode 3: Vortex street/chain (periodic array)
            w0 = torch.zeros((nx, nx), device=device)

            n_vortices = 6
            for j in range(n_vortices):
                angle = 2 * math.pi * j / n_vortices
                radius = 0.8 + 0.4 * torch.rand(1, device=device).item()
                x_center = math.pi + radius * math.cos(angle)  # FIX: math.cos
                y_center = math.pi + radius * math.sin(angle)  # FIX: math.sin

                # Alternating sign vortices
                strength = 5.0 * ((-1)**j)
                vortex_radius = 0.1 + 0.1 * torch.rand(1, device=device).item()

                dx = physics.X - x_center
                dy = physics.Y - y_center
                physics._apply_periodic_distance(dx, dy)
                r_sq = dx**2 + dy**2

                w0 += strength * torch.exp(-r_sq / (2 * vortex_radius**2))

            # Add radial perturbations
            w0 += 1.0 * torch.randn_like(w0) * torch.sin(physics.X + physics.Y)

        else:
            # Mode 4: Merging vortex patches with extreme gradients
            w0 = torch.zeros((nx, nx), device=device)
            n_patches = torch.randint(6, 10, (1,)).item()

            for j in range(n_patches):
                # Closely spaced patches for interaction
                angle = 2 * math.pi * j / n_patches  # Use math for float operations
                x_center = math.pi + 0.6 * math.cos(angle)  # FIX: math.cos
                y_center = math.pi + 0.6 * math.sin(angle)  # FIX: math.sin

                strength = (torch.rand(1, device=device).item() - 0.5) * 8.0
                radius = 0.08 + 0.12 * torch.rand(1, device=device).item()

                dx = physics.X - x_center
                dy = physics.Y - y_center
                physics._apply_periodic_distance(dx, dy)
                r_sq = dx**2 + dy**2

                # Sharp-edged patches
                w0 += strength * torch.exp(-r_sq / (2 * radius**2)) * torch.cos(3 * torch.sqrt(r_sq) / radius)

            # Add background strain
            strain = 2.0 * (physics.X - math.pi) * (physics.Y - math.pi) / (math.pi**2)
            w0 += strain

        # Add deterministic extreme perturbations
        if i % 3 == 0:
            # Shock-like discontinuity (sharp front)
            front_x = 0.5 * math.pi + 1.5 * math.pi * torch.rand(1, device=device).item()
            width = 0.02
            shock = 3.0 * torch.tanh((physics.X - front_x) / width)
            w0 += shock

        # Ultra-fine scale noise
        if i % 2 == 0:
            hf_noise = torch.randn((nx//2, nx//2), device=device) * 2.0
            hf_noise = torch.nn.functional.interpolate(hf_noise.unsqueeze(0).unsqueeze(0), size=(nx, nx), mode='bilinear', align_corners=False).squeeze()
            w0 += hf_noise

        # Extreme normalization for high amplitude
        w0 = w0 / (torch.max(torch.abs(w0)) + 1e-6) * (3.0 + 2.0 * torch.rand(1, device=device).item())

        # Solve trajectory WITH FORCING for sustained extreme dynamics
        traj = physics.solve_trajectory(w0, steps=steps, forcing=forcing)

        # Safety check
        if not torch.isfinite(traj).all():
            print(f"Warning: Non-finite values detected in sample {i}, regenerating with fallback...")
            w0 = 2.0 * torch.sin(physics.X) * torch.cos(2 * physics.Y)
            traj = physics.solve_trajectory(w0, steps=steps)

        data_x.append(traj[:-1])
        data_y.append(traj[1:])

        if (i+1) % 5 == 0:
            print(f"  Sample {i+1}/{num_samples} done (nu={nu:.4f}).")

    return torch.cat(data_x), torch.cat(data_y)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# 2. Expert Models (2D Versions)
# ==========================================

# --- A. Standard CNN 2D (Baseline Expert) ---
# --- A. Standard CNN 2D (Baseline Expert) ---
class Expert2D(nn.Module):
    def __init__(self, hidden_channels=32, resolution=32, base_res=32):
        super().__init__()
        
        # Scale dilation to maintain physical receptive field
        # resolution=32 -> dilation=1 (Standard Conv)
        # resolution=64 -> dilation=2
        # resolution=128 -> dilation=4
        dilation = max(1, resolution // base_res)
        
        # Calculate padding to ensure Output Size == Input Size
        # Formula: padding = dilation * (kernel_size - 1) / 2
        # For kernel_size=5: padding = dilation * 2
        padding = 2 * dilation

        self.net = nn.Sequential(
            # Input: (Batch, 1, H, W) -> Output: (Batch, Hidden, H, W)
            nn.Conv2d(1, hidden_channels, 5, padding=padding, dilation=dilation, padding_mode='circular'),
            nn.Tanh(),
            nn.Conv2d(hidden_channels, hidden_channels, 5, padding=padding, dilation=dilation, padding_mode='circular'),
            nn.Tanh(),
            nn.Conv2d(hidden_channels, 1, 5, padding=padding, dilation=dilation, padding_mode='circular')
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


class FNORouter2D(nn.Module):
    def __init__(self, modes=12, width=32, num_experts=4):
        super().__init__()
        # Use your existing FNO2D architecture
        self.fno = FNO2D(modes=modes, width=width)

        # Add a head that collapses the spatial dimensions (H, W)
        # to produce a single vector of weights per batch item
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width, num_experts)

    def forward(self, x):
        # 1. Standard FNO Forward (up to the spectral layers)
        # We modify the FNO2D slightly or use its internal features
        # For simplicity, let's assume FNO2D returns (Batch, H, W, Width)
        # before its final projection

        # If using your exact FNO2D:
        # It currently returns (Batch, H, W). We need (Batch, Experts).
        # We can reach into the FNO logic:

        grid = self.fno.get_grid(x.shape, x.device)
        x = torch.cat((x.unsqueeze(-1), grid), dim=-1)
        x = self.fno.fc0(x)
        x = x.permute(0, 3, 1, 2) # (Batch, Width, H, W)

        x1 = self.fno.conv0(x)
        x2 = self.fno.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.fno.conv1(x)
        x2 = self.fno.w1(x)
        x = F.gelu(x1 + x2)

        # --- THE ROUTER HEAD ---
        # Instead of projecting back to (H, W, 1), we pool spatially
        x = self.adaptive_pool(x) # Shape: (Batch, Width, 1, 1)
        x = torch.flatten(x, 1)    # Shape: (Batch, Width)
        logits = self.fc(x)        # Shape: (Batch, num_experts)
        return logits

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
    def __init__(self, experts, device, use_routing=False):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.register_buffer('costs', torch.tensor([e.cost for e in experts], device=device))
        #self.budget = budget
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

    def get_logits(self, x):
      if self.use_routing:
          return self.router(x)
      else:
          return self.theta # Returns the raw parameter vector
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

        total_loss = mse_loss +  self.lam_budget * budget_viol #self.lam_sum * sum_viol.mean() +
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




class PINNLoss2D(nn.Module):
    def __init__(self, mse_weight=10.0, physics_weight=1.0, nu=1e-3, dt=None):
        super().__init__()
        self.mse_weight = mse_weight
        self.physics_weight = physics_weight
        self.nu = nu
        self.dt = dt  # Time step size (needed for time derivative approximation)

    def get_velocity_and_grads(self, w):
        """
        Calculates velocity (u, v) and gradients (wx, wy, lap) from vorticity w
        using differentiable FFT operations.
        """
        batch, h, w_dim = w.shape
        # Create wavenumbers on the fly to match batch device
        device = w.device
        nx, ny = h, w_dim
        
        kx = torch.fft.fftfreq(nx, d=1/nx, device=device) * nx
        ky = torch.fft.fftfreq(ny, d=1/ny, device=device) * ny
        kx = kx.reshape(-1, 1)
        ky = ky.reshape(1, -1)
        
        k_sq = kx**2 + ky**2
        k_sq[0, 0] = 1.0  # Avoid division by zero
        
        # FFT
        w_hat = torch.fft.fftn(w, dim=(-2, -1))
        
        # 1. Solve Streamfunction: lap(psi) = -w  => psi_hat = w_hat / k_sq
        psi_hat = w_hat / k_sq
        
        # 2. Compute Velocities: u = psi_y, v = -psi_x
        # In Fourier: u_hat = iky * psi_hat, v_hat = -ikx * psi_hat
        u_hat = 1j * ky * psi_hat
        v_hat = -1j * kx * psi_hat
        
        # 3. Compute Gradients of Vorticity
        wx_hat = 1j * kx * w_hat
        wy_hat = 1j * ky * w_hat
        wlap_hat = -k_sq * w_hat
        
        # IFFT back to real
        u = torch.fft.ifftn(u_hat, dim=(-2, -1)).real
        v = torch.fft.ifftn(v_hat, dim=(-2, -1)).real
        wx = torch.fft.ifftn(wx_hat, dim=(-2, -1)).real
        wy = torch.fft.ifftn(wy_hat, dim=(-2, -1)).real
        wlap = torch.fft.ifftn(wlap_hat, dim=(-2, -1)).real
        
        return u, v, wx, wy, wlap

    def forward(self, pred_next, target_next, input_prev=None):
        """
        pred_next: The model's prediction for t+1
        target_next: Ground truth for t+1 (for MSE)
        input_prev: The state at t (optional, for temporal derivative)
        """
        # 1. Data Loss
        mse = F.mse_loss(pred_next, target_next)
        
        # 2. Physics Loss (Residual)
        # We check if the predicted state satisfies the NS equation
        u, v, wx, wy, wlap = self.get_velocity_and_grads(pred_next)
        
        # Non-linear Advection Term: (u * wx + v * wy)
        advection = u * wx + v * wy
        
        # Diffusion Term: nu * lap(w)
        diffusion = self.nu * wlap
        
        # Residual: w_t + Advection - Diffusion = 0
        # If we don't have w_t (time derivative), we can punish the spatial structure
        # assuming the update was correct. 
        # Ideally, residual = (pred_next - input_prev)/dt + advection - diffusion
        
        if input_prev is not None and self.dt is not None:
            w_t = (pred_next - input_prev) / self.dt
            residual = w_t + advection - diffusion
        else:
            # Fallback: Just regularize spatial smoothness consistency
            # (Less physically strict, but enforces NS relationships)
            residual = advection - diffusion 
            
        physics_loss = torch.mean(residual**2)

        return (self.mse_weight * mse) + (self.physics_weight * physics_loss)
    
# ==========================================
# 4. Utilities & Training (2D Adapted)
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

    # 1. Separate Parameters for Two Optimizers (Primal and Dual)
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
    opt_dual = optim.Adam(dual_params, lr=1e-3, maximize=True) if dual_params else None

    loss_fn = PINNLoss2D(mse_weight=10.0, physics_weight=1e-3).to(device)

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
                    combiner.lam_sum.add_(0.1 * (avg_sum - 1.0))
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
                val_mse += F.mse_loss(pred, by).item()

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
                test_mse += F.mse_loss(pred, by).item()
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


def train_experts_2d(device):
    resolutions = [32, 64, 128]
    costs = [1.0, 2.0, 4.0]
    experts = []

    # Track full history for 2D standalone performance
    pretrain_history = {
        res: {
            'train_loss': [], 'train_mse': [],
            'val_loss': [], 'val_mse': []
        } for res in resolutions
    }

    print("Pre-training 2D Experts...")
    for res, cost in zip(resolutions, costs):
        print(f"\n>>> Starting Expert Res: {res} (Native Resolution Training)")

        # 1. Data Generation (Matching native expert resolution)
        train_x, train_y = generate_2d_data(num_samples=100, nx=res, steps=20, device=device)
        val_x, val_y = generate_2d_data(num_samples=30, nx=res, steps=20, device=device)

        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=16, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=16)

        model = Expert2D(resolution=res, base_res=32).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        # Using 2D PINN Loss
        loss_fn = PINNLoss2D(mse_weight=10.0, physics_weight=1e-3).to(device)

        epochs = 50
        for epoch in range(epochs):
            # --- Training Phase ---
            model.train()
            epoch_tr_loss, epoch_tr_mse = 0, 0
            for bx, by in train_loader:
                optimizer.zero_grad()
                pred = model(bx)

                loss = loss_fn(pred, by)
                mse = F.mse_loss(pred, by)

                loss.backward()
                optimizer.step()

                epoch_tr_loss += loss.item()
                epoch_tr_mse += mse.item()

            # --- Validation Phase ---
            model.eval()
            epoch_val_loss, epoch_val_mse = 0, 0
            with torch.no_grad():
                for v_bx, v_by in val_loader:
                    v_pred = model(v_bx)
                    v_loss = loss_fn(v_pred, v_by)
                    v_mse = F.mse_loss(v_pred, v_by)

                    epoch_val_loss += v_loss.item()
                    epoch_val_mse += v_mse.item()

            # Average and Record
            avg_tr_l = epoch_tr_loss / len(train_loader)
            avg_tr_m = epoch_tr_mse / len(train_loader)
            avg_vl_l = epoch_val_loss / len(val_loader)
            avg_vl_m = epoch_val_mse / len(val_loader)

            pretrain_history[res]['train_loss'].append(avg_tr_l)
            pretrain_history[res]['train_mse'].append(avg_tr_m)
            pretrain_history[res]['val_loss'].append(avg_vl_l)
            pretrain_history[res]['val_mse'].append(avg_vl_m)

            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f" [Res {res:3}] Ep {epoch:2} | TrMSE: {avg_tr_m:.6f} | VlMSE: {avg_vl_m:.6f} | VlLoss: {avg_vl_l:.6f}")

        experts.append(MultiResExpertWrapper2D(model, res, cost))

    return experts, [str(a) for a in resolutions], pretrain_history

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running 2D Pipeline on {device}")

    # 1. Train Experts
    experts, expert_names, _ = train_experts_2d(device)
    for e in experts:
        e.eval()
        #for p in e.parameters(): p.requires_grad = False

    results = {}
    # 2. Generate Dataset (High Res Target)
    print("Generating High-Res (64x64) Datasets...")
    # Train/Val/Test
    train_x, train_y = generate_2d_data(num_samples=200, nx=128, steps=20, device=device)
    test_x, test_y = generate_2d_data(num_samples=50, nx=128, steps=20, device=device)

    val_x, val_y = generate_2d_data(num_samples=50, nx=128, steps=20, device=device,)


    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=32)


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
    #experts_config = [copy.deepcopy(experts)]*5
    # --- 1. Softmax ---
    model = SoftmaxCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "Softmax")
    results['Softmax'] = {'history': history, 'time': elapsed}
    # --- 2. Lagrangian ---
    experts, expert_names, _ = train_experts_2d(device)
    model = LagrangianCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "Lagrangian")
    results['Lagrangian'] = {'history': history, 'time': elapsed}

    experts, expert_names, _ = train_experts_2d(device)
    model = ImpAugLagrangianCombiner(experts, device, use_routing=True).to(device)
    history, elapsed = train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "ImpLagrangian")
    results['ImpAugLag_Routing'] = {'history': history, 'time': elapsed}

    # --- 3. Aug Lagrangian ---
    experts, expert_names, _ = train_experts_2d(device)
    model = AugLagrangianCombiner(experts, device, use_routing=True).to(device)
    history, elapsed =  train_combiner(model, train_loader, val_loader, test_loader, BUDGET, device, "AugLag (Routing)")
    results['AugLag_Routing'] = {'history': history, 'time': elapsed}
    # --- 4. ADMM ---
    experts, expert_names, _ = train_experts_2d(device)
    admm_model = ADMMCombiner(experts, device, use_routing=True).to(device)
    history, elapsed =  train_combiner(admm_model, train_loader, val_loader, test_loader, BUDGET, device, "ADMM (Routing)")
    results['ADMM_Routing'] = {'history': history, 'time': elapsed}

    # --- Final Summary ---


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

