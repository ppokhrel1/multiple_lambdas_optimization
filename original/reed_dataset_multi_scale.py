import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import copy
import os
import json
import warnings
import math

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ========== 0. CORE COMPONENTS ==========
class FourierBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.w = nn.Conv2d(in_channels, out_channels, 1)
        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros_like(x_ft)
        
        freq_size_x = x_ft.shape[2]
        freq_size_y = x_ft.shape[3]
        
        modes_x = min(self.weights1.shape[2], freq_size_x)
        modes_y = min(self.weights1.shape[3], freq_size_y)
        
        out_ft[:, :, :modes_x, :modes_y] = torch.einsum(
            "bixy,ioxy->boxy", 
            x_ft[:, :, :modes_x, :modes_y], 
            self.weights1[:, :, :modes_x, :modes_y]
        )
        
        x_spec = torch.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))
        
        # 2. Linear Skip Block + Normalization
        x_lin = self.w(x_in)
        x_combined = x_spec + x_lin
        x_out = self.norm(x_combined)
        fourier_output = x_spec + x_lin
        
        # --- Final Structure for Stability (Recommended in FNO) ---
        return self.norm(fourier_output)



# ========== 1. ORIGINAL DATASETS ==========
class NavierStokesDataset(Dataset):
    """
    Complex Navier-Stokes (Vorticity) dataset with realistic multi-scale physics.
    """
    def __init__(self, n_samples: int, domain_size: int = 128, 
                 n_fourier_modes: int = 10, time_step: int = 5, 
                 reynolds: float = 1000.0, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.domain_size = domain_size
        self.samples = []
        self.reynolds = reynolds
        self.nu = 1.0 / reynolds
        
        x = torch.linspace(0, 1, domain_size)
        y = torch.linspace(0, 1, domain_size)
        self.grid_x, self.grid_y = torch.meshgrid(x, y, indexing='ij')
        
        # Pre-compute wave numbers for spectral methods
        self.kx = 2 * torch.pi * torch.fft.fftfreq(domain_size, d=1.0/domain_size)
        self.ky = 2 * torch.pi * torch.fft.fftfreq(domain_size, d=1.0/domain_size)
        kx, ky = torch.meshgrid(self.kx, self.ky, indexing='ij')
        self.k_sq = kx**2 + ky**2
        self.k_sq[0, 0] = 1.0  # Avoid division by zero
        
        # Multiple seeds for diversity
        seeds = np.random.randint(0, 1000000, n_samples)
        
        for i, sample_seed in enumerate(seeds):
            torch.manual_seed(sample_seed)
            np.random.seed(sample_seed)
            
            # 1. Generate multi-scale initial vorticity field
            omega_0 = self._multi_scale_initial_field(n_fourier_modes, i)
            
            # 2. Simulate Navier-Stokes with spectral method
            omega_t = self._spectral_navier_stokes(omega_0, time_step)
            
            # 3. Add random forcing and noise
            omega_t = self._add_physical_effects(omega_t, omega_0, i)
            
            self.samples.append({
                'grid': torch.stack([self.grid_x, self.grid_y, omega_0], dim=0),
                'solution': omega_t.unsqueeze(0),
                'parameters': {'Re': reynolds, 'time': time_step, 'seed': sample_seed}
            })
    
    def _spectral_navier_stokes(self, omega_0: torch.Tensor, time_step: int) -> torch.Tensor:
        """Optimized spectral method for 2D Navier-Stokes"""
        omega = omega_0.clone()
        dt = 0.01
        
        # Pre-compute masks for boundary conditions
        border_mask = torch.ones_like(omega)
        border_size = 3
        border_mask[:border_size, :] = 0
        border_mask[-border_size:, :] = 0
        border_mask[:, :border_size] = 0
        border_mask[:, -border_size:] = 0
        
        for t in range(time_step):
            # Stream function: ∇²ψ = -ω
            psi_ft = -torch.fft.fft2(omega) / (self.k_sq + 1e-8)
            psi_ft[0, 0] = 0  # Zero mean
            
            # Velocity from stream function
            psi_ft_kx = 1j * self.kx.unsqueeze(1) * psi_ft
            psi_ft_ky = 1j * self.ky.unsqueeze(0) * psi_ft
            
            u = torch.fft.ifft2(psi_ft_ky).real  # u = ∂ψ/∂y
            v = -torch.fft.ifft2(psi_ft_kx).real  # v = -∂ψ/∂x
            
            # Advection term: u·∇ω
            omega_ft = torch.fft.fft2(omega)
            omega_x_ft = 1j * self.kx.unsqueeze(0) * omega_ft
            omega_y_ft = 1j * self.ky.unsqueeze(1) * omega_ft
            
            omega_x = torch.fft.ifft2(omega_x_ft).real
            omega_y = torch.fft.ifft2(omega_y_ft).real
            
            advection = -(u * omega_x + v * omega_y)
            
            # Time integration (semi-implicit)
            omega_ft_new = (torch.fft.fft2(omega + dt * advection) / 
                        (1 + dt * self.nu * self.k_sq))
            omega = torch.fft.ifft2(omega_ft_new).real
            
            # Apply boundary conditions
            omega = omega * border_mask
            
            # Add minimal non-linear forcing
            if t % 5 == 0:
                forcing = 0.02 * torch.sin(torch.tensor(2.0) * torch.pi * t / time_step) * torch.sin(4 * torch.pi * omega)
                omega += dt * forcing
                omega = omega * border_mask
        
        return omega

    def _multi_scale_initial_field(self, n_modes: int, sample_id: int) -> torch.Tensor:
        """Generates initial field with multiple spatial scales"""
        domain_size = self.domain_size
        omega = torch.zeros(domain_size, domain_size)
        
        # Scale 1: Large-scale coherent structures (vortices)
        n_large_modes = n_modes // 4
        for _ in range(2 + sample_id % 3):
            amplitude = 0.5 + 0.5 * torch.randn(1).item()
            kx_large = torch.randint(1, n_large_modes, (1,)).item()
            ky_large = torch.randint(1, n_large_modes, (1,)).item()
            phase_x = 2 * torch.pi * torch.rand(1).item()
            phase_y = 2 * torch.pi * torch.rand(1).item()

            omega += amplitude * (
                torch.sin(2 * torch.pi * kx_large * self.grid_x + phase_x) *
                torch.sin(2 * torch.pi * ky_large * self.grid_y + phase_y)
            )
        
        # Scale 2: Medium-scale waves
        n_medium_modes = n_modes // 2
        for mode in range(n_medium_modes // 2):
            kx_med = 4 + mode * 2
            ky_med = 4 + mode * 2
            amplitude = 0.2 + 0.1 * torch.randn(1).item()
            phase = 2 * torch.pi * torch.rand(1).item()
            
            kx_grid = kx_med * self.grid_x * 2 * torch.pi
            ky_grid = ky_med * self.grid_y * 2 * torch.pi
            omega += amplitude * torch.sin(kx_grid + ky_grid + phase)
        
        # Scale 3: Small-scale turbulence (random Fourier modes)
        omega_ft = torch.fft.fft2(omega)
        mask_size = n_modes
        turbulence = torch.zeros_like(omega_ft)
        
        # Random phases and amplitudes with Kolmogorov-like spectrum
        for kx in range(mask_size):
            for ky in range(mask_size):
                if kx == 0 and ky == 0:
                    continue
                k = torch.sqrt(torch.tensor(kx**2 + ky**2).float())
                # Kolmogorov -5/3 spectrum
                amplitude = torch.randn(1).item() * k**(-5/6) / (1 + k**2)
                phase = 2 * torch.pi * torch.rand(1).item() # phase is a Python float
                
                # FIX: Convert the float 'phase' to a tensor before calling torch.cos/sin
                phase_tensor = torch.tensor(phase, dtype=torch.float32)
                complex_exp = torch.complex(torch.cos(phase_tensor), torch.sin(phase_tensor))
                
                turbulence[kx, ky] = amplitude * complex_exp
                
                # Hermitian symmetry (check if indices are within the range of the tensor)
                if kx > 0 and ky > 0:
                    if -kx >= -domain_size and -ky >= -domain_size: # Ensure negative indices are valid
                        turbulence[-kx, -ky] = torch.conj(turbulence[kx, ky])
                    
        turbulence = torch.fft.ifft2(turbulence).real
        omega += 0.3 * turbulence / turbulence.std()
        
        # Add localized features (vortex sheets, shocks)
        if sample_id % 4 == 0:
            # Vortex sheet
            x0 = torch.rand(1).item()
            y0 = torch.rand(1).item()
            sigma = 0.05 + 0.05 * torch.rand(1).item()
            
            r = torch.sqrt((self.grid_x - x0)**2 + (self.grid_y - y0)**2)
            vortex = torch.exp(-r**2 / (2 * sigma**2)) * torch.sin(8 * torch.pi * r / sigma)
            omega += 0.4 * vortex
        
        if sample_id % 5 == 0:
            # Shock-like discontinuity
            shock_pos = 0.3 + 0.4 * torch.rand(1).item()
            shock_width = 0.02 + 0.03 * torch.rand(1).item()
            shock_strength = 0.5 + 0.5 * torch.rand(1).item()
            
            shock = shock_strength * torch.tanh((self.grid_x - shock_pos) / shock_width)
            omega += shock * torch.exp(-10 * (self.grid_y - 0.5)**2)
        
        # Enforce boundary conditions (no-slip)
        omega[0, :] = omega[-1, :] = omega[:, 0] = omega[:, -1] = 0
        
        # Smooth near boundaries
        for i in range(5):
            omega[i, :] *= i / 5
            omega[-i-1, :] *= i / 5
            omega[:, i] *= i / 5
            omega[:, -i-1] *= i / 5
        
        # Normalize
        omega = omega / (torch.std(omega) + 1e-6) * (0.5 + 0.5 * torch.rand(1).item())
        
        return omega
    
    def _multi_scale_initial_field(self, n_modes: int, sample_id: int) -> torch.Tensor:
        """Vectorized version of initial field generation"""
        domain_size = self.domain_size
        omega = torch.zeros(domain_size, domain_size)
        
        # 1. Large-scale coherent structures - vectorized
        n_large_modes = n_modes // 4
        for _ in range(2 + sample_id % 3):
            amplitude = 0.5 + 0.5 * torch.randn(1).item()
            kx_large = torch.randint(1, n_large_modes, (1,)).item()
            ky_large = torch.randint(1, n_large_modes, (1,)).item()
            phase_x = 2 * torch.pi * torch.rand(1).item()
            phase_y = 2 * torch.pi * torch.rand(1).item()

            omega += amplitude * (
                torch.sin(2 * torch.pi * kx_large * self.grid_x + phase_x) *
                torch.sin(2 * torch.pi * ky_large * self.grid_y + phase_y)
            )
        
        # 2. Medium-scale waves - vectorized
        n_medium_modes = n_modes // 2
        for mode in range(n_medium_modes // 2):
            kx_med = 4 + mode * 2
            ky_med = 4 + mode * 2
            amplitude = 0.2 + 0.1 * torch.randn(1).item()
            phase = 2 * torch.pi * torch.rand(1).item()
            
            omega += amplitude * torch.sin(
                2 * torch.pi * kx_med * self.grid_x + 
                2 * torch.pi * ky_med * self.grid_y + 
                phase
            )
        
        # 3. Small-scale turbulence - VECTORIZED VERSION
        mask_size = n_modes
        kx = torch.arange(mask_size, device=self.grid_x.device)
        ky = torch.arange(mask_size, device=self.grid_y.device)
        
        # Create meshgrid
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        
        # Compute k magnitude
        k_mag = torch.sqrt(KX.float()**2 + KY.float()**2)
        
        # Kolmogorov -5/3 spectrum
        amplitude = torch.randn(mask_size, mask_size, device=self.grid_x.device) * k_mag**(-5/6) / (1 + k_mag**2)
        phase = 2 * torch.pi * torch.rand(mask_size, mask_size, device=self.grid_x.device)
        
        # Set DC component to zero
        amplitude[0, 0] = 0
        phase[0, 0] = 0
        
        # Create complex spectrum with Hermitian symmetry
        turbulence_ft = torch.zeros(domain_size, domain_size, dtype=torch.cfloat, device=self.grid_x.device)
        turbulence_ft[:mask_size, :mask_size] = amplitude * torch.exp(1j * phase)
        
        # Apply Hermitian symmetry
        turbulence_ft[-mask_size+1:, -mask_size+1:] = torch.conj(
            torch.rot90(turbulence_ft[1:mask_size, 1:mask_size], k=2)
        )
        
        turbulence = torch.fft.ifft2(turbulence_ft).real
        
        if turbulence.std() > 0:
            omega += 0.3 * turbulence / turbulence.std()
        
        # 4. Localized features - vectorized
        if sample_id % 4 == 0:
            x0 = torch.rand(1).item()
            y0 = torch.rand(1).item()
            sigma = 0.05 + 0.05 * torch.rand(1).item()
            
            r = torch.sqrt((self.grid_x - x0)**2 + (self.grid_y - y0)**2)
            vortex = torch.exp(-r**2 / (2 * sigma**2)) * torch.sin(8 * torch.pi * r / sigma)
            omega += 0.4 * vortex
        
        if sample_id % 5 == 0:
            shock_pos = 0.3 + 0.4 * torch.rand(1).item()
            shock_width = 0.02 + 0.03 * torch.rand(1).item()
            shock_strength = 0.5 + 0.5 * torch.rand(1).item()
            
            shock = shock_strength * torch.tanh((self.grid_x - shock_pos) / shock_width)
            omega += shock * torch.exp(-10 * (self.grid_y - 0.5)**2)
        
        # Boundary conditions - vectorized
        omega[0, :] = omega[-1, :] = omega[:, 0] = omega[:, -1] = 0
        
        # Smooth boundaries - vectorized
        border_size = 5
        x_weights = torch.linspace(0, 1, border_size)
        y_weights = torch.linspace(0, 1, border_size)
        
        omega[:border_size, :] *= x_weights.view(-1, 1)
        omega[-border_size:, :] *= x_weights.flip(0).view(-1, 1)
        omega[:, :border_size] *= y_weights.view(1, -1)
        omega[:, -border_size:] *= y_weights.flip(0).view(1, -1)
        
        # Normalize
        std = torch.std(omega)
        if std > 0:
            omega = omega / std * (0.5 + 0.5 * torch.rand(1).item())
        
        return omega
    
    def _add_physical_effects(self, omega_t: torch.Tensor, omega_0: torch.Tensor, sample_id: int) -> torch.Tensor:
        """Add realistic physical effects to the solution"""
        
        # 1. Non-linear energy cascade effects
        # Forward cascade: large → small scales
        omega_ft = torch.fft.fft2(omega_t)
        omega_ft_abs = torch.abs(omega_ft)
        
        # Apply scale-dependent damping (simulating dissipation)
        for kx in range(self.domain_size):
            for ky in range(self.domain_size):
                k = torch.sqrt(torch.tensor(kx**2 + ky**2).float())
                if k > 10:  # Damp high wavenumbers
                    damping = torch.exp(-0.1 * k / self.domain_size)
                    omega_ft[kx, ky] *= damping
        
        omega_t = torch.fft.ifft2(omega_ft).real
        
        # 2. Inverse cascade effects (energy transfer to large scales)
        if sample_id % 3 == 0:
            # Coherent structure formation
            large_scale = F.avg_pool2d(omega_t.unsqueeze(0).unsqueeze(0), 
                                      kernel_size=5, stride=1, padding=2).squeeze()
            omega_t = 0.7 * omega_t + 0.3 * large_scale
        
        # 3. Memory effects (dependence on initial condition)
        sample_input = torch.tensor(sample_id * 0.1, dtype=torch.float32)
        memory_weight = 0.1 + 0.05 * torch.sin(sample_input)
        omega_t = (1 - memory_weight) * omega_t + memory_weight * omega_0
        
        # 4. Add small-scale noise (subgrid effects)
        noise_amplitude = 0.01 + 0.005 * torch.randn(1).item()
        noise = noise_amplitude * torch.randn_like(omega_t)
        # Only add to interior (not boundaries)
        noise[0, :] = noise[-1, :] = noise[:, 0] = noise[:, -1] = 0
        omega_t += noise
        
        # 5. Non-linear saturation effects
        saturation = torch.tanh(omega_t * 2.0) / 2.0
        omega_t = 0.8 * omega_t + 0.2 * saturation
        
        # 6. Scale interactions (explicit multi-scale coupling)
        # Compute multi-scale decomposition
        scales = [32, 64, 128]
        multi_scale = torch.zeros_like(omega_t)
        
        for scale in scales:
            if scale < self.domain_size:
                downsampled = F.interpolate(
                    omega_t.unsqueeze(0).unsqueeze(0),
                    size=(scale, scale),
                    mode='bilinear',
                    align_corners=False
                )
                upsampled = F.interpolate(
                    downsampled,
                    size=(self.domain_size, self.domain_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                
                # Different scale weights
                weight = 0.1 * (scale / self.domain_size)
                if scale == 32:
                    multi_scale += weight * upsampled * torch.sin(omega_t * 3)
                elif scale == 64:
                    multi_scale += weight * upsampled * torch.cos(omega_t * 2)
                else:
                    multi_scale += weight * upsampled
        
        omega_t += multi_scale
        
        # 7. Physical constraints (enstrophy conservation approximation)
        enstrophy_0 = torch.mean(omega_0**2)
        enstrophy_t = torch.mean(omega_t**2)
        if enstrophy_t > 0:
            omega_t = omega_t * torch.sqrt(enstrophy_0 / (enstrophy_t + 1e-6))
        
        # 8. Final non-linear transformation
        omega_t = omega_t + 0.1 * torch.sin(omega_t * 3) + 0.05 * torch.tanh(omega_t * 2)
        
        # 9. Smooth near boundaries
        for i in range(3):
            omega_t[i, :] = omega_t[i, :] * (i / 3)
            omega_t[-i-1, :] = omega_t[-i-1, :] * (i / 3)
            omega_t[:, i] = omega_t[:, i] * (i / 3)
            omega_t[:, -i-1] = omega_t[:, -i-1] * (i / 3)
        
        # Normalize to reasonable range
        std = torch.std(omega_t)
        if std > 0:
            omega_t = omega_t / std * 0.5
        
        return omega_t

    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx): 
        return self.samples[idx]

# ========== 2. MULTISCALE DATASET WRAPPER ==========
class MultiScaleDataset(Dataset):
    def __init__(self, base_dataset: Dataset, scales: List[int], 
                 mode: str = 'train', augment: bool = False):
        self.base_dataset = base_dataset
        self.scales = sorted(scales)
        self.mode = mode
        self.augment = augment and (mode == 'train')
    
    def __len__(self):
        return len(self.base_dataset)
    
    def _downsample(self, tensor: torch.Tensor, target_size: int) -> torch.Tensor:
        if tensor.shape[-1] == target_size:
            return tensor
        
        return F.interpolate(
            tensor.unsqueeze(0) if tensor.dim() == 3 else tensor,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
    
    def _augment(self, grid: torch.Tensor, solution: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.augment:
            return grid, solution
        
        if torch.rand(1) > 0.5:
            grid = torch.flip(grid, dims=[-1])
            solution = torch.flip(solution, dims=[-1])
        
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            grid = torch.rot90(grid, k, dims=[-2, -1])
            solution = torch.rot90(solution, k, dims=[-2, -1])
        
        return grid, solution
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        sample = self.base_dataset[idx]
        original_grid = sample['grid']
        original_solution = sample['solution']
        
        if self.augment:
            original_grid, original_solution = self._augment(original_grid, original_solution)
        
        multiscale_data = {}
        multiscale_solutions = {}
        
        for scale in self.scales:
            grid_scaled = self._downsample(original_grid, scale)
            solution_scaled = self._downsample(original_solution, scale)
            
            multiscale_data[f'grid_{scale}'] = grid_scaled
            multiscale_solutions[f'solution_{scale}'] = solution_scaled
        
        return {
            'multiscale_grids': multiscale_data,
            'multiscale_solutions': multiscale_solutions,
            'original_grid': original_grid,
            'original_solution': original_solution,
            'scales': self.scales,
            'parameters': sample.get('parameters', {})
        }
    

# ========== 3. SCALE-SPECIFIC MODELS ==========
class ScaleSpecificFNO(nn.Module):
    def __init__(self, scale: int, in_channels: int, out_channels: int, 
                 width: int = 256, depth: int = 4):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        modes = max(4, scale // 8)
        
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, width, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
            #nn.Dropout(0.05),
        )
        
        self.layers = nn.ModuleList([
            FourierBlock2D(width, width, modes, modes) 
            for _ in range(depth)
        ])
        
        self.project = nn.Sequential(
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, 1),
            #nn.Dropout(0.05),
        )
    
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        if grid.shape[-1] != self.scale or grid.shape[-2] != self.scale:
            grid = F.interpolate(grid, size=(self.scale, self.scale), mode='bilinear')
        
        x = self.lift(grid)
        for layer in self.layers:
            x = layer(x) + x
            x = F.gelu(x)
        output = self.project(x)
        
        return output
    

class MultiscaleSoftmaxRouter(nn.Module):
    def __init__(self, scales: List[int], in_channels: int, out_channels: int):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        self.last_router_weights = None
        
        # Get out_channels from parameters (it should be 1 for your case)
        self.out_channels = out_channels
        
        # Calculate the input dimension: 4x4 pooled from each scale, concatenated
        # Each scale gives: out_channels * 4 * 4 features
        router_input_dim = self.num_scales * out_channels * 4 * 4
        
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_scales),
            nn.Softmax(dim=-1)
        )
        
        # Keep the scale_biases for compatibility
        self.scale_biases = nn.Parameter(torch.zeros(self.num_scales, out_channels, 1, 1))
        
    def forward(self, multiscale_predictions: Dict[int, torch.Tensor], 
            input_grid: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size = input_grid.shape[0]
        target_size = max(self.scales)
        self.last_router_weights = None

        # Get the first available scale to determine number of channels
        available_scales = list(multiscale_predictions.keys())
        if not available_scales:
            device = input_grid.device
            return torch.zeros(batch_size, self.out_channels, 
                            target_size, target_size, device=device), {'router_weights': None}
        
        first_scale = available_scales[0]
        pred = multiscale_predictions[first_scale]
        
        # Ensure predictions have the correct shape [B, C, H, W]
        if pred.dim() == 2:  # [B, H*W] or similar
            # Reshape to [B, C, H, W] - assuming C=1
            pred = pred.unsqueeze(1)
            # Need to know spatial dimensions - try to make square
            if pred.shape[-1] > 1:
                sqrt_len = int(pred.shape[-1]**0.5)
                if sqrt_len * sqrt_len == pred.shape[-1]:
                    pred = pred.view(batch_size, 1, sqrt_len, sqrt_len)
        elif pred.dim() == 3:  # [B, H, W]
            pred = pred.unsqueeze(1)  # Add channel dimension
        
        out_channels = pred.shape[1]
        
        # Collect features from all scales (use zeros for missing scales)
        router_features = []
        compatible_predictions = {}
        
        for i, scale in enumerate(self.scales):
            if scale in multiscale_predictions:
                pred = multiscale_predictions[scale]
                
                # Ensure proper shape [B, C, H, W]
                if pred.dim() == 2:
                    pred = pred.unsqueeze(1)
                    # Try to reshape to square
                    if pred.shape[-1] > 1:
                        sqrt_len = int(pred.shape[-1]**0.5)
                        if sqrt_len * sqrt_len == pred.shape[-1]:
                            pred = pred.view(batch_size, pred.shape[1], sqrt_len, sqrt_len)
                elif pred.dim() == 3:
                    pred = pred.unsqueeze(1)
                
                compatible_predictions[scale] = pred
                
                # Pool to 4x4 for router input
                pooled = F.adaptive_avg_pool2d(pred, 4)
                router_features.append(pooled)
            else:
                # Create zeros for missing scale
                zeros = torch.zeros(batch_size, out_channels, 4, 4, 
                                device=input_grid.device)
                router_features.append(zeros)
        
        # Concatenate along channel dimension
        router_input = torch.cat(router_features, dim=1)  # [B, out_channels*num_scales, 4, 4]
        
        # Flatten spatial dimensions
        router_input = router_input.view(batch_size, -1)  # [B, out_channels*num_scales*4*4]
        
        router_weights = self.router(router_input)
        self.last_router_weights = router_weights.detach().clone()

        # Weighted combination
        combined = torch.zeros(batch_size, out_channels, 
                            target_size, target_size, device=input_grid.device)
        total_weight = torch.zeros(batch_size, 1, 1, 1, device=input_grid.device)
        
        for i, scale in enumerate(self.scales):
            if scale in compatible_predictions:
                pred = compatible_predictions[scale]
                
                if pred.shape[-1] != target_size:
                    pred_upscaled = F.interpolate(pred, size=(target_size, target_size), 
                                                mode='bilinear', align_corners=False)
                else:
                    pred_upscaled = pred
                
                pred_upscaled = pred_upscaled + self.scale_biases[i]
                weight = router_weights[:, i].view(-1, 1, 1, 1)
                combined = combined + weight * pred_upscaled
                total_weight = total_weight + weight
        
        # Normalize
        combined = combined / (total_weight + 1e-8)
        
        return combined, {
            'router_weights': router_weights,
            'scale_outputs': compatible_predictions,
            'combined': combined,
            'total_weight': total_weight
        }
    
class LagrangianSingleScaleCombiner(nn.Module):
    """Single time scale Lagrangian method for multiscale combination"""
    def __init__(self, scales: List[int], in_channels: int, out_channels: int, 
                 lambda_init: float = 1.0):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        
        # Lagrangian multipliers for each scale (log-space for positivity)
        self.log_lambdas = nn.Parameter(torch.ones(self.num_scales) * np.log(lambda_init))
        
        # Compatibility layer for scale differences
        self.compatibility_layers = nn.ModuleDict()
        for i, scale in enumerate(scales):
            self.compatibility_layers[str(scale)] = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )
    
    def forward(self, multiscale_predictions: Dict[int, torch.Tensor], 
                input_grid: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Lagrangian combination: z = Σ λ_i * f_i(x_i) / Σ λ_i
        target_size = max(self.scales)
        
        # Apply compatibility layers
        compatible_predictions = {}
        for i, scale in enumerate(self.scales):
            if scale in multiscale_predictions:
                pred = multiscale_predictions[scale]
                if pred.shape[-1] != target_size:
                    pred = F.interpolate(pred, size=(target_size, target_size), mode='bilinear')
                compatible_predictions[scale] = self.compatibility_layers[str(scale)](pred)
        
        # Get lambdas (ensure positivity via exponential)
        lambdas = torch.exp(self.log_lambdas)
        lambda_weights = lambdas / torch.sum(lambdas)
        
        # Weighted combination
        combined = None
        for i, scale in enumerate(self.scales):
            if scale in compatible_predictions:
                pred = compatible_predictions[scale]
                weight = lambda_weights[i]
                
                if combined is None:
                    combined = weight * pred
                else:
                    combined = combined + weight * pred
        
        # Handle case when no predictions
        if combined is None:
            device = input_grid.device
            combined = torch.zeros(1, multiscale_predictions.get(list(multiscale_predictions.keys())[0], 
                                      torch.zeros(1, 1, target_size, target_size, device=device)).shape[1], 
                                 target_size, target_size, device=device)
        
        return combined, {
            'lambda_weights': lambda_weights,
            'log_lambdas': self.log_lambdas,
            'compatible_predictions': compatible_predictions
        }

class LagrangianTwoScaleCombiner(nn.Module):
    """Two time scale Lagrangian method (primal-dual) simplified for a single-step combination,
    similar to LagrangianSingleScaleCombiner but with projection layers."""
    def __init__(self, scales: List[int], in_channels: int, out_channels: int,
                 primal_lr: float = 0.1, dual_lr: float = 0.01, num_iter: int = 5, # These are unused now
                 lambda_init: float = 1.0): # Added lambda_init argument for consistency
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        self.num_iter = num_iter
        
        # Lagrangian multipliers for each scale (log-space for positivity)
        self.log_lambdas = nn.Parameter(torch.ones(self.num_scales) * np.log(lambda_init))
        
        # NOTE: The combined_weight parameter is non-standard for the simple Lagrangian approach.
        # It is kept commented out to maintain the intended single-scale-like combination.
        # self.combined_weight = nn.Parameter(torch.ones(1, out_channels, 1, 1) * 0.01)
        
        # Compatibility/Projection layers for each scale
        self.projection_layers = nn.ModuleDict()
        for scale in scales:
            self.projection_layers[str(scale)] = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )
    
    def forward(self, multiscale_predictions: Dict[int, torch.Tensor],
                input_grid: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Combination: z = Σ λ_i * f_i(x_i) / Σ λ_i
        target_size = max(self.scales)
        
        # 1. Apply projection layers
        compatible_predictions = {}
        for i, scale in enumerate(self.scales):
            if scale in multiscale_predictions:
                pred = multiscale_predictions[scale]
                if pred.shape[-1] != target_size:
                    pred = F.interpolate(pred, size=(target_size, target_size), mode='bilinear')
                compatible_predictions[scale] = self.projection_layers[str(scale)](pred)
        
        if not compatible_predictions:
            device = input_grid.device
            # Ensure return shape is correct
            out_channels = multiscale_predictions.get(list(multiscale_predictions.keys())[0], 
                                                     torch.zeros(1, 1, target_size, target_size, device=device)).shape[1]
            return torch.zeros(input_grid.shape[0], out_channels, target_size, target_size, device=device), {}
        
        # 2. Get lambdas (ensure positivity via exponential) and normalize
        lambdas = torch.exp(self.log_lambdas)
        lambda_weights = lambdas / torch.sum(lambdas)
        
        # 3. Weighted combination
        combined = None
        for i, scale in enumerate(self.scales):
            if scale in compatible_predictions:
                pred = compatible_predictions[scale]
                weight = lambda_weights[i]
                
                if combined is None:
                    combined = weight * pred
                else:
                    combined = combined + weight * pred
        
        # 4. Final metadata
        return combined, {
            'lambda_weights': lambda_weights,
            'log_lambdas': self.log_lambdas,
            'compatible_predictions': compatible_predictions
        }
    
class ADMMCombiner(nn.Module):
    """ADMM (Alternating Direction Method of Multipliers) combiner with weight tracking"""
    def __init__(self, scales: List[int], in_channels: int, out_channels: int,
                 rho: float = 0.1, num_iter: int = 3, beta: float = 0.01):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        self.num_iter = num_iter
        
        # ADMM penalty parameter - trainable in log space
        self.log_rho = nn.Parameter(torch.tensor(np.log(rho)))
        
        # Learnable compatibility layers for each scale
        self.compatibility_layers = nn.ModuleDict()
        for scale in scales:
            self.compatibility_layers[str(scale)] = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )
        
        # Consensus refinement network
        self.consensus_refiner = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
        )
        
        # Parameter for soft consensus (instead of strict averaging)
        self.consensus_weight = nn.Parameter(torch.ones(1))
        
        # Store scale weights for analysis
        self.last_scale_weights = None
        self.epoch_scale_weights = []  # Store weights for each epoch
        
    def forward(self, multiscale_predictions: Dict[int, torch.Tensor],
                input_grid: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        target_size = max(self.scales)
        batch_size = input_grid.shape[0]
        
        # Get channel dimension from first prediction
        if multiscale_predictions:
            first_pred = next(iter(multiscale_predictions.values()))
            channel_dim = first_pred.shape[1]
            device = first_pred.device
        else:
            channel_dim = 1
            device = input_grid.device
        
        # Apply compatibility layers and upscale predictions
        compatible_predictions = {}
        for scale in self.scales:
            if scale in multiscale_predictions:
                pred = multiscale_predictions[scale]
                if pred.shape[-1] != target_size:
                    pred = F.interpolate(pred, size=(target_size, target_size), 
                                        mode='bilinear', align_corners=False)
                compatible_predictions[scale] = self.compatibility_layers[str(scale)](pred)
        
        if not compatible_predictions:
            return torch.zeros(batch_size, channel_dim, target_size, target_size, 
                             device=device), {}
        
        # Initialize ADMM variables
        rho = torch.exp(self.log_rho)
        
        # Initialize local variables x_i with compatible predictions
        x = {scale: pred.clone() for scale, pred in compatible_predictions.items()}
        
        # Initialize consensus variable z as weighted average of predictions
        weights = torch.ones(len(compatible_predictions), device=device) / len(compatible_predictions)
        z = sum(w * x[scale] for w, (scale, _) in zip(weights, compatible_predictions.items()))
        
        # Initialize dual variables u_i
        u = {scale: torch.zeros_like(pred) for scale, pred in compatible_predictions.items()}
        
        scale_outputs = {}
        
        # ADMM iterations
        for iter_idx in range(self.num_iter):
            # 1. x-update (local variable update for each scale)
            for scale in compatible_predictions.keys():
                # x_i = (pred_i + rho * (z - u_i)) / (1 + rho)
                x[scale] = (compatible_predictions[scale] + rho * (z - u[scale])) / (1 + rho)
                scale_outputs[scale] = x[scale]
            
            # 2. z-update (consensus update)
            # Soft consensus: z = (1/num_scales) * Σ(x_i + u_i) * softmax(consensus_weight)
            if len(x) > 0:
                z_numerator = torch.zeros_like(z)
                count = 0
                
                for scale, x_i in x.items():
                    if scale in u:
                        z_numerator = z_numerator + x_i + u[scale]
                        count += 1
                
                if count > 0:
                    # Use soft consensus instead of strict averaging
                    soft_consensus = torch.sigmoid(self.consensus_weight)
                    z_new = (z_numerator / count)
                    z = soft_consensus * z_new + (1 - soft_consensus) * z
            
            # 3. u-update (dual variable update)
            for scale in compatible_predictions.keys():
                if scale in x and scale in u:
                    u[scale] = u[scale] + (x[scale] - z)
        
        # Refine consensus
        z_refined = self.consensus_refiner(z)
        
        # Compute scale contributions to consensus
        scale_contributions = {}
        total_contribution = 0
        
        for scale in compatible_predictions.keys():
            if scale in x and scale in u:
                contribution = torch.norm(x[scale] + u[scale], p='fro').item()
                scale_contributions[scale] = contribution
                total_contribution += contribution
        
        # Normalize to get scale weights
        if total_contribution > 0:
            self.last_scale_weights = {
                scale: contribution / total_contribution
                for scale, contribution in scale_contributions.items()
            }
        else:
            self.last_scale_weights = {scale: 1.0/len(self.scales) for scale in self.scales}
        
        # Store metadata
        self.last_dual_vars = {scale: u[scale].detach().clone() 
                              for scale in u if scale in self.scales}
        
        return z_refined, {
            'consensus': z,
            'dual_vars': u,
            'scale_outputs': scale_outputs,
            'compatible_predictions': compatible_predictions,
            'rho': rho,
            'log_rho': self.log_rho,
            'consensus_weight': self.consensus_weight,
            'scale_weights': self.last_scale_weights  # Add scale weights to metadata
        }
    
    def reset_epoch_weights(self):
        """Reset epoch weight tracking"""
        self.epoch_scale_weights = []
    
    def add_epoch_weights(self, weights):
        """Add weights from current batch to epoch tracking"""
        if weights is not None:
            self.epoch_scale_weights.append(weights)
    
    def get_average_epoch_weights(self):
        """Get average weights for the epoch"""
        if not self.epoch_scale_weights:
            return {scale: 1.0/len(self.scales) for scale in self.scales}
        
        avg_weights = defaultdict(float)
        count = 0
        
        for batch_weights in self.epoch_scale_weights:
            for scale, weight in batch_weights.items():
                avg_weights[scale] += weight
            count += 1
        
        for scale in avg_weights:
            avg_weights[scale] /= count
        
        return dict(avg_weights)
    

class SingleScaleFNO(nn.Module):
    """Baseline: Standard FNO operating at single fixed scale"""
    def __init__(self, scale: int, in_channels: int, out_channels: int, 
                 width: int = 32, depth: int = 4):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        modes = max(4, scale // 8)
        
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, width, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 1)
        )
        
        self.fno_layers = nn.ModuleList([
            FourierBlock2D(width, width, modes, modes) 
            for _ in range(depth)
        ])
        
        self.project = nn.Sequential(
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, 1)
        )
        
        # Additional refinement for fair comparison
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is at the right scale
        if x.shape[-1] != self.scale or x.shape[-2] != self.scale:
            x = F.interpolate(x, size=(self.scale, self.scale), mode='bilinear')
        
        x = self.lift(x)
        for layer in self.fno_layers:
            x = layer(x) + x
            x = F.gelu(x)
        x = self.project(x)
        x = self.refine(x)
        
        return x
    
    def compute_loss(self, predictions: torch.Tensor, 
                    targets: torch.Tensor,
                    metadata: Dict = None) -> Tuple[torch.Tensor, Dict]:
        """Simple compute loss method for compatibility"""
        if metadata is None:
            metadata = {}
        
        losses = {}
        
        # Basic MSE loss
        if predictions.shape[-2:] != targets.shape[-2:]:
            predictions_resized = F.interpolate(
                predictions, 
                size=targets.shape[-2:], 
                mode='bilinear',
                align_corners=False
            )
            losses['reconstruction'] = F.mse_loss(predictions_resized, targets)
        else:
            losses['reconstruction'] = F.mse_loss(predictions, targets)
        
        total_loss = losses['reconstruction']
        losses['total'] = total_loss
        
        return total_loss, losses
    

class SingleScaleBaseline(nn.Module):
    """Wrapper for single scale baseline with consistent interface"""
    def __init__(self, scale: int, in_channels: int, out_channels: int):
        super().__init__()
        self.scale = scale
        self.model = SingleScaleFNO(scale, in_channels, out_channels)
        self.out_channels = out_channels  # Add this for loss computation
        self.is_baseline = True
    
    def forward(self, multiscale_inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        # Use the largest available scale input
        available_scales = [int(k.split('_')[1]) for k in multiscale_inputs.keys()]
        if not available_scales:
            device = next(self.parameters()).device
            target_size = self.scale
            return torch.zeros(1, self.out_channels, target_size, target_size, 
                             device=device), {}
        
        largest_scale = max(available_scales)
        input_key = f'grid_{largest_scale}'
        grid = multiscale_inputs[input_key]
        
        # Run single scale model
        output = self.model(grid)
        
        # Create metadata similar to multiscale methods
        metadata = {
            'scale_used': largest_scale,
            'output': output,
            'scale_predictions': {largest_scale: output}
        }
        
        return output, metadata
    
    def compute_loss(self, predictions: torch.Tensor, 
                    multiscale_targets: Dict[str, torch.Tensor],
                    metadata: Dict) -> Tuple[torch.Tensor, Dict]:
        """Compute loss for single scale baseline"""
        losses = {}
        
        # Get the scale used for prediction
        scale_used = metadata.get('scale_used', None)
        
        if scale_used is not None:
            # Look for target at the same scale
            target_key = f'solution_{scale_used}'
            if target_key in multiscale_targets:
                target = multiscale_targets[target_key]
                if predictions.shape[-2:] != target.shape[-2:]:
                    predictions_resized = F.interpolate(
                        predictions, 
                        size=target.shape[-2:], 
                        mode='bilinear',
                        align_corners=False
                    )
                    losses['reconstruction'] = F.mse_loss(predictions_resized, target)
                else:
                    losses['reconstruction'] = F.mse_loss(predictions, target)
            else:
                # If target at same scale doesn't exist, use the largest available target
                available_target_scales = [int(k.split('_')[1]) for k in multiscale_targets.keys()]
                if available_target_scales:
                    largest_target_scale = max(available_target_scales)
                    target_key = f'solution_{largest_target_scale}'
                    target = multiscale_targets[target_key]
                    if predictions.shape[-2:] != target.shape[-2:]:
                        predictions_resized = F.interpolate(
                            predictions, 
                            size=target.shape[-2:], 
                            mode='bilinear',
                            align_corners=False
                        )
                        losses['reconstruction'] = F.mse_loss(predictions_resized, target)
                    else:
                        losses['reconstruction'] = F.mse_loss(predictions, target)
                else:
                    # No target available
                    losses['reconstruction'] = torch.tensor(0.0, device=predictions.device)
        else:
            # Use the first available target
            first_target_key = list(multiscale_targets.keys())[0]
            target = multiscale_targets[first_target_key]
            if predictions.shape[-2:] != target.shape[-2:]:
                predictions_resized = F.interpolate(
                    predictions, 
                    size=target.shape[-2:], 
                    mode='bilinear',
                    align_corners=False
                )
                losses['reconstruction'] = F.mse_loss(predictions_resized, target)
            else:
                losses['reconstruction'] = F.mse_loss(predictions, target)
        
        # Calculate total loss
        total_loss = losses['reconstruction']
        losses['total'] = total_loss
        
        return total_loss, losses



# ========== MODIFIED TRAINER WITH WEIGHT LOGGING ==========
class StableMultiscaleTrainer:
    def __init__(self, model, 
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str,
                 lr: Optional[float] = None,
                 grad_clip: float = 1.0,
                 patience: int = 30,
                 log_weights_every: int = 5):  # NEW: Control weight logging frequency
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.patience = patience
        self.log_weights_every = log_weights_every  # NEW
        
        # Get the actual scales the model is designed for
        self.model_scales = model.scales
        self.method = model.combination_method
        
        # Method-specific learning rates
        if lr is None:
            lr_map = {
                'single_scale_baseline': 3e-3,
                'softmax': 1e-3,
                'lagrangian_single': 1e-3,
                'lagrangian_two': 1e-3,
                'admm': 1e-3,
            }
            print("combination method: ", model.combination_method)
            lr = lr_map.get(model.combination_method, 1e-3)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=1e-6,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.995
        )
        
        self.grad_clip = grad_clip
        self.best_val_loss = float('inf')
        self.val_loss_ema = None
        self.ema_alpha = 0.1
        
        self.history = defaultdict(list)
        self.weight_history = []  # NEW: Store detailed weight history
        self.epoch_weights_summary = []  # NEW: Store per-epoch weight summaries
        self.admm_weights_per_epoch = []  # NEW: Store ADMM weights per epoch

    def _filter_multiscale_data(self, batch_data, key_prefix):
        """Filter multiscale data to only include scales the model can handle"""
        filtered = {}
        for k, v in batch_data.items():
            try:
                scale = int(k.split('_')[1])
                if scale in self.model_scales:
                    filtered[k] = v
            except (ValueError, IndexError):
                continue
        return filtered
    
    def _extract_weights_from_metadata(self, metadata: Dict) -> Dict:
        """Extract weight information from combiner metadata"""
        weights_info = {}
        
        # Softmax router weights
        if 'router_weights' in metadata and metadata['router_weights'] is not None:
            router_weights = metadata['router_weights']
            if router_weights.dim() == 2:  # [B, num_scales]
                # Average over batch
                avg_weights = router_weights.mean(dim=0)
                for i, scale in enumerate(self.model_scales):
                    if i < avg_weights.shape[0]:
                        weights_info[f'scale_{scale}'] = avg_weights[i].item()
        
        # Lagrangian weights (single scale)
        elif 'lambda_weights' in metadata:
            lambda_weights = metadata['lambda_weights']
            for i, scale in enumerate(self.model_scales):
                if i < lambda_weights.shape[0]:
                    weights_info[f'scale_{scale}'] = lambda_weights[i].item()
        
        # Lagrangian weights (two-timescale)
        elif 'lagrangian_weights' in metadata:
            lagrangian_weights = metadata['lagrangian_weights']
            for i, scale in enumerate(self.model_scales):
                if i < lagrangian_weights.shape[0]:
                    weights_info[f'scale_{scale}'] = lagrangian_weights[i].item()
        
        # ADMM scale weights
        elif 'scale_weights' in metadata:
            scale_weights = metadata['scale_weights']
            for scale, weight in scale_weights.items():
                weights_info[f'scale_{scale}'] = weight
        
        # ADMM consensus information (fallback)
        elif 'consensus' in metadata and self.method == 'admm':
            # If no explicit scale weights, compute from scale outputs
            if 'scale_outputs' in metadata:
                scale_outputs = metadata['scale_outputs']
                total_norm = 0
                for scale, output in scale_outputs.items():
                    output_norm = output.norm().item()
                    weights_info[f'scale_{scale}_norm'] = output_norm
                    total_norm += output_norm
                
                # Normalize to get relative importance
                if total_norm > 0:
                    for scale, output in scale_outputs.items():
                        weights_info[f'scale_{scale}_rel'] = weights_info.get(f'scale_{scale}_norm', 0) / total_norm
        
        # For baseline single scale, track which scale was used
        elif self.model.is_baseline and 'scale_used' in metadata:
            scale_used = metadata['scale_used']
            weights_info[f'scale_{scale_used}'] = 1.0
        
        return weights_info
    
    def _print_weight_summary(self, epoch: int, weights: Dict, is_admm: bool = False):
        """Print formatted weight summary for current epoch"""
        if not weights:
            return
        
        if is_admm:
            print("  ADMM Scale Weights:")
        else:
            print("  Scale Weights:")
        
        # Separate actual weights from other metrics
        actual_weights = {}
        other_metrics = {}
        
        for key, value in weights.items():
            if key.startswith('scale_') and not (key.endswith('_norm') or key.endswith('_rel')):
                scale_num = int(key.replace('scale_', ''))
                actual_weights[scale_num] = value
            elif key.endswith('_rel'):
                scale_num = int(key.replace('scale_', '').replace('_rel', ''))
                other_metrics[f'scale_{scale_num}_rel'] = value
        
        # Sort scales
        sorted_scales = sorted(actual_weights.keys())
        
        # Print actual weights
        for scale in sorted_scales:
            weight_val = actual_weights.get(scale, 0.0)
            rel_val = other_metrics.get(f'scale_{scale}_rel', None)
            
            if rel_val is not None:
                print(f"    Scale {scale:3d}: {weight_val:6.4f} (relative: {rel_val:6.4f})")
            else:
                print(f"    Scale {scale:3d}: {weight_val:6.4f}")
        
        # Print summary statistics
        if len(actual_weights) > 0:
            weight_values = list(actual_weights.values())
            weight_sum = sum(weight_values)
            weight_max = max(weight_values)
            weight_min = min(weight_values)
            weight_std = np.std(weight_values) if len(weight_values) > 1 else 0.0
            
            print(f"    Summary: Sum={weight_sum:.4f}, Max={weight_max:.4f}, "
                  f"Min={weight_min:.4f}, Std={weight_std:.4f}")
            
            # Check for weight stability
            if len(self.epoch_weights_summary) > 1:
                prev_weights = self.epoch_weights_summary[-2]['weights']
                prev_weight_vals = []
                for scale in sorted_scales:
                    key = f'scale_{scale}'
                    if key in prev_weights:
                        prev_weight_vals.append(prev_weights[key])
                
                if prev_weight_vals and len(prev_weight_vals) == len(weight_values):
                    changes = [abs(w - pw) for w, pw in zip(weight_values, prev_weight_vals)]
                    avg_change = np.mean(changes)
                    max_change = max(changes)
                    print(f"    Changes: Avg={avg_change:.6f}, Max={max_change:.6f}")
    
    def train_epoch(self, epoch: int) -> float:  # ADDED epoch parameter
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        # Track weight statistics for this epoch
        epoch_weights = defaultdict(list)
        
        # For ADMM, track scale-specific weights
        if self.method == 'admm' and hasattr(self.model.combiner, 'reset_epoch_weights'):
            self.model.combiner.reset_epoch_weights()
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # Filter inputs to only include model-compatible scales
                multiscale_inputs = self._filter_multiscale_data(
                    batch['multiscale_grids'], 'grid'
                )
                multiscale_targets = self._filter_multiscale_data(
                    batch['multiscale_solutions'], 'solution'
                )
                
                # Skip batch if no compatible scales
                if not multiscale_inputs or not multiscale_targets:
                    continue
                
                # Move to device
                multiscale_inputs = {k: v.to(self.device) for k, v in multiscale_inputs.items()}
                multiscale_targets = {k: v.to(self.device) for k, v in multiscale_targets.items()}
                
                predictions, metadata = self.model(multiscale_inputs)
                
                # Track ADMM weights per batch
                if self.method == 'admm' and hasattr(self.model.combiner, 'add_epoch_weights'):
                    if hasattr(self.model.combiner, 'last_scale_weights') and self.model.combiner.last_scale_weights:
                        self.model.combiner.add_epoch_weights(self.model.combiner.last_scale_weights)
                
                # Extract and log weights from metadata
                weights_info = self._extract_weights_from_metadata(metadata)
                for scale_key, weight_value in weights_info.items():
                    epoch_weights[scale_key].append(weight_value)
                
                # Compute loss
                loss, loss_components = self.model.compute_loss(
                    predictions, multiscale_targets, metadata
                )
                
                if not torch.isfinite(loss):
                    print(f"Warning: Non-finite loss {loss.item()}, skipping batch")
                    continue
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_clip,
                    norm_type=2
                )
                
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: Gradient norm is {grad_norm.item()}, skipping update")
                    self.optimizer.zero_grad()
                    continue
                
                self.optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                batch_count += 1
                self.history['grad_norm'].append(grad_norm.item())
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"CUDA OOM at batch {batch_idx}, skipping")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"Error in training batch {batch_idx}: {e}")
                    continue
        
        # Update scheduler
        self.scheduler.step()
        
        # Compute average weights for this epoch
        avg_epoch_weights = {}
        for scale_key, weight_list in epoch_weights.items():
            if weight_list:  # Check if list is not empty
                avg_epoch_weights[scale_key] = np.mean(weight_list)
        
        # Store in history
        if avg_epoch_weights:
            self.epoch_weights_summary.append({
                'epoch': epoch,
                'weights': avg_epoch_weights,
                'total_loss': total_loss / max(batch_count, 1) if batch_count > 0 else float('inf')
            })
        
        # For ADMM, get average epoch weights
        if self.method == 'admm' and hasattr(self.model.combiner, 'get_average_epoch_weights'):
            admm_epoch_weights = self.model.combiner.get_average_epoch_weights()
            self.admm_weights_per_epoch.append(admm_epoch_weights)
            
            # Print ADMM weights if this is a logging epoch
            if epoch % self.log_weights_every == 0:
                print(f"\n  ADMM Epoch {epoch} Scale Weights:")
                total_weight = 0
                for scale, weight in admm_epoch_weights.items():
                    total_weight += weight
                    print(f"    Scale {scale}: {weight:.4f}")
                print(f"    Total: {total_weight:.4f}")
                
                # Print rho parameter
                if hasattr(self.model.combiner, 'log_rho'):
                    rho = torch.exp(self.model.combiner.log_rho).item()
                    print(f"    Rho parameter: {rho:.6f}")
        
        return total_loss / max(batch_count, 1) if batch_count > 0 else float('inf')
    
    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0
        batch_count = 0
        
        for batch in self.val_loader:
            # Filter inputs to only include model-compatible scales
            multiscale_inputs = self._filter_multiscale_data(
                batch['multiscale_grids'], 'grid'
            )
            multiscale_targets = self._filter_multiscale_data(
                batch['multiscale_solutions'], 'solution'
            )
            
            # Skip batch if no compatible scales
            if not multiscale_inputs or not multiscale_targets:
                continue
            
            # Move to device
            multiscale_inputs = {k: v.to(self.device) for k, v in multiscale_inputs.items()}
            multiscale_targets = {k: v.to(self.device) for k, v in multiscale_targets.items()}
            
            predictions, metadata = self.model(multiscale_inputs)
            loss, _ = self.model.compute_loss(predictions, multiscale_targets, metadata)
            
            if torch.isfinite(loss):
                total_loss += loss.item()
                batch_count += 1
        
        avg_val_loss = total_loss / max(batch_count, 1) if batch_count > 0 else float('inf')
        
        # Exponential moving average
        if self.val_loss_ema is None:
            self.val_loss_ema = avg_val_loss
        else:
            self.val_loss_ema = self.ema_alpha * avg_val_loss + (1 - self.ema_alpha) * self.val_loss_ema
        
        return avg_val_loss, self.val_loss_ema
    
    def train(self, num_epochs: int = 100, verbose: bool = True):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_loss_ema = self.validate()
            
            # Check for valid losses
            if train_loss == float('inf') or val_loss == float('inf'):
                print(f"Warning: Invalid loss at epoch {epoch+1}, skipping epoch")
                continue
            
            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_loss_ema'].append(val_loss_ema)
            
            # Save best model
            if val_loss_ema < best_val_loss:
                best_val_loss = val_loss_ema
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'weight_history': self.epoch_weights_summary,  # NEW: save weight history
                    'admm_weights_per_epoch': self.admm_weights_per_epoch if self.method == 'admm' else None,
                }, f'best_model_{self.model.combination_method}.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            # if patience_counter >= self.patience:
            #     print(f"Early stopping at epoch {epoch+1}")
            #     # Print final weight summary
            #     if self.epoch_weights_summary:
            #         print("\nFinal Weight Distribution:")
            #         self._print_weight_summary(epoch, self.epoch_weights_summary[-1]['weights'])
            #     break
            
            # Logging with weight information
            if verbose and (epoch % self.log_weights_every == 0 or epoch < 5 or epoch == num_epochs - 1):
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"\nEpoch {epoch+1}/{num_epochs}:")
                print(f"  Train Loss = {train_loss:.4e}")
                print(f"  Val Loss = {val_loss:.4e} (EMA: {val_loss_ema:.4e})")
                print(f"  LR = {current_lr:.2e}, Best Val = {best_val_loss:.4e}")
                print(f"  Patience = {patience_counter}/{self.patience}")
                
                # Print weight summary for non-ADMM methods
                if self.method != 'admm' and self.epoch_weights_summary:
                    latest_weights = self.epoch_weights_summary[-1]['weights']
                    self._print_weight_summary(epoch, latest_weights)
        
        # Plot weight evolution after training
        if len(self.epoch_weights_summary) > 1:
            self._plot_weight_evolution()
        
        # For ADMM, print detailed weight evolution
        if self.method == 'admm' and len(self.admm_weights_per_epoch) > 0:
            self._print_admm_weight_evolution()
        
        return self.history
    
    def _plot_weight_evolution(self):
        """Plot the evolution of scale weights over training"""
        if not self.epoch_weights_summary:
            return
        
        # Extract data
        epochs = [entry['epoch'] for entry in self.epoch_weights_summary]
        
        # Get all scale keys that appear in any epoch
        all_scale_keys = set()
        for entry in self.epoch_weights_summary:
            all_scale_keys.update(entry['weights'].keys())
        
        # Filter for actual weight keys (not metrics)
        weight_keys = [k for k in all_scale_keys if k.startswith('scale_') and 
                      not (k.endswith('_norm') or k.endswith('_rel'))]
        
        if not weight_keys:
            return
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot each scale's weight evolution
        for key in sorted(weight_keys):
            # Extract scale number
            try:
                scale_num = int(key.replace('scale_', ''))
            except:
                continue
            
            # Get weight values across epochs
            weights = []
            for entry in self.epoch_weights_summary:
                if key in entry['weights']:
                    weights.append(entry['weights'][key])
                else:
                    weights.append(0.0)
            
            plt.plot(epochs, weights, 'o-', label=f'Scale {scale_num}', linewidth=2, markersize=4)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Weight Value', fontsize=12)
        plt.title(f'Scale Weight Evolution - {self.model.combination_method.replace("_", " ").title()}', fontsize=14)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f'weight_evolution_{self.model.combination_method}.png', dpi=300)
        plt.close()
        
        # Create a summary table of final weights
        print(f"\n{'='*60}")
        print(f"FINAL WEIGHT SUMMARY - {self.model.combination_method.upper().replace('_', ' ')}")
        print(f"{'='*60}")
        
        # Get final weights
        final_weights = self.epoch_weights_summary[-1]['weights']
        
        # Sort by scale
        weight_items = []
        for key, value in final_weights.items():
            if key.startswith('scale_') and not (key.endswith('_norm') or key.endswith('_rel')):
                try:
                    scale_num = int(key.replace('scale_', ''))
                    weight_items.append((scale_num, value))
                except:
                    continue
        
        if weight_items:
            weight_items.sort()
            
            print(f"\n{'Scale':<10} {'Weight':<15} {'Percentage':<15}")
            print(f"{'-'*40}")
            
            for scale, weight in weight_items:
                # Calculate percentage if weights sum to something
                total = sum(w for _, w in weight_items)
                percentage = (weight / total * 100) if total > 0 else 0
                print(f"{scale:<10} {weight:<15.6f} {percentage:<15.2f}%")
            
            print(f"\nTotal sum: {total:.6f}")
            
            # Calculate entropy (measure of weight diversity)
            entropy = 0
            for scale, weight in weight_items:
                if weight > 0:
                    p = weight / total
                    entropy -= p * math.log(p)
            
            max_entropy = math.log(len(weight_items))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            print(f"Entropy: {entropy:.4f} (Normalized: {normalized_entropy:.4f})")
            
            if normalized_entropy < 0.3:
                print("Note: Low entropy - one scale dominates")
            elif normalized_entropy > 0.7:
                print("Note: High entropy - weights are evenly distributed")
    
    def _print_admm_weight_evolution(self):
        """Print ADMM weight evolution per epoch"""
        if not self.admm_weights_per_epoch:
            return
        
        print(f"\n{'='*80}")
        print("ADMM SCALE WEIGHT EVOLUTION PER EPOCH")
        print(f"{'='*80}")
        
        # Get scales from the first epoch's weights
        first_epoch_weights = self.admm_weights_per_epoch[0]
        scales = sorted(list(first_epoch_weights.keys()))
        
        # Print header
        print(f"\n{'Epoch':<8}", end="")
        for scale in scales:
            print(f"{'Scale ' + str(scale):<12}", end="")
        print(f"{'Total':<12}")
        
        print("-" * (8 + 12 * (len(scales) + 1)))
        
        # Print weights for each epoch
        for epoch_idx, epoch_weights in enumerate(self.admm_weights_per_epoch):
            print(f"{epoch_idx:<8}", end="")
            total_weight = 0.0
            for scale in scales:
                weight = epoch_weights.get(scale, 0.0)
                total_weight += weight
                print(f"{weight:<12.4f}", end="")
            print(f"{total_weight:<12.4f}")
        
        # Print summary statistics
        print(f"\n{'='*80}")
        print("ADMM WEIGHTS SUMMARY STATISTICS")
        print(f"{'='*80}")
        
        # Calculate final weights
        final_weights = self.admm_weights_per_epoch[-1]
        print(f"\nFinal Epoch Weights (Epoch {len(self.admm_weights_per_epoch)}):")
        print(f"{'Scale':<10} {'Weight':<10} {'Percentage':<12}")
        print("-" * 32)
        for scale, weight in final_weights.items():
            print(f"{scale:<10} {weight:<10.4f} {weight*100:<11.2f}%")
        
        # Create visualization of weight evolution
        plt.figure(figsize=(12, 6))
        
        # Plot weight evolution
        for scale in scales:
            weights = [epoch_weights.get(scale, 0.0) for epoch_weights in self.admm_weights_per_epoch]
            epochs = range(1, len(weights) + 1)
            plt.plot(epochs, weights, marker='o', label=f'Scale {scale}', linewidth=2, markersize=4)
        
        plt.xlabel('Epoch')
        plt.ylabel('Weight')
        plt.title('ADMM Scale Weight Evolution During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('admm_weight_evolution_detailed.png', dpi=300)
        plt.close()
        
        print(f"\nADMM weight evolution visualization saved to 'admm_weight_evolution_detailed.png'")


# ========== IMPROVED MULTISCALE SOLVER WITH BETTER INITIALIZATION ==========
class StableMultiscalePDEsolver(nn.Module):
    def __init__(self, scales: List[int], in_channels: int, out_channels: int,
                 model_type: str = 'fno', combination_method: str = 'softmax', 
                 method_config: Optional[Dict] = None):
        super().__init__()
        self.scales = sorted(scales)
        self.num_scales = len(scales)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_type = model_type
        self.combination_method = combination_method
        self.method = combination_method
        
        # Special case: single scale baseline
        if combination_method == 'single_scale_baseline':
            self.is_baseline = True
            self.baseline_scale = max(scales)
            self.model = SingleScaleBaseline(self.baseline_scale, in_channels, out_channels)
            return
        
        self.is_baseline = False
        
        # Default configs with better stability
        self.method_config = method_config or self._get_stable_config()
        
        # Initialize scale models with better stability
        self.scale_models = nn.ModuleDict()
        for scale in self.scales:
            model = ScaleSpecificFNO(scale, in_channels, out_channels)
            # Apply careful initialization
            self._initialize_model(model)
            self.scale_models[str(scale)] = model
        
        # Initialize combiner with stability in mind
        self.combiner = self._create_stable_combiner()
        
        # Enhanced fine-tuning with residual connections
        self.fine_tune = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 2, 3, padding=1),
            nn.GroupNorm(min(8, out_channels * 2), out_channels * 2),
            nn.GELU(),
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
        )
        
        # Residual weight
        self.skip_weight = nn.Parameter(torch.tensor(0.5))
        
        # Final refinement
        self.final_proj = nn.Conv2d(out_channels, out_channels, 1)
    
    def _get_stable_config(self) -> Dict:
        """Get method-specific stable configurations"""
        configs = {
            'softmax': {
                'router_sparsity_weight': 0.001,  # Reduced
                'scale_consistency_weight': 0.01,  # Reduced
                'reconstruction_weight': 1.0,
            },
            'lagrangian_single': {
                'lambda_init': 0.1,  # Smaller initialization
                'lambda_reg_weight': 0.001,  # Much reduced
                'scale_consistency_weight': 0.01,
                'reconstruction_weight': 1.0,
            },
            'lagrangian_two': {
                'primal_lr': 5e-4,  # Increase from 1e-3 to 5e-3
                'dual_lr': 1e-3,    # Increase from 1e-4 to 1e-3
                'num_iter': 5,      # Increase iterations for better convergence
                'dual_reg_weight': 0.001,  # Reduce regularization
                'scale_consistency_weight': 0.001,
                'reconstruction_weight': 1.0,
            },
            'admm': {
                'rho': 0.1,  # Smaller penalty
                'num_iter': 5,  # Reduced iterations
                'beta': 0.01,  # Smaller beta
                'admm_consensus_weight': 0.001,  # Much reduced
                'scale_consistency_weight': 0.01,
                'reconstruction_weight': 1.0,
            }
        }
        return configs.get(self.combination_method, {})
    
    def _initialize_model(self, model: nn.Module):
        """Careful initialization for stability"""
        for name, param in model.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param, gain=0.5)  # Smaller gain
                else:
                    nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def _create_stable_combiner(self):
        """Create combiner with stability measures"""
        if self.combination_method == 'softmax':
            combiner = MultiscaleSoftmaxRouter(
                self.scales, self.in_channels, self.out_channels
            )
        elif self.combination_method == 'lagrangian_single':
            combiner = LagrangianSingleScaleCombiner(
                self.scales, self.in_channels, self.out_channels,
                lambda_init=self.method_config.get('lambda_init', 0.1)
            )
        elif self.combination_method == 'lagrangian_two':
            combiner = LagrangianTwoScaleCombiner(
                self.scales, self.in_channels, self.out_channels,
                primal_lr=self.method_config.get('primal_lr', 1e-3),
                dual_lr=self.method_config.get('dual_lr', 1e-4),
                num_iter=self.method_config.get('num_iter', 5)
            )
        elif self.combination_method == 'admm':
            combiner = ADMMCombiner(
                self.scales, self.in_channels, self.out_channels,
                rho=self.method_config.get('rho', 0.1),
                num_iter=self.method_config.get('num_iter', 1),
                beta=self.method_config.get('beta', 0.01)
            )
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
        
        # Initialize combiner carefully
        for name, param in combiner.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param, gain=0.5)
                else:
                    nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'log_' in name or 'lambda' in name:  # Log parameters
                nn.init.constant_(param, np.log(0.1))  # Small initialization
        
        return combiner
    
    def forward(self, multiscale_inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        if self.is_baseline:
            return self.model(multiscale_inputs)
        
        scale_predictions = {}
        
        # Get predictions for each scale with gradient checkpointing for stability
        for scale in self.scales:
            input_key = f'grid_{scale}'
            if input_key in multiscale_inputs:
                grid = multiscale_inputs[input_key]
                pred = self.scale_models[str(scale)](grid)
                scale_predictions[scale] = pred
        
        if not scale_predictions:
            target_size = max(self.scales)
            device = next(self.parameters()).device
            return torch.zeros(1, self.out_channels, target_size, target_size, 
                             device=device), {'scale_predictions': {}}
        
        # Use largest input for combiner
        available_scales_in_input = [int(k.split('_')[1]) for k in multiscale_inputs.keys()]
        largest_input_scale = max(available_scales_in_input)
        largest_grid_key = f'grid_{largest_input_scale}'
        largest_grid = multiscale_inputs[largest_grid_key]
        
        # Combine predictions
        combined, meta = self.combiner(scale_predictions, largest_grid)
        meta['scale_predictions'] = scale_predictions
        
        # Refine output with residual connection
        refined = self.fine_tune(combined)
        output = self.skip_weight * refined + (1 - self.skip_weight) * combined
        output = self.final_proj(output) + output  # Residual
        
        meta['final_output'] = output
        
        return output, meta
    
    def compute_loss(self, predictions: torch.Tensor, 
                multiscale_targets: Dict[str, torch.Tensor],
                metadata: Dict) -> Tuple[torch.Tensor, Dict]:
        """Compute loss with better gradient flow and physics constraints"""
        
        if self.is_baseline:
            return self.model.compute_loss(predictions, multiscale_targets, metadata)
        
        losses = {}
        
        # 1. Multi-scale reconstruction loss
        total_recon_loss = 0.0
        num_targets = 0
        
        for scale in self.scales:
            target_key = f'solution_{scale}'
            if target_key in multiscale_targets:
                target = multiscale_targets[target_key]
                
                # Resize prediction to target scale if needed
                if predictions.shape[-2:] != target.shape[-2:]:
                    pred_resized = F.interpolate(
                        predictions, 
                        size=target.shape[-2:], 
                        mode='bilinear',
                        align_corners=False
                    )
                    loss = F.mse_loss(pred_resized, target)
                else:
                    loss = F.mse_loss(predictions, target)
                
                # Weight by scale importance
                scale_weight = (scale / max(self.scales)) ** 0.5
                total_recon_loss += loss * scale_weight
                num_targets += 1
        
        if num_targets > 0:
            losses['reconstruction'] = total_recon_loss / num_targets
        else:
            losses['reconstruction'] = torch.tensor(0.0, device=predictions.device)
        
        # 2. Individual scale losses (helps gradient flow to scale models)
        scale_predictions = metadata.get('scale_predictions', {})
        scale_losses = []
        
        for scale, pred in scale_predictions.items():
            target_key = f'solution_{scale}'
            if target_key in multiscale_targets:
                target = multiscale_targets[target_key]
                scale_loss = F.mse_loss(pred, target)
                scale_losses.append(scale_loss)
        
        if scale_losses:
            losses['scale_individual'] = torch.mean(torch.stack(scale_losses)) * 0.1
        
        # 3. Weight entropy regularization (only for softmax)
        if self.combination_method == 'softmax' and 'router_weights' in metadata:
            router_weights = metadata['router_weights']
            num_scales = router_weights.shape[1]  # Get num_scales from router_weights
            
            # Encourage balanced weights (avoid collapse to single scale)
            max_entropy = torch.log(torch.tensor(float(num_scales), device=router_weights.device))
            entropy = -torch.sum(router_weights * torch.log(router_weights + 1e-8), dim=1)
            
            # Loss = squared difference from max entropy (encourages exploration)
            entropy_loss = torch.mean((entropy - max_entropy) ** 2)
            losses['weight_entropy'] = entropy_loss * 0.01
        
        # ========== ADDED: PHYSICS LOSS COMPONENTS ==========
        # 4. Physics-informed loss for Navier-Stokes equations
        physics_loss_components = self._compute_physics_loss(predictions, metadata)
        losses.update(physics_loss_components)
        
        # 5. Calculate total loss
        total_loss = losses.get('reconstruction', 0.0)
        
        for loss_name, loss_value in losses.items():
            if loss_name != 'reconstruction':
                # Scale physics losses appropriately
                if loss_name in ['physics_pde', 'physics_conservation', 'physics_boundary']:
                    total_loss = total_loss + loss_value * 1e-3  # Reduced weight for physics
                else:
                    total_loss = total_loss + loss_value
        
        losses['total'] = total_loss
        
        return total_loss, losses
    
    def _compute_physics_loss(self, predictions: torch.Tensor, metadata: Dict) -> Dict:
        """Compute physics-constrained loss based on Navier-Stokes equations"""
        physics_losses = {}
        
        # Get batch size and spatial dimensions
        batch_size = predictions.shape[0]
        H, W = predictions.shape[2], predictions.shape[3]
        
        # Skip physics loss if we don't have predictions or metadata
        if batch_size == 0 or 'scale_predictions' not in metadata:
            return physics_losses
        
        # Get predictions for the largest scale (highest resolution)
        largest_scale = max(self.scales)
        if largest_scale in metadata['scale_predictions']:
            pred_field = metadata['scale_predictions'][largest_scale]
            
            # Ensure we have valid field
            if pred_field.shape[-2] >= 4 and pred_field.shape[-1] >= 4:
                # Compute spatial gradients for physics constraints
                dx = 1.0 / (W - 1)  # Assuming domain [0,1]x[0,1]
                dy = 1.0 / (H - 1)
                
                # Compute gradients using finite differences
                grad_x = (pred_field[:, :, :, 2:] - pred_field[:, :, :, :-2]) / (2 * dx)
                grad_y = (pred_field[:, :, 2:, :] - pred_field[:, :, :-2, :]) / (2 * dy)
                
                # Compute second derivatives for Laplacian
                grad_xx = (pred_field[:, :, :, 2:] - 2 * pred_field[:, :, :, 1:-1] + pred_field[:, :, :, :-2]) / (dx**2)
                grad_yy = (pred_field[:, :, 2:, :] - 2 * pred_field[:, :, 1:-1, :] + pred_field[:, :, :-2, :]) / (dy**2)
                
                # Laplacian (diffusion term)
                laplacian = grad_xx[:, :, 1:-1, :] + grad_yy[:, :, :, 1:-1]
                
                # Vorticity transport equation residual (simplified)
                # ∂ω/∂t + u·∇ω = ν∇²ω
                # For steady state or assuming small changes: u·∇ω ≈ ν∇²ω
                
                # We'll use a simplified physics loss that encourages smoothness
                # while preserving vorticity structures
                
                # 4a. PDE residual loss (encourages vorticity field to satisfy diffusion equation)
                # Using a simple diffusion model: ∂ω/∂t ≈ D∇²ω
                diffusion_coeff = 0.01  # Small diffusion coefficient
                pde_residual = torch.abs(laplacian) * diffusion_coeff
                physics_losses['physics_pde'] = torch.mean(pde_residual**2) * 0.01
                
                # 4b. Enstrophy conservation loss (for 2D inviscid flow)
                # Enstrophy = ∫ ω² dA should be approximately conserved
                # We'll encourage conservation between predictions at different scales
                if len(metadata['scale_predictions']) > 1:
                    enstrophy_values = []
                    for scale, pred in metadata['scale_predictions'].items():
                        # Compute enstrophy density (mean of ω²)
                        enstrophy = torch.mean(pred**2, dim=[2, 3])
                        enstrophy_values.append(enstrophy)
                    
                    if len(enstrophy_values) >= 2:
                        # Encourage conservation across scales
                        enstrophy_var = torch.var(torch.stack(enstrophy_values, dim=0), dim=0)
                        physics_losses['physics_conservation'] = torch.mean(enstrophy_var) * 0.001
                
                # 4c. Boundary condition loss (no-slip for vorticity)
                # Vorticity should be zero at boundaries for no-slip conditions
                boundary_loss = torch.mean(pred_field[:, :, 0, :]**2) + \
                              torch.mean(pred_field[:, :, -1, :]**2) + \
                              torch.mean(pred_field[:, :, :, 0]**2) + \
                              torch.mean(pred_field[:, :, :, -1]**2)
                physics_losses['physics_boundary'] = boundary_loss * 0.01
                
                # 4d. Smoothness/regularity loss (encourages physical vorticity fields)
                # Total variation regularization for smooth vorticity fields
                tv_loss = torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y))
                physics_losses['physics_smoothness'] = tv_loss * 0.001
        
        return physics_losses

def stable_multiscale_comparison():
    print(f"Using device: {device}")
    
    # Create base dataset
    train_dataset = NavierStokesDataset(
        n_samples=5000, domain_size=256, seed=42
    )
    val_dataset = NavierStokesDataset(
        n_samples=1000, domain_size=256, seed=43
    )
    test_dataset = NavierStokesDataset(
        n_samples=1000, domain_size=256, seed=44
    )
    
    # Define scales
    train_scales = [32, 64]  # Training scales (add 64 for better generalization)
    val_scale = [64]
    eval_scales = [128, 256]       # Evaluation scales (include trained scales)

    
    # Create multiscale datasets
    train_multiscale = MultiScaleDataset(
        train_dataset, train_scales, mode='train', augment=True
    )
    val_multiscale = MultiScaleDataset(
        val_dataset, val_scale, mode='val', augment=False
    )
    test_multiscale = MultiScaleDataset(
        test_dataset, eval_scales, mode='test', augment=False
    )
    
    train_set, val_set = train_multiscale, val_multiscale  
    
    # Data loaders
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_multiscale, batch_size=4, shuffle=False, num_workers=2)
    
    # Get sample to determine dimensions
    sample = train_multiscale[0]
    in_channels = sample['original_grid'].shape[0]
    out_channels = sample['original_solution'].shape[0]
    
    print(f"\nDataset Info:")
    print(f"  Input channels: {in_channels}, Output channels: {out_channels}")
    print(f"  Training scales: {train_scales}")
    print(f"  Evaluation scales: {eval_scales}")
    print(f"  Train samples: {len(train_set)}")
    print(f"  Val samples: {len(val_set)}")
    print(f"  Test samples: {len(test_multiscale)}")

    # Test methods
    combination_methods = [
        #'single_scale_baseline',
        #'softmax', 
        'lagrangian_single',
        'lagrangian_two', 
        'admm'
    ]
    
    results = {}
    
    for method in combination_methods:
        print(f"\n{'='*60}")
        print(f"Training {method.upper().replace('_', ' ')}")
        print(f"{'='*60}")
        
        # Create model with training scales
        model = StableMultiscalePDEsolver(
            scales=train_scales,  # Model only knows training scales
            in_channels=in_channels,
            out_channels=out_channels,
            model_type='fno',
            combination_method=method
        ).to(device)
        
        print(f"  Model type: StableMultiscalePDEsolver")
        if hasattr(model, 'scale_models'):
            print(f"  Number of scale models: {len(model.scale_models)}")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create trainer with weight logging
        trainer = StableMultiscaleTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            grad_clip=1.0,
            patience=30,
            log_weights_every=5  # NEW: Log weights every 5 epochs
        )
        
        # Train
        try:
            history = trainer.train(num_epochs=500, verbose=True)
            
            # Test on evaluation set (includes unseen scales)
            model.eval()
            # ========== COMPUTE TEST MSE ==========
            print("Computing Test MSE...")
            test_losses = {}
            test_predictions = []
            test_targets = []

            with torch.no_grad():
                for batch in test_loader:
                    # IMPORTANT: Use all available scales in test set, not just training scales
                    # The model should generalize to unseen scales
                    multiscale_inputs = {}
                    for key, val in batch['multiscale_grids'].items():
                        multiscale_inputs[key] = val.to(device)
                    
                    multiscale_targets = {}
                    for key, val in batch['multiscale_solutions'].items():
                        multiscale_targets[key] = val.to(device)
                    
                    # Skip if no inputs
                    if not multiscale_inputs:
                        continue
                    
                    predictions, metadata = model(multiscale_inputs)
                    
                    # For single scale baseline, we need special handling
                    if model.is_baseline:
                        # Baseline model always outputs at its training scale
                        baseline_scale = model.scale
                        
                        # Find the closest target scale to the baseline's output scale
                        available_target_scales = []
                        for target_key in multiscale_targets.keys():
                            try:
                                scale = int(target_key.split('_')[1])
                                available_target_scales.append(scale)
                            except:
                                continue
                        
                        if not available_target_scales:
                            continue
                        
                        # Use the largest available target scale
                        target_scale = max(available_target_scales)
                        target_key = f'solution_{target_scale}'
                        
                        if target_key in multiscale_targets:
                            target = multiscale_targets[target_key]
                            
                            # Resize prediction to target scale
                            if predictions.shape[-2:] != target.shape[-2:]:
                                pred_resized = F.interpolate(
                                    predictions,
                                    size=target.shape[-2:],
                                    mode='bilinear',
                                    align_corners=False
                                )
                                loss = F.mse_loss(pred_resized, target)
                            else:
                                loss = F.mse_loss(predictions, target)
                            
                            if target_scale not in test_losses:
                                test_losses[target_scale] = []
                            test_losses[target_scale].append(loss.item())
                            
                            # Store for overall metrics
                            test_predictions.append(pred_resized.cpu() if 'pred_resized' in locals() else predictions.cpu())
                            test_targets.append(target.cpu())
                    else:
                        # For multiscale methods, compute loss for each evaluation scale
                        for scale in eval_scales:
                            target_key = f'solution_{scale}'
                            if target_key in multiscale_targets:
                                target = multiscale_targets[target_key]
                                
                                if predictions.shape[-2:] != target.shape[-2:]:
                                    pred_resized = F.interpolate(
                                        predictions,
                                        size=target.shape[-2:],
                                        mode='bilinear',
                                        align_corners=False
                                    )
                                    loss = F.mse_loss(pred_resized, target)
                                else:
                                    loss = F.mse_loss(predictions, target)
                                
                                if scale not in test_losses:
                                    test_losses[scale] = []
                                test_losses[scale].append(loss.item())
                                
                                # Store for overall metrics (use the first evaluation scale)
                                if scale == eval_scales[0]:
                                    test_predictions.append(pred_resized.cpu() if 'pred_resized' in locals() else predictions.cpu())
                                    test_targets.append(target.cpu())
                                    
            
            # Compute average losses per scale
            avg_test_losses = {}
            for scale, losses in test_losses.items():
                avg_test_losses[scale] = np.mean(losses) if losses else float('inf')
            
            # Overall metrics (using largest evaluation scale)
            if test_predictions and test_targets:
                all_preds = torch.cat(test_predictions, dim=0)
                all_targets = torch.cat(test_targets, dim=0)
                
                # Resize if needed
                if all_preds.shape[-2:] != all_targets.shape[-2:]:
                    all_preds = F.interpolate(
                        all_preds, 
                        size=all_targets.shape[-2:], 
                        mode='bilinear',
                        align_corners=False
                    )
                
                rmse = torch.sqrt(F.mse_loss(all_preds, all_targets)).item()
                relative_error = torch.mean(
                    torch.abs(all_preds - all_targets) / (torch.abs(all_targets) + 1e-6)
                ).item()
            else:
                rmse = float('inf')
                relative_error = float('inf')
            
            # Store results
            results[method] = {
                'history': history,
                'weight_history': trainer.epoch_weights_summary,  # NEW: save weight history
                'admm_weights_per_epoch': trainer.admm_weights_per_epoch if method == 'admm' else [],  # NEW: save ADMM weights
                'final_val_loss': history['val_loss'][-1] if history['val_loss'] else float('inf'),
                'test_losses': avg_test_losses,
                'test_rmse': rmse,
                'test_relative_error': relative_error,
                'best_val_loss': min(history['val_loss']) if history['val_loss'] else float('inf'),
                'num_parameters': sum(p.numel() for p in model.parameters()),
                'success': True
            }
            
            print(f"\n  Test Results for {method.upper().replace('_', ' ')}:")
            for scale, loss in avg_test_losses.items():
                print(f"    Scale {scale}x{scale} Loss: {loss:.4e}")
            print(f"    Overall RMSE: {rmse:.4e}")
            print(f"    Relative Error: {relative_error:.4%}")
            print(f"    Best Validation Loss: {results[method]['best_val_loss']:.4e}")
            
        except Exception as e:
            print(f"  Training failed for {method}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[method] = {
                'success': False,
                'error': str(e)
            }
    
    return results, train_scales, eval_scales



# ========== SIMPLIFIED ANALYSIS ==========
def analyze_stable_results(results: Dict, train_scales: List[int]):
    print(f"\n{'='*80}")
    print("STABLE METHOD COMPARISON")
    print(f"{'='*80}")
    
    # Filter successful methods
    successful_methods = {k: v for k, v in results.items() if v.get('success', False)}
    
    if not successful_methods:
        print("No methods successfully trained!")
        return None, []
    
    # Create summary table
    print(f"\n{'Method':<25} {'Best Val Loss':<15} {'Test Loss':<15} {'Rel Error':<15} {'Params':<15}")
    print(f"{'-'*80}")
    
    for method, result in successful_methods.items():
        method_name = method.replace('_', ' ').title()
        print(f"{method_name:<25} {result['best_val_loss']:<15.4e} "
              f"{result['test_loss']:<15.4e} {result['test_relative_error']:<15.4%} "
              f"{result['num_parameters']:<15,}")
    
    # Simple bar plot
    plt.figure(figsize=(12, 6))
    
    methods = list(successful_methods.keys())
    test_losses = [successful_methods[m]['test_loss'] for m in methods]
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    bars = plt.bar(methods, test_losses, color=colors[:len(methods)])
    
    plt.yscale('log')
    plt.ylabel('Test Loss (log scale)')
    plt.title('Test Loss Comparison of Multiscale Methods')
    plt.xticks(rotation=45, ha='right')
    
    # Add values on bars
    for bar, loss in zip(bars, test_losses):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.2e}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('stable_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find best method
    best_method = min(successful_methods, key=lambda m: successful_methods[m]['test_loss'])
    best_loss = successful_methods[best_method]['test_loss']
    
    print(f"\nBest Method: {best_method.replace('_', ' ').title()} with loss {best_loss:.4e}")
    
    # Compare to baseline if available
    if 'single_scale_baseline' in successful_methods:
        baseline_loss = successful_methods['single_scale_baseline']['test_loss']
        print(f"Baseline (Single Scale) loss: {baseline_loss:.4e}")
        
        for method in methods:
            if method != 'single_scale_baseline':
                improvement = (baseline_loss - successful_methods[method]['test_loss']) / baseline_loss * 100
                print(f"  {method.replace('_', ' ').title():<25} Improvement: {improvement:+.1f}%")
    
    return best_method, list(successful_methods.keys())

def plot_weight_evolution_analysis(results: Dict, train_scales: List[int]):
    """Create comprehensive weight evolution analysis"""
    
    # Filter out single_scale_baseline since it doesn't have lambda weights
    successful_methods = {k: v for k, v in results.items() 
                         if v.get('success', False) and k != 'single_scale_baseline' and 'weight_history' in v}
    
    if not successful_methods:
        print("No successful methods with weight history available!")
        return
    
    # Create a comprehensive figure
    n_methods = len(successful_methods)
    fig, axes = plt.subplots(2, max(2, n_methods), figsize=(5*max(2, n_methods), 10))
    
    if n_methods == 1:
        axes = axes.reshape(2, 1)
    
    # Colors for scales
    scale_colors = plt.cm.Set3(np.linspace(0, 1, len(train_scales)))
    
    for idx, (method, result) in enumerate(successful_methods.items()):
        ax1 = axes[0, idx] if n_methods > 1 else axes[0]
        ax2 = axes[1, idx] if n_methods > 1 else axes[1]
        
        weight_history = result['weight_history']
        
        # Extract epoch data
        epochs = [entry['epoch'] for entry in weight_history]
        
        # Get all scale weights over time
        scale_weight_matrix = []
        scale_labels = []
        
        for scale in train_scales:
            scale_key = f'scale_{scale}'
            weights = []
            for entry in weight_history:
                if scale_key in entry['weights']:
                    weights.append(entry['weights'][scale_key])
                else:
                    weights.append(0.0)
            scale_weight_matrix.append(weights)
            scale_labels.append(f'Scale {scale}')
        
        scale_weight_matrix = np.array(scale_weight_matrix)
        
        # 1. Weight evolution line plot
        for i, (weights, label) in enumerate(zip(scale_weight_matrix, scale_labels)):
            ax1.plot(epochs, weights, 'o-', label=label, color=scale_colors[i], 
                    markersize=3, linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Weight Value')
        ax1.set_title(f'{method.upper()}: Weight Evolution')
        ax1.legend(fontsize=8, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. Final weight distribution (pie chart)
        if len(weight_history) > 0:
            final_weights = weight_history[-1]['weights']
            
            # Extract weights for scales
            pie_weights = []
            pie_labels = []
            for scale in train_scales:
                scale_key = f'scale_{scale}'
                if scale_key in final_weights:
                    pie_weights.append(final_weights[scale_key])
                    pie_labels.append(f'Scale {scale}')
            
            if pie_weights and sum(pie_weights) > 0:
                # Normalize
                pie_weights = np.array(pie_weights)
                pie_weights = pie_weights / pie_weights.sum()
                
                # Create pie chart
                wedges, texts, autotexts = ax2.pie(pie_weights, labels=pie_labels, colors=scale_colors[:len(pie_weights)],
                                                  autopct='%1.1f%%', startangle=90)
                
                # Enhance readability
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                ax2.set_title(f'{method.upper()}: Final Weight Distribution')
            else:
                ax2.text(0.5, 0.5, 'No weight data', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title(f'{method.upper()}: No Weight Data')
    
    # Hide unused subplots
    for idx in range(n_methods, axes.shape[1]):
        axes[0, idx].axis('off')
        axes[1, idx].axis('off')
    
    plt.suptitle('Weight Evolution and Distribution Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('comprehensive_weight_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComprehensive weight analysis saved to 'comprehensive_weight_analysis.png'")
    
    # Print detailed weight statistics
    print(f"\n{'='*80}")
    print("DETAILED WEIGHT STATISTICS")
    print(f"{'='*80}")
    
    for method, result in successful_methods.items():
        weight_history = result['weight_history']
        
        if not weight_history:
            continue
        
        final_weights = weight_history[-1]['weights']
        
        print(f"\n{method.upper().replace('_', ' ')}:")
        print(f"{'-'*40}")
        
        # Extract and sort scale weights
        scale_weights = []
        for key, value in final_weights.items():
            if key.startswith('scale_') and not (key.endswith('_norm') or key.endswith('_rel')):
                try:
                    scale_num = int(key.replace('scale_', ''))
                    scale_weights.append((scale_num, value))
                except:
                    continue
        
        scale_weights.sort()
        
        if scale_weights:
            total_weight = sum(w for _, w in scale_weights)
            if total_weight > 0:
                for scale, weight in scale_weights:
                    percentage = (weight / total_weight) * 100
                    print(f"  Scale {scale:3d}: {weight:8.6f} ({percentage:6.2f}%)")
                
                # Calculate statistics
                weights_only = [w for _, w in scale_weights]
                entropy = 0
                for w in weights_only:
                    if w > 0:
                        p = w / total_weight
                        entropy -= p * math.log(p)
                
                max_entropy = math.log(len(weights_only))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                print(f"\n  Total weight: {total_weight:.6f}")
                print(f"  Entropy: {entropy:.4f} (Normalized: {normalized_entropy:.4f})")
                
                # Interpretation
                if normalized_entropy < 0.3:
                    print("  Interpretation: One scale dominates (low diversity)")
                elif normalized_entropy > 0.7:
                    print("  Interpretation: Weights are well distributed (high diversity)")
                else:
                    print("  Interpretation: Moderate weight distribution")
                
                # Check weight convergence
                if len(weight_history) > 10:
                    early_weights = weight_history[5]['weights']
                    late_weights = weight_history[-1]['weights']
                    
                    early_total = 0
                    late_total = 0
                    for scale, _ in scale_weights:
                        early_key = f'scale_{scale}'
                        late_key = f'scale_{scale}'
                        early_weight = early_weights.get(early_key, 0)
                        late_weight = late_weights.get(late_key, 0)
                        early_total += early_weight
                        late_total += late_weight
                    
                    if early_total > 0 and late_total > 0:
                        # Calculate change in distribution
                        changes = []
                        for scale, _ in scale_weights:
                            early_key = f'scale_{scale}'
                            late_key = f'scale_{scale}'
                            early_norm = early_weights.get(early_key, 0) / early_total
                            late_norm = late_weights.get(late_key, 0) / late_total
                            changes.append(abs(late_norm - early_norm))
                        
                        avg_change = np.mean(changes)
                        print(f"  Average distribution change: {avg_change:.6f}")
                        
                        if avg_change < 0.01:
                            print("  ✅ Weights have converged well")
                        else:
                            print("  ⚠️  Weights may still be changing")

def print_admm_weight_summary(results: Dict, train_scales: List[int]):
    """Print detailed ADMM weight summary"""
    if 'admm' not in results or not results['admm'].get('success', False):
        print("ADMM method not found or not successful!")
        return
    
    admm_result = results['admm']
    
    if 'admm_weights_per_epoch' not in admm_result or not admm_result['admm_weights_per_epoch']:
        print("No ADMM weight data available!")
        return
    
    admm_weights = admm_result['admm_weights_per_epoch']
    
    print(f"\n{'='*80}")
    print("ADMM DETAILED WEIGHT EVOLUTION")
    print(f"{'='*80}")
    
    # Get scales
    first_epoch_weights = admm_weights[0]
    scales = sorted(list(first_epoch_weights.keys()))
    
    # Print header
    print(f"\n{'Epoch':<8}", end="")
    for scale in scales:
        print(f"{'Scale ' + str(scale):<12}", end="")
    print(f"{'Total':<12} {'Rho':<12}")
    
    print("-" * (8 + 12 * (len(scales) + 2)))
    
    # Print weights for each epoch
    for epoch_idx, epoch_weights in enumerate(admm_weights):
        print(f"{epoch_idx:<8}", end="")
        total_weight = 0.0
        for scale in scales:
            weight = epoch_weights.get(scale, 0.0)
            total_weight += weight
            print(f"{weight:<12.4f}", end="")
        print(f"{total_weight:<12.4f}", end="")
        
        # Get rho if available
        if epoch_idx == 0 or epoch_idx == len(admm_weights) - 1:
            print(f"{'N/A':<12}")
        else:
            print(f"{'':<12}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("ADMM FINAL WEIGHT DISTRIBUTION")
    print(f"{'='*80}")
    
    final_weights = admm_weights[-1]
    total_final = sum(final_weights.values())
    
    print(f"\n{'Scale':<10} {'Weight':<10} {'Percentage':<12}")
    print("-" * 32)
    for scale in scales:
        weight = final_weights.get(scale, 0.0)
        percentage = (weight / total_final * 100) if total_final > 0 else 0
        print(f"{scale:<10} {weight:<10.4f} {percentage:<11.2f}%")
    
    # Calculate weight evolution
    if len(admm_weights) > 1:
        initial_weights = admm_weights[0]
        final_weights = admm_weights[-1]
        
        print(f"\n{'='*80}")
        print("ADMM WEIGHT CHANGES")
        print(f"{'='*80}")
        
        print(f"\n{'Scale':<10} {'Initial':<10} {'Final':<10} {'Change':<10} {'% Change':<10}")
        print("-" * 50)
        
        for scale in scales:
            initial = initial_weights.get(scale, 0.0)
            final = final_weights.get(scale, 0.0)
            change = final - initial
            percent_change = (change / initial * 100) if initial > 0 else 0
            print(f"{scale:<10} {initial:<10.4f} {final:<10.4f} {change:<10.4f} {percent_change:<9.1f}%")
        
        # Calculate stability metrics
        if len(admm_weights) > 5:
            last_5_weights = admm_weights[-5:]
            stability_metrics = {}
            
            for scale in scales:
                weights_over_last_5 = [w.get(scale, 0.0) for w in last_5_weights]
                mean_weight = np.mean(weights_over_last_5)
                std_weight = np.std(weights_over_last_5)
                cv = (std_weight / mean_weight * 100) if mean_weight > 0 else 0
                stability_metrics[scale] = {'mean': mean_weight, 'std': std_weight, 'cv': cv}
            
            print(f"\n{'='*80}")
            print("ADMM WEIGHT STABILITY (LAST 5 EPOCHS)")
            print(f"{'='*80}")
            
            print(f"\n{'Scale':<10} {'Mean':<10} {'Std Dev':<10} {'CV (%)':<10} {'Stability':<15}")
            print("-" * 55)
            
            for scale in scales:
                metrics = stability_metrics[scale]
                cv = metrics['cv']
                if cv < 5:
                    stability = "Very Stable"
                elif cv < 10:
                    stability = "Stable"
                elif cv < 20:
                    stability = "Moderate"
                else:
                    stability = "Unstable"
                
                print(f"{scale:<10} {metrics['mean']:<10.4f} {metrics['std']:<10.4f} "
                      f"{cv:<9.1f}% {stability:<15}")


# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    print("=" * 80)
    print("STABLE MULTISCALE PDE SOLVER COMPARISON WITH WEIGHT LOGGING")
    print("Will print ADMM scale weights every 5 epochs")
    print("=" * 80)
    
    # Run stable comparison
    results, train_scales, eval_scales = stable_multiscale_comparison()
    
    # Analyze results
    if results:
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE - SUMMARY")
        print(f"{'='*80}")
        
        successful_methods = {k: v for k, v in results.items() if v.get('success', False)}
        
        if successful_methods:
            print(f"\nSuccessful methods: {', '.join(successful_methods.keys())}")
            print(f"Training scales: {train_scales}")
            print(f"Evaluation scales: {eval_scales}")
            
            # Display results
            print(f"\n{'Method':<25} {'Best Val Loss':<15} {'Overall RMSE':<15}")
            print(f"{'-'*55}")
            for method, result in successful_methods.items():
                method_name = method.replace('_', ' ').title()
                print(f"{method_name:<25} {result['best_val_loss']:<15.4e} "
                      f"{result['test_rmse']:<15.4e}")
            
            # Print ADMM weight summary
            if 'admm' in successful_methods:
                print_admm_weight_summary(results, train_scales)
            
            # Plot weight evolution analysis
            plot_weight_evolution_analysis(results, train_scales)
            
        else:
            print("No methods successfully trained!")
    else:
        print("No results obtained!")