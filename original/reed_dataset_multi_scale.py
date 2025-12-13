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
                 width: int = 32, depth: int = 4):
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
    """ADMM (Alternating Direction Method of Multipliers) combiner - Fixed Implementation"""
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
            'consensus_weight': self.consensus_weight
        }
    

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



class StableMultiscaleTrainer:
    def __init__(self, model, 
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str,
                 lr: Optional[float] = None,
                 grad_clip: float = 1.0,
                 patience: int = 30):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.patience = patience
        
        # Get the actual scales the model is designed for
        self.model_scales = model.scales
        self.method = model.combination_method
        
        # Method-specific learning rates
        if lr is None:
            lr_map = {
                'single_scale_baseline': 3e-4,
                'softmax': 1e-3,
                'lagrangian_single': 1e-3,
                'lagrangian_two': 1e-3,
                'admm': 1e-3,
            }
            lr = lr_map.get(model.combination_method, 1e-3)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            epochs=100,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0
        )
        
        self.grad_clip = grad_clip
        self.best_val_loss = float('inf')
        self.val_loss_ema = None
        self.ema_alpha = 0.1
        
        self.history = defaultdict(list)
        self.history['combiner_weights'] = []

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
    
    def _log_combiner_weights(self):
        """Extract and log lambda weights for all combiners"""
        if self.method == 'single_scale_baseline':
            # Don't log weights for baseline
            return
            
        combiner = self.model.combiner
        weights_to_log = {}
        
        # Extract lambda weights for all methods
        if (self.method == 'lagrangian_single' or self.method == 'lagrangian_two') and hasattr(combiner, 'log_lambdas'):
            # Direct lambda weights (normalized)
            lambdas = torch.exp(combiner.log_lambdas).detach().cpu().numpy()
            lambda_weights = lambdas / np.sum(lambdas)
            weights_to_log['lambda'] = lambda_weights
            
        # elif self.method == 'lagrangian_two':
        #     # Dual variables from softmax (already sum to 1)
        #     if hasattr(combiner, 'dual_logits'):
        #         lambda_weights = F.softmax(combiner.dual_logits, dim=0).detach().cpu().numpy()
        #         weights_to_log['lambda'] = lambda_weights
            
        elif self.method == 'admm':
            # For ADMM, check if we have last_dual_vars
            if hasattr(combiner, 'last_dual_vars') and combiner.last_dual_vars:
                # Extract mean magnitude of dual variables for each scale
                lambda_weights = []
                scales_with_vars = sorted(combiner.last_dual_vars.keys())
                
                for scale in self.model_scales:
                    if scale in combiner.last_dual_vars:
                        dual_var = combiner.last_dual_vars[scale]
                        weight = torch.mean(torch.abs(dual_var)).item()
                        lambda_weights.append(weight)
                    else:
                        # If scale not in dual_vars, use 0
                        lambda_weights.append(0.0)
                
                if lambda_weights and sum(lambda_weights) > 0:
                    # Normalize to sum to 1
                    lambda_weights = np.array(lambda_weights)
                    lambda_weights = lambda_weights / np.sum(lambda_weights)
                    weights_to_log['lambda'] = lambda_weights
                else:
                    # Fallback: use equal weights
                    weights_to_log['lambda'] = np.ones(len(self.model_scales)) / len(self.model_scales)
            else:
                # Fallback: use equal weights
                weights_to_log['lambda'] = np.ones(len(self.model_scales)) / len(self.model_scales)
                    
        elif self.method == 'softmax':
            # For softmax, we need to store router weights during forward pass
            if hasattr(combiner, 'last_router_weights') and combiner.last_router_weights is not None:
                # Take mean across batch dimension
                router_weights = combiner.last_router_weights.detach().cpu().numpy()
                lambda_weights = np.mean(router_weights, axis=0)  # Shape: [num_scales]
                weights_to_log['lambda'] = lambda_weights
            else:
                # Fallback: use equal weights
                weights_to_log['lambda'] = np.ones(len(self.model_scales)) / len(self.model_scales)
        
        if weights_to_log:
            self.history['combiner_weights'].append(weights_to_log)
            
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        batch_count = 0
        
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
                self.scheduler.step()
                
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
        self._log_combiner_weights()
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
            train_loss = self.train_epoch()
            
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
                }, f'best_model_{self.model.combination_method}.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Logging
            if verbose and (epoch % 5 == 0 or epoch < 5 or epoch == num_epochs - 1):
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{num_epochs}:")
                print(f"  Train Loss = {train_loss:.4e}")
                print(f"  Val Loss = {val_loss:.4e} (EMA: {val_loss_ema:.4e})")
                print(f"  LR = {current_lr:.2e}, Best Val = {best_val_loss:.4e}")
                print(f"  Patience = {patience_counter}/{self.patience}")
        
        # Load best model if available
        try:
            checkpoint = torch.load(f'best_model_{self.model.combination_method}.pth')
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except FileNotFoundError:
            print(f"Warning: No best model saved for {self.model.combination_method}")
        
        return self.history



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
                'primal_lr': 5e-3,  # Increase from 1e-3 to 5e-3
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
        """Compute loss with better gradient flow"""
        
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
        
        # 4. Calculate total loss
        total_loss = losses.get('reconstruction', 0.0)
        
        for loss_name, loss_value in losses.items():
            if loss_name != 'reconstruction':
                total_loss = total_loss + loss_value
        
        losses['total'] = total_loss
        
        return total_loss, losses

def stable_multiscale_comparison():
    print(f"Using device: {device}")
    
    # Create base dataset
    train_dataset = NavierStokesDataset(
        n_samples=2000, domain_size=128, seed=42
    )
    val_dataset = NavierStokesDataset(
        n_samples=500, domain_size=128, seed=43
    )
    test_dataset = NavierStokesDataset(
        n_samples=500, domain_size=128, seed=44
    )
    
    # Define scales
    train_scales = [32, 64]  # Training scales (add 64 for better generalization)
    val_scale = [64]
    eval_scales = [64, 128]       # Evaluation scales (include trained scales)

    
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
    
    # # Split train into train/val
    # train_size = int(0.8 * len(train_multiscale))
    # val_size = len(train_multiscale) - train_size
    # train_set, val_set = torch.utils.data.random_split(
    #     train_multiscale, [train_size, val_size]
    # )
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
        'single_scale_baseline',
        'softmax', 
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
        
        # Create trainer
        trainer = StableMultiscaleTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            grad_clip=1.0,
            patience=30
        )
        
        # Train
        try:
            history = trainer.train(num_epochs=10, verbose=True)
            
            # Test on evaluation set (includes unseen scales)
            model.eval()
            test_losses = {}
            test_predictions = []
            test_targets = []
            
            with torch.no_grad():
                for batch in test_loader:
                    # Filter to only include scales the model was trained on
                    # This simulates real multiscale inference
                    multiscale_inputs = {}
                    for scale in train_scales:
                        key = f'grid_{scale}'
                        if key in batch['multiscale_grids']:
                            multiscale_inputs[key] = batch['multiscale_grids'][key].to(device)
                    
                    multiscale_targets = {}
                    for scale in train_scales:  # Use train_scales, not eval_scales
                        target_key = f'solution_{scale}'
                        if target_key in batch['multiscale_solutions']:
                            multiscale_targets[target_key] = batch['multiscale_solutions'][target_key].to(device)
                            
                    # Skip if no valid inputs
                    if not multiscale_inputs:
                        continue
                    
                    predictions, metadata = model(multiscale_inputs)
                    
                    # Compute loss for each evaluation scale
                    for scale in eval_scales:
                        target_key = f'solution_{scale}'
                        if target_key in multiscale_targets:
                            target = multiscale_targets[target_key]
                            
                            # Ensure predictions match target size
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
                    
                    # Store for analysis
                    test_predictions.append(predictions.cpu())
                    # Use largest evaluation scale for overall metrics
                    largest_train_scale = max(train_scales)  # This should be 64
                    target_key = f'solution_{largest_train_scale}'
                    if target_key in multiscale_targets:
                        test_targets.append(multiscale_targets[target_key].cpu())
            
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
    plt.show()
    
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

def plot_loss_evolution(results: Dict, log_scale: bool = True):
    """Plot the validation loss evolution for all successful methods in a single graph."""
    
    successful_methods = {k: v for k, v in results.items() if v.get('success', False)}
    
    if not successful_methods:
        print("No successful methods to plot loss evolution for!")
        return
        
    plt.figure(figsize=(12, 6))
    
    for method, result in successful_methods.items():
        history = result['history']
        method_name = method.replace('_', ' ').title()
        print(method_name)
        if method_name == 'Single Scale Baseline':
            continue #Skip baseline
        #print(method_name, history)
        # if 'val_loss_ema' in history:
        #     # Use Exponential Moving Average (EMA) for a smoother, clearer plot
        #     val_loss = history['val_loss_ema']
        #     label = f'{method_name} (EMA)'
            
        #     # Pad with last value if necessary to ensure all lines are the same length
        #     max_len = max(len(h['val_loss_ema']) for h in successful_methods.values())
            
        #     if len(val_loss) < max_len:
        #         last_val = val_loss[-1]
        #         val_loss.extend([last_val] * (max_len - len(val_loss)))
                
        #     plt.plot(val_loss, label=label, linewidth=2)
            
        if 'val_loss' in history:
            # Fallback to raw validation loss if EMA is missing
            val_loss = history['val_loss']
            label = f'{method_name} (Raw)'
            
            # Pad with last value if necessary
            max_len = max(len(h['val_loss']) for h in successful_methods.values())
            
            if len(val_loss) < max_len:
                last_val = val_loss[-1]
                val_loss.extend([last_val] * (max_len - len(val_loss)))

            plt.plot(val_loss, label=label, linestyle='--', linewidth=1)
            
        else:
            print(f"Warning: No loss data found for {method_name}")


    if log_scale:
        plt.yscale('log')
        plt.ylabel('Validation Loss (Log Scale)')
    else:
        plt.ylabel('Validation Loss (Linear Scale)')
        
    plt.xlabel('Epoch')
    plt.title('Validation Loss Evolution of Multiscale Methods')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('loss_evolution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nLoss evolution plot saved to 'loss_evolution_comparison.png'")

# Add this function at the end of your code, right before if __name__ == "__main__":


def plot_method_comparison_summary(results: Dict, train_scales: List[int]):
    """Create a comprehensive summary plot similar to the example"""
    
    successful_methods = {k: v for k, v in results.items() if v.get('success', False)}
    
    if not successful_methods:
        print("No successful methods to plot summary!")
        return
    
    # Method colors and names
    method_colors = {
        'single_scale_baseline': '#8c564b',
        'softmax': '#1f77b4',
        'lagrangian_single': '#2ca02c',
        'lagrangian_two': '#d62728',
        'admm': '#9467bd',
    }
    
    method_names = {
        'single_scale_baseline': 'Single Scale',
        'softmax': 'Softmax',
        'lagrangian_single': 'Lagrangian (S)',
        'lagrangian_two': 'Lagrangian (T)',
        'admm': 'ADMM',
    }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Training Loss (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    for method, result in successful_methods.items():
        if 'history' in result and 'train_loss' in result['history']:
            ax1.plot(result['history']['train_loss'], 
                    label=method_names[method],
                    color=method_colors[method],
                    linewidth=2)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation Loss (Top Middle Left)
    ax2 = fig.add_subplot(gs[0, 1])
    for method, result in successful_methods.items():
        if 'history' in result and 'val_loss' in result['history']:
            ax2.plot(result['history']['val_loss'], 
                    label=method_names[method],
                    color=method_colors[method],
                    linewidth=2)
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Test Loss by Scale (Top Middle Right)
    ax3 = fig.add_subplot(gs[0, 2])
    for method, result in successful_methods.items():
        if 'test_losses' in result:
            scales = list(result['test_losses'].keys())
            losses = list(result['test_losses'].values())
            ax3.plot(scales, losses, 'o-', 
                    label=method_names[method],
                    color=method_colors[method],
                    linewidth=2, markersize=8)
    ax3.set_title('Test Loss by Scale', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Scale')
    ax3.set_ylabel('Loss')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    
    # 4. Final Test Performance (Top Right)
    ax4 = fig.add_subplot(gs[0, 3])
    methods = list(successful_methods.keys())
    test_rmses = [successful_methods[m]['test_rmse'] for m in methods]
    colors = [method_colors[m] for m in methods]
    labels = [method_names[m] for m in methods]
    
    bars = ax4.bar(labels, test_rmses, color=colors, alpha=0.8)
    ax4.set_title('Final Test RMSE', fontsize=14, fontweight='bold')
    ax4.set_ylabel('RMSE')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, rmse in zip(bars, test_rmses):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse:.2e}', ha='center', va='bottom', fontsize=10)
    
    # 5. Weight Distribution Heatmap (Middle Row, full width)
    ax5 = fig.add_subplot(gs[1, :])
    
    # Collect weight data
    weight_data = []
    weight_methods = []
    for method, result in successful_methods.items():
        if method != 'single_scale_baseline' and 'history' in result:
            history = result['history']
            if 'combiner_weights' in history and history['combiner_weights']:
                last_weights = history['combiner_weights'][-1].get('lambda', None)
                if last_weights is not None and len(last_weights) == len(train_scales):
                    weight_data.append(last_weights)
                    weight_methods.append(method_names[method])
    
    if weight_data:
        weight_matrix = np.array(weight_data)
        im = ax5.imshow(weight_matrix, cmap='YlOrRd', aspect='auto', 
                       vmin=0, vmax=1, interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax5)
        cbar.set_label('Weight Value', rotation=270, labelpad=15)
        
        # Set labels
        ax5.set_xticks(np.arange(len(train_scales)))
        ax5.set_yticks(np.arange(len(weight_methods)))
        ax5.set_xticklabels([f'Scale {s}' for s in train_scales], fontsize=11)
        ax5.set_yticklabels(weight_methods, fontsize=11)
        
        # Add text annotations
        for i in range(len(weight_methods)):
            for j in range(len(train_scales)):
                text = ax5.text(j, i, f'{weight_matrix[i, j]:.3f}',
                              ha="center", va="center", 
                              color="black" if weight_matrix[i, j] < 0.7 else "white",
                              fontsize=10)
        
        ax5.set_title('Weight Distribution Heatmap', fontsize=14, fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'No weight data available', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Weight Distribution Heatmap', fontsize=14, fontweight='bold')
    
    # 6. Weight Evolution (Bottom Left)
    ax6 = fig.add_subplot(gs[2, 0])
    for method, result in successful_methods.items():
        if method != 'single_scale_baseline' and 'history' in result:
            history = result['history']
            if 'combiner_weights' in history and history['combiner_weights']:
                weights_evo = []
                for epoch_weights in history['combiner_weights']:
                    if 'lambda' in epoch_weights:
                        weights = epoch_weights['lambda']
                        if weights is not None:
                            weights_evo.append(np.mean(weights))
                
                if weights_evo:
                    ax6.plot(weights_evo, 
                            label=method_names[method],
                            color=method_colors[method],
                            linewidth=2)
    
    ax6.set_title('Average Weight Evolution', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Mean Weight')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # 7. Parameter Count (Bottom Middle Left)
    ax7 = fig.add_subplot(gs[2, 1])
    methods = list(successful_methods.keys())
    param_counts = [successful_methods[m]['num_parameters'] for m in methods]
    labels = [method_names[m] for m in methods]
    colors = [method_colors[m] for m in methods]
    
    bars = ax7.bar(labels, param_counts, color=colors, alpha=0.8)
    ax7.set_title('Model Complexity', fontsize=14, fontweight='bold')
    ax7.set_ylabel('Number of Parameters')
    ax7.tick_params(axis='x', rotation=45)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, count in zip(bars, param_counts):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{count/1e6:.2f}M', ha='center', va='bottom', fontsize=10)
    
    # 8. Relative Improvement Over Baseline (Bottom Middle Right)
    ax8 = fig.add_subplot(gs[2, 2])
    if 'single_scale_baseline' in successful_methods:
        baseline_loss = successful_methods['single_scale_baseline']['test_rmse']
        improvements = []
        method_labels = []
        
        for method, result in successful_methods.items():
            if method != 'single_scale_baseline':
                improvement = ((baseline_loss - result['test_rmse']) / baseline_loss) * 100
                improvements.append(improvement)
                method_labels.append(method_names[method])
        
        colors = [method_colors[m] for m in successful_methods.keys() 
                 if m != 'single_scale_baseline']
        
        bars = ax8.bar(method_labels, improvements, color=colors, alpha=0.8)
        ax8.set_title('Improvement Over Baseline', fontsize=14, fontweight='bold')
        ax8.set_ylabel('Improvement (%)')
        ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax8.tick_params(axis='x', rotation=45)
        ax8.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            color = 'green' if imp > 0 else 'red'
            ax8.text(bar.get_x() + bar.get_width()/2., 
                    height + (1 if imp >= 0 else -1),
                    f'{imp:+.1f}%', ha='center', va='bottom' if imp >= 0 else 'top',
                    color=color, fontsize=10)
    
    # 9. Best Validation Loss (Bottom Right)
    ax9 = fig.add_subplot(gs[2, 3])
    best_val_losses = [successful_methods[m]['best_val_loss'] for m in methods]
    labels = [method_names[m] for m in methods]
    colors = [method_colors[m] for m in methods]
    
    bars = ax9.bar(labels, best_val_losses, color=colors, alpha=0.8)
    ax9.set_title('Best Validation Loss', fontsize=14, fontweight='bold')
    ax9.set_ylabel('Loss')
    ax9.tick_params(axis='x', rotation=45)
    ax9.set_yscale('log')
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, loss in zip(bars, best_val_losses):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.2e}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Multiscale Method Comparison Summary', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('method_comparison_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nComprehensive comparison summary saved to 'method_comparison_summary.png'")


def plot_weight_evolution(results: Dict, train_scales: List[int]):
    """Plot the evolution of lambda weights during training"""
    
    # Filter out single_scale_baseline since it doesn't have lambda weights
    successful_methods = {k: v for k, v in results.items() 
                         if v.get('success', False) and k != 'single_scale_baseline'}
    
    if not successful_methods:
        print("No successful methods to plot weight evolution!")
        return
    
    # Create subplots for all successful methods
    n_methods = len(successful_methods)
    n_cols = min(2, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Flatten axes if we have multiple rows/columns
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for idx, (method, result) in enumerate(successful_methods.items()):
        ax = axes[idx]
        history = result['history']
        
        if 'combiner_weights' in history and history['combiner_weights']:
            weights_history = history['combiner_weights']
            
            # Extract lambda weights over epochs
            lambda_weights_history = []
            for epoch_weights in weights_history:
                if 'lambda' in epoch_weights:
                    lambda_weights = epoch_weights['lambda']
                    if lambda_weights is not None:
                        lambda_weights_history.append(lambda_weights)
            
            if lambda_weights_history:
                lambda_weights_arr = np.array(lambda_weights_history)
                
                # Plot each lambda weight evolution
                for i in range(lambda_weights_arr.shape[1]):
                    label = f'Scale {train_scales[i]}' if i < len(train_scales) else f'λ{i}'
                    ax.plot(range(len(lambda_weights_arr)), lambda_weights_arr[:, i], 
                           label=label, marker='o', markersize=3, linewidth=1.5)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Lambda Weight')
                ax.set_title(f'{method.upper()}: Lambda Weights Evolution')
                ax.legend(fontsize=9, loc='best')
                ax.grid(True, alpha=0.3)
                
                # Add annotation for final values
                if len(lambda_weights_arr) > 0:
                    final_weights = lambda_weights_arr[-1]
                    # Check if weights sum to approximately 1
                    weight_sum = np.sum(final_weights)
                    ax.text(0.02, 0.98, f'Sum: {weight_sum:.3f}',
                           transform=ax.transAxes, fontsize=8,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    for i, weight in enumerate(final_weights):
                        ax.text(0.02, 0.90 - i*0.05, f'S{i+1}: {weight:.3f}',
                               transform=ax.transAxes, fontsize=8,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.text(0.5, 0.5, 'No lambda weights data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{method.upper()}')
        else:
            ax.text(0.5, 0.5, 'No weight evolution data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{method.upper()}')
    
    # Hide unused subplots
    for idx in range(len(successful_methods), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Lambda Weight Evolution During Training (Excluding Single Scale Baseline)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('lambda_weight_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nLambda weight evolution plot saved to 'lambda_weight_evolution.png'")

if __name__ == "__main__":
    print("=" * 80)
    print("STABLE MULTISCALE PDE SOLVER COMPARISON")
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
            
            # Plot results
            plt.figure(figsize=(12, 6))
            methods = list(successful_methods.keys())
            test_rmse = [successful_methods[m]['test_rmse'] for m in methods]
            
            colors = ['blue', 'green', 'red', 'orange', 'purple']
            bars = plt.bar(methods, test_rmse, color=colors[:len(methods)])
            
            plt.yscale('log')
            plt.ylabel('Test RMSE (log scale)')
            plt.title('Multiscale Method Comparison - Extrapolation to Higher Resolutions')
            plt.xticks(rotation=45, ha='right')
            
            # Add values on bars
            for bar, rmse in zip(bars, test_rmse):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{rmse:.2e}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('stable_comparison_fixed.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"\nVisualization saved to 'stable_comparison_fixed.png'")
            
            # Plot the combiner weights
            plot_method_comparison_summary(results, train_scales)
            
            # Plot weight evolution during training
            plot_weight_evolution(results, train_scales)
            #plot_loss_evolution(results)
            
        else:
            print("No methods successfully trained!")
    else:
        print("No results obtained!")    