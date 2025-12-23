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
class FourierBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )
        self.w = nn.Conv1d(in_channels, out_channels, 1)
        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros_like(x_ft)
        
        freq_size = x_ft.shape[2]
        modes = min(self.weights1.shape[2], freq_size)
        
        out_ft[:, :, :modes] = torch.einsum(
            "bix,iox->box", 
            x_ft[:, :, :modes], 
            self.weights1[:, :, :modes]
        )
        
        x_spec = torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)
        
        # 2. Linear Skip Block + Normalization
        x_lin = self.w(x_in)
        x_combined = x_spec + x_lin
        x_out = self.norm(x_combined)
        
        # --- Final Structure for Stability (Recommended in FNO) ---
        return self.norm(x_spec + x_lin)


# ========== 1. ORIGINAL DATASETS ==========
class BurgersDataset1D(Dataset):
    """
    1D Burgers' equation dataset with realistic physics.
    """
    def __init__(self, n_samples: int, domain_size: int = 256, 
                 n_fourier_modes: int = 20, time_step: int = 5, 
                 viscosity: float = 0.01, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.domain_size = domain_size
        self.samples = []
        self.nu = viscosity
        
        x = torch.linspace(0, 1, domain_size)
        self.grid = x.unsqueeze(0)  # Shape: [1, domain_size]
        
        # Pre-compute wave numbers for spectral methods
        self.k = 2 * torch.pi * torch.fft.fftfreq(domain_size, d=1.0/domain_size)
        self.k_sq = self.k**2
        self.k_sq[0] = 1.0  # Avoid division by zero
        
        # Multiple seeds for diversity
        seeds = np.random.randint(0, 1000000, n_samples)
        
        for i, sample_seed in enumerate(seeds):
            torch.manual_seed(sample_seed)
            np.random.seed(sample_seed)
            
            # 1. Generate multi-scale initial condition
            u_0 = self._multi_scale_initial_field(n_fourier_modes, i)
            
            # 2. Simulate Burgers' equation with spectral method
            u_t = self._spectral_burgers(u_0, time_step)
            
            # 3. Add physical effects
            u_t = self._add_physical_effects(u_t, u_0, i)
            
            self.samples.append({
                'grid': torch.stack([self.grid.squeeze(), u_0], dim=0),  # [2, domain_size]
                'solution': u_t.unsqueeze(0),  # [1, domain_size]
                'parameters': {'nu': viscosity, 'time': time_step, 'seed': sample_seed}
            })
    
    def _spectral_burgers(self, u_0: torch.Tensor, time_step: int) -> torch.Tensor:
        """Spectral method for 1D Burgers' equation"""
        u = u_0.clone()
        dt = 0.001
        
        # Pre-compute masks for boundary conditions
        border_mask = torch.ones_like(u)
        border_size = 3
        border_mask[:border_size] = 0
        border_mask[-border_size:] = 0
        
        for t in range(time_step):
            # Nonlinear term: u * ∂u/∂x
            u_ft = torch.fft.fft(u)
            
            # Compute gradient in Fourier space
            u_x_ft = 1j * self.k * u_ft
            u_x = torch.fft.ifft(u_x_ft).real
            
            # Burgers' equation: ∂u/∂t + u * ∂u/∂x = ν * ∂²u/∂x²
            nonlinear = -u * u_x
            
            # Time integration (semi-implicit)
            u_ft_new = (torch.fft.fft(u + dt * nonlinear) / 
                       (1 + dt * self.nu * self.k_sq))
            u = torch.fft.ifft(u_ft_new).real
            
            # Apply boundary conditions (periodic or Dirichlet)
            u = u * border_mask
            
            # Add minimal forcing
            if t % 10 == 0:
                forcing = 0.01 * math.sin(2 * math.pi * t / time_step) * torch.sin(4 * torch.pi * u)
                u += dt * forcing
                u = u * border_mask
        
        return u

    def _multi_scale_initial_field(self, n_modes: int, sample_id: int) -> torch.Tensor:
        """Generates initial field with multiple spatial scales"""
        domain_size = self.domain_size
        u = torch.zeros(domain_size)
        
        # Scale 1: Large-scale waves
        n_large_modes = n_modes // 4
        for _ in range(2 + sample_id % 3):
            amplitude = 0.5 + 0.5 * torch.randn(1).item()
            k_large = torch.randint(1, n_large_modes, (1,)).item()
            phase = 2 * torch.pi * torch.rand(1).item()
            
            u += amplitude * torch.sin(2 * torch.pi * k_large * self.grid.squeeze() + phase)
        
        # Scale 2: Medium-scale waves
        n_medium_modes = n_modes // 2
        for mode in range(n_medium_modes // 2):
            k_med = 4 + mode * 2
            amplitude = 0.2 + 0.1 * torch.randn(1).item()
            phase = 2 * torch.pi * torch.rand(1).item()
            
            u += amplitude * torch.sin(2 * torch.pi * k_med * self.grid.squeeze() + phase)
        
        # Scale 3: Small-scale turbulence (random Fourier modes)
        mask_size = n_modes
        k = torch.arange(mask_size, device=self.grid.device)
        
        # Kolmogorov -5/3 spectrum (adapted for 1D)
        k_mag = k.float()
        amplitude = torch.randn(mask_size, device=self.grid.device) * (k_mag + 1e-6)**(-2/3) / (1 + k_mag**2)
        phase = 2 * torch.pi * torch.rand(mask_size, device=self.grid.device)
        
        # Set DC component to zero
        amplitude[0] = 0
        phase[0] = 0
        
        # Create complex spectrum
        turbulence_ft = torch.zeros(domain_size, dtype=torch.cfloat, device=self.grid.device)
        turbulence_ft[:mask_size] = amplitude * torch.exp(1j * phase)
        
        # Hermitian symmetry for real signal
        turbulence_ft[-mask_size+1:] = torch.conj(turbulence_ft[1:mask_size].flip(0))
        
        turbulence = torch.fft.ifft(turbulence_ft).real
        
        if turbulence.std() > 0:
            u += 0.3 * turbulence / turbulence.std()
        
        # Add localized features (shocks)
        if sample_id % 4 == 0:
            # Shock-like discontinuity
            shock_pos = 0.3 + 0.4 * torch.rand(1).item()
            shock_width = 0.02 + 0.03 * torch.rand(1).item()
            shock_strength = 0.5 + 0.5 * torch.rand(1).item()
            
            shock = shock_strength * torch.tanh((self.grid.squeeze() - shock_pos) / shock_width)
            u += shock
        
        # Enforce boundary conditions
        u[0] = u[-1] = 0
        
        # Smooth near boundaries
        for i in range(5):
            u[i] *= i / 5
            u[-i-1] *= i / 5
        
        # Normalize
        std = torch.std(u)
        if std > 0:
            u = u / std * (0.5 + 0.5 * torch.rand(1).item())
        
        return u
    
    def _add_physical_effects(self, u_t: torch.Tensor, u_0: torch.Tensor, sample_id: int) -> torch.Tensor:
        """Add realistic physical effects to the solution"""
        
        # 1. Non-linear energy cascade effects
        u_ft = torch.fft.fft(u_t)
        
        # Apply scale-dependent damping (simulating dissipation)
        for k in range(self.domain_size):
            k_val = torch.abs(torch.tensor(k).float())
            if k_val > 10:  # Damp high wavenumbers
                damping = torch.exp(-0.1 * k_val / self.domain_size)
                u_ft[k] *= damping
        
        u_t = torch.fft.ifft(u_ft).real
        
        # 2. Inverse cascade effects
        if sample_id % 3 == 0:
            # Coherent structure formation
            large_scale = F.avg_pool1d(u_t.unsqueeze(0).unsqueeze(0), 
                                      kernel_size=5, stride=1, padding=2).squeeze()
            u_t = 0.7 * u_t + 0.3 * large_scale
        
        # 3. Memory effects
        memory_weight = 0.1 + 0.05 * torch.sin(torch.tensor(sample_id * 0.1, dtype=torch.float32))
        u_t = (1 - memory_weight) * u_t + memory_weight * u_0
        
        # 4. Add small-scale noise
        noise_amplitude = 0.01 + 0.005 * torch.randn(1).item()
        noise = noise_amplitude * torch.randn_like(u_t)
        noise[0] = noise[-1] = 0
        u_t += noise
        
        # 5. Non-linear saturation effects
        saturation = torch.tanh(u_t * 2.0) / 2.0
        u_t = 0.8 * u_t + 0.2 * saturation
        
        # 6. Scale interactions
        scales = [64, 128, 256]
        multi_scale = torch.zeros_like(u_t)
        
        for scale in scales:
            if scale < self.domain_size:
                downsampled = F.interpolate(
                    u_t.unsqueeze(0).unsqueeze(0),
                    size=scale,
                    mode='linear',
                    align_corners=False
                )
                upsampled = F.interpolate(
                    downsampled,
                    size=self.domain_size,
                    mode='linear',
                    align_corners=False
                ).squeeze()
                
                weight = 0.1 * (scale / self.domain_size)
                if scale == 64:
                    multi_scale += weight * upsampled * torch.sin(u_t * 3)
                elif scale == 128:
                    multi_scale += weight * upsampled * torch.cos(u_t * 2)
                else:
                    multi_scale += weight * upsampled
        
        u_t += multi_scale
        
        # 7. Physical constraints (energy conservation approximation)
        energy_0 = torch.mean(u_0**2)
        energy_t = torch.mean(u_t**2)
        if energy_t > 0:
            u_t = u_t * torch.sqrt(energy_0 / (energy_t + 1e-6))
        
        # 8. Final non-linear transformation
        u_t = u_t + 0.1 * torch.sin(u_t * 3) + 0.05 * torch.tanh(u_t * 2)
        
        # 9. Smooth near boundaries
        for i in range(3):
            u_t[i] = u_t[i] * (i / 3)
            u_t[-i-1] = u_t[-i-1] * (i / 3)
        
        # Normalize to reasonable range
        std = torch.std(u_t)
        if std > 0:
            u_t = u_t / std * 0.5
        
        return u_t

    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx): 
        return self.samples[idx]


# ========== 2. MULTISCALE DATASET WRAPPER ==========
class MultiScaleDataset1D(Dataset):
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
        
        # Add channel dimension if needed
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)  # [C, L]
        
        return F.interpolate(
            tensor.unsqueeze(0) if tensor.dim() == 2 else tensor,
            size=target_size,
            mode='linear',
            align_corners=False
        ).squeeze(0)
    
    def _augment(self, grid: torch.Tensor, solution: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.augment:
            return grid, solution
        
        # Flip augmentation
        if torch.rand(1) > 0.5:
            grid = torch.flip(grid, dims=[-1])
            solution = torch.flip(solution, dims=[-1])
        
        return grid, solution
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        sample = self.base_dataset[idx]
        original_grid = sample['grid']  # [2, L]
        original_solution = sample['solution']  # [1, L]
        
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
class ScaleSpecificFNO1D(nn.Module):
    def __init__(self, scale: int, in_channels: int, out_channels: int, 
                 width: int = 64, depth: int = 4):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        modes = max(4, scale // 16)
        
        self.lift = nn.Sequential(
            nn.Conv1d(in_channels, width, 1),
            nn.GELU(),
            nn.Conv1d(width, width, 1),
        )
        
        self.layers = nn.ModuleList([
            FourierBlock1D(width, width, modes) 
            for _ in range(depth)
        ])
        
        self.project = nn.Sequential(
            nn.Conv1d(width, width, 1),
            nn.GELU(),
            nn.Conv1d(width, out_channels, 1),
        )
    
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        if grid.shape[-1] != self.scale:
            grid = F.interpolate(grid, size=self.scale, mode='linear')
        
        x = self.lift(grid)
        for layer in self.layers:
            x = layer(x) + x
            x = F.gelu(x)
        output = self.project(x)
        
        return output


class MultiscaleSoftmaxRouter1D(nn.Module):
    def __init__(self, scales: List[int], in_channels: int, out_channels: int):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        self.last_router_weights = None
        
        self.out_channels = out_channels
        
        # Calculate the input dimension: 4 pooled from each scale, concatenated
        router_input_dim = self.num_scales * out_channels * 4
        
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_scales),
            nn.Softmax(dim=-1)
        )
        
        self.scale_biases = nn.Parameter(torch.zeros(self.num_scales, out_channels, 1))
        
    def forward(self, multiscale_predictions: Dict[int, torch.Tensor], 
                input_grid: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size = input_grid.shape[0]
        target_size = max(self.scales)
        self.last_router_weights = None

        available_scales = list(multiscale_predictions.keys())
        if not available_scales:
            device = input_grid.device
            return torch.zeros(batch_size, self.out_channels, 
                            target_size, device=device), {'router_weights': None}
        
        first_scale = available_scales[0]
        pred = multiscale_predictions[first_scale]
        
        # Ensure predictions have the correct shape [B, C, L]
        if pred.dim() == 2:  # [B, L]
            pred = pred.unsqueeze(1)  # [B, 1, L]
        
        out_channels = pred.shape[1]
        
        router_features = []
        compatible_predictions = {}
        
        for i, scale in enumerate(self.scales):
            if scale in multiscale_predictions:
                pred = multiscale_predictions[scale]
                
                if pred.dim() == 2:
                    pred = pred.unsqueeze(1)
                
                compatible_predictions[scale] = pred
                
                # Pool to size 4 for router input
                pooled = F.adaptive_avg_pool1d(pred, 4)
                router_features.append(pooled)
            else:
                zeros = torch.zeros(batch_size, out_channels, 4, 
                                device=input_grid.device)
                router_features.append(zeros)
        
        # Concatenate along channel dimension
        router_input = torch.cat(router_features, dim=1)  # [B, out_channels*num_scales, 4]
        
        # Flatten
        router_input = router_input.view(batch_size, -1)  # [B, out_channels*num_scales*4]
        
        router_weights = self.router(router_input)
        self.last_router_weights = router_weights.detach().clone()

        # Weighted combination
        combined = torch.zeros(batch_size, out_channels, 
                            target_size, device=input_grid.device)
        total_weight = torch.zeros(batch_size, 1, 1, device=input_grid.device)
        
        for i, scale in enumerate(self.scales):
            if scale in compatible_predictions:
                pred = compatible_predictions[scale]
                
                if pred.shape[-1] != target_size:
                    pred_upscaled = F.interpolate(pred, size=target_size, 
                                                mode='linear', align_corners=False)
                else:
                    pred_upscaled = pred
                
                pred_upscaled = pred_upscaled + self.scale_biases[i]
                weight = router_weights[:, i].view(-1, 1, 1)
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


class LagrangianSingleScaleCombiner1D(nn.Module):
    def __init__(self, scales: List[int], in_channels: int, out_channels: int, 
                 lambda_init: float = 1.0):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        
        self.log_lambdas = nn.Parameter(torch.ones(self.num_scales) * np.log(lambda_init))
        
        self.compatibility_layers = nn.ModuleDict()
        for i, scale in enumerate(scales):
            self.compatibility_layers[str(scale)] = nn.Sequential(
                nn.Conv1d(out_channels, out_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv1d(out_channels, out_channels, 3, padding=1)
            )
    
    def forward(self, multiscale_predictions: Dict[int, torch.Tensor], 
                input_grid: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        target_size = max(self.scales)
        
        compatible_predictions = {}
        for i, scale in enumerate(self.scales):
            if scale in multiscale_predictions:
                pred = multiscale_predictions[scale]
                if pred.shape[-1] != target_size:
                    pred = F.interpolate(pred, size=target_size, mode='linear')
                compatible_predictions[scale] = self.compatibility_layers[str(scale)](pred)
        
        lambdas = torch.exp(self.log_lambdas)
        lambda_weights = lambdas / torch.sum(lambdas)
        
        combined = None
        for i, scale in enumerate(self.scales):
            if scale in compatible_predictions:
                pred = compatible_predictions[scale]
                weight = lambda_weights[i]
                
                if combined is None:
                    combined = weight * pred
                else:
                    combined = combined + weight * pred
        
        if combined is None:
            device = input_grid.device
            combined = torch.zeros(1, multiscale_predictions.get(list(multiscale_predictions.keys())[0], 
                                      torch.zeros(1, 1, target_size, device=device)).shape[1], 
                                 target_size, device=device)
        
        return combined, {
            'lambda_weights': lambda_weights,
            'log_lambdas': self.log_lambdas,
            'compatible_predictions': compatible_predictions
        }


class AugmentedLagrangianCombiner1D(nn.Module):
    def __init__(self, scales: List[int], in_channels: int, out_channels: int,
                 lambda_init: float = 1.0, rho_init: float = 0.1, 
                 num_iter: int = 3, penalty_growth: float = 1.1):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        self.num_iter = num_iter
        self.penalty_growth = penalty_growth
        
        # Lagrange multipliers for each scale
        self.log_lambdas = nn.Parameter(torch.ones(self.num_scales) * np.log(lambda_init))
        
        # Penalty parameters (primal and dual)
        self.log_rho_primal = nn.Parameter(torch.tensor(np.log(rho_init)))
        self.log_rho_dual = nn.Parameter(torch.tensor(np.log(rho_init * 0.5)))
        
        # Compatibility layers for each scale
        self.compatibility_layers = nn.ModuleDict()
        for i, scale in enumerate(scales):
            self.compatibility_layers[str(scale)] = nn.Sequential(
                nn.Conv1d(out_channels, out_channels * 2, 3, padding=1),
                nn.GELU(),
                nn.Conv1d(out_channels * 2, out_channels, 3, padding=1)
            )
        
        # Consensus refiner (acts on the primal variable)
        self.primal_refiner = nn.Sequential(
            nn.Conv1d(out_channels, out_channels * 2, 3, padding=1),
            nn.GroupNorm(min(8, out_channels * 2), out_channels * 2),
            nn.GELU(),
            nn.Conv1d(out_channels * 2, out_channels, 3, padding=1)
        )
        
        # Dual variable refiner (for better dual updates)
        self.dual_refiner = nn.Sequential(
            nn.Conv1d(out_channels, out_channels * 2, 3, padding=1),
            nn.GroupNorm(min(8, out_channels * 2), out_channels * 2),
            nn.GELU(),
            nn.Conv1d(out_channels * 2, out_channels, 3, padding=1)
        )
        
        # Store iteration history for monitoring
        self.primal_history = []
        self.dual_history = []
        self.last_primal_vars = None
        self.last_dual_vars = None
        self.last_consensus = None
    
    def forward(self, multiscale_predictions: Dict[int, torch.Tensor], 
                input_grid: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        target_size = max(self.scales)
        batch_size = input_grid.shape[0]
        
        if multiscale_predictions:
            first_pred = next(iter(multiscale_predictions.values()))
            channel_dim = first_pred.shape[1]
            device = first_pred.device
        else:
            channel_dim = 1
            device = input_grid.device
        
        # Apply compatibility layers
        compatible_predictions = {}
        for scale in self.scales:
            if scale in multiscale_predictions:
                pred = multiscale_predictions[scale]
                if pred.shape[-1] != target_size:
                    pred = F.interpolate(pred, size=target_size, mode='linear')
                compatible_predictions[scale] = self.compatibility_layers[str(scale)](pred)
        
        if not compatible_predictions:
            return torch.zeros(batch_size, channel_dim, target_size, 
                             device=device), {}
        
        # Get penalty parameters
        rho_primal = torch.exp(self.log_rho_primal)
        rho_dual = torch.exp(self.log_rho_dual)
        
        # Initialize primal and dual variables
        x = {scale: pred.clone() for scale, pred in compatible_predictions.items()}
        z = sum(pred for pred in x.values()) / len(x)  # Initial consensus
        y = {scale: torch.zeros_like(pred) for scale, pred in x.items()}  # Dual variables
        
        # Get Lagrange multipliers
        lambdas = torch.exp(self.log_lambdas)
        lambda_weights = lambdas / torch.sum(lambdas)
        
        # Two-time scale updates
        scale_outputs_history = []
        primal_residuals = []
        dual_residuals = []
        
        for iter_idx in range(self.num_iter):
            # Scale index mapping for lambda weights
            scale_to_idx = {scale: i for i, scale in enumerate(self.scales)}
            
            # ----- PRIMAL UPDATE (faster timescale) -----
            for scale in compatible_predictions.keys():
                idx = scale_to_idx[scale]
                lambda_weight = lambda_weights[idx]
                
                # Augmented Lagrangian primal update with momentum
                primal_grad = (compatible_predictions[scale] - x[scale]) + \
                            rho_primal * (x[scale] - z + y[scale])
                
                # Apply lambda-weighted update
                x[scale] = x[scale] + lambda_weight * primal_grad
                
                # Add refinement
                x[scale] = self.primal_refiner(x[scale])
            
            # ----- CONSENSUS UPDATE -----
            z_new = torch.zeros_like(z)
            total_weight = 0
            
            for scale in compatible_predictions.keys():
                idx = scale_to_idx[scale]
                weight = lambda_weights[idx] + rho_primal * 0.1  # Add penalty influence
                z_new = z_new + weight * (x[scale] + y[scale])
                total_weight += weight
            
            if total_weight > 0:
                z = z_new / total_weight
            
            # ----- DUAL UPDATE (slower timescale) -----
            for scale in compatible_predictions.keys():
                # Augmented Lagrangian dual update
                dual_grad = x[scale] - z
                y[scale] = y[scale] + rho_dual * dual_grad
                
                # Refine dual variables
                y[scale] = self.dual_refiner(y[scale])
            
            # Store history for monitoring
            scale_outputs_history.append({scale: x[scale].detach().clone() 
                                         for scale in x.keys()})
            
            # Compute residuals
            if iter_idx > 0:
                primal_residual = sum(torch.norm(x[scale] - z, p=2).item() 
                                     for scale in x.keys()) / len(x)
                primal_residuals.append(primal_residual)
                
                if iter_idx > 1:
                    # Dual residual based on change in consensus
                    dual_residual = torch.norm(z - scale_outputs_history[-2]['consensus'], 
                                              p=2).item() if 'consensus' in scale_outputs_history[-2] else 0
                    dual_residuals.append(dual_residual)
            
            # Store consensus in history
            if iter_idx < len(scale_outputs_history):
                scale_outputs_history[iter_idx]['consensus'] = z.detach().clone()
        
        # Final refinement of consensus
        z_refined = self.primal_refiner(z)
        
        # Scale contributions based on final Lagrange multipliers
        scale_contributions = {}
        total_contribution = 0
        
        for i, scale in enumerate(self.scales):
            if scale in x:
                contribution = lambda_weights[i].item() * torch.norm(x[scale], p='fro').item()
                scale_contributions[scale] = contribution
                total_contribution += contribution
        
        # Normalize contributions
        if total_contribution > 0:
            scale_weights = {scale: contrib / total_contribution 
                           for scale, contrib in scale_contributions.items()}
        else:
            scale_weights = {scale: 1.0 / len(self.scales) for scale in self.scales}
        
        # Store for monitoring
        self.last_primal_vars = {scale: x[scale].detach().clone() 
                                for scale in x.keys()}
        self.last_dual_vars = {scale: y[scale].detach().clone() 
                              for scale in y.keys()}
        self.last_consensus = z_refined.detach().clone()
        
        return z_refined, {
            'primal_vars': x,
            'dual_vars': y,
            'consensus': z,
            'refined_consensus': z_refined,
            'lambda_weights': lambda_weights,
            'rho_primal': rho_primal,
            'rho_dual': rho_dual,
            'scale_weights': scale_weights,
            'scale_outputs': x,
            'primal_residuals': primal_residuals,
            'dual_residuals': dual_residuals,
            'scale_outputs_history': scale_outputs_history,
            'compatible_predictions': compatible_predictions
        }
    
    def get_lambda_statistics(self) -> Dict:
        """Get statistics about Lagrange multipliers"""
        lambdas = torch.exp(self.log_lambdas).detach().cpu().numpy()
        lambda_weights = lambdas / np.sum(lambdas)
        
        return {
            'raw_lambdas': lambdas,
            'lambda_weights': lambda_weights,
            'lambda_entropy': -np.sum(lambda_weights * np.log(lambda_weights + 1e-8)),
            'lambda_max': np.max(lambda_weights),
            'lambda_min': np.min(lambda_weights)
        }


class ADMMCombiner1D(nn.Module):
    def __init__(self, scales: List[int], in_channels: int, out_channels: int,
                 rho: float = 0.1, num_iter: int = 3, beta: float = 0.01):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        self.num_iter = num_iter
        
        self.log_rho = nn.Parameter(torch.tensor(np.log(rho)))
        
        self.compatibility_layers = nn.ModuleDict()
        for scale in scales:
            self.compatibility_layers[str(scale)] = nn.Sequential(
                nn.Conv1d(out_channels, out_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv1d(out_channels, out_channels, 3, padding=1)
            )
        
        self.consensus_refiner = nn.Sequential(
            nn.Conv1d(out_channels, out_channels * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(out_channels * 2, out_channels, 3, padding=1)
        )
        
        self.consensus_weight = nn.Parameter(torch.ones(1))
        
        self.last_scale_weights = None
        self.epoch_scale_weights = []
        
    def forward(self, multiscale_predictions: Dict[int, torch.Tensor],
                input_grid: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        target_size = max(self.scales)
        batch_size = input_grid.shape[0]
        
        if multiscale_predictions:
            first_pred = next(iter(multiscale_predictions.values()))
            channel_dim = first_pred.shape[1]
            device = first_pred.device
        else:
            channel_dim = 1
            device = input_grid.device
        
        compatible_predictions = {}
        for scale in self.scales:
            if scale in multiscale_predictions:
                pred = multiscale_predictions[scale]
                if pred.shape[-1] != target_size:
                    pred = F.interpolate(pred, size=target_size, mode='linear')
                compatible_predictions[scale] = self.compatibility_layers[str(scale)](pred)
        
        if not compatible_predictions:
            return torch.zeros(batch_size, channel_dim, target_size, 
                             device=device), {}
        
        rho = torch.exp(self.log_rho)
        
        x = {scale: pred.clone() for scale, pred in compatible_predictions.items()}
        
        weights = torch.ones(len(compatible_predictions), device=device) / len(compatible_predictions)
        z = sum(w * x[scale] for w, (scale, _) in zip(weights, compatible_predictions.items()))
        
        u = {scale: torch.zeros_like(pred) for scale, pred in compatible_predictions.items()}
        
        scale_outputs = {}
        
        for iter_idx in range(self.num_iter):
            for scale in compatible_predictions.keys():
                x[scale] = (compatible_predictions[scale] + rho * (z - u[scale])) / (1 + rho)
                scale_outputs[scale] = x[scale]
            
            if len(x) > 0:
                z_numerator = torch.zeros_like(z)
                count = 0
                
                for scale, x_i in x.items():
                    if scale in u:
                        z_numerator = z_numerator + x_i + u[scale]
                        count += 1
                
                if count > 0:
                    soft_consensus = torch.sigmoid(self.consensus_weight)
                    z_new = (z_numerator / count)
                    z = soft_consensus * z_new + (1 - soft_consensus) * z
            
            for scale in compatible_predictions.keys():
                if scale in x and scale in u:
                    u[scale] = u[scale] + (x[scale] - z)
        
        z_refined = self.consensus_refiner(z)
        
        scale_contributions = {}
        total_contribution = 0
        
        for scale in compatible_predictions.keys():
            if scale in x and scale in u:
                contribution = torch.norm(x[scale] + u[scale], p='fro').item()
                scale_contributions[scale] = contribution
                total_contribution += contribution
        
        if total_contribution > 0:
            self.last_scale_weights = {
                scale: contribution / total_contribution
                for scale, contribution in scale_contributions.items()
            }
        else:
            self.last_scale_weights = {scale: 1.0/len(self.scales) for scale in self.scales}
        
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
            'scale_weights': self.last_scale_weights
        }
    
    def reset_epoch_weights(self):
        self.epoch_scale_weights = []
    
    def add_epoch_weights(self, weights):
        if weights is not None:
            self.epoch_scale_weights.append(weights)
    
    def get_average_epoch_weights(self):
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


class SingleScaleFNO1D(nn.Module):
    def __init__(self, scale: int, in_channels: int, out_channels: int, 
                 width: int = 32, depth: int = 4):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        modes = max(4, scale // 16)
        
        self.lift = nn.Sequential(
            nn.Conv1d(in_channels, width, 1),
            nn.GELU(),
            nn.Conv1d(width, width, 1)
        )
        
        self.fno_layers = nn.ModuleList([
            FourierBlock1D(width, width, modes) 
            for _ in range(depth)
        ])
        
        self.project = nn.Sequential(
            nn.Conv1d(width, width, 1),
            nn.GELU(),
            nn.Conv1d(width, out_channels, 1)
        )
        
        self.refine = nn.Sequential(
            nn.Conv1d(out_channels, out_channels * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(out_channels * 2, out_channels, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.scale:
            x = F.interpolate(x, size=self.scale, mode='linear')
        
        x = self.lift(x)
        for layer in self.fno_layers:
            x = layer(x) + x
            x = F.gelu(x)
        x = self.project(x)
        x = self.refine(x)
        
        return x


class SingleScaleBaseline1D(nn.Module):
    def __init__(self, scale: int, in_channels: int, out_channels: int):
        super().__init__()
        self.scale = scale
        self.model = SingleScaleFNO1D(scale, in_channels, out_channels)
        self.out_channels = out_channels
        self.is_baseline = True
    
    def forward(self, multiscale_inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        available_scales = [int(k.split('_')[1]) for k in multiscale_inputs.keys()]
        if not available_scales:
            device = next(self.parameters()).device
            target_size = self.scale
            return torch.zeros(1, self.out_channels, target_size, 
                             device=device), {}
        
        largest_scale = max(available_scales)
        input_key = f'grid_{largest_scale}'
        grid = multiscale_inputs[input_key]
        
        output = self.model(grid)
        
        metadata = {
            'scale_used': largest_scale,
            'output': output,
            'scale_predictions': {largest_scale: output}
        }
        
        return output, metadata


# ========== 4. MULTISCALE SOLVER ==========
class StableMultiscalePDEsolver1D(nn.Module):
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
        
        if combination_method == 'single_scale_baseline':
            self.is_baseline = True
            self.baseline_scale = max(scales)
            self.model = SingleScaleBaseline1D(self.baseline_scale, in_channels, out_channels)
            return
        
        self.is_baseline = False
        
        self.method_config = method_config or self._get_stable_config()
        
        self.scale_models = nn.ModuleDict()
        for scale in self.scales:
            model = ScaleSpecificFNO1D(scale, in_channels, out_channels)
            self._initialize_model(model)
            self.scale_models[str(scale)] = model
        
        self.combiner = self._create_stable_combiner()
        
        self.fine_tune = nn.Sequential(
            nn.Conv1d(out_channels, out_channels * 2, 3, padding=1),
            nn.GroupNorm(min(8, out_channels * 2), out_channels * 2),
            nn.GELU(),
            nn.Conv1d(out_channels * 2, out_channels, 3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
        )
        
        self.skip_weight = nn.Parameter(torch.tensor(0.5))
        self.final_proj = nn.Conv1d(out_channels, out_channels, 1)
    
    def _get_stable_config(self) -> Dict:
        configs = {
            'softmax': {
                'router_sparsity_weight': 0.001,
                'scale_consistency_weight': 0.01,
                'reconstruction_weight': 1.0,
            },
            'lagrangian_single': {
                'lambda_init': 0.1,
                'lambda_reg_weight': 0.001,
                'scale_consistency_weight': 0.01,
                'reconstruction_weight': 1.0,
            },
            'lagrangian_augmented': {  # ADDED
                'lambda_init': 0.1,
                'rho_init': 0.1,
                'num_iter': 3,
                'penalty_growth': 1.1,
                'lambda_reg_weight': 0.001,
                'primal_dual_balance_weight': 0.01,
                'scale_consistency_weight': 0.01,
                'reconstruction_weight': 1.0,
            },
            'admm': {
                'rho': 0.1,
                'num_iter': 3,
                'beta': 0.01,
                'admm_consensus_weight': 0.001,
                'scale_consistency_weight': 0.01,
                'reconstruction_weight': 1.0,
            }
        }
        return configs.get(self.combination_method, {})
    
    def _initialize_model(self, model: nn.Module):
        for name, param in model.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param, gain=0.5)
                else:
                    nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def _create_stable_combiner(self):
        if self.combination_method == 'softmax':
            combiner = MultiscaleSoftmaxRouter1D(
                self.scales, self.in_channels, self.out_channels
            )
        elif self.combination_method == 'lagrangian_single':
            combiner = LagrangianSingleScaleCombiner1D(
                self.scales, self.in_channels, self.out_channels,
                lambda_init=self.method_config.get('lambda_init', 0.1)
            )
        elif self.combination_method == 'lagrangian_augmented':  # ADDED
            combiner = AugmentedLagrangianCombiner1D(
                self.scales, self.in_channels, self.out_channels,
                lambda_init=self.method_config.get('lambda_init', 0.1),
                rho_init=self.method_config.get('rho_init', 0.1),
                num_iter=self.method_config.get('num_iter', 3),
                penalty_growth=self.method_config.get('penalty_growth', 1.1)
            )
        elif self.combination_method == 'admm':
            combiner = ADMMCombiner1D(
                self.scales, self.in_channels, self.out_channels,
                rho=self.method_config.get('rho', 0.1),
                num_iter=self.method_config.get('num_iter', 3),
                beta=self.method_config.get('beta', 0.01)
            )
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
        
        for name, param in combiner.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param, gain=0.5)
                else:
                    nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'log_' in name or 'lambda' in name:
                nn.init.constant_(param, np.log(0.1))
        
        return combiner
    
    def forward(self, multiscale_inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        if self.is_baseline:
            return self.model(multiscale_inputs)
        
        scale_predictions = {}
        
        for scale in self.scales:
            input_key = f'grid_{scale}'
            if input_key in multiscale_inputs:
                grid = multiscale_inputs[input_key]
                pred = self.scale_models[str(scale)](grid)
                scale_predictions[scale] = pred
        
        if not scale_predictions:
            target_size = max(self.scales)
            device = next(self.parameters()).device
            return torch.zeros(1, self.out_channels, target_size, 
                             device=device), {'scale_predictions': {}}
        
        available_scales_in_input = [int(k.split('_')[1]) for k in multiscale_inputs.keys()]
        largest_input_scale = max(available_scales_in_input)
        largest_grid_key = f'grid_{largest_input_scale}'
        largest_grid = multiscale_inputs[largest_grid_key]
        
        combined, meta = self.combiner(scale_predictions, largest_grid)
        meta['scale_predictions'] = scale_predictions
        
        refined = self.fine_tune(combined)
        output = self.skip_weight * refined + (1 - self.skip_weight) * combined
        output = self.final_proj(output) + output
        
        meta['final_output'] = output
        
        return output, meta
    
    def compute_loss(self, predictions: torch.Tensor, 
                     multiscale_targets: Dict[str, torch.Tensor],
                     metadata: Dict) -> Tuple[torch.Tensor, Dict]:
        if self.is_baseline:
            # Simple baseline loss
            losses = {}
            largest_scale = max(self.scales)
            target_key = f'solution_{largest_scale}'
            
            if target_key in multiscale_targets:
                target = multiscale_targets[target_key]
                if predictions.shape[-1] != target.shape[-1]:
                    predictions_resized = F.interpolate(
                        predictions, 
                        size=target.shape[-1], 
                        mode='linear',
                        align_corners=False
                    )
                    losses['reconstruction'] = F.mse_loss(predictions_resized, target)
                else:
                    losses['reconstruction'] = F.mse_loss(predictions, target)
            else:
                losses['reconstruction'] = torch.tensor(0.0, device=predictions.device)
            
            total_loss = losses['reconstruction']
            losses['total'] = total_loss
            
            return total_loss, losses
        
        losses = {}
        
        # 1. Multi-scale reconstruction loss
        total_recon_loss = 0.0
        num_targets = 0
        
        for scale in self.scales:
            target_key = f'solution_{scale}'
            if target_key in multiscale_targets:
                target = multiscale_targets[target_key]
                
                if predictions.shape[-1] != target.shape[-1]:
                    pred_resized = F.interpolate(
                        predictions, 
                        size=target.shape[-1], 
                        mode='linear',
                        align_corners=False
                    )
                    loss = F.mse_loss(pred_resized, target)
                else:
                    loss = F.mse_loss(predictions, target)
                
                scale_weight = (scale / max(self.scales)) ** 0.5
                total_recon_loss += loss * scale_weight
                num_targets += 1
        
        if num_targets > 0:
            losses['reconstruction'] = total_recon_loss / num_targets
        else:
            losses['reconstruction'] = torch.tensor(0.0, device=predictions.device)
        
        # 2. Individual scale losses
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
        
        # 3. Weight entropy regularization
        if self.combination_method == 'softmax' and 'router_weights' in metadata:
            router_weights = metadata['router_weights']
            num_scales = router_weights.shape[1]
            
            max_entropy = torch.log(torch.tensor(float(num_scales), device=router_weights.device))
            entropy = -torch.sum(router_weights * torch.log(router_weights + 1e-8), dim=1)
            
            entropy_loss = torch.mean((entropy - max_entropy) ** 2)
            losses['weight_entropy'] = entropy_loss * 0.01
        
        # 4. Lagrangian regularization (for Lagrangian methods)
        if self.combination_method in ['lagrangian_single', 'lagrangian_augmented']:
            if 'lambda_weights' in metadata:
                lambda_weights = metadata['lambda_weights']
                
                # Encourage balanced weights
                lambda_entropy = -torch.sum(lambda_weights * torch.log(lambda_weights + 1e-8))
                max_entropy = torch.log(torch.tensor(float(len(lambda_weights)), 
                                                    device=lambda_weights.device))
                
                # Penalize extreme lambda values
                lambda_reg = torch.mean((lambda_weights - 1.0/len(lambda_weights)) ** 2)
                
                losses['lambda_regularization'] = lambda_reg * 0.001
                losses['lambda_entropy'] = (max_entropy - lambda_entropy) * 0.0005
            
            # Additional loss for augmented Lagrangian
            if self.combination_method == 'lagrangian_augmented':
                # Primal-dual balance loss
                if 'primal_residuals' in metadata and 'dual_residuals' in metadata:
                    primal_res = torch.tensor(metadata['primal_residuals'][-1] 
                                            if metadata['primal_residuals'] else 0.0,
                                            device=predictions.device)
                    dual_res = torch.tensor(metadata['dual_residuals'][-1] 
                                           if metadata['dual_residuals'] else 0.0,
                                           device=predictions.device)
                    
                    # Encourage balanced primal-dual residuals
                    balance_loss = torch.abs(primal_res - dual_res) / (primal_res + dual_res + 1e-8)
                    losses['primal_dual_balance'] = balance_loss * 0.01
                
                # Penalty parameter regularization
                if 'rho_primal' in metadata and 'rho_dual' in metadata:
                    rho_primal = metadata['rho_primal']
                    rho_dual = metadata['rho_dual']
                    
                    # Encourage reasonable ratio (typically rho_primal > rho_dual)
                    ratio_loss = F.relu(rho_dual - rho_primal * 0.8)  # rho_dual should be <= 0.8*rho_primal
                    losses['penalty_ratio'] = ratio_loss * 0.001
        
        # 5. ADMM-specific losses
        if self.combination_method == 'admm' and 'consensus' in metadata:
            consensus = metadata['consensus']
            scale_outputs = metadata.get('scale_outputs', {})
            
            if scale_outputs:
                consensus_loss = 0.0
                for scale, output in scale_outputs.items():
                    consensus_loss += F.mse_loss(output, consensus)
                
                losses['admm_consensus'] = consensus_loss * 0.001
        
        # ========== ADDED: PHYSICS LOSS FOR 1D ==========
        # 6. Physics-informed loss for 1D Burgers' equation
        physics_loss_components = self._compute_physics_loss_1d(predictions, metadata)
        losses.update(physics_loss_components)
        
        # 7. Calculate total loss
        total_loss = losses.get('reconstruction', 0.0)
        
        for loss_name, loss_value in losses.items():
            if loss_name != 'reconstruction':
                # Scale physics losses appropriately
                if loss_name.startswith('physics_'):
                    total_loss = total_loss + loss_value * 0.1  # Reduced weight for physics
                else:
                    total_loss = total_loss + loss_value
        
        losses['total'] = total_loss
        
        return total_loss, losses
    
    def _compute_physics_loss_1d(self, predictions: torch.Tensor, metadata: Dict) -> Dict:
        """Compute physics-constrained loss for 1D Burgers' equation"""
        physics_losses = {}
        
        # Get batch size and spatial dimensions
        batch_size = predictions.shape[0]
        L = predictions.shape[2]  # Length of 1D domain
        
        # Skip physics loss if we don't have predictions or metadata
        if batch_size == 0 or 'scale_predictions' not in metadata:
            return physics_losses
        
        # Get predictions for the largest scale (highest resolution)
        largest_scale = max(self.scales)
        if largest_scale in metadata['scale_predictions']:
            pred_field = metadata['scale_predictions'][largest_scale]
            
            # Ensure we have valid field
            if pred_field.shape[-1] >= 4:
                # Compute spatial gradients for physics constraints
                dx = 1.0 / (L - 1)  # Assuming domain [0,1]
                
                # Compute gradients using finite differences (central difference)
                grad = (pred_field[:, :, 2:] - pred_field[:, :, :-2]) / (2 * dx)
                
                # Compute second derivatives for diffusion term
                grad_xx = (pred_field[:, :, 2:] - 2 * pred_field[:, :, 1:-1] + pred_field[:, :, :-2]) / (dx**2)
                
                # Burgers' equation: ∂u/∂t + u * ∂u/∂x = ν * ∂²u/∂x²
                # For steady state or assuming small changes: u * ∂u/∂x ≈ ν * ∂²u/∂x²
                
                # 4a. PDE residual loss for Burgers' equation
                viscosity = 0.01  # Same as dataset
                convective_term = pred_field[:, :, 1:-1] * grad
                diffusion_term = viscosity * grad_xx
                
                # Residual: ∂u/∂t + u * ∂u/∂x - ν * ∂²u/∂x² ≈ 0
                # For steady state approximation: u * ∂u/∂x - ν * ∂²u/∂x² ≈ 0
                pde_residual = torch.abs(convective_term - diffusion_term)
                physics_losses['physics_pde'] = torch.mean(pde_residual**2) * 0.01
                
                # 4b. Energy conservation loss
                # For Burgers' equation without forcing, energy should decay
                # We'll encourage smooth energy decay across scales
                if len(metadata['scale_predictions']) > 1:
                    energy_values = []
                    for scale, pred in metadata['scale_predictions'].items():
                        # Compute kinetic energy density (mean of u²)
                        energy = torch.mean(pred**2, dim=[1, 2])
                        energy_values.append(energy)
                    
                    if len(energy_values) >= 2:
                        # Encourage monotonic energy decay with scale (more dissipation at finer scales)
                        energy_decay_loss = 0.0
                        for i in range(len(energy_values) - 1):
                            # Higher scale (finer resolution) should have lower energy due to dissipation
                            energy_diff = F.relu(energy_values[i] - energy_values[i+1] + 0.01)  # Allow small tolerance
                            energy_decay_loss += torch.mean(energy_diff)
                        
                        if len(energy_values) > 1:
                            physics_losses['physics_energy_decay'] = energy_decay_loss * 0.001
                
                # 4c. Boundary condition loss (Dirichlet boundaries)
                boundary_loss = torch.mean(pred_field[:, :, 0]**2) + \
                              torch.mean(pred_field[:, :, -1]**2)
                physics_losses['physics_boundary'] = boundary_loss * 0.01
                
                # 4d. Smoothness/regularity loss
                # Total variation regularization for smooth velocity fields
                tv_loss = torch.mean(torch.abs(grad))
                physics_losses['physics_smoothness'] = tv_loss * 0.001
                
                # 4e. Shock capturing loss (encourage proper shock formation)
                # Burgers' equation can develop shocks - we want gradients to be finite
                shock_loss = torch.mean(torch.abs(grad_xx) / (torch.abs(grad) + 1e-6))
                physics_losses['physics_shock'] = shock_loss * 0.0001
        
        return physics_losses


# ========== 5. TRAINER ==========
class StableMultiscaleTrainer1D:
    def __init__(self, model, 
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str,
                 lr: Optional[float] = None,
                 grad_clip: float = 1.0,
                 patience: int = 30,
                 log_weights_every: int = 5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.patience = patience
        self.log_weights_every = log_weights_every
        
        self.model_scales = model.scales
        self.method = model.combination_method
        
        if lr is None:
            lr_map = {
                'single_scale_baseline': 3e-3,
                'softmax': 5e-5,
                'lagrangian_single': 6e-5,
                'lagrangian_augmented': 5e-5,  # ADDED
                'admm': 1e-5,
            }
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
        self.weight_history = []
        self.epoch_weights_summary = []
        self.admm_weights_per_epoch = []

    def _filter_multiscale_data(self, batch_data, key_prefix):
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
        weights_info = {}
        
        if 'router_weights' in metadata and metadata['router_weights'] is not None:
            router_weights = metadata['router_weights']
            if router_weights.dim() == 2:
                avg_weights = router_weights.mean(dim=0)
                for i, scale in enumerate(self.model_scales):
                    if i < avg_weights.shape[0]:
                        weights_info[f'scale_{scale}'] = avg_weights[i].item()
        
        elif 'lambda_weights' in metadata:
            lambda_weights = metadata['lambda_weights']
            for i, scale in enumerate(self.model_scales):
                if i < lambda_weights.shape[0]:
                    weights_info[f'scale_{scale}'] = lambda_weights[i].item()
        
        elif 'scale_weights' in metadata:
            scale_weights = metadata['scale_weights']
            for scale, weight in scale_weights.items():
                weights_info[f'scale_{scale}'] = weight
        
        elif self.model.is_baseline and 'scale_used' in metadata:
            scale_used = metadata['scale_used']
            weights_info[f'scale_{scale_used}'] = 1.0
        
        return weights_info
    
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        epoch_weights = defaultdict(list)
        
        if self.method == 'admm' and hasattr(self.model.combiner, 'reset_epoch_weights'):
            self.model.combiner.reset_epoch_weights()
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                multiscale_inputs = self._filter_multiscale_data(
                    batch['multiscale_grids'], 'grid'
                )
                multiscale_targets = self._filter_multiscale_data(
                    batch['multiscale_solutions'], 'solution'
                )
                
                if not multiscale_inputs or not multiscale_targets:
                    continue
                
                multiscale_inputs = {k: v.to(self.device) for k, v in multiscale_inputs.items()}
                multiscale_targets = {k: v.to(self.device) for k, v in multiscale_targets.items()}
                
                predictions, metadata = self.model(multiscale_inputs)
                
                if self.method == 'admm' and hasattr(self.model.combiner, 'add_epoch_weights'):
                    if hasattr(self.model.combiner, 'last_scale_weights') and self.model.combiner.last_scale_weights:
                        self.model.combiner.add_epoch_weights(self.model.combiner.last_scale_weights)
                
                weights_info = self._extract_weights_from_metadata(metadata)
                for scale_key, weight_value in weights_info.items():
                    epoch_weights[scale_key].append(weight_value)
                
                loss, loss_components = self.model.compute_loss(
                    predictions, multiscale_targets, metadata
                )
                
                if not torch.isfinite(loss):
                    print(f"Warning: Non-finite loss {loss.item()}, skipping batch")
                    continue
                
                self.optimizer.zero_grad()
                loss.backward()
                
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
        
        self.scheduler.step()
        
        avg_epoch_weights = {}
        for scale_key, weight_list in epoch_weights.items():
            if weight_list:
                avg_epoch_weights[scale_key] = np.mean(weight_list)
        
        if avg_epoch_weights:
            self.epoch_weights_summary.append({
                'epoch': epoch,
                'weights': avg_epoch_weights,
                'total_loss': total_loss / max(batch_count, 1) if batch_count > 0 else float('inf')
            })
        
        if self.method == 'admm' and hasattr(self.model.combiner, 'get_average_epoch_weights'):
            admm_epoch_weights = self.model.combiner.get_average_epoch_weights()
            self.admm_weights_per_epoch.append(admm_epoch_weights)
            
            if epoch % self.log_weights_every == 0:
                print(f"\n  ADMM Epoch {epoch} Scale Weights:")
                total_weight = 0
                for scale, weight in admm_epoch_weights.items():
                    total_weight += weight
                    print(f"    Scale {scale}: {weight:.4f}")
                print(f"    Total: {total_weight:.4f}")
                
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
            multiscale_inputs = self._filter_multiscale_data(
                batch['multiscale_grids'], 'grid'
            )
            multiscale_targets = self._filter_multiscale_data(
                batch['multiscale_solutions'], 'solution'
            )
            
            if not multiscale_inputs or not multiscale_targets:
                continue
            
            multiscale_inputs = {k: v.to(self.device) for k, v in multiscale_inputs.items()}
            multiscale_targets = {k: v.to(self.device) for k, v in multiscale_targets.items()}
            
            predictions, metadata = self.model(multiscale_inputs)
            loss, _ = self.model.compute_loss(predictions, multiscale_targets, metadata)
            
            if torch.isfinite(loss):
                total_loss += loss.item()
                batch_count += 1
        
        avg_val_loss = total_loss / max(batch_count, 1) if batch_count > 0 else float('inf')
        
        if self.val_loss_ema is None:
            self.val_loss_ema = avg_val_loss
        else:
            self.val_loss_ema = self.ema_alpha * avg_val_loss + (1 - self.ema_alpha) * self.val_loss_ema
        
        return avg_val_loss, self.val_loss_ema
    
    def train(self, num_epochs: int = 100, verbose: bool = True):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            
            val_loss, val_loss_ema = self.validate()
            
            if train_loss == float('inf') or val_loss == float('inf'):
                print(f"Warning: Invalid loss at epoch {epoch+1}, skipping epoch")
                continue
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_loss_ema'].append(val_loss_ema)
            
            if val_loss_ema < best_val_loss:
                best_val_loss = val_loss_ema
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'weight_history': self.epoch_weights_summary,
                    'admm_weights_per_epoch': self.admm_weights_per_epoch if self.method == 'admm' else None,
                }, f'best_model_{self.model.combination_method}_1d.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and (epoch % self.log_weights_every == 0 or epoch < 5 or epoch == num_epochs - 1):
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"\nEpoch {epoch+1}/{num_epochs}:")
                print(f"  Train Loss = {train_loss:.4e}")
                print(f"  Val Loss = {val_loss:.4e} (EMA: {val_loss_ema:.4e})")
                print(f"  LR = {current_lr:.2e}, Best Val = {best_val_loss:.4e}")
                print(f"  Patience = {patience_counter}/{self.patience}")
                
                if self.method != 'admm' and self.epoch_weights_summary:
                    latest_weights = self.epoch_weights_summary[-1]['weights']
                    print("  Scale Weights:")
                    for key, value in latest_weights.items():
                        if key.startswith('scale_'):
                            scale_num = int(key.replace('scale_', ''))
                            print(f"    Scale {scale_num}: {value:.4f}")
        
        return self.history


def stable_multiscale_comparison_1d():
    print(f"Using device: {device}")
    
    # Create base dataset
    train_dataset = BurgersDataset1D(
        n_samples=5000, domain_size=256, seed=42
    )
    val_dataset = BurgersDataset1D(
        n_samples=500, domain_size=256, seed=43
    )
    test_dataset = BurgersDataset1D(
        n_samples=1000, domain_size=256, seed=44
    )
    
    # FIXED: Use consistent scales for training and evaluation
    # Train on both low and high resolutions, evaluate on same scales
    train_scales = [32, 64]
    val_scales = [64]  # Validation at highest training scale
    eval_scales = [128, 256]  # Evaluate at same scale as training
    
    # Create multiscale datasets
    train_multiscale = MultiScaleDataset1D(
        train_dataset, train_scales, mode='train', augment=True
    )
    val_multiscale = MultiScaleDataset1D(
        val_dataset, val_scales, mode='val', augment=False
    )
    test_multiscale = MultiScaleDataset1D(
        test_dataset, eval_scales, mode='test', augment=False
    )
    
    # Data loaders
    train_loader = DataLoader(train_multiscale, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_multiscale, batch_size=8, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_multiscale, batch_size=8, shuffle=False, num_workers=2)
    
    # Get sample to determine dimensions
    sample = train_multiscale[0]
    in_channels = sample['original_grid'].shape[0]
    out_channels = sample['original_solution'].shape[0]
    
    print(f"\nDataset Info:")
    print(f"  Input channels: {in_channels}, Output channels: {out_channels}")
    print(f"  Training scales: {train_scales}")
    print(f"  Evaluation scales: {eval_scales}")
    print(f"  Train samples: {len(train_multiscale)}")
    print(f"  Val samples: {len(val_multiscale)}")
    print(f"  Test samples: {len(test_multiscale)}")

    # Test methods
    combination_methods = [
        #'single_scale_baseline',
        #'softmax', 
        'lagrangian_single',
        'lagrangian_augmented',
        'admm'
    ]
    
    results = {}
    
    for method in combination_methods:
        print(f"\n{'='*60}")
        print(f"Training {method.upper().replace('_', ' ')} (1D)")
        print(f"{'='*60}")
        
        # FIXED: Create model with proper config to reduce regularization
        method_config = {
            'reconstruction_weight': 1.0,
            'scale_consistency_weight': 0.001,  # Reduced
            'lambda_reg_weight': 0.0001,  # Much reduced
            'physics_weight': 0.0001,  # Minimal physics loss
        }
        
        # For baseline, use the largest training scale
        if method == 'single_scale_baseline':
            model = StableMultiscalePDEsolver1D(
                scales=[max(train_scales)],  # Only largest scale for baseline
                in_channels=in_channels,
                out_channels=out_channels,
                model_type='fno',
                combination_method=method,
                method_config=method_config
            ).to(device)
        else:
            model = StableMultiscalePDEsolver1D(
                scales=train_scales,
                in_channels=in_channels,
                out_channels=out_channels,
                model_type='fno',
                combination_method=method,
                method_config=method_config
            ).to(device)
        
        print(f"  Model type: StableMultiscalePDEsolver1D")
        if hasattr(model, 'scale_models'):
            print(f"  Number of scale models: {len(model.scale_models)}")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # FIXED: Use appropriate learning rates
        lr_map = {
            'single_scale_baseline': 5e-4,
            'softmax': 5e-4,
            'lagrangian_single': 5e-4,
            'lagrangian_augmented': 5e-4,  # Lower LR for augmented Lagrangian
            'admm': 1e-4,  # Lower LR for ADMM
        }
        lr = lr_map.get(method, 1e-3)
        
        # Create trainer with modified settings
        trainer = StableMultiscaleTrainer1D(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=lr,
            grad_clip=0.5,  # Reduced gradient clipping
            patience=20,
            log_weights_every=10
        )
        
        # Train
        try:
            history = trainer.train(num_epochs=50, verbose=True)
            
            # ========== COMPUTE TRAIN MSE ==========
            print("\nComputing Train MSE...")
            model.eval()
            train_mse_values = []
            train_samples_processed = 0
            
            with torch.no_grad():
                for batch in train_loader:
                    multiscale_inputs = {}
                    for scale in train_scales:
                        key = f'grid_{scale}'
                        if key in batch['multiscale_grids']:
                            multiscale_inputs[key] = batch['multiscale_grids'][key].to(device)
                    
                    multiscale_targets = {}
                    for scale in train_scales:
                        target_key = f'solution_{scale}'
                        if target_key in batch['multiscale_solutions']:
                            multiscale_targets[target_key] = batch['multiscale_solutions'][target_key].to(device)
                    
                    if not multiscale_inputs:
                        continue
                    
                    predictions, metadata = model(multiscale_inputs)
                    
                    # Compute loss for the largest training scale
                    largest_train_scale = max(train_scales)
                    target_key = f'solution_{largest_train_scale}'
                    if target_key in multiscale_targets:
                        target = multiscale_targets[target_key]
                        
                        if predictions.shape[-1] != target.shape[-1]:
                            pred_resized = F.interpolate(
                                predictions, 
                                size=target.shape[-1], 
                                mode='linear',
                                align_corners=False
                            )
                            mse = F.mse_loss(pred_resized, target).item()
                        else:
                            mse = F.mse_loss(predictions, target).item()
                        
                        train_mse_values.append(mse)
                        train_samples_processed += target.shape[0]
            
            avg_train_mse = np.mean(train_mse_values) if train_mse_values else float('inf')
            
            # ========== COMPUTE TEST MSE ==========
            print("Computing Test MSE...")
            test_mse_values = []
            test_samples_processed = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    multiscale_inputs = {}
                    for scale in train_scales:
                        key = f'grid_{scale}'
                        if key in batch['multiscale_grids']:
                            multiscale_inputs[key] = batch['multiscale_grids'][key].to(device)
                    
                    multiscale_targets = {}
                    for scale in eval_scales:
                        target_key = f'solution_{scale}'
                        if target_key in batch['multiscale_solutions']:
                            multiscale_targets[target_key] = batch['multiscale_solutions'][target_key].to(device)
                    
                    if not multiscale_inputs:
                        continue
                    
                    predictions, metadata = model(multiscale_inputs)
                    
                    # Compute MSE for each evaluation scale
                    for scale in eval_scales:
                        target_key = f'solution_{scale}'
                        if target_key in multiscale_targets:
                            target = multiscale_targets[target_key]
                            
                            if predictions.shape[-1] != target.shape[-1]:
                                pred_resized = F.interpolate(
                                    predictions, 
                                    size=target.shape[-1], 
                                    mode='linear',
                                    align_corners=False
                                )
                                mse = F.mse_loss(pred_resized, target).item()
                            else:
                                mse = F.mse_loss(predictions, target).item()
                            
                            test_mse_values.append(mse)
                            test_samples_processed += target.shape[0]
            
            avg_test_mse = np.mean(test_mse_values) if test_mse_values else float('inf')
            
            # Store results
            results[method] = {
                'history': history,
                'weight_history': trainer.epoch_weights_summary,
                'admm_weights_per_epoch': trainer.admm_weights_per_epoch if method == 'admm' else [],
                'final_val_loss': history['val_loss'][-1] if history['val_loss'] else float('inf'),
                'train_mse': avg_train_mse,
                'test_mse': avg_test_mse,
                'test_losses': {scale: avg_test_mse for scale in eval_scales},  # Same for all eval scales
                'best_val_loss': min(history['val_loss']) if history['val_loss'] else float('inf'),
                'num_parameters': sum(p.numel() for p in model.parameters()),
                'train_samples': train_samples_processed,
                'test_samples': test_samples_processed,
                'success': True
            }
            
            # ========== PRINT MSE RESULTS ==========
            print(f"\n{'='*60}")
            print(f"FINAL MSE RESULTS for {method.upper().replace('_', ' ')}:")
            print(f"{'='*60}")
            print(f"Train MSE:     {avg_train_mse:.6e} (over {train_samples_processed} samples)")
            print(f"Test MSE:      {avg_test_mse:.6e} (over {test_samples_processed} samples)")
            print(f"Best Val Loss: {results[method]['best_val_loss']:.6e}")
            
            # Print final scale weights if available
            if method != 'single_scale_baseline' and trainer.epoch_weights_summary:
                latest_weights = trainer.epoch_weights_summary[-1]['weights']
                print("\nFinal Scale Weights:")
                for key, value in latest_weights.items():
                    if key.startswith('scale_'):
                        scale_num = int(key.replace('scale_', ''))
                        print(f"  Scale {scale_num}: {value:.4f}")
            
        except Exception as e:
            print(f"  Training failed for {method}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[method] = {
                'success': False,
                'error': str(e)
            }
    
    return results, train_scales, eval_scales


# Also need to fix the compute_loss method to reduce regularization
def compute_loss_fixed(self, predictions: torch.Tensor, 
                       multiscale_targets: Dict[str, torch.Tensor],
                       metadata: Dict) -> Tuple[torch.Tensor, Dict]:
    """Fixed compute_loss with reduced regularization"""
    if self.is_baseline:
        # Simple baseline loss
        losses = {}
        largest_scale = max(self.scales)
        target_key = f'solution_{largest_scale}'
        
        if target_key in multiscale_targets:
            target = multiscale_targets[target_key]
            if predictions.shape[-1] != target.shape[-1]:
                predictions_resized = F.interpolate(
                    predictions, 
                    size=target.shape[-1], 
                    mode='linear',
                    align_corners=False
                )
                losses['reconstruction'] = F.mse_loss(predictions_resized, target)
            else:
                losses['reconstruction'] = F.mse_loss(predictions, target)
        else:
            losses['reconstruction'] = torch.tensor(0.0, device=predictions.device)
        
        total_loss = losses['reconstruction']
        losses['total'] = total_loss
        
        return total_loss, losses
    
    losses = {}
    
    # 1. Main reconstruction loss (most important)
    total_recon_loss = 0.0
    num_targets = 0
    
    for scale in self.scales:
        target_key = f'solution_{scale}'
        if target_key in multiscale_targets:
            target = multiscale_targets[target_key]
            
            if predictions.shape[-1] != target.shape[-1]:
                pred_resized = F.interpolate(
                    predictions, 
                    size=target.shape[-1], 
                    mode='linear',
                    align_corners=False
                )
                loss = F.mse_loss(pred_resized, target)
            else:
                loss = F.mse_loss(predictions, target)
            
            # Weight by scale (higher scales more important)
            scale_weight = (scale / max(self.scales))
            total_recon_loss += loss * scale_weight
            num_targets += 1
    
    if num_targets > 0:
        losses['reconstruction'] = total_recon_loss / num_targets
    else:
        losses['reconstruction'] = torch.tensor(0.0, device=predictions.device)
    
    # 2. Individual scale losses (help gradient flow)
    scale_predictions = metadata.get('scale_predictions', {})
    if scale_predictions:
        scale_losses = []
        for scale, pred in scale_predictions.items():
            target_key = f'solution_{scale}'
            if target_key in multiscale_targets:
                target = multiscale_targets[target_key]
                scale_loss = F.mse_loss(pred, target)
                scale_losses.append(scale_loss)
        
        if scale_losses:
            # Much smaller weight for scale individual loss
            losses['scale_individual'] = torch.mean(torch.stack(scale_losses)) * 0.01
    
    # 3. Minimal regularization for softmax
    if self.combination_method == 'softmax' and 'router_weights' in metadata:
        router_weights = metadata['router_weights']
        num_scales = router_weights.shape[1]
        
        # Very small entropy regularization
        max_entropy = torch.log(torch.tensor(float(num_scales), device=router_weights.device))
        entropy = -torch.sum(router_weights * torch.log(router_weights + 1e-8), dim=1)
        entropy_loss = torch.mean((entropy - max_entropy) ** 2)
        losses['weight_entropy'] = entropy_loss * 0.0001  # Very small
    
    # 4. Very small Lagrangian regularization
    if self.combination_method in ['lagrangian_single', 'lagrangian_augmented']:
        if 'lambda_weights' in metadata:
            lambda_weights = metadata['lambda_weights']
            # Tiny regularization to prevent collapse
            lambda_reg = torch.mean((lambda_weights - 1.0/len(lambda_weights)) ** 2)
            losses['lambda_regularization'] = lambda_reg * 0.00001  # Very small
    
    # 5. Minimal ADMM consensus loss
    if self.combination_method == 'admm' and 'consensus' in metadata:
        consensus = metadata['consensus']
        scale_outputs = metadata.get('scale_outputs', {})
        
        if scale_outputs:
            consensus_loss = 0.0
            for scale, output in scale_outputs.items():
                consensus_loss += F.mse_loss(output, consensus)
            
            losses['admm_consensus'] = consensus_loss * 0.0001  # Very small
    
    # 6. Calculate total loss - reconstruction is dominant
    total_loss = losses.get('reconstruction', 0.0)
    
    # Add other losses with very small weights
    for loss_name, loss_value in losses.items():
        if loss_name != 'reconstruction':
            total_loss = total_loss + loss_value * 0.01  # All other losses are 1% of reconstruction
    
    losses['total'] = total_loss
    
    return total_loss, losses


# Replace the compute_loss method in StableMultiscalePDEsolver1D
StableMultiscalePDEsolver1D.compute_loss = compute_loss_fixed


# ========== 7. VISUALIZATION ==========
def visualize_1d_results(results: Dict, train_scales: List[int], test_dataset: Dataset):
    """Visualize predictions vs ground truth for 1D case"""
    
    successful_methods = {k: v for k, v in results.items() if v.get('success', False)}
    if not successful_methods:
        print("No successful methods to visualize!")
        return
    
    # Get a test sample
    test_sample = test_dataset[0]
    grid = test_sample['grid']  # [2, L]
    solution = test_sample['solution']  # [1, L]
    
    # Create a simple test
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))  # Updated for 6 methods
    axes = axes.flatten()
    
    x = grid[0].cpu().numpy()
    y_true = solution.squeeze().cpu().numpy()
    
    for idx, (method, result) in enumerate(successful_methods.items()):
        if idx >= 6:
            break
        
        # Load model
        model = StableMultiscalePDEsolver1D(
            scales=train_scales,
            in_channels=grid.shape[0],
            out_channels=solution.shape[0],
            combination_method=method
        ).to(device)
        
        try:
            # Load best model
            checkpoint = torch.load(f'best_model_{method}_1d.pth', map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Prepare input
            test_input = {}
            for scale in train_scales:
                # Downsample to each scale
                grid_scaled = F.interpolate(grid.unsqueeze(0), size=scale, mode='linear').squeeze(0)
                test_input[f'grid_{scale}'] = grid_scaled.unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                prediction, _ = model(test_input)
                y_pred = prediction.squeeze().cpu().numpy()
            
            # Plot
            ax = axes[idx]
            ax.plot(x, y_true, 'b-', linewidth=2, label='Ground Truth', alpha=0.7)
            ax.plot(x, y_pred, 'r--', linewidth=2, label='Prediction', alpha=0.7)
            ax.set_xlabel('x')
            ax.set_ylabel('u(x)')
            ax.set_title(f'{method.replace("_", " ").title()} (1D)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Calculate error
            mse = np.mean((y_true - y_pred)**2)
            ax.text(0.05, 0.95, f'MSE: {mse:.2e}', transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
        except Exception as e:
            print(f"Could not visualize {method}: {e}")
            ax = axes[idx]
            ax.text(0.5, 0.5, f'Error loading {method}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{method.replace("_", " ").title()} (Error)')
    
    # Hide any unused subplots
    for idx in range(len(successful_methods), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('1d_comparison_visualization.png', dpi=300, bbox_inches='tight')
    #plt.show()


# ========== 8. MAIN ==========
if __name__ == "__main__":
    print("=" * 80)
    print("1D MULTISCALE PDE SOLVER COMPARISON WITH PHYSICS LOSS")
    print("=" * 80)
    
    # Run 1D comparison
    results, train_scales, eval_scales = stable_multiscale_comparison_1d()
    
    # Analyze results
    if results:
        print(f"\n{'='*80}")
        print("1D TRAINING COMPLETE - SUMMARY")
        print(f"{'='*80}")
        
        successful_methods = {k: v for k, v in results.items() if v.get('success', False)}
        
        if successful_methods:
            print(f"\nSuccessful methods: {', '.join(successful_methods.keys())}")
            print(f"Training scales: {train_scales}")
            print(f"Evaluation scales: {eval_scales}")
            
            # Display results with detailed MSE
            #analyze_results_1d(results, train_scales, eval_scales)
            
            # Create test dataset for visualization
            test_dataset = BurgersDataset1D(n_samples=10, domain_size=256, seed=999)
            
            # Visualize results
            visualize_1d_results(results, train_scales, test_dataset)
            
        else:
            print("No methods successfully trained!")
    else:
        print("No results obtained!")