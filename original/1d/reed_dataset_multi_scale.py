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
import pickle

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========== 0. CORE COMPONENTS - CONVERTED TO 1D ==========
class FourierBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        # Use smaller scale for initialization
        self.scale = 0.1 / (in_channels * out_channels)  # Reduced from 1.0 to 0.1
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )
        self.w = nn.Conv1d(in_channels, out_channels, 1)
        self.norm = nn.GroupNorm(1, out_channels)
        
        # Initialize conv layers with smaller weights
        for layer in [self.w]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros_like(x_ft)
        
        freq_size = x_ft.shape[-1]
        modes = min(self.weights1.shape[2], freq_size)
        
        out_ft[:, :, :modes] = torch.einsum(
            "bix,iox->box", 
            x_ft[:, :, :modes], 
            self.weights1[:, :, :modes]
        )
        
        x_spec = torch.fft.irfft(out_ft, n=x.shape[-1])
        
        # Linear Skip Block + Normalization
        x_lin = self.w(x_in)
        x_combined = x_spec + x_lin
        x_out = self.norm(x_combined)
        
        return x_out


# ========== 1. 1D BURGERS/NAVIER-STOKES DATASET ==========

class Burgers1DDataset(Dataset):
    """
    1D Burgers/Navier-Stokes equation dataset with proper normalization.
    """
    def __init__(self, n_samples: int, domain_size: int = 256,
                 time_step: int = 10, nu: float = 0.01, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.domain_size = domain_size
        self.nu = nu  # viscosity
        self.samples = []
        
        x = torch.linspace(0, 2*np.pi, domain_size)
        self.grid = x
        
        # Pre-compute wave numbers for spectral method
        self.k = 2 * torch.pi * torch.fft.fftfreq(domain_size, d=1.0/domain_size)
        self.k_sq = self.k**2
        self.k_sq[0] = 1.0  # Avoid division by zero
        
        seeds = np.random.randint(0, 1000000, n_samples)
        
        # Collect statistics for normalization
        all_u0 = []
        all_u_t = []
        
        for i, sample_seed in enumerate(seeds):
            torch.manual_seed(sample_seed)
            np.random.seed(sample_seed)
            
            # Generate initial condition
            u0 = self._generate_initial_condition(i)
            
            # Solve Burgers equation with spectral method
            u_t = self._spectral_burgers(u0, time_step)
            
            all_u0.append(u0)
            all_u_t.append(u_t)
            
        # Compute global statistics
        self.u0_mean = torch.stack(all_u0).mean()
        self.u0_std = torch.stack(all_u0).std()
        self.u_t_mean = torch.stack(all_u_t).mean()
        self.u_t_std = torch.stack(all_u_t).std()
        
        print(f"Dataset normalization stats:")
        print(f"  u0: mean={self.u0_mean:.4f}, std={self.u0_std:.4f}")
        print(f"  u_t: mean={self.u_t_mean:.4f}, std={self.u_t_std:.4f}")
        
        for i, sample_seed in enumerate(seeds):
            torch.manual_seed(sample_seed)
            np.random.seed(sample_seed)
            
            u0 = all_u0[i]
            u_t = all_u_t[i]
            
            # Normalize data
            u0_norm = (u0 - self.u0_mean) / (self.u0_std + 1e-8)
            u_t_norm = (u_t - self.u_t_mean) / (self.u_t_std + 1e-8)
            
            # Also normalize grid to [0, 1]
            grid_norm = x / (2 * np.pi)
            
            self.samples.append({
                'grid': torch.stack([grid_norm, u0_norm], dim=0).unsqueeze(0),  # [1, 2, N]
                'solution': u_t_norm.unsqueeze(0).unsqueeze(0),  # [1, 1, N]
                'parameters': {'nu': nu, 'time': time_step, 'seed': sample_seed}
            })
    
    def _generate_initial_condition(self, sample_id: int) -> torch.Tensor:
        """Generate multi-scale initial condition for 1D Burgers equation"""
        domain_size = self.domain_size
        u = torch.zeros(domain_size)
        
        # Add multiple frequencies with controlled amplitudes
        n_freqs = 8 + sample_id % 5
        
        for freq in range(1, n_freqs + 1):
            # Use smaller amplitudes
            amplitude = 0.1 * torch.randn(1).item() / (freq ** 0.5)
            phase = 2 * torch.pi * torch.rand(1).item()
            u += amplitude * torch.sin(freq * self.grid + phase)
        
        # Add shock-like discontinuity for some samples
        if sample_id % 3 == 0:
            shock_pos = torch.rand(1).item() * 2 * np.pi
            shock_width = 0.1 + 0.1 * torch.rand(1).item()
            shock_strength = 0.2 + 0.2 * torch.rand(1).item()
            u += shock_strength * torch.tanh((self.grid - shock_pos) / shock_width)
        
        # Add Gaussian pulse for some samples
        if sample_id % 4 == 0:
            pulse_pos = torch.rand(1).item() * 2 * np.pi
            pulse_width = 0.2 + 0.2 * torch.rand(1).item()
            pulse_strength = 0.1 + 0.1 * torch.rand(1).item()
            u += pulse_strength * torch.exp(-(self.grid - pulse_pos)**2 / (2 * pulse_width**2))
        
        # Don't normalize here - let global normalization handle it
        return u
    
    def _spectral_burgers(self, u0: torch.Tensor, time_step: int) -> torch.Tensor:
        """Spectral method for 1D Burgers equation"""
        u = u0.clone()
        dt = 0.005  # Smaller timestep for stability
        
        for t in range(time_step):
            # Compute Fourier transform
            u_ft = torch.fft.fft(u)
            
            # Non-linear term in physical space (u * u_x)
            u_x = torch.fft.ifft(1j * self.k * u_ft).real
            nonlinear = u * u_x
            
            # Transform non-linear term to Fourier space
            nl_ft = torch.fft.fft(nonlinear)
            
            # Semi-implicit time stepping
            u_ft_new = (u_ft - dt * nl_ft) / (1 + dt * self.nu * self.k_sq)
            
            # Inverse transform
            u = torch.fft.ifft(u_ft_new).real
            
            # Add small forcing for some samples
            if t % 20 == 0:
                forcing = 0.005 * torch.sin(self.grid + t * 0.1)
                u += dt * forcing
        
        return u
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.samples)
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        """Returns one sample from the dataset."""
        return self.samples[idx]
    
# ========== 2. MULTISCALE DATASET WRAPPER - CONVERTED TO 1D ==========
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
        
        # Remove the extra batch dimension that's causing the issue
        if tensor.dim() == 4:  # [B, 1, C, L] from original dataset
            tensor = tensor.squeeze(1)  # Remove the extra dimension: [B, C, L]
        
        if tensor.dim() == 2:  # [C, L] for single sample
            tensor = tensor.unsqueeze(0)  # [1, C, L]
        
        # Now tensor should be [B, C, L] or [1, C, L]
        resized = F.interpolate(
            tensor,
            size=target_size,
            mode='linear',
            align_corners=False
        )
        
        return resized
    
    def _augment(self, grid: torch.Tensor, solution: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.augment:
            return grid, solution
        
        # Squeeze extra dimension if needed
        if grid.dim() == 4:
            grid = grid.squeeze(1)
        if solution.dim() == 4:
            solution = solution.squeeze(1)
        
        # Random flip (reverse spatial direction)
        if torch.rand(1) > 0.5:
            grid = torch.flip(grid, dims=[-1])
            solution = torch.flip(solution, dims=[-1])
        
        # Random translation (periodic boundary)
        if torch.rand(1) > 0.5:
            shift = torch.randint(0, grid.shape[-1], (1,)).item()
            grid = torch.roll(grid, shifts=shift, dims=-1)
            solution = torch.roll(solution, shifts=shift, dims=-1)
        
        return grid, solution
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        sample = self.base_dataset[idx]
        original_grid = sample['grid']  # [1, 2, L] from dataset
        original_solution = sample['solution']  # [1, 1, L] from dataset
        
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
    
# ========== 3. SCALE-SPECIFIC MODELS - CONVERTED TO 1D ==========
class ScaleSpecificFNO1D(nn.Module):
    def __init__(self, scale: int, in_channels: int, out_channels: int, 
                 width: int = 32, depth: int = 4):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        modes = max(4, scale // 8)
        
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
        
        # Initialize with smaller weights
        for layer in [self.lift[0], self.lift[2], self.project[0], self.project[2]]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        # Ensure input is correct shape: [B, C, L]
        if grid.dim() == 4:  # [B, 1, C, L]
            grid = grid.squeeze(1)  # Remove extra dimension
        
        if grid.shape[-1] != self.scale:
            grid = F.interpolate(grid, size=self.scale, mode='linear', align_corners=False)
        
        x = self.lift(grid)
        for layer in self.layers:
            x = layer(x) + x * 0.1  # Add small skip connection
            x = F.gelu(x)
        output = self.project(x)
        
        return output


# ========== 4. COMBINERS - ADAPTED FOR 1D ==========
class MultiscaleSoftmaxRouter1D(nn.Module):
    def __init__(self, scales: List[int], in_channels: int, out_channels: int):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        self.out_channels = out_channels
        
        router_input_dim = self.num_scales * out_channels * 4
        
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_scales),
            nn.Softmax(dim=-1)
        )
        
        self.scale_biases = nn.Parameter(torch.zeros(self.num_scales, out_channels, 1))
        
    def forward(self, multiscale_predictions: Dict[int, torch.Tensor], 
                input_grid: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size = input_grid.shape[0]
        target_size = max(self.scales)
        
        # Get first prediction to determine channels
        available_scales = list(multiscale_predictions.keys())
        if not available_scales:
            device = input_grid.device
            return torch.zeros(batch_size, self.out_channels, target_size, device=device), {'router_weights': None}
        
        router_features = []
        compatible_predictions = {}
        
        for i, scale in enumerate(self.scales):
            if scale in multiscale_predictions:
                pred = multiscale_predictions[scale]
                
                # Ensure proper shape [B, C, L]
                if pred.dim() == 2:  # [B, L] or similar
                    pred = pred.unsqueeze(1)
                elif pred.dim() == 3:  # [B, C, L]
                    pass  # Already correct
                
                compatible_predictions[scale] = pred
                
                # Pool to fixed size for router input
                pooled = F.adaptive_avg_pool1d(pred, 4)
                router_features.append(pooled)
            else:
                # Create zeros for missing scale
                zeros = torch.zeros(batch_size, self.out_channels, 4, device=input_grid.device)
                router_features.append(zeros)
        
        # Concatenate along channel dimension
        router_input = torch.cat(router_features, dim=1)  # [B, out_channels*num_scales, 4]
        router_input = router_input.view(batch_size, -1)  # [B, out_channels*num_scales*4]
        
        router_weights = self.router(router_input)
        
        # Weighted combination
        combined = torch.zeros(batch_size, self.out_channels, target_size, device=input_grid.device)
        total_weight = torch.zeros(batch_size, 1, 1, device=input_grid.device)
        
        for i, scale in enumerate(self.scales):
            if scale in compatible_predictions:
                pred = compatible_predictions[scale]
                
                if pred.shape[-1] != target_size:
                    pred_upscaled = F.interpolate(pred, size=target_size, mode='linear', align_corners=False)
                else:
                    pred_upscaled = pred
                
                pred_upscaled = pred_upscaled + self.scale_biases[i]
                weight = router_weights[:, i].view(-1, 1, 1)
                combined = combined + weight * pred_upscaled
                total_weight = total_weight + weight
        
        combined = combined / (total_weight + 1e-8)
        
        return combined, {
            'router_weights': router_weights,
            'scale_outputs': compatible_predictions
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
                    pred = F.interpolate(pred, size=target_size, mode='linear', align_corners=False)
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
            combined = torch.zeros(1, 1, target_size, device=device)
        
        return combined, {
            'lambda_weights': lambda_weights,
            'log_lambdas': self.log_lambdas,
            'compatible_predictions': compatible_predictions
        }

class LagrangianTwoTimeScaleCombiner1D(nn.Module):
    """Two-timescale Lagrangian combiner (missing from original code)"""
    def __init__(self, scales: List[int], in_channels: int, out_channels: int,
                 primal_lr: float = 1e-4, dual_lr_init: float = 1e-3,
                 rho: float = 0.1, dual_decay: float = 0.999):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        self.primal_lr = primal_lr
        self.dual_lr_init = dual_lr_init
        self.dual_lr = dual_lr_init
        self.dual_decay = dual_decay
        self.rho = rho
        
        # Primal variables (learnable weights per scale)
        self.primal_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        
        # Dual variables (Lagrange multipliers)
        self.dual_params = nn.Parameter(torch.zeros(self.num_scales))
        
        # Adaptive learning rate tracking
        self.constraint_history = []
        self.max_history = 100
        
        # Compatibility layers - FIXED: Removed GroupNorm for scalar output
        self.compatibility_layers = nn.ModuleDict()
        for i, scale in enumerate(scales):
            self.compatibility_layers[str(scale)] = nn.Sequential(
                nn.Conv1d(out_channels, out_channels, 3, padding=1),
                # Removed GroupNorm since out_channels=1 for scalar field
                nn.GELU(),
                nn.Conv1d(out_channels, out_channels, 3, padding=1)
            )
    
    def forward(self, multiscale_predictions: Dict[int, torch.Tensor],
                input_grid: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with two-timescale Lagrangian optimization"""
        target_size = max(self.scales)
        batch_size = input_grid.shape[0]
        
        # Get compatible predictions
        compatible_predictions = {}
        for i, scale in enumerate(self.scales):
            if scale in multiscale_predictions:
                pred = multiscale_predictions[scale]
                if pred.shape[-1] != target_size:
                    pred = F.interpolate(pred, size=target_size, mode='linear', align_corners=False)
                compatible_predictions[scale] = self.compatibility_layers[str(scale)](pred)
        
        if not compatible_predictions:
            device = input_grid.device
            return torch.zeros(batch_size, 1, target_size, device=device), {}
        
        # ===== PRIMAL UPDATE (fast timescale) =====
        # Compute constraints
        weights = F.softmax(self.primal_weights, dim=-1)
        
        # Sum-to-one constraint
        sum_constraint = torch.abs(weights.sum() - 1.0)
        
        # Non-negativity constraint (already satisfied by softmax)
        
        # ===== DUAL UPDATE (slow timescale) =====
        # Update dual learning rate adaptively
        self.constraint_history.append(sum_constraint.item())
        if len(self.constraint_history) > self.max_history:
            self.constraint_history.pop(0)
        
        # Adjust dual learning rate based on constraint satisfaction
        avg_constraint = np.mean(self.constraint_history) if self.constraint_history else 0.0
        
        if avg_constraint > 1e-2:  # High violation
            self.dual_lr = min(self.dual_lr * 1.01, 5e-3)
        elif avg_constraint < 1e-4:  # Low violation
            self.dual_lr = max(self.dual_lr * 0.99, 1e-5)
        
        # Decay dual learning rate
        self.dual_lr *= self.dual_decay
        
        # Update dual variables with adaptive LR
        with torch.no_grad():
            dual_update = self.dual_lr * sum_constraint
            self.dual_params.data += dual_update * torch.randn_like(self.dual_params) * 0.1
        
        # ===== AUGMENTED LAGRANGIAN =====
        # Compute augmented Lagrangian weights
        lagrangian_weights = F.softmax(
            self.primal_weights + self.rho * self.dual_params,
            dim=-1
        )
        
        # Combine predictions
        combined = None
        for i, scale in enumerate(self.scales):
            if scale in compatible_predictions:
                pred = compatible_predictions[scale]
                weight = lagrangian_weights[i]
                
                if combined is None:
                    combined = weight * pred
                else:
                    combined = combined + weight * pred
        
        if combined is None:
            combined = torch.zeros(batch_size, 1, target_size, device=input_grid.device)
        
        # Compute constraint violation for loss
        constraint_violation = sum_constraint + F.relu(-lagrangian_weights).mean()
        
        return combined, {
            'lagrangian_weights': lagrangian_weights,
            'primal_weights': weights,
            'dual_params': self.dual_params,
            'constraint_violation': constraint_violation,
            'dual_lr': self.dual_lr,
            'compatible_predictions': compatible_predictions
        }
    
class ADMMCombiner1D(nn.Module):
    def __init__(self, scales: List[int], in_channels: int, out_channels: int,
                 rho: float = 0.1, num_iter: int = 3):
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
    
    def forward(self, multiscale_predictions: Dict[int, torch.Tensor],
                input_grid: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        target_size = max(self.scales)
        batch_size = input_grid.shape[0]
        
        compatible_predictions = {}
        for scale in self.scales:
            if scale in multiscale_predictions:
                pred = multiscale_predictions[scale]
                if pred.shape[-1] != target_size:
                    pred = F.interpolate(pred, size=target_size, mode='linear', align_corners=False)
                compatible_predictions[scale] = self.compatibility_layers[str(scale)](pred)
        
        if not compatible_predictions:
            return torch.zeros(batch_size, 1, target_size, device=input_grid.device), {}
        
        rho = torch.exp(self.log_rho)
        x = {scale: pred.clone() for scale, pred in compatible_predictions.items()}
        
        weights = torch.ones(len(compatible_predictions), device=input_grid.device) / len(compatible_predictions)
        z = sum(w * x[scale] for w, (scale, _) in zip(weights, compatible_predictions.items()))
        
        u = {scale: torch.zeros_like(pred) for scale, pred in compatible_predictions.items()}
        
        for _ in range(self.num_iter):
            for scale in compatible_predictions.keys():
                x[scale] = (compatible_predictions[scale] + rho * (z - u[scale])) / (1 + rho)
            
            if len(x) > 0:
                z_numerator = torch.zeros_like(z)
                count = 0
                
                for scale, x_i in x.items():
                    if scale in u:
                        contribution = x_i + u[scale]
                        z_numerator = z_numerator + contribution
                        count += 1
                
                if count > 0:
                    soft_consensus = torch.sigmoid(self.consensus_weight)
                    z_new = (z_numerator / count)
                    z = soft_consensus * z_new + (1 - soft_consensus) * z
            
            for scale in compatible_predictions.keys():
                if scale in x and scale in u:
                    u[scale] = u[scale] + (x[scale] - z)
        
        z_refined = self.consensus_refiner(z)
        
        return z_refined, {
            'consensus': z,
            'dual_vars': u,
            'rho': rho
        }

# ========== 5. FIXED MAIN MULTISCALE SOLVER ==========
# ========== 5. FIXED MAIN MULTISCALE SOLVER ==========
class FixedMultiscalePDEsolver1D(nn.Module):
    def __init__(self, scales: List[int], in_channels: int, out_channels: int,
                 combination_method: str = 'softmax', 
                 physics_weight: float = 0.01, constraint_weight: float = 0.01):
        super().__init__()
        self.scales = sorted(scales)
        self.num_scales = len(scales)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.combination_method = combination_method
        self.physics_weight = physics_weight
        self.constraint_weight = constraint_weight
        
        if combination_method == 'single_scale_baseline':
            self.is_baseline = True
            self.baseline_scale = max(scales)
            self.model = ScaleSpecificFNO1D(self.baseline_scale, in_channels, out_channels)
            return
        
        self.is_baseline = False
        
        # Scale-specific models
        self.scale_models = nn.ModuleDict()
        for scale in self.scales:
            self.scale_models[str(scale)] = ScaleSpecificFNO1D(scale, in_channels, out_channels)
        
        # Select combiner (ADDED lagrangian_two_scale)
        if combination_method == 'softmax':
            self.combiner = MultiscaleSoftmaxRouter1D(scales, in_channels, out_channels)
        elif combination_method == 'lagrangian_single':
            self.combiner = LagrangianSingleScaleCombiner1D(scales, in_channels, out_channels)
        elif combination_method == 'lagrangian_two_scale':  # NEW!
            self.combiner = LagrangianTwoTimeScaleCombiner1D(scales, in_channels, out_channels)
        elif combination_method == 'admm':
            self.combiner = ADMMCombiner1D(scales, in_channels, out_channels)
        else:
            raise ValueError(f"Unknown combination method: {combination_method}")
        
        # Fine-tuning layer
        self.fine_tune = nn.Sequential(
            nn.Conv1d(out_channels, out_channels * 2, 3, padding=1),
            nn.GroupNorm(min(8, out_channels * 2), out_channels * 2),
            nn.GELU(),
            nn.Conv1d(out_channels * 2, out_channels, 3, padding=1),
        )
        
        # FIXED: Only initialize weights for Conv1d layers, not GELU
        self._initialize_fine_tune_weights()
    
    def _initialize_fine_tune_weights(self):
        """Initialize only the convolutional layers in fine_tune"""
        for layer in self.fine_tune:
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
    
    def compute_physics_loss(self, u_pred: torch.Tensor, u0: torch.Tensor, 
                           grid: torch.Tensor, nu: float = 0.01) -> torch.Tensor:
        """Physics-constrained loss for Burgers equation"""
        # Convert normalized grid back to [0, 2π]
        x = grid * (2 * np.pi)
        dx = 2 * np.pi / (x.shape[-1] - 1)
        
        # Compute spatial derivatives
        u_x = torch.gradient(u_pred, spacing=dx, dim=-1)[0]
        u_xx = torch.gradient(u_x, spacing=dx, dim=-1)[0]
        
        # Approximate time derivative (assume dt from dataset)
        dt = 0.005 * 10  # time_step from dataset
        u_t = (u_pred - u0) / dt
        
        # Burgers equation residual
        residual = u_t + u_pred * u_x - nu * u_xx
        
        return torch.mean(residual**2)
    
    def forward(self, multiscale_inputs: Dict[str, torch.Tensor],
                return_all: bool = False) -> Tuple[torch.Tensor, Dict]:
        """Forward pass that works for all scales"""
        if self.is_baseline:
            # Use largest scale input
            largest_scale = max(self.scales)
            input_key = f'grid_{largest_scale}'
            if input_key not in multiscale_inputs:
                # Try any available scale
                input_key = list(multiscale_inputs.keys())[0]
            grid = multiscale_inputs[input_key]
            output = self.model(grid)
            return (output, {'scale_used': largest_scale}) if return_all else output
        
        # Collect predictions from all scales
        scale_predictions = {}
        for scale in self.scales:
            input_key = f'grid_{scale}'
            if input_key in multiscale_inputs:
                grid = multiscale_inputs[input_key]
                pred = self.scale_models[str(scale)](grid)
                scale_predictions[scale] = pred
        
        if not scale_predictions:
            device = next(self.parameters()).device
            output = torch.zeros(1, self.out_channels, max(self.scales), device=device)
            return (output, {}) if return_all else output
        
        # Get target grid for combiner
        target_scale = max(self.scales)
        target_grid_key = f'grid_{target_scale}'
        target_grid = multiscale_inputs.get(target_grid_key, next(iter(multiscale_inputs.values())))
        
        # Combine predictions
        combined, meta = self.combiner(scale_predictions, target_grid)
        
        # Fine-tune with residual connection
        refined = self.fine_tune(combined)
        output = refined * 0.1 + combined * 0.9
        
        meta['scale_predictions'] = scale_predictions
        meta['final_output'] = output
        
        return (output, meta) if return_all else output
    
    def compute_loss(self, predictions: torch.Tensor,
                    multiscale_targets: Dict[str, torch.Tensor],
                    multiscale_inputs: Dict[str, torch.Tensor],
                    u0: torch.Tensor,
                    metadata: Dict) -> Tuple[torch.Tensor, Dict]:
        """Fixed loss function with physics and constraints"""
        losses = {}
        
        # 1. Reconstruction loss (MSE)
        recon_loss = 0.0
        num_targets = 0
        
        for scale in self.scales:
            target_key = f'solution_{scale}'
            if target_key in multiscale_targets:
                target = multiscale_targets[target_key]
                
                # Resize prediction if needed
                if predictions.shape[-1] != target.shape[-1]:
                    pred_resized = F.interpolate(
                        predictions, size=target.shape[-1],
                        mode='linear', align_corners=False
                    )
                    loss = F.mse_loss(pred_resized, target)
                else:
                    loss = F.mse_loss(predictions, target)
                
                # Weight by scale importance (higher resolution more important)
                scale_weight = (scale / max(self.scales)) ** 0.5
                recon_loss += loss * scale_weight
                num_targets += 1
        
        if num_targets > 0:
            recon_loss = recon_loss / num_targets
        losses['reconstruction'] = recon_loss
        
        # 2. Physics loss (if we have grid)
        physics_loss = torch.tensor(0.0, device=predictions.device)
        grid_key = f'grid_{max(self.scales)}'
        if grid_key in multiscale_inputs:
            grid = multiscale_inputs[grid_key]
            # Extract spatial grid (channel 0)
            spatial_grid = grid[:, 0:1, :]
            physics_loss = self.compute_physics_loss(predictions, u0, spatial_grid)
        losses['physics'] = physics_loss
        
        # 3. Constraint loss (from combiner)
        constraint_loss = torch.tensor(0.0, device=predictions.device)
        if 'constraint_violation' in metadata:
            constraint_loss = metadata['constraint_violation']
        elif 'lambda_weights' in metadata:
            # For Lagrangian methods, encourage diversity
            weights = metadata['lambda_weights']
            constraint_loss = -torch.sum(weights * torch.log(weights + 1e-10))
        losses['constraint'] = constraint_loss
        
        # 4. Total loss
        total_loss = (recon_loss + 
                     self.physics_weight * physics_loss +
                     self.constraint_weight * constraint_loss)
        
        losses['total'] = total_loss
        
        return total_loss, losses


# ========== 6. FIXED TRAINER ==========
class FixedMultiscaleTrainer1D:
    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader,
                 device: str, lr: float = 1e-3, grad_clip: float = 1.0, 
                 patience: int = 50, eval_scales: List[int] = None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.patience = patience
        self.model_scales = model.scales
        self.eval_scales = eval_scales if eval_scales else model.scales
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10,
        )
        
        self.grad_clip = grad_clip
        self.best_val_loss = float('inf')
        self.history = defaultdict(list)
    
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in self.train_loader:
            # Get u0 from original grid (channel 1 is initial condition)
            u0 = batch['original_grid'][:, :, 1:2, :].to(self.device)
            
            # Process inputs and targets
            multiscale_inputs = self._filter_multiscale_data(batch['multiscale_grids'])
            multiscale_targets = self._filter_multiscale_data(batch['multiscale_solutions'])
            
            if not multiscale_inputs or not multiscale_targets:
                continue
            
            # Process dimensions
            processed_inputs = {}
            for k, v in multiscale_inputs.items():
                v = v.to(self.device)
                if v.dim() == 4:  # [B, 1, C, L]
                    v = v.squeeze(1)  # [B, C, L]
                processed_inputs[k] = v
            
            processed_targets = {}
            for k, v in multiscale_targets.items():
                v = v.to(self.device)
                if v.dim() == 4:
                    v = v.squeeze(1)
                processed_targets[k] = v
            
            # Forward pass
            predictions, metadata = self.model(processed_inputs, return_all=True)
            
            # Compute loss with physics and constraints
            loss, loss_components = self.model.compute_loss(
                predictions, processed_targets, processed_inputs, u0, metadata
            )
            
            if not torch.isfinite(loss):
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip,
                norm_type=2
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        return total_loss / max(batch_count, 1) if batch_count > 0 else float('inf')
    
    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0
        batch_count = 0
        
        for batch in self.val_loader:
            u0 = batch['original_grid'][:, :, 1:2, :].to(self.device)
            
            multiscale_inputs = self._filter_multiscale_data(batch['multiscale_grids'])
            multiscale_targets = self._filter_multiscale_data(batch['multiscale_solutions'])
            
            if not multiscale_inputs or not multiscale_targets:
                continue
            
            processed_inputs = {}
            for k, v in multiscale_inputs.items():
                v = v.to(self.device)
                if v.dim() == 4:
                    v = v.squeeze(1)
                processed_inputs[k] = v
            
            processed_targets = {}
            for k, v in multiscale_targets.items():
                v = v.to(self.device)
                if v.dim() == 4:
                    v = v.squeeze(1)
                processed_targets[k] = v
            
            predictions, metadata = self.model(processed_inputs, return_all=True)
            
            loss, _ = self.model.compute_loss(
                predictions, processed_targets, processed_inputs, u0, metadata
            )
            
            if torch.isfinite(loss):
                total_loss += loss.item()
                batch_count += 1
        
        return total_loss / max(batch_count, 1) if batch_count > 0 else float('inf')
    
    def _filter_multiscale_data(self, batch_data):
        filtered = {}
        for k, v in batch_data.items():
            try:
                scale = int(k.split('_')[1])
                if scale in self.model_scales:
                    filtered[k] = v
            except (ValueError, IndexError):
                continue
        return filtered
    
    def train(self, num_epochs: int = 100, verbose: bool = True):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.scheduler.step(val_loss)
            
            if train_loss == float('inf') or val_loss == float('inf'):
                print(f"Warning: Invalid loss at epoch {epoch+1}")
                continue
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }, f'best_model_1d_{self.model.combination_method}.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if verbose and (epoch % 10 == 0 or epoch < 5 or epoch == num_epochs - 1):
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{num_epochs}:")
                print(f"  Train Loss = {train_loss:.4e}")
                print(f"  Val Loss = {val_loss:.4e}")
                print(f"  LR = {current_lr:.2e}, Best Val = {best_val_loss:.4e}")
                print(f"  Patience = {patience_counter}/{self.patience}")
        
        return self.history


# ========== 7. FIXED TEST FUNCTION ==========
def test_fixed_1d_navier_stokes():
    print(f"Using device: {device}")
    
    # CRITICAL FIX: Train and test on SAME scales!
    train_scales = [64, 128, 256]  # Train on ALL scales including 256
    eval_scales = [128, 256]  # Test on these (subset of training scales)
    
    # Create datasets
    train_dataset = Burgers1DDataset(
        n_samples=2000, domain_size=256, time_step=10, nu=0.01, seed=42
    )
    val_dataset = Burgers1DDataset(
        n_samples=500, domain_size=256, time_step=10, nu=0.01, seed=43
    )
    test_dataset = Burgers1DDataset(
        n_samples=500, domain_size=256, time_step=10, nu=0.01, seed=44
    )
    
    # Create multiscale datasets
    train_multiscale = MultiScaleDataset1D(
        train_dataset, train_scales, mode='train', augment=True
    )
    val_multiscale = MultiScaleDataset1D(
        val_dataset, train_scales, mode='val', augment=False
    )
    test_multiscale = MultiScaleDataset1D(
        test_dataset, eval_scales, mode='test', augment=False
    )
    
    # Data loaders
    train_loader = DataLoader(train_multiscale, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_multiscale, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_multiscale, batch_size=8, shuffle=False)
    
    # Get sample dimensions
    sample = train_multiscale[0]
    in_channels = sample['original_grid'].shape[1]  # Should be 2
    out_channels = sample['original_solution'].shape[1]  # Should be 1
    
    print(f"\nFixed Dataset Info:")
    print(f"  Input channels: {in_channels}, Output channels: {out_channels}")
    print(f"  Training scales: {train_scales}")  # Now includes 256!
    print(f"  Evaluation scales: {eval_scales}")
    print(f"  Train samples: {len(train_multiscale)}")
    print(f"  Val samples: {len(val_multiscale)}")
    print(f"  Test samples: {len(test_multiscale)}")
    
    # Test all methods including new two-timescale Lagrangian
    combination_methods = [
        'single_scale_baseline',
        'softmax',
        'lagrangian_single',
        'lagrangian_two_scale',  # NEW!
        'admm'
    ]
    
    results = {}
    
    for method in combination_methods:
        print(f"\n{'='*60}")
        print(f"Training {method.upper().replace('_', ' ')}")
        print(f"{'='*60}")
        
        # Create model
        model = FixedMultiscalePDEsolver1D(
            scales=train_scales,
            in_channels=in_channels,
            out_channels=out_channels,
            combination_method=method,
            physics_weight=1e-3,
            constraint_weight=0.01
        ).to(device)
        
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create trainer
        trainer = FixedMultiscaleTrainer1D(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=1e-4,  
            grad_clip=1.0,
            #patience=20,
            eval_scales=eval_scales
        )
        
        # Train
        try:
            history = trainer.train(num_epochs=500, verbose=True)
            
            # Test evaluation
            model.eval()
            test_losses = {}
            
            with torch.no_grad():
                for batch in test_loader:
                    # Get u0
                    u0 = batch['original_grid'][:, :, 1:2, :].to(device)
                    
                    # Get inputs and targets
                    multiscale_inputs = {}
                    for scale in train_scales:  # Use training scales for input
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
                    
                    predictions, metadata = model(multiscale_inputs, return_all=True)
                    
                    # Compute loss using the model's compute_loss
                    loss, loss_dict = model.compute_loss(
                        predictions, multiscale_targets, multiscale_inputs, u0, metadata
                    )
                    
                    # Also compute simple MSE for each scale
                    for scale in eval_scales:
                        target_key = f'solution_{scale}'
                        if target_key in multiscale_targets:
                            target = multiscale_targets[target_key]
                            
                            if predictions.shape[-1] != target.shape[-1]:
                                pred_resized = F.interpolate(
                                    predictions, size=target.shape[-1],
                                    mode='linear', align_corners=False
                                )
                                mse = F.mse_loss(pred_resized, target)
                            else:
                                mse = F.mse_loss(predictions, target)
                            
                            if scale not in test_losses:
                                test_losses[scale] = []
                            test_losses[scale].append(mse.item())
            
            # Compute average losses
            avg_test_losses = {}
            for scale, losses in test_losses.items():
                avg_test_losses[scale] = np.mean(losses) if losses else float('inf')
            
            # Overall test loss (average across eval scales)
            overall_test_loss = np.mean(list(avg_test_losses.values())) if avg_test_losses else float('inf')
            
            results[method] = {
                'history': history,
                'final_val_loss': history['val_loss'][-1] if history['val_loss'] else float('inf'),
                'test_losses': avg_test_losses,
                'overall_loss': overall_test_loss,
                'best_val_loss': min(history['val_loss']) if history['val_loss'] else float('inf'),
                'success': True
            }
            
            print(f"\n  Test Results:")
            for scale, loss in avg_test_losses.items():
                print(f"    Scale {scale}: {loss:.4e}")
            print(f"    Average test loss: {overall_test_loss:.4e}")
            
            # Check for overfitting
            if overall_test_loss / results[method]['best_val_loss'] > 10:
                print(f"  ⚠️  Warning: Severe overfitting detected!")
            
        except Exception as e:
            print(f"  Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results[method] = {'success': False, 'error': str(e)}
    
    return results, train_scales, eval_scales


# ========== 8. IMPROVED VISUALIZATION ==========
def visualize_fixed_results(results: Dict, train_scales: List[int], eval_scales: List[int]):
    """Visualize results with better analysis"""
    
    successful_methods = {k: v for k, v in results.items() if v.get('success', False)}
    
    if not successful_methods:
        print("No methods successfully trained!")
        return
    
    # Create summary plot
    plt.figure(figsize=(16, 12))
    
    # 1. Test loss comparison (bar chart)
    plt.subplot(2, 3, 1)
    methods = list(successful_methods.keys())
    method_names = [m.replace('_', ' ').title() for m in methods]
    test_losses = [successful_methods[m]['overall_loss'] for m in methods]
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    bars = plt.bar(method_names, test_losses, color=colors[:len(methods)])
    
    plt.yscale('log')
    plt.ylabel('Test Loss (log scale)')
    plt.title('Method Comparison - Test Loss')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, loss in zip(bars, test_losses):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height*1.05,
                f'{loss:.2e}', ha='center', va='bottom', fontsize=9)
    
    # 2. Loss evolution (training and validation)
    plt.subplot(2, 3, 2)
    for method, result in successful_methods.items():
        history = result['history']
        if 'val_loss' in history:
            method_name = method.replace('_', ' ').title()
            epochs = range(1, len(history['val_loss']) + 1)
            plt.plot(epochs, history['val_loss'], '--', label=f'{method_name} (val)', linewidth=1.5)
            if 'train_loss' in history and len(history['train_loss']) == len(history['val_loss']):
                plt.plot(epochs, history['train_loss'], '-', label=f'{method_name} (train)', alpha=0.7, linewidth=1)
    
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 3. Overfitting analysis (test/val ratio)
    plt.subplot(2, 3, 3)
    overfitting_ratios = []
    for method in methods:
        if method in successful_methods:
            val_loss = successful_methods[method]['best_val_loss']
            test_loss = successful_methods[method]['overall_loss']
            if val_loss > 0:
                ratio = test_loss / val_loss
                overfitting_ratios.append(ratio)
            else:
                overfitting_ratios.append(float('inf'))
    
    bars = plt.bar(method_names, overfitting_ratios, color=['red' if r > 5 else 'green' for r in overfitting_ratios])
    plt.axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='Good (2x)')
    plt.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Overfitting (5x)')
    plt.ylabel('Test/Val Loss Ratio')
    plt.title('Overfitting Analysis')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Scale-wise performance
    plt.subplot(2, 3, 4)
    for method, result in successful_methods.items():
        if 'test_losses' in result:
            scales = sorted(list(result['test_losses'].keys()))
            losses = [result['test_losses'][s] for s in scales]
            method_name = method.replace('_', ' ').title()
            plt.plot(scales, losses, 'o-', label=method_name, linewidth=2, markersize=6)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Resolution Scale')
    plt.ylabel('Test Loss')
    plt.title('Loss vs Resolution')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3, which='both')
    
    # 5. Example prediction
    plt.subplot(2, 3, 5)
    # Load a test sample
    test_dataset = Burgers1DDataset(n_samples=1, domain_size=256, seed=45)
    sample = test_dataset[0]
    x = sample['grid'][0, 0].numpy()  # Grid positions
    u0 = sample['grid'][0, 1].numpy()  # Initial condition
    u_true = sample['solution'][0, 0].numpy()  # True solution
    
    plt.plot(x, u0, 'b-', alpha=0.5, linewidth=1, label='Initial Condition')
    plt.plot(x, u_true, 'k-', alpha=0.8, linewidth=2, label='True Solution')
    
    # Highlight shock regions
    u0_grad = np.gradient(u0)
    shock_regions = np.where(np.abs(u0_grad) > np.percentile(np.abs(u0_grad), 90))[0]
    if len(shock_regions) > 0:
        plt.scatter(x[shock_regions], u0[shock_regions], c='red', s=10, alpha=0.3, label='Shock regions')
    
    plt.xlabel('x (normalized)')
    plt.ylabel('u(x) (normalized)')
    plt.title('Example: Initial and True Solution')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 6. Summary table
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Create summary table
    table_data = [['Method', 'Val Loss', 'Test Loss', 'Ratio']]
    for method in methods:
        if method in successful_methods:
            method_name = method.replace('_', ' ').title()
            val_loss = successful_methods[method]['best_val_loss']
            test_loss = successful_methods[method]['overall_loss']
            ratio = test_loss / val_loss if val_loss > 0 else float('inf')
            
            # Format with scientific notation
            val_str = f"{val_loss:.2e}"
            test_str = f"{test_loss:.2e}"
            ratio_str = f"{ratio:.2f}"
            
            # Color code based on overfitting
            if ratio > 5:
                ratio_str = f"\\textbf{{{ratio_str}}}"
            
            table_data.append([method_name, val_str, test_str, ratio_str])
    
    # Create table
    table = plt.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#404040')
        table[(0, i)].set_text_props(color='white', weight='bold')
    
    plt.title('Performance Summary', fontsize=12, y=0.98)
    
    plt.tight_layout()
    plt.savefig('fixed_burgers_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed summary
    print(f"\n{'='*80}")
    print("FIXED 1D BURGERS EQUATION - DETAILED RESULTS")
    print(f"{'='*80}")
    
    print(f"\nTraining scales: {train_scales}")
    print(f"Evaluation scales: {eval_scales}")
    
    print(f"\n{'Method':<25} {'Best Val Loss':<15} {'Test Loss':<15} {'Ratio':<10} {'Status':<10}")
    print(f"{'-'*85}")
    
    for method, result in successful_methods.items():
        method_name = method.replace('_', ' ').title()
        val_loss = result['best_val_loss']
        test_loss = result['overall_loss']
        ratio = test_loss / val_loss if val_loss > 0 else float('inf')
        
        # Determine status
        if ratio < 2:
            status = "✓ Good"
        elif ratio < 5:
            status = "⚠️ Moderate"
        else:
            status = "✗ Overfit"
        
        print(f"{method_name:<25} {val_loss:<15.4e} {test_loss:<15.4e} {ratio:<10.2f} {status:<10}")
        
        # Show scale-wise performance
        if 'test_losses' in result:
            scale_info = ", ".join([f"{scale}:{loss:.2e}" for scale, loss in result['test_losses'].items()])
            print(f"  Scale losses: {scale_info}")
    
    # Save results
    with open("fixed_burgers_results.pkl", "wb") as f:
        pickle.dump({
            'results': results,
            'train_scales': train_scales,
            'eval_scales': eval_scales,
            'timestamp': time.time()
        }, f)
        print(f"\nResults saved to 'fixed_burgers_results.pkl'")


# ========== 9. MAIN EXECUTION ==========
if __name__ == "__main__":
    print("=" * 100)
    print("FIXED 1D NAVIER-STOKES/BURGERS EQUATION MULTISCALE SOLVER")
    print("Now includes: Two-timescale Lagrangian, Physics-constrained loss, Proper scale training")
    print("=" * 100)
    
    # Run fixed test
    results, train_scales, eval_scales = test_fixed_1d_navier_stokes()
    
    # Visualize results
    if results:
        visualize_fixed_results(results, train_scales, eval_scales)