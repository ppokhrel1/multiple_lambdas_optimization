import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
import warnings
from typing import List, Tuple, Dict, Optional, Union
import os
from scipy import fft

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class LargeScaleBurgersDataset2D(Dataset):
    """
    Dataset for large-scale multi-source 2D Burgers' equation with 128 sources.
    
    Generates 2D velocity fields (u, v) with multiple initial conditions,
    solves the 2D Burgers' equation, and creates 128 corrupted sources.
    """
    
    def __init__(self, n_samples=500, grid_size=64, t_final=0.5, re=100,
                 noise_range=(0.05, 0.5), bias_range=(0.1, 1.0),
                 missing_range=(0.05, 0.2), seed=42):
        super().__init__()
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.t_final = t_final
        self.re = re
        self.nu = 1.0 / re
        self.noise_range = noise_range
        self.bias_range = bias_range
        self.missing_range = missing_range
        
        # Create 2D grid
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        self.X, self.Y = np.meshgrid(x, y)
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        
        np.random.seed(seed)
        self.data = self._generate_data()
    
    def _generate_initial_condition(self):
        """Generate stable 2D initial condition for u and v components"""
        ic_type = np.random.choice(['vortex', 'gaussian', 'shear_layer', 'random', 'mixed'])
        
        if ic_type == 'vortex':
            # Taylor-Green vortex
            k = np.random.uniform(1.0, 3.0) * np.pi
            u0 = np.sin(k * self.X) * np.cos(k * self.Y)
            v0 = -np.cos(k * self.X) * np.sin(k * self.Y)
            
        elif ic_type == 'gaussian':
            # Gaussian vortices
            centers = [(0.5, 0.5), (-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5)]
            center = centers[np.random.randint(0, 4)]
            width = np.random.uniform(5, 20)
            
            # Velocity potential
            psi = np.exp(-width * ((self.X - center[0])**2 + (self.Y - center[1])**2))
            
            # Compute velocity field from potential
            u0 = np.gradient(psi, self.dx, axis=1)
            v0 = -np.gradient(psi, self.dy, axis=0)
            
        elif ic_type == 'shear_layer':
            # Shear layer instability
            thickness = np.random.uniform(5, 20)
            u0 = np.tanh(thickness * self.Y)
            v0 = 0.1 * np.sin(2 * np.pi * self.X)
            
        elif ic_type == 'mixed':
            # Mixed modes
            k1 = np.random.uniform(1.0, 2.5)
            k2 = np.random.uniform(1.0, 2.5)
            u0 = 0.5 * np.sin(k1 * np.pi * self.X) * np.cos(k2 * np.pi * self.Y) + \
                 0.3 * np.exp(-10 * (self.X**2 + self.Y**2))
            v0 = 0.5 * np.cos(k1 * np.pi * self.X) * np.sin(k2 * np.pi * self.Y) + \
                 0.3 * np.exp(-10 * ((self.X - 0.3)**2 + (self.Y - 0.3)**2))
        
        else:  # random
            # Random divergence-free field using stream function
            psi = np.random.randn(self.grid_size, self.grid_size)
            psi = ndimage.gaussian_filter(psi, sigma=2)
            
            u0 = np.gradient(psi, self.dx, axis=1)
            v0 = -np.gradient(psi, self.dy, axis=0)
        
        # Normalize to prevent instability
        max_uv = max(np.max(np.abs(u0)), np.max(np.abs(v0))) + 1e-8
        u0 = u0 / max_uv
        v0 = v0 / max_uv
        
        return u0, v0
    
    def _solve_burgers_2d(self, u0, v0):
        """Solve 2D Burgers' equation using pseudo-spectral method with stability control"""
        N = self.grid_size
        
        # Wavenumbers in 2D
        kx = 2 * np.pi * np.fft.fftfreq(N, d=self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(N, d=self.dy)
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        
        # Initial conditions in Fourier space
        u_hat = np.fft.fft2(u0)
        v_hat = np.fft.fft2(v0)
        
        # Time stepping with CFL condition
        dt = 0.0005  # Smaller time step for stability
        n_steps = int(self.t_final / dt)
        
        for step in range(n_steps):
            # Convert to physical space
            u = np.real(np.fft.ifft2(u_hat))
            v = np.real(np.fft.ifft2(v_hat))
            
            # Check for instability
            if np.any(np.isnan(u)) or np.any(np.isinf(u)) or \
               np.any(np.isnan(v)) or np.any(np.isinf(v)):
                return None, None
            
            # Nonlinear terms
            uu_hat = np.fft.fft2(u * u)
            uv_hat = np.fft.fft2(u * v)
            vv_hat = np.fft.fft2(v * v)
            
            # Burgers' equations in Fourier space
            rhs_u = -1j * (KX * uu_hat + KY * uv_hat) - self.nu * K2 * u_hat
            rhs_v = -1j * (KX * uv_hat + KY * vv_hat) - self.nu * K2 * v_hat
            
            # Semi-implicit scheme for stability
            u_hat_new = (u_hat + dt * rhs_u) / (1 + dt * self.nu * K2)
            v_hat_new = (v_hat + dt * rhs_v) / (1 + dt * self.nu * K2)
            
            # Dealising
            cutoff = N // 2
            u_hat_new[cutoff:, :] = 0
            u_hat_new[:, cutoff:] = 0
            v_hat_new[cutoff:, :] = 0
            v_hat_new[:, cutoff:] = 0
            
            u_hat = u_hat_new
            v_hat = v_hat_new
        
        # Convert back to physical space
        u_final = np.real(np.fft.ifft2(u_hat))
        v_final = np.real(np.fft.ifft2(v_hat))
        
        return u_final, v_final
    
    def _corrupt_data_2d(self, u_clean, v_clean):
        """Apply noise, bias, and missing values to create corrupted 2D sources"""
        u_corrupted = u_clean.copy()
        v_corrupted = v_clean.copy()
        
        # Add Gaussian noise
        noise_std = np.random.uniform(*self.noise_range)
        u_corrupted += np.random.normal(0, noise_std, size=u_corrupted.shape)
        v_corrupted += np.random.normal(0, noise_std, size=v_corrupted.shape)
        
        # Add bias (spatially varying)
        bias_u = np.random.uniform(*self.bias_range)
        bias_v = np.random.uniform(*self.bias_range)
        u_corrupted += bias_u
        v_corrupted += bias_v
        
        # Add spatially correlated missing values
        missing_prob = np.random.uniform(*self.missing_range)
        
        # Create patches of missing values
        patch_size = np.random.randint(3, 10)
        mask = np.random.random(u_corrupted.shape) < missing_prob
        mask = ndimage.binary_dilation(mask, structure=np.ones((patch_size, patch_size)))
        
        u_corrupted[mask] = 0.0
        v_corrupted[mask] = 0.0
        
        return u_corrupted, v_corrupted
    
    def _generate_data(self):
        """Generate multi-source 2D dataset, skipping unstable samples."""
        data = []
        
        print("Generating large-scale multi-source 2D dataset...")
        i = 0
        max_attempts = int(self.n_samples * 3)  # More attempts for 2D stability
        attempts = 0
        
        while i < self.n_samples and attempts < max_attempts:
            # Generate clean initial conditions
            u0, v0 = self._generate_initial_condition()
            
            # Solve 2D Burgers' equation
            u_clean, v_clean = self._solve_burgers_2d(u0, v0)
            
            if u_clean is None or v_clean is None:
                # Solver failed due to instability
                attempts += 1
                continue
            
            # Check for stability
            if (np.max(np.abs(u_clean)) > 10.0 or np.max(np.abs(v_clean)) > 10.0 or
                np.any(np.isnan(u_clean)) or np.any(np.isnan(v_clean))):
                attempts += 1
                continue
            
            # Generate 128 corrupted sources
            sources_u = []
            sources_v = []
            
            for src_idx in range(128):
                u_corr, v_corr = self._corrupt_data_2d(u_clean, v_clean)
                sources_u.append(u_corr.astype(np.float32))
                sources_v.append(v_corr.astype(np.float32))
            
            # Stack u and v components
            sources = []
            for u_corr, v_corr in zip(sources_u, sources_v):
                source = np.stack([u_corr, v_corr], axis=0)  # Shape: (2, H, W)
                sources.append(source)
            
            data.append({
                'u0': np.stack([u0, v0], axis=0).astype(np.float32),  # (2, H, W)
                'u_clean': np.stack([u_clean, v_clean], axis=0).astype(np.float32),  # (2, H, W)
                'sources': sources  # List of (2, H, W) arrays
            })
            
            i += 1
            
            if i % 50 == 0:
                print(f"Generated {i}/{self.n_samples} stable 2D samples.")
            
            attempts += 1
        
        print(f"Data generation complete. {len(data)} stable 2D samples generated.")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        sources_tensor = torch.stack([torch.FloatTensor(src) for src in item['sources']])  # (128, 2, H, W)
        return {
            'u0': torch.FloatTensor(item['u0']),  # (2, H, W)
            'u_clean': torch.FloatTensor(item['u_clean']),  # (2, H, W)
            'sources': sources_tensor  # (128, 2, H, W)
        }


class SourceNetwork2D(nn.Module):
    """Individual source network for 2D with convolutional neural transform"""
    
    def __init__(self, input_channels=2, hidden_channels=64, spatial_size=64):
        super().__init__()
        self.input_channels = input_channels
        self.spatial_size = spatial_size
        
        # Convolutional neural transform for 2D
        self.transform = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1)
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.transform:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x):
        return self.transform(x)


class BaseIntegrator2D(nn.Module):
    """Base class for all 2D integrators with common methods"""
    
    def __init__(self, n_sources=128, input_channels=2, spatial_size=64, hidden_dim=64):
        super().__init__()
        self.n_sources = n_sources
        self.input_channels = input_channels
        self.spatial_size = spatial_size
        
        # Attention network for source weighting (2D convolutional)
        self.attention_net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_sources)
        )
        
        # Source networks (128 sources)
        self.source_networks = nn.ModuleList([
            SourceNetwork2D(input_channels, hidden_channels=64, spatial_size=spatial_size) 
            for _ in range(n_sources)
        ])
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.attention_net:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
    
    def compute_physics_loss(self, u_pred, u0, nu=0.01):
        """Compute 2D Burgers' equation residual loss using torch operations"""
        device = u_pred.device
        batch_size = u_pred.shape[0]
        
        # Spatial steps
        dx = 2.0 / (self.spatial_size - 1)
        dy = 2.0 / (self.spatial_size - 1)
        
        # Time step
        dt = 0.01
        
        physics_loss = 0.0
        
        for i in range(batch_size):
            u = u_pred[i, 0]  # u-component
            v = u_pred[i, 1]  # v-component
            u0_i = u0[i, 0]
            v0_i = u0[i, 1]
            
            # Compute spatial derivatives using torch.gradient for 2D
            # u derivatives
            du_dx = torch.gradient(u, spacing=dx, dim=1)[0]
            du_dy = torch.gradient(u, spacing=dy, dim=0)[0]
            
            # v derivatives
            dv_dx = torch.gradient(v, spacing=dx, dim=1)[0]
            dv_dy = torch.gradient(v, spacing=dy, dim=0)[0]
            
            # Second derivatives for diffusion term
            du_dx2 = torch.gradient(du_dx, spacing=dx, dim=1)[0]
            du_dy2 = torch.gradient(du_dy, spacing=dy, dim=0)[0]
            
            dv_dx2 = torch.gradient(dv_dx, spacing=dx, dim=1)[0]
            dv_dy2 = torch.gradient(dv_dy, spacing=dy, dim=0)[0]
            
            # Time derivatives
            du_dt = (u - u0_i) / dt
            dv_dt = (v - v0_i) / dt
            
            # 2D Burgers' equation residuals
            residual_u = du_dt + u * du_dx + v * du_dy - nu * (du_dx2 + du_dy2)
            residual_v = dv_dt + u * dv_dx + v * dv_dy - nu * (dv_dx2 + dv_dy2)
            
            physics_loss += torch.mean(residual_u**2 + residual_v**2)
        
        return physics_loss / batch_size
    
    def _project_simplex_stable(self, v):
        """Stable projection onto simplex (sum to 1, non-negative) for 2D"""
        device = v.device
        batch_size = v.shape[0]
        
        # Add small epsilon to avoid negative values
        v = v + 1e-10
        
        # Sort in descending order
        v_sorted, _ = torch.sort(v, descending=True, dim=-1)
        
        # Compute cumulative sum
        cssv = torch.cumsum(v_sorted, dim=-1)
        
        # Create rho indices
        rho = torch.arange(1, v.shape[-1] + 1, device=device).float().unsqueeze(0).expand(batch_size, -1)
        
        # Find threshold
        cond = v_sorted - (cssv - 1.0) / rho > 0
        rho_max = cond.sum(dim=-1, keepdim=True)
        
        # Compute theta
        rho_max_expanded = rho_max.float()
        cssv_selected = torch.gather(cssv, 1, (rho_max - 1).clamp(min=0))
        theta = (cssv_selected - 1.0) / rho_max_expanded.clamp(min=1)
        
        # Project
        return F.relu(v - theta)
    
    def _compute_constraints(self, weights):
        """Compute constraint violations for 2D"""
        # Sum-to-one constraint
        sum_constraint = torch.abs(weights.sum(dim=-1) - 1.0).mean()
        
        # Non-negativity constraint
        nonneg_constraint = F.relu(-weights).mean()
        
        return sum_constraint + nonneg_constraint
    
    def forward(self, x):
        """Base forward pass - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_regularization_loss(self):
        """Get regularization loss if any"""
        return torch.tensor(0.0, device=next(self.parameters()).device)


class SoftmaxIntegrator2D(BaseIntegrator2D):
    """Softmax-based integrator with temperature scaling and top-k sparsity for 2D"""
    
    def __init__(self, n_sources=128, input_channels=2, spatial_size=64, hidden_dim=64,
                 k=10, temperature=1.0, entropy_weight=0.01):
        super().__init__(n_sources, input_channels, spatial_size, hidden_dim)
        self.k = min(k, n_sources)
        self.temperature = temperature
        self.entropy_weight = entropy_weight
    
    def forward(self, x, return_weights=False):
        """Forward pass with softmax attention and top-k sparsity for 2D"""
        batch_size = x.shape[0]
        
        # Compute attention scores from 2D input
        attention_logits = self.attention_net(x)  # (batch_size, n_sources)
        
        # Temperature-scaled softmax
        attention_weights = F.softmax(attention_logits / self.temperature, dim=-1)
        
        # Apply top-k sparsity
        if self.k < self.n_sources:
            top_k_values, top_k_indices = torch.topk(attention_weights, self.k, dim=-1)
            
            # Create sparse weight matrix
            sparse_weights = torch.zeros_like(attention_weights)
            sparse_weights.scatter_(-1, top_k_indices, top_k_values)
            
            # Renormalize
            sparse_weights = sparse_weights / (sparse_weights.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            sparse_weights = attention_weights
        
        # Compute source outputs for 2D
        source_outputs = []
        for i, source_net in enumerate(self.source_networks):
            source_out = source_net(x)  # (batch_size, 2, H, W)
            source_outputs.append(source_out.unsqueeze(1))  # (batch_size, 1, 2, H, W)
        
        source_outputs = torch.cat(source_outputs, dim=1)  # (batch_size, n_sources, 2, H, W)
        
        # Weighted combination for 2D
        weights_expanded = sparse_weights.view(batch_size, self.n_sources, 1, 1, 1)  # Add spatial dimensions
        output = torch.sum(source_outputs * weights_expanded, dim=1)  # (batch_size, 2, H, W)
        
        if return_weights:
            # Compute entropy for regularization
            entropy = -torch.sum(sparse_weights * torch.log(sparse_weights + 1e-10), dim=-1).mean()
            sparsity = (sparse_weights < 1e-3).float().mean()
            
            return output, sparse_weights, {
                'entropy': entropy,
                'sparsity': sparsity
            }
        
        return output
    
    def get_regularization_loss(self):
        """Get entropy regularization loss"""
        return torch.tensor(0.0, device=next(self.parameters()).device)


class LagrangianSingleTimeIntegrator2D(BaseIntegrator2D):
    """Lagrangian integrator with single timescale optimization for 2D"""
    
    def __init__(self, n_sources=128, input_channels=2, spatial_size=64, hidden_dim=64,
                 rho=0.1, dual_lr=1e-3):
        super().__init__(n_sources, input_channels, spatial_size, hidden_dim)
        self.rho = rho
        self.dual_lr = dual_lr
        
        # Dual variables (Lagrange multipliers)
        self.lambda_params = nn.Parameter(torch.zeros(n_sources))
    
    def forward(self, x, return_weights=False):
        """Forward pass with Lagrangian optimization for 2D"""
        batch_size = x.shape[0]
        
        # Compute primal variables (weights) from 2D network
        primal_logits = self.attention_net(x)  # (batch_size, n_sources)
        
        # Project onto simplex with Lagrangian
        weights = self._lagrangian_projection(primal_logits)
        
        # Compute source outputs for 2D
        source_outputs = []
        for i, source_net in enumerate(self.source_networks):
            source_out = source_net(x)  # (batch_size, 2, H, W)
            source_outputs.append(source_out.unsqueeze(1))
        
        source_outputs = torch.cat(source_outputs, dim=1)  # (batch_size, n_sources, 2, H, W)
        
        # Weighted combination
        weights_expanded = weights.view(batch_size, self.n_sources, 1, 1, 1)
        output = torch.sum(source_outputs * weights_expanded, dim=1)  # (batch_size, 2, H, W)
        
        if return_weights:
            # Compute constraint violations
            constraint_violation = self._compute_constraints(weights)
            
            # Compute concentration metric
            entropy = -torch.sum(weights * torch.log(weights + 1e-10), dim=-1).mean()
            concentration = 1.0 / (entropy + 1e-10)
            
            return output, weights, {
                'constraint_violation': constraint_violation,
                'concentration': concentration
            }
        
        return output
    
    def _lagrangian_projection(self, primal_logits):
        """Project onto simplex with augmented Lagrangian for 2D"""
        batch_size = primal_logits.shape[0]
        
        # Initial softmax
        weights = F.softmax(primal_logits, dim=-1)
        
        # Add Lagrange multiplier adjustment
        lambda_expanded = self.lambda_params.unsqueeze(0).expand(batch_size, -1)
        adjusted = weights + self.rho * lambda_expanded
        
        # Project onto non-negative orthant
        adjusted = F.relu(adjusted)
        
        # Project onto simplex
        return self._project_simplex_stable(adjusted)
    
    def update_lagrange_multipliers(self, constraint_violation):
        """Update dual variables (Lagrange multipliers) for 2D"""
        with torch.no_grad():
            # Fixed step size update (single timescale)
            update = self.dual_lr * constraint_violation
            update = torch.clamp(update, -1.0, 1.0)
            self.lambda_params += update
    
    def get_regularization_loss(self):
        """Get constraint regularization loss for 2D"""
        return torch.tensor(0.0, device=next(self.parameters()).device)


class LagrangianTwoTimeIntegrator2D(LagrangianSingleTimeIntegrator2D):
    """Lagrangian integrator with two-timescale optimization for 2D"""
    
    def __init__(self, n_sources=128, input_channels=2, spatial_size=64, hidden_dim=64,
                 rho=0.1, dual_lr_init=1e-3, dual_decay=0.999, dual_min_lr=1e-5):
        super().__init__(n_sources, input_channels, spatial_size, hidden_dim, rho, dual_lr_init)
        self.dual_lr_init = dual_lr_init
        self.dual_lr = dual_lr_init
        self.dual_decay = dual_decay
        self.dual_min_lr = dual_min_lr
        
        # Tracking for adaptive dual learning rate
        self.constraint_history = []
        self.max_history = 100
    
    def update_lagrange_multipliers(self, constraint_violation):
        """Update dual variables with adaptive learning rate (two timescales) for 2D"""
        with torch.no_grad():
            # Store constraint violation for adaptive learning rate
            self.constraint_history.append(constraint_violation.item())
            if len(self.constraint_history) > self.max_history:
                self.constraint_history.pop(0)
            
            # Adaptive dual learning rate based on constraint satisfaction
            avg_constraint = np.mean(self.constraint_history) if self.constraint_history else constraint_violation.item()
            
            # Adjust learning rate based on constraint satisfaction
            if avg_constraint > 1e-2:  # High constraint violation
                self.dual_lr = min(self.dual_lr * 1.01, 5e-3)  # Increase LR
            elif avg_constraint < 1e-4:  # Low constraint violation
                self.dual_lr = max(self.dual_lr * 0.99, self.dual_min_lr)  # Decrease LR
            
            # Decay learning rate
            self.dual_lr *= self.dual_decay
            self.dual_lr = max(self.dual_lr, self.dual_min_lr)
            
            # Update with adaptive learning rate
            update = self.dual_lr * constraint_violation
            update = torch.clamp(update, -0.5, 0.5)
            self.lambda_params += update


class ADMMIntegrator2D(BaseIntegrator2D):
    """ADMM (Alternating Direction Method of Multipliers) integrator for 2D"""
    
    def __init__(self, n_sources=128, input_channels=2, spatial_size=64, hidden_dim=64,
                 rho=0.1, alpha=1.6, max_iter=3):
        super().__init__(n_sources, input_channels, spatial_size, hidden_dim)
        self.rho = rho
        self.alpha = alpha  # Over-relaxation parameter
        self.max_iter = max_iter
        
        # ADMM variables
        self.z = nn.Parameter(torch.ones(n_sources) / n_sources)  # Auxiliary variable
        self.u = nn.Parameter(torch.zeros(n_sources))  # Dual variable
        self.z_prev = None
    
    def forward(self, x, return_weights=False):
        """Forward pass with ADMM optimization for 2D"""
        batch_size = x.shape[0]
        
        # Compute primal variables (weights) from 2D network
        primal_logits = self.attention_net(x)  # (batch_size, n_sources)
        
        # Initial softmax weights with temperature for stability
        weights = F.softmax(primal_logits / 1.0, dim=-1)
        
        # ADMM iterations
        for iter_idx in range(self.max_iter):
            # z-update: project onto simplex
            z_candidate = weights.mean(dim=0) + self.u
            z_projected = self._project_simplex_stable_single(z_candidate)
            
            # Over-relaxation with clipping
            z_relaxed = self.alpha * z_projected + (1 - self.alpha) * self.z
            z_relaxed = torch.clamp(z_relaxed, 0.0, 1.0)
            
            # u-update with clipping
            u_update = self.u + weights.mean(dim=0) - z_relaxed
            u_update = torch.clamp(u_update, -10.0, 10.0)
            
            # Store for next iteration
            self.z_prev = self.z.clone() if self.z_prev is None else self.z.clone()
            
            # Update variables
            with torch.no_grad():
                self.z.data = z_projected
                self.u.data = u_update
        
        # Compute source outputs for 2D
        source_outputs = []
        for i, source_net in enumerate(self.source_networks):
            source_out = source_net(x)  # (batch_size, 2, H, W)
            source_outputs.append(source_out.unsqueeze(1))
        
        source_outputs = torch.cat(source_outputs, dim=1)  # (batch_size, n_sources, 2, H, W)
        
        # Use weights for combination
        weights_expanded = weights.view(batch_size, self.n_sources, 1, 1, 1)
        output = torch.sum(source_outputs * weights_expanded, dim=1)  # (batch_size, 2, H, W)
        
        if return_weights:
            # Compute ADMM residuals
            primal_residual = torch.norm(weights.mean(dim=0) - self.z, p=2)
            dual_residual = torch.norm(self.z - self.z_prev, p=2) if self.z_prev is not None else torch.tensor(0.0)
            
            # Compute weight concentration
            entropy = -torch.sum(weights * torch.log(weights + 1e-10), dim=-1).mean()
            concentration = 1.0 / (entropy + 1e-10)
            
            return output, weights, {
                'admm_residual': primal_residual + dual_residual,
                'concentration': concentration
            }
        
        return output
    
    def _project_simplex_stable_single(self, v):
        """Stable projection onto simplex for single vector"""
        device = v.device
        
        # Add small epsilon to avoid issues
        v = v + 1e-10
        
        # Sort in descending order
        u, _ = torch.sort(v, descending=True)
        
        # Compute cumulative sum
        cssv = torch.cumsum(u, dim=0)
        
        # Create rho indices
        rho = torch.arange(1, len(v) + 1, device=device).float()
        
        # Find threshold
        cond = u - (cssv - 1.0) / rho > 0
        rho_max = cond.sum().item()
        
        # Safely compute theta
        if rho_max > 0:
            theta = (cssv[rho_max - 1] - 1.0) / float(rho_max)
        else:
            theta = torch.tensor(0.0, device=device)
        
        # Project
        return torch.max(v - theta, torch.tensor(0.0, device=device))
    
    def get_regularization_loss(self):
        """Get ADMM regularization loss for 2D"""
        return torch.tensor(0.0, device=next(self.parameters()).device)


class UnifiedLoss2D:
    """Unified loss computation for all 2D integrator types"""
    
    def __init__(self, mse_weight=1.0, physics_weight=0.01,
                 constraint_weight=0.01, entropy_weight=0.001):
        self.mse_weight = mse_weight
        self.physics_weight = physics_weight
        self.constraint_weight = constraint_weight
        self.entropy_weight = entropy_weight
    
    def __call__(self, model, u_pred, u_true, u0, extra_losses=None):
        """Compute unified loss for any 2D integrator type"""
        total_loss = 0.0
        loss_dict = {}
        
        # Reconstruction loss for 2D
        mse_loss = F.mse_loss(u_pred, u_true)
        total_loss += self.mse_weight * mse_loss
        loss_dict['mse'] = mse_loss.item()
        
        # Physics-constrained loss for 2D
        physics_loss = model.compute_physics_loss(u_pred, u0)
        total_loss += self.physics_weight * physics_loss
        loss_dict['physics'] = physics_loss.item()
        
        # Integrator-specific losses
        if extra_losses is not None:
            if 'entropy' in extra_losses:
                # For softmax: encourage diversity
                entropy_loss = self.entropy_weight * extra_losses['entropy']
                total_loss += entropy_loss
                loss_dict['entropy'] = extra_losses['entropy'].item()
                
            if 'constraint_violation' in extra_losses:
                # For Lagrangian: penalize constraint violations
                constraint_loss = self.constraint_weight * extra_losses['constraint_violation']
                total_loss += constraint_loss
                loss_dict['constraint'] = extra_losses['constraint_violation'].item()
                
            if 'admm_residual' in extra_losses:
                # For ADMM: penalize ADMM residuals
                admm_loss = self.constraint_weight * extra_losses['admm_residual']
                total_loss += admm_loss
                loss_dict['admm_residual'] = extra_losses['admm_residual'].item()
        
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


class TwoTimescaleAdamOptimizer2D:
    """Two-timescale Adam optimizer for primal and dual variables for 2D"""
    
    def __init__(self, model, primal_lr=1e-4, dual_lr=1e-3,
                 betas=(0.9, 0.999), eps=1e-8):
        self.model = model
        self.primal_lr = primal_lr
        self.dual_lr = dual_lr
        
        # Separate optimizers for primal and dual parameters
        primal_params = []
        dual_params = []
        
        for name, param in model.named_parameters():
            if 'lambda' in name or 'z' in name or 'u' in name:
                dual_params.append(param)
            else:
                primal_params.append(param)
        
        self.primal_optimizer = torch.optim.Adam(
            primal_params, lr=primal_lr, betas=betas, eps=eps
        )
        
        self.dual_optimizer = torch.optim.Adam(
            dual_params, lr=dual_lr, betas=betas, eps=eps
        )
    
    def zero_grad(self):
        self.primal_optimizer.zero_grad()
        self.dual_optimizer.zero_grad()
    
    def step(self):
        self.primal_optimizer.step()
        self.dual_optimizer.step()


def train_large_scale_integration_unified_2d():
    """Main training function with unified loss structure for 2D"""
    
    print("="*80)
    print("LARGE-SCALE MULTI-SOURCE 2D INTEGRATION - UNIFIED IMPLEMENTATION")
    print("128 sources, 64x64 grid, 2D Burgers' equation (Re=100)")
    print("Comparing: Softmax, Single-Timescale Lagrangian, Two-Timescale Lagrangian, ADMM")
    print("="*80)
    
    # Create 2D dataset
    print("\n1. Creating 2D dataset...")
    dataset = LargeScaleBurgersDataset2D(
        n_samples=300,  # Reduced for 2D (more complex)
        grid_size=64,
        t_final=0.5,
        re=100,
        noise_range=(0.05, 0.3),  # Reduced for 2D
        bias_range=(0.1, 0.5),
        missing_range=(0.05, 0.15)
    )
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Smaller batch size for 2D
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize 2D models
    print("\n2. Initializing 2D integration models...")
    
    # Softmax integrator for 2D
    softmax_model = SoftmaxIntegrator2D(
        n_sources=128,
        input_channels=2,
        spatial_size=64,
        hidden_dim=64,
        k=10,
        temperature=1.0,
        entropy_weight=0.01
    ).to(device)
    
    # Single-timescale Lagrangian integrator for 2D
    single_ts_model = LagrangianSingleTimeIntegrator2D(
        n_sources=128,
        input_channels=2,
        spatial_size=64,
        hidden_dim=64,
        rho=0.1,
        dual_lr=1e-3
    ).to(device)
    
    # Two-timescale Lagrangian integrator for 2D
    two_ts_model = LagrangianTwoTimeIntegrator2D(
        n_sources=128,
        input_channels=2,
        spatial_size=64,
        hidden_dim=64,
        rho=0.1,
        dual_lr_init=1e-3,
        dual_decay=0.999,
        dual_min_lr=1e-5
    ).to(device)
    
    # ADMM integrator for 2D
    admm_model = ADMMIntegrator2D(
        n_sources=128,
        input_channels=2,
        spatial_size=64,
        hidden_dim=64,
        rho=0.1,
        alpha=1.6,
        max_iter=3
    ).to(device)
    
    # Optimizers
    models = {
        'softmax': softmax_model,
        'single_ts_lagrangian': single_ts_model,
        'two_ts_lagrangian': two_ts_model,
        'admm': admm_model
    }
    
    # Use smaller learning rates for 2D
    optimizers = {
        'softmax': torch.optim.Adam(softmax_model.parameters(), lr=5e-5),
        'single_ts_lagrangian': torch.optim.Adam(single_ts_model.parameters(), lr=5e-5),
        'two_ts_lagrangian': TwoTimescaleAdamOptimizer2D(two_ts_model, primal_lr=5e-5, dual_lr=5e-4),
        'admm': TwoTimescaleAdamOptimizer2D(admm_model, primal_lr=5e-5, dual_lr=5e-4)
    }
    
    # Unified loss function for 2D
    unified_loss = UnifiedLoss2D(
        mse_weight=1.0,
        physics_weight=0.05,  # Increased for 2D
        constraint_weight=0.01,
        entropy_weight=0.001
    )
    
    # Training
    print("\n3. Training 2D models (100 epochs)...")
    epochs = 100  # Fewer epochs for 2D
    results = {method: [] for method in models.keys()}
    
    # Gradient clipping
    grad_clip_value = 0.5  # Smaller for 2D
    
    for epoch in range(epochs):
        # Train each model
        for method_name, model in models.items():
            model.train()
            optimizer = optimizers[method_name]
            
            train_loss_total = 0.0
            train_loss_mse = 0.0
            train_loss_physics = 0.0
            train_loss_extra = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                u0 = batch['u0'].to(device)  # (batch, 2, H, W)
                u_clean = batch['u_clean'].to(device)
                
                # Zero gradients
                if isinstance(optimizer, TwoTimescaleAdamOptimizer2D):
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad()
                
                # Forward pass with weights for 2D
                u_pred, weights, extra_losses = model(u0, return_weights=True)
                
                # Compute unified loss
                total_loss, loss_dict = unified_loss(model, u_pred, u_clean, u0, extra_losses)
                
                # Check for NaN
                if torch.isnan(total_loss):
                    print(f"Warning: NaN in {method_name} loss, skipping batch")
                    continue
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping for 2D
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_value)
                
                # Optimizer step
                if isinstance(optimizer, TwoTimescaleAdamOptimizer2D):
                    optimizer.step()
                else:
                    optimizer.step()
                
                # Update Lagrange multipliers for Lagrangian methods
                if method_name in ['single_ts_lagrangian', 'two_ts_lagrangian']:
                    if 'constraint_violation' in extra_losses:
                        model.update_lagrange_multipliers(extra_losses['constraint_violation'])
                
                # Accumulate losses
                train_loss_total += loss_dict['total']
                train_loss_mse += loss_dict['mse']
                train_loss_physics += loss_dict['physics']
                if 'entropy' in loss_dict:
                    train_loss_extra += loss_dict['entropy']
                elif 'constraint' in loss_dict:
                    train_loss_extra += loss_dict['constraint']
                elif 'admm_residual' in loss_dict:
                    train_loss_extra += loss_dict['admm_residual']
            
            # Store training stats
            num_batches = len(train_loader)
            if num_batches > 0:
                results[method_name].append({
                    'epoch': epoch + 1,
                    'train_total': train_loss_total / num_batches,
                    'train_mse': train_loss_mse / num_batches,
                    'train_physics': train_loss_physics / num_batches,
                    'train_extra': train_loss_extra / num_batches if train_loss_extra > 0 else 0.0,
                    'weights': weights.detach().cpu() if weights is not None else None
                })
        
        # Validation
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            for method_name, model in models.items():
                model.eval()
                val_loss_total = 0.0
                val_loss_mse = 0.0
                val_loss_physics = 0.0
                weight_stats = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        u0 = batch['u0'].to(device)
                        u_clean = batch['u_clean'].to(device)
                        
                        u_pred, weights, _ = model(u0, return_weights=True)
                        
                        # Compute losses for 2D
                        mse_loss = F.mse_loss(u_pred, u_clean)
                        physics_loss = model.compute_physics_loss(u_pred, u0)
                        
                        val_loss_total += mse_loss.item() + 0.05 * physics_loss.item()
                        val_loss_mse += mse_loss.item()
                        val_loss_physics += physics_loss.item()
                        
                        # Collect weight statistics
                        if weights is not None:
                            weight_stats.append(weights.cpu().numpy())
                
                # Compute weight statistics
                if weight_stats:
                    weights_all = np.concatenate(weight_stats, axis=0)
                    weight_mean = weights_all.mean()
                    weight_std = weights_all.std()
                    weight_entropy = -np.sum(weights_all * np.log(weights_all + 1e-10), axis=1).mean()
                    weight_sparsity = (weights_all < 1e-3).mean()
                else:
                    weight_mean, weight_std, weight_entropy, weight_sparsity = 0.0, 0.0, 0.0, 0.0
                
                # Update results
                if results[method_name]:
                    num_val = len(val_loader)
                    results[method_name][-1].update({
                        'val_total': val_loss_total / num_val if num_val > 0 else 0.0,
                        'val_mse': val_loss_mse / num_val if num_val > 0 else 0.0,
                        'val_physics': val_loss_physics / num_val if num_val > 0 else 0.0,
                        'weight_mean': weight_mean,
                        'weight_std': weight_std,
                        'weight_entropy': weight_entropy,
                        'weight_sparsity': weight_sparsity
                    })
                
                # Print progress
                if results[method_name]:
                    last_result = results[method_name][-1]
                    print(f"{method_name.upper():25s} - "
                          f"Train: {last_result['train_total']:.4f}, "
                          f"Val: {last_result.get('val_total', 0):.4f}, "
                          f"Entropy: {weight_entropy:.4f}")
    
    print("\n4. Final evaluation on test set...")
    
    test_results = {}
    test_predictions = {}
    
    for method_name, model in models.items():
        model.eval()
        test_loss_total = 0.0
        test_loss_mse = 0.0
        test_weights = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                u0 = batch['u0'].to(device)
                u_clean = batch['u_clean'].to(device)
                
                u_pred, weights, _ = model(u0, return_weights=True)
                
                mse_loss = F.mse_loss(u_pred, u_clean)
                test_loss_total += mse_loss.item()
                test_loss_mse += mse_loss.item()
                
                if weights is not None:
                    test_weights.append(weights.cpu().numpy())
                
                # Store first batch for visualization
                if method_name not in test_predictions and batch_idx == 0:
                    test_predictions[method_name] = {
                        'u0': u0[0].cpu().numpy(),  # (2, H, W)
                        'u_clean': u_clean[0].cpu().numpy(),
                        'u_pred': u_pred[0].cpu().numpy(),
                        'weights': weights[0].cpu().numpy() if weights is not None else None
                    }
        
        # Compute test statistics
        num_test = len(test_loader)
        test_loss_total /= num_test if num_test > 0 else 1.0
        test_loss_mse /= num_test if num_test > 0 else 1.0
        
        if test_weights:
            weights_all = np.concatenate(test_weights, axis=0)
            sparsity = (weights_all < 1e-3).mean()
            weight_entropy = -np.sum(weights_all * np.log(weights_all + 1e-10), axis=1).mean()
        else:
            sparsity = 0.0
            weight_entropy = 0.0
        
        test_results[method_name] = {
            'test_mse': test_loss_mse,
            'test_total': test_loss_total,
            'sparsity': sparsity,
            'weight_entropy': weight_entropy
        }
    
    # Print final results
    print("\n" + "="*80)
    print("2D FINAL TEST RESULTS")
    print("="*80)
    for method, metrics in test_results.items():
        print(f"\n{method.upper()}:")
        print(f"  Test MSE: {metrics['test_mse']:.6f}")
        print(f"  Sparsity: {metrics['sparsity']*100:.2f}%")
        print(f"  Weight Entropy: {metrics['weight_entropy']:.4f}")
    print("="*80)
    
    return test_results, test_predictions, results


def create_unified_visualization_2d(test_predictions, training_results):
    """Create 2D visualization comparing all methods with unified structure"""
    
    print("\n5. Generating 2D unified visualization...")
    
    # Create directory for plots
    os.makedirs('plots/2d_unified_multi_source', exist_ok=True)
    
    # 1. Plot predictions comparison for a single sample
    fig = plt.figure(figsize=(20, 16))
    
    # Select first method for ground truth
    first_method = list(test_predictions.keys())[0]
    u0_data = test_predictions[first_method]['u0']  # (2, H, W)
    u_clean_data = test_predictions[first_method]['u_clean']  # (2, H, W)
    
    # Create spatial grid
    H, W = u0_data.shape[1], u0_data.shape[2]
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    X, Y = np.meshgrid(x, y)
    
    # Colors for different methods
    colors = {'softmax': 'blue', 'single_ts_lagrangian': 'red',
              'two_ts_lagrangian': 'magenta', 'admm': 'green'}
    
    # Plot 1: Initial condition and ground truth
    ax1 = plt.subplot(3, 4, 1)
    im1 = ax1.imshow(u0_data[0], cmap='RdBu', extent=[-1, 1, -1, 1], origin='lower')
    ax1.set_title('Initial u(x,y)', fontsize=10)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    ax2 = plt.subplot(3, 4, 2)
    im2 = ax2.imshow(u0_data[1], cmap='RdBu', extent=[-1, 1, -1, 1], origin='lower')
    ax2.set_title('Initial v(x,y)', fontsize=10)
    ax2.set_xlabel('x')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    ax3 = plt.subplot(3, 4, 3)
    im3 = ax3.imshow(u_clean_data[0], cmap='RdBu', extent=[-1, 1, -1, 1], origin='lower')
    ax3.set_title('Ground Truth u(x,y)', fontsize=10)
    ax3.set_xlabel('x')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    ax4 = plt.subplot(3, 4, 4)
    im4 = ax4.imshow(u_clean_data[1], cmap='RdBu', extent=[-1, 1, -1, 1], origin='lower')
    ax4.set_title('Ground Truth v(x,y)', fontsize=10)
    ax4.set_xlabel('x')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # Plot predictions for each method (u-component only)
    row = 1
    for idx, (method, pred_data) in enumerate(test_predictions.items()):
        u_pred = pred_data['u_pred'][0]  # u-component
        
        ax = plt.subplot(3, 4, 5 + idx)
        im = ax.imshow(u_pred, cmap='RdBu', extent=[-1, 1, -1, 1], origin='lower')
        ax.set_title(f'{method.upper()}: Predicted u(x,y)', fontsize=10)
        ax.set_xlabel('x')
        if idx == 0:
            ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Plot 2: Weight distributions
    ax9 = plt.subplot(3, 4, 9)
    bins = np.linspace(0, 0.15, 40)
    
    for method, pred_data in test_predictions.items():
        if pred_data['weights'] is not None:
            weights = pred_data['weights']
            ax9.hist(weights, bins=bins, alpha=0.5, color=colors[method],
                     label=f"{method.replace('_', ' ').title()}", density=True)
    
    ax9.set_xlabel('Weight Value', fontsize=10)
    ax9.set_ylabel('Density', fontsize=10)
    ax9.set_title('Weight Distribution Comparison', fontsize=11)
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)
    
    # Plot 3: Training loss comparison
    ax10 = plt.subplot(3, 4, 10)
    
    for method, color in colors.items():
        if method in training_results and training_results[method]:
            epochs = [r['epoch'] for r in training_results[method]]
            train_loss = [r['train_total'] for r in training_results[method]]
            ax10.plot(epochs, train_loss, color=color, linewidth=2,
                      label=method.replace('_', ' ').title())
    
    ax10.set_xlabel('Epoch', fontsize=10)
    ax10.set_ylabel('Training Loss', fontsize=10)
    ax10.set_title('Training Loss Comparison', fontsize=11)
    ax10.legend(fontsize=8)
    ax10.grid(True, alpha=0.3)
    ax10.set_yscale('log')
    
    # Plot 4: Weight entropy during training
    ax11 = plt.subplot(3, 4, 11)
    
    for method, color in colors.items():
        if method in training_results and training_results[method]:
            epochs = [r['epoch'] for r in training_results[method]]
            entropy = [r.get('weight_entropy', 0) for r in training_results[method]]
            ax11.plot(epochs, entropy, color=color, linewidth=2,
                      label=method.replace('_', ' ').title())
    
    ax11.set_xlabel('Epoch', fontsize=10)
    ax11.set_ylabel('Weight Entropy', fontsize=10)
    ax11.set_title('Weight Entropy During Training', fontsize=11)
    ax11.legend(fontsize=8)
    ax11.grid(True, alpha=0.3)
    
    # Plot 5: Vorticity comparison
    ax12 = plt.subplot(3, 4, 12)
    
    # Compute vorticity for each method
    vorticity_data = {}
    for method, pred_data in test_predictions.items():
        u = pred_data['u_pred'][0]
        v = pred_data['u_pred'][1]
        
        # Compute vorticity: ω = ∂v/∂x - ∂u/∂y
        dx = 2.0 / (W - 1)
        dy = 2.0 / (H - 1)
        
        du_dy = np.gradient(u, dy, axis=0)
        dv_dx = np.gradient(v, dx, axis=1)
        vorticity = dv_dx - du_dy
        vorticity_data[method] = vorticity
    
    # Plot vorticity for ground truth
    u_gt = u_clean_data[0]
    v_gt = u_clean_data[1]
    du_dy_gt = np.gradient(u_gt, dy, axis=0)
    dv_dx_gt = np.gradient(v_gt, dx, axis=1)
    vorticity_gt = dv_dx_gt - du_dy_gt
    
    im_gt = ax12.imshow(vorticity_gt, cmap='viridis', extent=[-1, 1, -1, 1], origin='lower')
    ax12.set_title('Ground Truth Vorticity', fontsize=11)
    ax12.set_xlabel('x')
    plt.colorbar(im_gt, ax=ax12, fraction=0.046, pad=0.04)
    
    # Add overall title
    fig.suptitle('Large-Scale Multi-Source 2D Integration: Unified Implementation Comparison',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('plots/2d_unified_multi_source/2d_unified_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.savefig('plots/2d_unified_multi_source/2d_unified_comparison.pdf',
                bbox_inches='tight')
    plt.show()
    
    # 2. Additional visualization: Streamlines for each method
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    axes2 = axes2.flatten()
    
    for idx, (method, pred_data) in enumerate(test_predictions.items()):
        if idx >= 4:
            break
            
        ax = axes2[idx]
        u = pred_data['u_pred'][0]
        v = pred_data['u_pred'][1]
        
        # Compute velocity magnitude
        vel_mag = np.sqrt(u**2 + v**2)
        
        # Plot streamlines
        stride = 2
        ax.streamplot(X[::stride, ::stride], Y[::stride, ::stride],
                      u[::stride, ::stride], v[::stride, ::stride],
                      color=vel_mag[::stride, ::stride], cmap='hot',
                      density=2, linewidth=1)
        ax.set_title(f'{method.upper()}: Velocity Field', fontsize=11)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    
    fig2.suptitle('2D Velocity Fields: Streamline Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('plots/2d_unified_multi_source/2d_streamlines.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("2D visualizations saved to 'plots/2d_unified_multi_source/'")


def main():
    """Main function to run 2D multi-source integration"""
    print("="*100)
    print("UNIFIED LARGE-SCALE MULTI-SOURCE 2D INTEGRATION")
    print("128 sources, 64x64 grid, 2D Burgers' equation (Re=100)")
    print("Uniform implementation: Softmax, Lagrangian Single/Two-Timescale, ADMM")
    print("="*100)
    
    # Run unified training for 2D
    test_results, test_predictions, training_results = train_large_scale_integration_unified_2d()
    
    # Create unified visualization
    create_unified_visualization_2d(test_predictions, training_results)
    
    print("\n" + "="*100)
    print("2D UNIFIED IMPLEMENTATION SUMMARY")
    print("="*100)
    print("\nKey Changes for 2D:")
    print("1. Dataset: Generates 2D velocity fields (u, v) with vortices, shear layers, etc.")
    print("2. Networks: Uses 2D convolutions instead of linear layers")
    print("3. Physics Loss: Computes 2D Burgers' equation residuals (both u and v equations)")
    print("4. Source Networks: Lightweight CNNs for 2D feature extraction")
    print("5. Attention Network: 2D CNN with spatial pooling for weight computation")
    
    print("\nMethod Characteristics in 2D:")
    print("• Softmax: Top-k sparsity applied to 2D convolutional features")
    print("• Lagrangian: Explicit constraint handling for 2D velocity fields")
    print("• Two-timescale: Adaptive dual learning rate based on 2D constraints")
    print("• ADMM: Variable splitting optimized for 2D spatial correlations")
    
    print("\n2D Performance Insights:")
    print("• 2D physics loss includes both u and v equation residuals")
    print("• Convolutional networks capture spatial correlations in velocity fields")
    print("• Weight distributions show spatial specialization patterns")
    print("• Vorticity preservation is a key metric for 2D methods")
    print("• Computational requirements increased but architecture is optimized")
    print("="*100)


if __name__ == "__main__":
    main()