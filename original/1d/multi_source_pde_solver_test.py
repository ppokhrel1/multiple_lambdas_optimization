"""
Four-way comparison for 1-D PDE solver ensemble
------------------------------------------------
1. Softmax router
2. Single-time-scale Lagrangian
3. Two-time-scale Lagrangian
4. ADMM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import os
from matplotlib.lines import Line2D
from scipy.fft import fft, ifft  # Added import

# ------------------------------------------------------------------
# 0. 1D Fourier layer
# ------------------------------------------------------------------
class FourierBlock1D(nn.Module):
    """1-D Fourier layer"""
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )
        self.w = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFT
        x_ft = torch.fft.rfft(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            x.shape[0], self.w.out_channels, x.shape[-1] // 2 + 1,
            device=x.device, dtype=torch.cfloat
        )
        
        modes = min(self.weights.shape[2], x_ft.shape[-1])
        out_ft[:, :, :modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :modes], self.weights[:, :, :modes]
        )
        
        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.shape[-1])
        return x + self.w(x)

# ------------------------------------------------------------------
# 1. 1D Burgers' Equation Dataset
# ------------------------------------------------------------------
class BurgersEquationDataset(Dataset):
    """1D Burgers' equation with varying parameters"""
    def __init__(self, n_samples: int, domain_size: int = 128, 
                 dt: float = 0.001, n_steps: int = 100, 
                 nu_range: Tuple[float, float] = (0.001, 0.1),
                 noise_level: float = 0.01, seed: int = 42):
        torch.manual_seed(seed)
        self.domain_size = domain_size
        self.dt = dt
        self.dx = 2.0 * np.pi / domain_size
        self.nu_range = nu_range
        self.samples = []
        
        x = torch.linspace(0, 2 * np.pi, domain_size)
        
        for i in range(n_samples):
            # Random viscosity coefficient
            nu = torch.empty(1).uniform_(*nu_range).item()
            
            # Random initial condition (combination of sinusoids)
            u0 = self.generate_initial_condition(x)
            
            # Solve Burgers' equation
            solution = self.solve_burgers(u0, nu, n_steps)
            
            # Add noise
            solution = solution + noise_level * torch.randn_like(solution)
            
            # Store grid and solution
            self.samples.append({
                'grid': x.unsqueeze(0),  # [1, domain_size]
                'solution': solution[-1].unsqueeze(0),  # [1, domain_size]
                'parameters': {'nu': nu, 'domain_size': domain_size}
            })
    
    def generate_initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """Generate random initial condition for Burgers' equation"""
        n_components = torch.randint(1, 5, (1,)).item()
        u0 = torch.zeros_like(x)
        
        for _ in range(n_components):
            k = torch.randint(1, 6, (1,)).item()
            a = torch.randn(1).item() * 0.5
            phi = torch.rand(1).item() * 2 * np.pi
            u0 += a * torch.sin(k * x + phi)
        
        return u0
    
    def solve_burgers(self, u0: torch.Tensor, nu: float, n_steps: int) -> torch.Tensor:
        """Solve 1D Burgers' equation using spectral method"""
        u = u0.numpy()
        n = len(u)
        k = np.fft.fftfreq(n) * n
        
        # Time-stepping using integrating factor method
        solutions = [torch.from_numpy(u).float()]
        
        for _ in range(n_steps):
            u_hat = fft(u)
            
            # Nonlinear term (pseudo-spectral)
            u_squared = 0.5 * u**2
            u_squared_hat = fft(u_squared)
            
            # Time step in Fourier space
            u_hat = u_hat - self.dt * (1j * k * u_squared_hat) - nu * self.dt * (k**2 * u_hat)
            
            # Inverse transform
            u = np.real(ifft(u_hat))
            solutions.append(torch.from_numpy(u).float())
        
        return torch.stack(solutions)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ------------------------------------------------------------------
# 2. 1D Solver Definitions
# ------------------------------------------------------------------
class FourierPDESolver1D(nn.Module):
    """1D Fourier Neural Operator"""
    def __init__(self, domain_size: int, modes: int = 16, width: int = 16):
        super().__init__()
        self.lift = nn.Sequential(
            nn.Conv1d(1, width, 1), nn.GELU(), nn.Conv1d(width, width, 1)
        )
        self.fourier_layers = nn.ModuleList([
            FourierBlock1D(width, width, modes) for _ in range(4)
        ])
        self.project = nn.Sequential(
            nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for layer in self.fourier_layers:
            x = layer(x)
            x = F.gelu(x)
        return self.project(x)

class WENOSolver1D(nn.Module):
    """1D WENO-like solver with shock detection"""
    def __init__(self, domain_size: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, 1)
        )
        self.shock_detector = nn.Sequential(
            nn.Conv1d(2, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, 1), nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.net(x)
        u_x = self.compute_gradient(u)
        shock_weights = self.shock_detector(torch.cat([x, u_x], dim=1))
        return u * shock_weights
    
    def compute_gradient(self, u: torch.Tensor) -> torch.Tensor:
        u_x = F.pad(u[:, :, 1:] - u[:, :, :-1], (0, 1))
        return u_x

class DeepONetSolver1D(nn.Module):
    """1D DeepONet"""
    def __init__(self, domain_size: int, hidden_dim: int = 64, branch_dim: int = 40):
        super().__init__()
        self.branch_net = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden_dim, branch_dim, 1)
        )
        
        # Create coordinate grid
        x = torch.linspace(0, 2 * np.pi, domain_size)
        self.register_buffer('coords', x.unsqueeze(0).unsqueeze(0))  # [1, 1, domain_size]
        
        self.trunk_net = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, branch_dim)
        )
        
        self.projection = nn.Conv1d(branch_dim, 1, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = self.branch_net(x)
        
        # Process coordinates
        coords_expanded = self.coords.expand(x.shape[0], -1, -1)
        trunk = self.trunk_net(coords_expanded.permute(0, 2, 1))
        trunk = trunk.permute(0, 2, 1)
        
        output = branch * trunk
        return self.projection(output)

class MultiResolutionSolver1D(nn.Module):
    """1D Multi-resolution solver"""
    def __init__(self, domain_size: int, scales: List[int] = None):
        super().__init__()
        scales = scales or [1, 2, 4, 8]
        self.scales = scales
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 64, 3, padding=1), nn.ReLU(),
                nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
                nn.Conv1d(64, 1, 1)
            ) for _ in scales
        ])
        self.fusion = nn.Sequential(
            nn.Conv1d(len(scales), 64, 1), nn.ReLU(),
            nn.Conv1d(64, 1, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for scale, net in zip(self.scales, self.nets):
            if scale > 1:
                # Downsample
                x_down = F.avg_pool1d(x, scale)
                o = net(x_down)
                # Upsample back to original size
                o = F.interpolate(o, size=x.shape[-1], mode='linear', align_corners=True)
            else:
                o = net(x)
            outs.append(o)
        
        return self.fusion(torch.cat(outs, dim=1))

# ------------------------------------------------------------------
# 3. Router Definitions (Adapted for 1D)
# ------------------------------------------------------------------
class AdaptiveRouter1D(nn.Module):
    """1D Adaptive router"""
    def __init__(self, domain_size: int, n_solvers: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16), nn.Flatten(),
            nn.Linear(16 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_solvers)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        logits = self.net(x)
        weights = F.softmax(logits, dim=-1)
        return weights, {'logits': logits, 'weights': weights}

# ------------------------------------------------------------------
# 4. Multi-solver System
# ------------------------------------------------------------------
class MultiSolverSystem1D(nn.Module):
    """1D version of multi-solver system"""
    def __init__(self, domain_size: int, solvers: List[nn.Module] = None, 
                 hidden_dim: int = 64, use_lagrangian: bool = True, rho: float = 1.0):
        super().__init__()
        self.solvers = nn.ModuleList(solvers) if solvers is not None else nn.ModuleList([
            FourierPDESolver1D(domain_size),
            WENOSolver1D(domain_size),
            DeepONetSolver1D(domain_size),
            MultiResolutionSolver1D(domain_size)
        ])
        self.n_solvers = len(self.solvers)
        self.use_lagrangian = use_lagrangian
        self.rho = rho
        
        if use_lagrangian:
            self.lambda_weights = nn.Parameter(torch.ones(self.n_solvers) / self.n_solvers)
            self.mu = nn.Parameter(torch.zeros(1))
            self.nu = nn.Parameter(torch.zeros(self.n_solvers))
        else:
            self.router = AdaptiveRouter1D(domain_size, self.n_solvers, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, domain_size]
        
        outs = []
        for solver in self.solvers:
            o = solver(x)
            outs.append(o)
        
        outs = torch.stack(outs, dim=1)  # [B, n_solvers, 1, domain_size]
        
        if self.use_lagrangian:
            w = self.lambda_weights.softmax(0).view(1, -1, 1, 1)
        else:
            w, router_info = self.router(x)
            w = w.view(-1, self.n_solvers, 1, 1)
        
        combined = (outs * w).sum(dim=1)  # [B, 1, domain_size]
        
        return combined, {
            'solver_outputs': outs,
            'weights': w.squeeze(-1).squeeze(-1),
            'lambda_weights': self.lambda_weights if self.use_lagrangian else None,
            'combined_output': combined
        }
    
    def compute_loss(self, x: torch.Tensor, target: torch.Tensor, meta: Dict) -> Tuple[torch.Tensor, Dict]:
        if self.use_lagrangian:
            w = self.lambda_weights.softmax(0).view(1, -1, 1, 1)
            combined_output = (meta['solver_outputs'] * w).sum(dim=1)
            recon = F.mse_loss(combined_output, target)
            
            # Constraint terms
            g = 1 - self.lambda_weights.sum()
            h = -self.lambda_weights
            constraint_penalty = self.mu * g + (self.nu * h).sum() + (self.rho / 2) * (g ** 2 + (torch.relu(h) ** 2).sum())
            loss = recon + constraint_penalty
        else:
            w = meta['weights'].view(-1, self.n_solvers, 1, 1)
            combined_output = (meta['solver_outputs'] * w).sum(dim=1)
            recon = F.mse_loss(combined_output, target)
            sparsity = 0.1 * (1 - (meta['weights'] > 0.1).float().mean())
            loss = recon + sparsity
        
        return loss, {
            'loss': loss,
            'recon_loss': recon.item(),
            'total_loss': loss.item(),
            'weight_sum': self.lambda_weights.sum().item() if self.use_lagrangian else meta['weights'].sum().item(),
            'weight_mean': self.lambda_weights.mean().item() if self.use_lagrangian else meta['weights'].mean().item()
        }

# ------------------------------------------------------------------
# 5. ADMM Router (1D)
# ------------------------------------------------------------------
class ADMMRouter1D(nn.Module):
    """1D ADMM router"""
    def __init__(self, domain_size: int, solvers: List[nn.Module] = None, rho: float = 0.1):
        super().__init__()
        self.solvers = nn.ModuleList(solvers) if solvers else nn.ModuleList([
            FourierPDESolver1D(domain_size),
            WENOSolver1D(domain_size),
            DeepONetSolver1D(domain_size),
            MultiResolutionSolver1D(domain_size)
        ])
        self.n_solvers = len(self.solvers)
        self.rho = rho
        
        # Global consensus variables
        self.z = nn.Parameter(torch.ones(self.n_solvers) / self.n_solvers)
        self.u = nn.Parameter(torch.zeros(self.n_solvers))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        outs = torch.stack([solver(x) for solver in self.solvers], dim=1)
        w = F.softmax(self.z, dim=0).view(1, -1, 1, 1)
        combined = (outs * w).sum(dim=1)
        
        return combined, {
            'solver_outputs': outs,
            'weights': F.softmax(self.z, dim=0),
            'z': self.z,
            'u': self.u
        }
    
    def compute_loss(self, x: torch.Tensor, target: torch.Tensor, meta: Dict) -> Tuple[torch.Tensor, Dict]:
        return self.admm_loss(x, target, meta['solver_outputs'])
    
    def admm_loss(self, x: torch.Tensor, target: torch.Tensor, 
                  solver_outputs: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        w = F.softmax(self.z, dim=0).view(1, -1, 1, 1)
        combined_output = (solver_outputs * w).sum(dim=1)
        recon = F.mse_loss(combined_output, target)
        
        consensus = 0.5 * self.rho * self.z.pow(2).sum()
        dual = (self.u * (F.softmax(self.z, dim=0) - 0.25 * torch.ones_like(self.z))).sum()
        
        total_loss = recon + consensus + dual
        
        return total_loss, {
            'loss': total_loss,
            'total_loss': total_loss.item(),  # FIXED: Added total_loss key
            'recon_loss': recon.item(),
            'consensus': consensus.item(),
            'weights': F.softmax(self.z, dim=0).detach().cpu().numpy()
        }

# ------------------------------------------------------------------
# 6. Optimizers (Same logic, now for 1D)
# ------------------------------------------------------------------
class SingleTimeScaleOptimizer1D:
    def __init__(self, model: MultiSolverSystem1D, lr: float = 1e-4, rho: float = 10.0):
        self.model = model
        self.rho = rho
        
        self.theta_opt = torch.optim.Adam([
            {'params': [p for n, p in model.named_parameters() 
                       if 'lambda' not in n and 'mu' not in n and 'nu' not in n], 
             'lr': lr},
            {'params': [model.lambda_weights], 'lr': lr * 5},
            {'params': [model.mu, model.nu], 'lr': lr * 5}
        ])
    
    def zero_grad(self):
        self.theta_opt.zero_grad()
    
    def step(self, loss_dict):
        loss_dict['loss'].backward()
        self.theta_opt.step()
        
        with torch.no_grad():
            g = 1 - self.model.lambda_weights.sum()
            h = -self.model.lambda_weights
            
            self.model.mu.add_(self.rho * g)
            self.model.nu.add_(self.rho * torch.relu(h))
            
            self.model.lambda_weights.data = self.project_simplex(self.model.lambda_weights.data)
    
    @staticmethod
    def project_simplex(v: torch.Tensor) -> torch.Tensor:
        v_sorted, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(v_sorted, 0)
        rho = (cssv - 1) / torch.arange(1, len(v) + 1, device=v.device, dtype=v.dtype)
        idx = torch.where(v_sorted > rho)[0]
        rho_star = rho[idx[-1]] if idx.numel() else 0
        return torch.maximum(v - rho_star, torch.zeros_like(v))

class TwoTimeScaleOptimizer1D:
    def __init__(self, model: MultiSolverSystem1D, lr_theta: float = 1e-3, 
                 lr_lambda: float = 1e-4, rho: float = 10.0):
        self.model = model
        self.rho = rho
        
        solver_params = [p for n, p in model.named_parameters() 
                        if n not in ['lambda_weights', 'nu', 'mu']]
        self.theta_opt = torch.optim.Adam(solver_params, lr=lr_theta)
        self.lambda_opt = torch.optim.SGD([model.lambda_weights, model.nu, model.mu], lr=lr_lambda)
    
    def zero_grad(self):
        self.theta_opt.zero_grad()
        self.lambda_opt.zero_grad()
    
    def step(self, loss_dict):
        loss_dict['loss'].backward()
        self.theta_opt.step()
        self.lambda_opt.step()
        
        with torch.no_grad():
            g = 1 - self.model.lambda_weights.sum()
            h = -self.model.lambda_weights
            
            self.model.mu.add_(self.rho * g)
            self.model.nu.add_(self.rho * torch.relu(h))
            
            self.model.lambda_weights.data = self.project_simplex(self.model.lambda_weights.data)
    
    @staticmethod
    def project_simplex(v: torch.Tensor) -> torch.Tensor:
        v_sorted, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(v_sorted, 0)
        rho = (cssv - 1) / torch.arange(1, len(v) + 1, device=v.device, dtype=v.dtype)
        idx = torch.where(v_sorted > rho)[0]
        rho_star = rho[idx[-1]] if idx.numel() else torch.tensor(0.0, device=v.device, dtype=v.dtype)
        return torch.maximum(v - rho_star, torch.zeros_like(v))

class ADMMOptimizer1D:
    def __init__(self, model: ADMMRouter1D, lr: float = 1e-4, rho: float = 0.1):
        self.model = model
        self.rho = rho
        solver_params = [p for n, p in model.named_parameters()]
        self.theta_opt = torch.optim.Adam(solver_params, lr=lr)
    
    def zero_grad(self):
        self.theta_opt.zero_grad()
    
    def step(self, loss_dict):
        self.theta_opt.zero_grad()
        loss_dict['loss'].backward(retain_graph=True)
        self.theta_opt.step()
        
        with torch.no_grad():
            z_sm = F.softmax(self.model.z, dim=0)
            self.model.z.copy_(self.model.z + self.rho * (z_sm - 0.25))
        
        with torch.no_grad():
            z_sm = F.softmax(self.model.z, dim=0)
            self.model.u.add_(self.rho * (z_sm - 0.25 * torch.ones_like(z_sm)))

# ------------------------------------------------------------------
# 7. Trainer (Adapted for 1D)
# ------------------------------------------------------------------
class Trainer1D:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu', optimizer=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optimizer
        self.metrics = defaultdict(list)
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        epoch_metrics = defaultdict(list)
        for batch in self.train_loader:
            m = self.train_step(batch)
            for k, v in m.items():
                epoch_metrics[k].append(v)
        return {k: np.mean([x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in v]) 
                for k, v in epoch_metrics.items()}
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        grid = batch['grid'].to(self.device)
        target = batch['solution'].to(self.device)
        output, meta = self.model(grid)
        
        if isinstance(self.model, ADMMRouter1D):
            loss, loss_meta = self.model.admm_loss(grid, target, meta['solver_outputs'])
        else:
            loss, loss_meta = self.model.compute_loss(grid, target, meta)
        
        self.optimizer.zero_grad()
        if isinstance(self.optimizer, (ADMMOptimizer1D, TwoTimeScaleOptimizer1D, SingleTimeScaleOptimizer1D)):
            self.optimizer.step(loss_meta)
        else:
            loss.backward()
            self.optimizer.step()
        
        # FIX: Handle missing keys in loss_meta
        return {
            'total_loss': loss_meta.get('total_loss', loss_meta.get('loss', torch.tensor(0.0)).item()),
            'recon_loss': loss_meta.get('recon_loss', 0.0),
            'weight_mean': meta['weights'].mean().item() if 'weights' in meta else 0.0,
            'weight_std': meta['weights'].std().item() if 'weights' in meta else 0.0
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        val_metrics = defaultdict(list)
        for batch in self.val_loader:
            grid = batch['grid'].to(self.device)
            target = batch['solution'].to(self.device)
            output, meta = self.model(grid)
            
            if isinstance(self.model, ADMMRouter1D):
                loss, loss_meta = self.model.admm_loss(grid, target, meta['solver_outputs'])
            else:
                loss, loss_meta = self.model.compute_loss(grid, target, meta)
            
            # FIX: Collect only available metrics
            for k, v in loss_meta.items():
                if isinstance(v, (int, float, np.number)):
                    val_metrics[k].append(v)
        
        return {k: np.mean(v) for k, v in val_metrics.items()}

# ------------------------------------------------------------------
# 8. Plotting Utilities (Adapted for 1D)
# ------------------------------------------------------------------
def plot_error_distribution_1d(models: Dict[str, nn.Module], test_loader: DataLoader, 
                               device: str, save_path: str, epoch: int):
    """Plot error distribution for 1D models"""
    errors = {n: [] for n in models}
    
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                grid = batch['grid'].to(device)
                target = batch['solution'].to(device)
                pred, _ = model(grid)
                errors[name].extend(torch.abs(pred - target).mean(dim=(1, 2)).cpu().numpy())
    
    plt.figure(figsize=(10, 6))
    for name, err in errors.items():
        n_bins = min(50, len(err) // 2)
        plt.hist(err, bins=n_bins, alpha=0.5, label=name)
    
    plt.title('Error Distribution (1D)')
    plt.xlabel('Mean Absolute Error')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/error_dist_1d_epoch_{epoch}.png')
    plt.close()

def plot_solution_comparison_1d(models: Dict[str, nn.Module], sample_batch: Dict[str, torch.Tensor],
                               device: str, save_path: str, epoch: int):
    """Plot 1D solution comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, (name, model) in enumerate(models.items()):
        if idx >= 4:
            break
        
        model.eval()
        with torch.no_grad():
            grid = sample_batch['grid'][:1].to(device)
            target = sample_batch['solution'][:1].to(device)
            pred, meta = model(grid)
            
            x = grid[0, 0].cpu().numpy()
            target_np = target[0, 0].cpu().numpy()
            pred_np = pred[0, 0].cpu().numpy()
            
            ax = axes[idx]
            ax.plot(x, target_np, 'k-', label='True', linewidth=2)
            ax.plot(x, pred_np, 'r--', label='Predicted', linewidth=1.5)
            ax.fill_between(x, target_np, pred_np, alpha=0.3)
            
            ax.set_title(f'{name}')
            ax.set_xlabel('x')
            ax.set_ylabel('u(x)')
            ax.legend()
            ax.grid(True)
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/solution_comparison_1d_epoch_{epoch}.png')
    plt.close()

def plot_loss_curves_1d(history: Dict[str, Dict[str, List]], save_path: str, plot_type: str = 'total'):
    """Plot training curves for 1D models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    loss_key = 'train_loss' if plot_type == 'total' else 'train_recon_loss'
    title_suffix = 'Total' if plot_type == 'total' else 'Reconstruction'
    
    # Training Loss
    for name, hist in history.items():
        epochs = range(1, len(hist[loss_key]) + 1)
        ax1.plot(epochs, hist[loss_key], label=f'{name} (Train)', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title(f'{title_suffix} Training Loss (1D)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_yscale('log')
    
    loss_key_val = 'val_loss' if plot_type == 'total' else 'val_recon_loss'
    # Validation Loss
    for name, hist in history.items():
        epochs = range(1, len(hist[loss_key_val]) + 1)
        ax2.plot(epochs, hist[loss_key_val], label=f'{name} (Val)', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title(f'{title_suffix} Validation Loss (1D)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/loss_evolution_1d_{plot_type}.png')
    plt.close()

def plot_weight_evolution_1d(history: Dict[str, Dict[str, List]], 
                            solver_names: List[str], save_path: str):
    """Plot weight evolution for 1D solvers"""
    n_methods = len(history)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, hist) in enumerate(history.items()):
        ax = axes[idx]
        weights_history = hist['weights']
        epochs = range(1, len(weights_history) + 1)
        
        if len(weights_history) == 0 or weights_history[0] is None:
            ax.text(0.5, 0.5, 'No weight data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{name} - Weights')
            continue
        
        # Convert weights to numpy array
        weights_array = []
        for w in weights_history:
            if isinstance(w, (list, np.ndarray)):
                weights_array.append(w)
            elif torch.is_tensor(w):
                weights_array.append(w.detach().cpu().numpy())
            else:
                weights_array.append(np.array(w))
        
        weights_array = np.array(weights_array)
        
        for i, solver_name in enumerate(solver_names):
            if i < weights_array.shape[1]:
                ax.plot(epochs, weights_array[:, i], label=solver_name, linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight')
        ax.set_title(f'{name} - Solver Weights (1D)')
        ax.legend()
        ax.grid(True)
    
    for i in range(len(history), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/weight_evolution_1d.png')
    plt.close()

# ------------------------------------------------------------------
# 9. Main Function
# ------------------------------------------------------------------
def main_1d():
    n_samples = 100  # Reduced for faster training
    domain_size = 128
    batch_size = 16
    n_epochs = 100  # Reduced epochs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'results_1d'
    
    print(f"Using device: {device}")
    print(f"Domain size: {domain_size}")
    
    # Create datasets
    train_dataset = BurgersEquationDataset(n_samples, domain_size)
    val_dataset = BurgersEquationDataset(n_samples // 5, domain_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"Dataset created: {len(train_dataset)} training samples")
    
    # Create models
    models = {
        'softmax': MultiSolverSystem1D(domain_size, use_lagrangian=False),
        'single_time_lr': MultiSolverSystem1D(domain_size, use_lagrangian=True),
        'two_time_lr': MultiSolverSystem1D(domain_size, use_lagrangian=True),
        'admm': ADMMRouter1D(domain_size)
    }
    
    # Create trainers with appropriate optimizers
    trainers = {}
    for name, model in models.items():
        if name == 'admm':
            opt = ADMMOptimizer1D(model, lr=1e-4)
            trainers[name] = Trainer1D(model, train_loader, val_loader, device=device, optimizer=opt)
        elif name == 'two_time_lr':
            opt = TwoTimeScaleOptimizer1D(model, lr_theta=1e-4, lr_lambda=1e-4)
            trainers[name] = Trainer1D(model, train_loader, val_loader, device=device, optimizer=opt)
        elif name == 'single_time_lr':
            opt = SingleTimeScaleOptimizer1D(model, lr=1e-4)
            trainers[name] = Trainer1D(model, train_loader, val_loader, device=device, optimizer=opt)
        else:  # softmax
            opt = torch.optim.Adam(model.parameters(), lr=1e-4)
            trainers[name] = Trainer1D(model, train_loader, val_loader, device=device, optimizer=opt)
    
    # Training history
    training_history = {name: {
        'train_loss': [], 
        'val_loss': [], 
        'train_recon_loss': [],
        'val_recon_loss': [],
        'weights': []
    } for name in models}
    
    solver_names = ['Fourier', 'WENO', 'DeepONet', 'MultiRes']
    
    # Get a sample for visualization
    sample_batch = next(iter(val_loader))
    
    # Training loop
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        for name, trainer in trainers.items():
            train_m = trainer.train_epoch()
            val_m = trainer.validate()
            
            # Store history - use get() with defaults for missing keys
            training_history[name]['train_loss'].append(train_m.get('total_loss', 0.0))
            training_history[name]['val_loss'].append(val_m.get('loss', val_m.get('total_loss', 0.0)))
            training_history[name]['train_recon_loss'].append(train_m.get('recon_loss', 0.0))
            training_history[name]['val_recon_loss'].append(val_m.get('recon_loss', 0.0))
            
            # Extract weights
            model = models[name]
            if isinstance(model, ADMMRouter1D):
                weights = model.z.softmax(0).detach().cpu().numpy()
            elif model.use_lagrangian:
                weights = model.lambda_weights.softmax(0).detach().cpu().numpy()
            else:
                with torch.no_grad():
                    grid_sample = sample_batch['grid'][:1].to(device)
                    _, meta = model(grid_sample)
                    weights = meta['weights'].mean(dim=0).cpu().numpy() if 'weights' in meta else np.zeros(model.n_solvers)
            
            training_history[name]['weights'].append(weights)
            
            print(f"{name:15} – train MSE: {train_m.get('recon_loss', 0.0):.4e}  val MSE: {val_m.get('recon_loss', 0.0):.4e}")
        
        # Plot intermediate results
        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            plot_error_distribution_1d(
                {'Softmax': models['softmax'],
                 'Single-Lagr': models['single_time_lr'],
                 'Two-Lagr': models['two_time_lr'],
                 'ADMM': models['admm']},
                val_loader, device, save_dir, epoch + 1
            )
            
            plot_solution_comparison_1d(
                {'Softmax': models['softmax'],
                 'Single-Lagr': models['single_time_lr'],
                 'Two-Lagr': models['two_time_lr'],
                 'ADMM': models['admm']},
                sample_batch, device, save_dir, epoch + 1
            )
    
    # Final plots
    print("\nGenerating final plots...")
    plot_loss_curves_1d(training_history, save_dir, plot_type='total')
    plot_loss_curves_1d(training_history, save_dir, plot_type='recon')
    plot_weight_evolution_1d(training_history, solver_names, save_dir)
    
    # Print final weights
    print("\nFinal solver weights:")
    for name, model in models.items():
        if isinstance(model, ADMMRouter1D):
            weights = model.z.softmax(0).detach().cpu().numpy()
        elif model.use_lagrangian:
            weights = model.lambda_weights.softmax(0).detach().cpu().numpy()
        else:
            with torch.no_grad():
                grid_sample = sample_batch['grid'][:1].to(device)
                _, meta = model(grid_sample)
                weights = meta['weights'].mean(dim=0).cpu().numpy() if 'weights' in meta else np.zeros(model.n_solvers)
        
        print(f"{name:15}: {weights}")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main_1d()