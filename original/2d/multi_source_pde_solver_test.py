"""
Four-way comparison for 2-D Navier–Stokes / PDE solver ensemble
---------------------------------------------------------------
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

# ------------------------------------------------------------------
# 0.  Missing helper classes  (minimal补齐)
# ------------------------------------------------------------------
class FourierBlock2D(nn.Module):
    """2-D Fourier layer used inside FourierPDESolver"""
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.w = nn.Conv2d(in_channels, out_channels, 1)  # local linear transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFT
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            x.shape[0], self.w.out_channels, x.shape[-2], x.shape[-1] // 2 + 1,
            device=x.device, dtype=torch.cfloat
        )
        out_ft[:, :, :self.weights1.shape[2], :self.weights1.shape[3]] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.weights1.shape[2], :self.weights1.shape[3]], self.weights1
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))
        return x + self.w(x)  # residual
    

# ---------- 1. 2-D Navier-Stokes data set (your original code) ----------
class NavierStokesDataset(Dataset):
    def __init__(self, n_samples: int, domain_size: int = 64, dt: float = 0.001,
                 n_steps: int = 100, Re: float = 100, noise_level: float = 0.01, seed: int = 42):
        torch.manual_seed(seed)
        self.domain_size = domain_size
        self.samples = []
        self.dt, self.dx, self.Re = dt, 2.0 / domain_size, Re
        self.nu = 1.0 / Re
        x = torch.linspace(-1, 1, domain_size)
        y = torch.linspace(-1, 1, domain_size)
        self.grid_x, self.grid_y = torch.meshgrid(x, y, indexing='ij')

        for i in range(n_samples):
            u, v, w = self.generate_initial_conditions()
            solution = self.solve_navier_stokes(u, v, w, n_steps)
            solution = solution + noise_level * torch.randn_like(solution)
            self.samples.append({
                'grid': torch.stack([self.grid_x, self.grid_y], dim=0),
                'solution': solution[-1],
                'parameters': {'Re': self.Re}
            })

    def generate_initial_conditions(self):
        u = torch.zeros((self.domain_size, self.domain_size))
        v = torch.zeros((self.domain_size, self.domain_size))
        w = torch.zeros((self.domain_size, self.domain_size))
        u[-1, :] = 1.0
        w = self.compute_vorticity(u, v)
        return u, v, w

    def compute_vorticity(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        du_dy = (u[1:, :] - u[:-1, :]) / self.dx
        dv_dx = (v[:, 1:] - v[:, :-1]) / self.dx
        du_dy = F.pad(du_dy, (0, 0, 0, 1))
        dv_dx = F.pad(dv_dx, (0, 1, 0, 0))
        return dv_dx - du_dy

    def solve_poisson(self, f: torch.Tensor, tol: float = 1e-4, max_iter: int = 1000) -> torch.Tensor:
        psi = torch.zeros_like(f)
        dx2 = self.dx * self.dx
        error, iter_count = 1.0, 0
        while error > tol and iter_count < max_iter:
            psi_old = psi.clone()
            psi[1:-1, 1:-1] = 0.25 * (psi_old[1:-1, 2:] + psi_old[1:-1, :-2] +
                                        psi_old[2:, 1:-1] + psi_old[:-2, 1:-1] - dx2 * f[1:-1, 1:-1])
            psi[0, :], psi[-1, :], psi[:, 0], psi[:, -1] = 0, 0, 0, 0
            error = torch.max(torch.abs(psi - psi_old)).item()
            iter_count += 1
        return psi

    def compute_velocity_from_stream(self, psi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = torch.zeros_like(psi)
        v = torch.zeros_like(psi)
        u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * self.dx)
        v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * self.dx)
        return u, v

    def solve_navier_stokes(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, n_steps: int) -> torch.Tensor:
        solutions = []
        for _ in range(n_steps):
            psi = self.solve_poisson(-w)
            u, v = self.compute_velocity_from_stream(psi)
            w_new = w.clone()
            du_dy = (u[1:, :] - u[:-1, :]) / self.dx
            dv_dx = (v[:, 1:] - v[:, :-1]) / self.dx
            d2w_dx2 = torch.zeros_like(w)
            d2w_dy2 = torch.zeros_like(w)
            dw_dx = torch.zeros_like(w)
            dw_dy = torch.zeros_like(w)
            dw_dx[1:-1, 1:-1] = (w[1:-1, 2:] - w[1:-1, :-2]) / (2 * self.dx)
            dw_dy[1:-1, 1:-1] = (w[2:, 1:-1] - w[:-2, 1:-1]) / (2 * self.dx)
            d2w_dx2[1:-1, 1:-1] = (w[1:-1, 2:] - 2 * w[1:-1, 1:-1] + w[1:-1, :-2]) / (self.dx ** 2)
            d2w_dy2[1:-1, 1:-1] = (w[2:, 1:-1] - 2 * w[1:-1, 1:-1] + w[:-2, 1:-1]) / (self.dx ** 2)
            w_new[1:-1, 1:-1] = w[1:-1, 1:-1] + self.dt * (
                -u[1:-1, 1:-1] * dw_dx[1:-1, 1:-1] - v[1:-1, 1:-1] * dw_dy[1:-1, 1:-1]
                + self.nu * (d2w_dx2[1:-1, 1:-1] + d2w_dy2[1:-1, 1:-1])
            )
            w_new[0, :] = -2 * psi[1, :] / (self.dx ** 2)
            w_new[-1, :] = -2 * psi[-2, :] / (self.dx ** 2)
            w_new[:, 0] = -2 * psi[:, 1] / (self.dx ** 2)
            w_new[:, -1] = -2 * psi[:, -2] / (self.dx ** 2)
            w = w_new
            solutions.append(torch.stack([u, v, w], dim=0))
        return torch.stack(solutions)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------- 2. Solver definitions (unchanged) ----------
class FourierPDESolver(nn.Module):
    def __init__(self, domain_size: int, mode1: int = 16, mode2: int = 9, width: int = 16):
        super().__init__()
        self.lift = nn.Sequential(nn.Conv2d(2, width, 1), nn.GELU(), nn.Conv2d(width, width, 1))
        self.fourier_layers = nn.ModuleList([FourierBlock2D(width, width, mode1, mode2) for _ in range(4)])
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, 3, 1))

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        x = self.lift(grid)
        for layer in self.fourier_layers:
            x = layer(x)
            x = F.gelu(x)
        return self.project(x)


class WENOSolver(nn.Module):
    def __init__(self, domain_size: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, 3, 1)
        )
        self.shock_detector = nn.Sequential(
            nn.Conv2d(4, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 1), nn.Sigmoid()
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        u = self.net(grid)                       # [B, 3, H, W]
        u_slice = u[:, :1, :, :]                 # <-- use only 1st channel for gradients
        u_x, u_y = self.compute_gradients(u_slice)
        shock_weights = self.shock_detector(torch.cat([grid, u_x, u_y], dim=1))
        return u * shock_weights

    def compute_gradients(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u_x = F.pad(u[:, :, 1:, :] - u[:, :, :-1, :], (0, 0, 0, 1))
        u_y = F.pad(u[:, :, :, 1:] - u[:, :, :, :-1], (0, 1, 0, 0))
        return u_x, u_y


class DeepONetSolver(nn.Module):
    def __init__(self, domain_size: int, hidden_dim: int = 64, branch_dim: int = 40):
        super().__init__()
        self.branch_net = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_dim, branch_dim, 1)
        )
        x = torch.linspace(-1, 1, domain_size)
        xx, yy = torch.meshgrid(x, x, indexing='ij')
        self.register_buffer('coords', torch.stack([xx, yy], dim=0).unsqueeze(0))
        self.trunk_net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, branch_dim)
        )
        self.projection = nn.Conv2d(branch_dim, 3, 1)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        branch = self.branch_net(grid)
        trunk = self.trunk_net(self.coords.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        output = branch * trunk
        return self.projection(output)


class MultiResolutionSolver(nn.Module):
    def __init__(self, domain_size: int, scales: List[int] = None):
        super().__init__()
        scales = scales or [1, 2, 4]
        self.scales = scales
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 1, 1)
            ) for _ in scales
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(len(scales), 64, 1), nn.ReLU(),
            nn.Conv2d(64, 3, 1)
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        outs = []
        for scale, net in zip(self.scales, self.nets):
            x = F.avg_pool2d(grid, scale) if scale > 1 else grid
            o = net(x)
            outs.append(F.interpolate(o, size=grid.shape[-2:], mode='bilinear', align_corners=True))
        return self.fusion(torch.cat(outs, dim=1))


# ---------- 3. Router definitions ----------
class AdaptiveRouter(nn.Module):
    def __init__(self, domain_size: int, n_solvers: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(8), nn.Flatten(),
            nn.Linear(64 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_solvers)
        )

    def forward(self, grid: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        logits = self.net(grid)
        weights = F.softmax(logits, dim=-1)
        return weights, {'logits': logits, 'weights': weights}


# ---------- 4. Multi-solver system (Lagrangian / Softmax) ----------
class MultiSolverSystem(nn.Module):
    def __init__(self, domain_size: int, solvers: List[nn.Module] = None, hidden_dim: int = 64, use_lagrangian: bool = True, rho: float = 1.0):
        super().__init__()
        # Use shared solvers if provided, otherwise create new instances
        self.solvers = nn.ModuleList(solvers) if solvers is not None else nn.ModuleList([
            FourierPDESolver(domain_size),
            WENOSolver(domain_size),
            DeepONetSolver(domain_size),
            MultiResolutionSolver(domain_size)
        ])
        self.n_solvers = len(self.solvers)
        self.use_lagrangian = use_lagrangian
        self.rho = rho

        if use_lagrangian:
            self.lambda_weights = nn.Parameter(torch.ones(self.n_solvers) / self.n_solvers)
            self.mu = nn.Parameter(torch.zeros(1))
            self.nu = nn.Parameter(torch.zeros(self.n_solvers))
        else:
            self.router = AdaptiveRouter(domain_size, self.n_solvers, hidden_dim)

    def forward(self, grid: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        outs = []
        for solver in self.solvers:
            o = solver(grid)
            outs.append(o)
        outs = torch.stack(outs, dim=1)  # [B, n_solvers, 3, H, W]

        if self.use_lagrangian:
            w = self.lambda_weights.softmax(0).view(1, -1, 1, 1, 1)
        else:
            w, router_info = self.router(grid)
            w = w.view(-1, self.n_solvers, 1, 1, 1)

        combined = (outs * w).sum(dim=1)  # [B, 3, H, W]
        return combined, {
            'solver_outputs': outs,
            'weights': w.squeeze(-1).squeeze(-1).squeeze(-1),  # Shape depends on case
            'lambda_weights': self.lambda_weights if self.use_lagrangian else None,
            'combined_output': combined  # NEW: Store for loss computation
        }

    def compute_loss(self, grid: torch.Tensor, target: torch.Tensor, meta: Dict) -> Tuple[torch.Tensor, Dict]:
        # FIX: Use weighted combination for reconstruction loss in ALL cases
        if self.use_lagrangian:
            # For Lagrangian: weights are static per batch, get from parameter
            w = self.lambda_weights.softmax(0).view(1, -1, 1, 1, 1)
            combined_output = (meta['solver_outputs'] * w).sum(dim=1)
            recon = F.mse_loss(combined_output, target)
            
            # Constraint terms ONLY for Lagrangian
            g = 1 - self.lambda_weights.sum()
            h = -self.lambda_weights
            constraint_penalty = self.mu * g + (self.nu * h).sum() + (self.rho / 2) * (g ** 2 + (torch.relu(h) ** 2).sum())
            loss = recon + constraint_penalty
        else:
            # Softmax router: weights are dynamic per sample
            w = meta['weights'].view(-1, self.n_solvers, 1, 1, 1)
            combined_output = (meta['solver_outputs'] * w).sum(dim=1)
            recon = F.mse_loss(combined_output, target)
            
            # FIX: Sparsity penalty - explicitly cast to float first
            sparsity = 0.1 * (1 - (meta['weights'] > 0.1).float().mean())
            loss = recon + sparsity

        return loss, {
            'loss': loss,
            'recon_loss': recon.item(),
            'total_loss': loss.item(),
            'weight_sum': self.lambda_weights.sum().item() if self.use_lagrangian else meta['weights'].sum().item(),
            'weight_mean': self.lambda_weights.mean().item() if self.use_lagrangian else meta['weights'].mean().item()
        }


# ---------- 5. ADMM Router (Global Weights) ----------
class ADMMRouter(nn.Module):
    def __init__(self, domain_size: int, solvers: List[nn.Module] = None, rho: float = 10.0):
        super().__init__()
        self.solvers = nn.ModuleList(solvers) if solvers else nn.ModuleList([
            FourierPDESolver(domain_size),
            WENOSolver(domain_size),
            DeepONetSolver(domain_size),
            MultiResolutionSolver(domain_size)
        ])
        self.n_solvers = len(self.solvers)
        self.rho = rho
        
        # Global consensus variable (1D, not spatial)
        self.z = nn.Parameter(torch.ones(self.n_solvers) / self.n_solvers)
        self.u = nn.Parameter(torch.zeros(self.n_solvers))

    def forward(self, grid: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        
        # Get solver outputs
        outs = torch.stack([solver(grid) for solver in self.solvers], dim=1)  # [B, n_solvers, 3, H, W]
        
        # Generate weights from z
        w = F.softmax(self.z, dim=0).view(1, -1, 1, 1, 1)  # [1, n_solvers, 1, 1, 1]
        combined = (outs * w).sum(dim=1)  # [B, 3, H, W]
        
        return combined, {
            'solver_outputs': outs,
            'weights': F.softmax(self.z, dim=0),
            'z': self.z,
            'u': self.u
        }

    def compute_loss(self, grid: torch.Tensor, target: torch.Tensor, meta: Dict) -> Tuple[torch.Tensor, Dict]:
        """Unified interface for Trainer"""
        return self.admm_loss(grid, target, meta['solver_outputs'])

    def admm_loss(self, grid: torch.Tensor, target: torch.Tensor,
                  solver_outputs: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Global weights
        w = F.softmax(self.z, dim=0).view(1, -1, 1, 1, 1)
        combined_output = (solver_outputs * w).sum(dim=1)
        recon = F.mse_loss(combined_output, target)
        
        # Consensus: encourage z to be sparse
        consensus = 0.5 * self.rho * self.z.pow(2).sum()
        
        # Dual term
        dual = (self.u * (F.softmax(self.z, dim=0) - 0.25 * torch.ones_like(self.z))).sum()
        
        total_loss = recon + consensus + dual
        
        return total_loss, {
            'loss': total_loss,
            'recon_loss': recon.item(),
            'consensus': consensus.item(),
            'weights': F.softmax(self.z, dim=0).detach().cpu().numpy()
        }
    def admm_loss(self, grid: torch.Tensor, target: torch.Tensor,
                  solver_outputs: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # target: [B, 3, H, W]  --->  keep identical shape
        w = self.z.softmax(0).view(1, -1, 1, 1, 1)
        combined_output = (solver_outputs * w).sum(dim=1)
        recon = F.mse_loss(combined_output, target)
        
        consensus = 0.5 * self.z.pow(2).sum()  # Encourage sparsity via projection
        loss = recon + consensus
        return loss, {'loss': loss, 
                      'total_loss': loss.item(),
                      'recon_loss': recon.item(),
                      'consensus': consensus,
                      'weights': self.z.softmax(0)}
    
# ---------- 6. Optimisers ----------
class SingleTimeScaleOptimizer:
    def __init__(self, model: MultiSolverSystem, lr: float = 1e-4, rho: float = 10.0):
        self.model = model
        self.rho = rho
        
        self.theta_opt = torch.optim.Adam([
                {'params': [p for n, p in model.named_parameters() 
                        if 'lambda' not in n and 'mu' not in n and 'nu' not in n], 
                'lr': lr},  # Solver parameters
                {'params': [model.lambda_weights], 'lr': lr * 5},  # Faster for weights
                {'params': [model.mu, model.nu], 'lr': lr * 5}     # Faster for multipliers
            ])
    def zero_grad(self):
        self.theta_opt.zero_grad()

    def step(self, loss_dict):
        # θ-update
        loss_dict['loss'].backward()
        self.theta_opt.step()
        
        # Dual ascent for μ, ν (separate update rule)
        with torch.no_grad():
            g = 1 - self.model.lambda_weights.sum()
            h = -self.model.lambda_weights
            
            self.model.mu.add_(self.rho * g)  # μ ← μ + ρg
            self.model.nu.add_(self.rho * torch.relu(h))  # ν ← ν + ρ·max(0,h)
            
            # Project λ to simplex
            self.model.lambda_weights.data = self.project_simplex(self.model.lambda_weights.data)


    @staticmethod
    def project_simplex(v: torch.Tensor) -> torch.Tensor:
        v_sorted, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(v_sorted, 0)
        rho = (cssv - 1) / torch.arange(1, len(v) + 1, device=v.device, dtype=v.dtype)
        idx = torch.where(v_sorted > rho)[0]
        rho_star = rho[idx[-1]] if idx.numel() else 0
        return torch.maximum(v - rho_star, torch.zeros_like(v))
    
    def state_dict(self):
        return {'opt': self.opt.state_dict()}

    def load_state_dict(self, d):
        self.opt.load_state_dict(d['opt'])


class TwoTimeScaleOptimizer:
    def __init__(self, model: MultiSolverSystem, lr_theta: float = 1e-3, lr_lambda: float = 1e-4, rho: float = 10.0):
        self.model = model
        self.rho = rho
        
        # Fast: solver parameters
        solver_params = [p for n, p in model.named_parameters() 
                        if n not in ['lambda_weights', 'nu', 'mu']]
        self.theta_opt = torch.optim.Adam(solver_params, lr=lr_theta)
        
        # Slow: λ only
        self.lambda_opt = torch.optim.SGD([model.lambda_weights, model.nu, model.mu], lr=lr_lambda)


    def zero_grad(self):
        self.theta_opt.zero_grad()
        self.lambda_opt.zero_grad()

    def step(self, loss_dict):
        # θ-update (fast timescale)
        loss_dict['loss'].backward()
        self.theta_opt.step()
        
        # λ-update (slow timescale)
        self.lambda_opt.step()
        
        # Dual ascent for μ, ν (separate timescale)
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

    def state_dict(self):
        return {'theta_opt': self.theta_opt.state_dict(), 'lambda_opt': self.lambda_opt.state_dict()}

    def load_state_dict(self, d):
        self.theta_opt.load_state_dict(d['theta_opt'])
        self.lambda_opt.load_state_dict(d['lambda_opt'])

class ADMMSolver(nn.Module):
    def __init__(self, domain_size: int, solvers: List[nn.Module] = None, 
                 rho: float = 0.1):
        super().__init__()
        self.solvers = nn.ModuleList(solvers) if solvers else nn.ModuleList([
            FourierPDESolver(domain_size),
            WENOSolver(domain_size),
            DeepONetSolver(domain_size),
            MultiResolutionSolver(domain_size)
        ])
        self.n_solvers = len(self.solvers)
        self.rho = rho
        
        # Consensus variable (shared across batch) - should match output shape
        self.z = nn.Parameter(torch.ones(1, self.n_solvers, 3, domain_size, domain_size) / self.n_solvers)
        self.u = nn.Parameter(torch.zeros_like(self.z))  # Dual variable

    def forward(self, grid: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        
        # Get solver outputs
        outs = torch.stack([solver(grid) for solver in self.solvers], dim=1)
        
        # Weighted combination using SOFTMAX of z (per-solver spatial weights)
        w = self.z.softmax(dim=1)  # Normalize over solver dimension
        combined = (outs * w).sum(dim=1)
        
        return combined, {
            'solver_outputs': outs,
            'z': self.z,
            'u': self.u,
            'weights': w.mean(dim=(0,2,3,4)).detach()  # For logging
        }

    def compute_loss(self, grid: torch.Tensor, target: torch.Tensor, meta: Dict) -> Tuple[torch.Tensor, Dict]:
        outs = meta['solver_outputs']  # [B, n_solvers, 3, H, W]
        
        # ADMM Augmented Lagrangian
        # 1. Reconstruction loss (weighted combination)
        w = self.z.softmax(dim=1)
        combined = (outs * w).sum(dim=1)
        recon = F.mse_loss(combined, target)
        
        # 2. Consensus constraint: each solver should match consensus z
        # Reshape z and u to match batch size
        z_expanded = self.z.expand(outs.shape[0], -1, -1, -1, -1)
        u_expanded = self.u.expand(outs.shape[0], -1, -1, -1, -1)
        
        # Consensus violation: ||solver_i - z_i||²
        consensus_violation = (outs - z_expanded).pow(2).mean()
        
        # 3. Augmented Lagrangian terms
        # uᵀ(solver - z) + (ρ/2)||solver - z||²
        dual_term = (u_expanded * (outs - z_expanded)).mean()
        penalty_term = 0.5 * self.rho * consensus_violation
        
        total_loss = recon + dual_term + penalty_term
        
        return total_loss, {
            'loss': total_loss,
            'recon_loss': recon.item(),
            'consensus': consensus_violation.item(),
            'weights': meta['weights']
        }
    
class ADMMOptimizer:
    def __init__(self, model: ADMMSolver, lr: float = 1e-4, 
                 rho: float = 0.1):
        self.model = model
        self.rho = rho
        
        # Only optimize solver parameters, not z or u
        solver_params = [p for n, p in model.named_parameters() ]
        self.theta_opt = torch.optim.Adam(solver_params, lr=lr)


    def zero_grad(self):
        self.theta_opt.zero_grad()
        #self.z_opt.zero_grad()

    def step(self, loss_dict):
        # θ-update
        self.theta_opt.zero_grad()
        loss_dict['loss'].backward(retain_graph=True)
        self.theta_opt.step()
        
        # z-update (dual proximal step)
        with torch.no_grad():
            # Project to simplex
            z_sm = F.softmax(self.model.z, dim=0)
            self.model.z.copy_(self.model.z + self.rho * (z_sm - 0.25))
        
        # u-update (dual ascent)
        with torch.no_grad():
            z_sm = F.softmax(self.model.z, dim=0)
            self.model.u.add_(self.rho * (z_sm - 0.25 * torch.ones_like(z_sm)))

# ---------- 7. Trainer ----------
class Trainer:
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
        return {k: np.mean([x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in v]) for k, v in epoch_metrics.items()}
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        grid = batch['grid'].to(self.device)
        target = batch['solution'].to(self.device)
        output, meta = self.model(grid)
        if isinstance(self.model, ADMMRouter):
            loss, loss_meta = self.model.admm_loss(grid, target, meta['solver_outputs'])
        else:
            loss, loss_meta = self.model.compute_loss(grid, target, meta)

        self.optimizer.zero_grad()
        if isinstance(self.optimizer, (ADMMOptimizer, TwoTimeScaleOptimizer, SingleTimeScaleOptimizer)):
            # custom optimisers call .backward() internally
            self.optimizer.step(loss_meta)
        else:
            # standard PyTorch optimisers need .backward() here
            loss.backward()
            self.optimizer.step()
        return {
            'total_loss': loss_meta['total_loss'],
            'recon_loss': loss_meta['recon_loss'],
            'weight_mean': meta['weights'].mean().item(),
            'weight_std': meta['weights'].std().item()
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        val_metrics = defaultdict(list)
        for batch in self.val_loader:
            grid = batch['grid'].to(self.device)
            target = batch['solution'].to(self.device)
            output, meta = self.model(grid)
            if isinstance(self.model, ADMMRouter):
                loss, loss_meta = self.model.admm_loss(grid, target, meta['solver_outputs'])
            else:
                loss, loss_meta = self.model.compute_loss(grid, target, meta)
            for k, v in loss_meta.items():
                val_metrics[k].append(v)
        return {k: np.mean(v) for k, v in val_metrics.items()}


# ---------- 8. Plotting utilities ----------
def plot_error_distribution(models: Dict[str, nn.Module], test_loader: DataLoader, device: str,
                            save_path: str, epoch: int):
    errors = {n: [] for n in models}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                grid = batch['grid'].to(device)
                target = batch['solution'].to(device)
                pred, _ = model(grid)
                errors[name].extend(torch.abs(pred - target).mean(dim=(1, 2, 3)).cpu().numpy())
    plt.figure(figsize=(10, 6))
    for name, err in errors.items():
        n_bins = min(50, len(err) // 2)
        plt.hist(err, bins=n_bins, alpha=0.5, label=name)
    plt.title('Error Distribution')
    plt.xlabel('Mean Absolute Error')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/error_dist_epoch_{epoch}.png')
    plt.close()


def plot_loss_curves(history: Dict[str, Dict[str, List]], save_path: str, plot_type: str = 'total'):
    """Plot training and validation loss evolution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    loss_key = 'train_loss' if plot_type == 'total' else 'train_recon_loss'
    title_suffix = 'Total' if plot_type == 'total' else 'Reconstruction'
    
    # Training Loss
    for name, hist in history.items():
        epochs = range(1, len(hist[loss_key]) + 1)
        ax1.plot(epochs, hist[loss_key], label=f'{name} (Train)', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title(f'{title_suffix} Training Loss Evolution')
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
    ax2.set_title(f'{title_suffix} Validation Loss Evolution')
    ax2.legend()
    ax2.grid(True)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/loss_evolution_{plot_type}.png')
    plt.close()


def plot_weight_evolution(history: Dict[str, Dict[str, List]], solver_names: List[str], save_path: str):
    """Plot solver weight evolution for each method."""
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
            
        weights_array = np.array(weights_history)  # Shape: (epochs, n_solvers)
        
        for i, solver_name in enumerate(solver_names):
            ax.plot(epochs, weights_array[:, i], label=solver_name, linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight')
        ax.set_title(f'{name} - Solver Weights')
        ax.legend()
        ax.grid(True)
    
    # Hide unused subplots if any
    for i in range(len(history), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/weight_evolution.png')
    plt.close()


# ---------- 9. Main ----------
def main():
    n_samples, domain_size, batch_size, n_epochs = 1024, 32, 16, 500
    n_samples = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'results'
    print(f"Using device: {device}")

    train_dataset = NavierStokesDataset(n_samples, domain_size)
    val_dataset = NavierStokesDataset(n_samples // 5, domain_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create shared solver instances for fair comparison
    # All models will use the exact same solver parameters
    print("Creating shared solver instances...")
    # shared_solvers = [
    #     FourierPDESolver(domain_size),
    #     WENOSolver(domain_size),
    #     DeepONetSolver(domain_size),
    #     MultiResolutionSolver(domain_size)
    # ]

    # 4 models (now with shared solvers)
    models = {
        'softmax':      MultiSolverSystem(domain_size,  use_lagrangian=False),
        'single_time_lr': MultiSolverSystem(domain_size,  use_lagrangian=True),   # single-time-scale
        'two_time_lr': MultiSolverSystem(domain_size,  use_lagrangian=True),   # two-time-scale
        'admm':         ADMMRouter(domain_size,)                               # ADMM
    }

    # 4 trainers
    trainers = {}
    for name, model in models.items():
        if name == 'admm':
            opt = ADMMOptimizer(model, lr=1e-4)
            trainers[name] = Trainer(model, train_loader, val_loader, device=device, optimizer=opt)
        elif name == 'two_time_lr':
            opt = TwoTimeScaleOptimizer(model, lr_theta=1e-4, lr_lambda=1e-4)
            trainers[name] = Trainer(model, train_loader, val_loader, device=device, optimizer=opt)
        elif name == 'single_time_lr':
            opt = SingleTimeScaleOptimizer(model, lr=1e-4)
            trainers[name] = Trainer(model, train_loader, val_loader, device=device, optimizer=opt)
        else:  # softmax
            opt = torch.optim.Adam(model.parameters(), lr=1e-4)
            trainers[name] = Trainer(model, train_loader, val_loader, device=device, optimizer=opt)

    # Initialize training history
    training_history = {name: {
        'train_loss': [], 
        'val_loss': [], 
        'train_recon_loss': [],  # NEW
        'val_recon_loss': [],     # NEW
        'weights': []
    } for name in models}
    solver_names = ['Fourier', 'WENO', 'DeepONet', 'MultiRes']
    
    # Get a single sample for weight extraction
    sample_batch = next(iter(val_loader))
    grid_sample = sample_batch['grid'][:1].to(device)

    # training loop
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        for name, trainer in trainers.items():
            train_m = trainer.train_epoch()
            val_m = trainer.validate()
            
            # Store loss history
            training_history[name]['train_loss'].append(train_m['total_loss'])
            training_history[name]['val_loss'].append(val_m['total_loss'])
            training_history[name]['train_recon_loss'].append(train_m['recon_loss'])  # NEW
            training_history[name]['val_recon_loss'].append(val_m['recon_loss'])      # NEW
            
            # Extract and store weights
            model = models[name]
            if isinstance(model, ADMMRouter):
                weights = model.z.softmax(0).detach().cpu().numpy()
            elif model.use_lagrangian:
                weights = model.lambda_weights.softmax(0).detach().cpu().numpy()
            else:  # softmax router - get average weight from sample
                with torch.no_grad():
                    _, meta = model(grid_sample)
                    weights = meta['weights'].mean(dim=0).cpu().numpy()
            training_history[name]['weights'].append(weights)
            
            print(f"{name:12} – train MSE: {train_m['recon_loss']:.4e}  val MSE: {val_m['recon_loss']:.4e}")


        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            plot_error_distribution(
                {'Softmax': models['softmax'],
                 'Single-Lagr': models['single_time_lr'],
                 'Two-Lagr': models['two_time_lr'],
                 'ADMM': models['admm']},
                val_loader, device, save_dir, epoch + 1
            )

    # Plot loss and weight evolution after training
    print("\nGenerating training plots...")
    plot_loss_curves(training_history, save_dir, plot_type='total')
    plot_loss_curves(training_history, save_dir, plot_type='recon')  # Also plot reconstruction loss
    plot_weight_evolution(training_history, solver_names, save_dir)
    print("\nTraining completed!")

if __name__ == "__main__":
    main()