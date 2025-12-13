import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from solvers.base_solver import BasePDESolver
from common.loss_functions import HuberLoss
from .router import LagrangianExpertRouter

class LagrangianExpertSystem(nn.Module):
    def __init__(self, solvers: List[BasePDESolver], input_dim: int, hidden_dim: int = 64, 
                 rho: float = 1.0, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.solvers = nn.ModuleList(solvers)
        self.n_experts = len(solvers)
        self.input_dim = input_dim
        self.device = device
        self.rho = rho
      
        self.solver_characteristics = [solver.characteristics for solver in solvers]
        self.router = LagrangianExpertRouter(
            n_experts=self.n_experts,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            rho=rho,
            device=device
        )
        self.router.solver_characteristics = self.solver_characteristics
      
        self.lambda_weights = nn.Parameter(torch.ones(self.n_experts) / self.n_experts)
        self.mu = nn.Parameter(torch.zeros(1))
        self.nu = nn.Parameter(torch.zeros(self.n_experts))
        self.huber = HuberLoss(delta=1.0)
  
    def forward(self, x: torch.Tensor, return_all: bool = False) -> Tuple[torch.Tensor, Dict]:
        weights, router_metadata = self.router(x, return_regime=True)
        solver_outputs = []
        for solver in self.solvers:
            output = solver(x)
            solver_outputs.append(output)
      
        solver_outputs = torch.stack(solver_outputs, dim=1)
        combined_weights = self.lambda_weights.view(1, -1, 1).expand(x.size(0), -1, x.size(1))
        combined = (solver_outputs * combined_weights).sum(dim=1)
      
        metadata = {
            'weights': self.lambda_weights,
            'router_metadata': router_metadata,
            'solver_outputs': solver_outputs
        }
      
        return combined, metadata
  
    def compute_loss(self, x: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        _, metadata = self.forward(x, return_all=True)
        regime_weights = metadata['router_metadata']['regime_weights']
        solver_outputs = metadata['solver_outputs']
      
        source_losses = []
        for i in range(self.n_experts):
            loss = F.mse_loss(solver_outputs[:, i], target)
            source_losses.append(loss)
        source_losses = torch.stack(source_losses)
      
        recon_loss = (self.lambda_weights * source_losses).sum()
        constraint_scale = 10.0
        g_lambda = 1 - self.lambda_weights.sum()
        h_lambda = -self.lambda_weights
      
        lagrangian = recon_loss + constraint_scale * (
            self.mu * g_lambda + (self.nu * h_lambda).sum() + 
            (self.rho/2) * (g_lambda**2 + (torch.relu(h_lambda)**2).sum())
        )
      
        entropy = -(F.softmax(self.lambda_weights, dim=0) * F.log_softmax(self.lambda_weights, dim=0)).sum()
        lagrangian = lagrangian - 0.01 * entropy
      
        metadata = {
            'g_lambda': g_lambda,
            'h_lambda': h_lambda,
            'recon_loss': recon_loss.item(),
            'weights': self.lambda_weights,
            'regime_weights': regime_weights,
            'source_losses': source_losses
        }
      
        return lagrangian, metadata

class MultiSolverSystem(nn.Module):
    def __init__(self, domain_size: int, hidden_dim: int = 64, use_lagrangian: bool = True, rho: float = 1.0):
        super().__init__()
        from solvers import FourierPDESolver, WENOSolver, DeepONetSolver, MultiResolutionSolver
      
        self.solvers = nn.ModuleList([
            FourierPDESolver(domain_size),
            WENOSolver(domain_size),
            DeepONetSolver(domain_size),
            MultiResolutionSolver(domain_size)
        ])
      
        self.domain_size = domain_size
        self.hidden_dim = hidden_dim
        self.use_lagrangian = use_lagrangian
        self.rho = rho
      
        if not use_lagrangian:
            from .router import AdaptiveRouter
            self.router = AdaptiveRouter(
                domain_size=domain_size,
                n_solvers=len(self.solvers),
                hidden_dim=hidden_dim
            )
      
        if use_lagrangian:
            self.lambda_weights = nn.Parameter(torch.randn(len(self.solvers)) * 0.01 + 1.0/len(self.solvers))
            self.mu = nn.Parameter(torch.zeros(1))
            self.nu = nn.Parameter(torch.zeros(len(self.solvers)))
  
    def forward(self, grid: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
      
        solver_outputs = []
        for solver in self.solvers:
            output = solver(grid)
            solver_outputs.append(output)
        solver_outputs = torch.stack(solver_outputs, dim=1)
      
        if self.use_lagrangian:
            weights = self.lambda_weights
            weights = weights.view(1, -1, 1, 1, 1).expand(grid.size(0), -1, 1, 1, 1)
        else:
            weights, router_info = self.router(grid)
            weights = weights.view(-1, len(self.solvers), 1, 1, 1)
      
        output = (solver_outputs * weights).sum(dim=1)
      
        return output, {
            'solver_outputs': solver_outputs,
            'weights': weights.squeeze(-1).squeeze(-1).squeeze(-1),
            'lambda_weights': self.lambda_weights if self.use_lagrangian else None
        }
  
    def compute_loss(self, grid: torch.Tensor, target: torch.Tensor, meta: Dict) -> Tuple[torch.Tensor, Dict]:
        target_expanded = target.unsqueeze(1).expand(-1, len(self.solvers), -1, -1, -1)
        recon_loss = F.mse_loss(meta['solver_outputs'], target_expanded)
      
        if self.use_lagrangian:
            g_lambda = 1 - self.lambda_weights.sum()
            h_lambda = -self.lambda_weights
            loss = (recon_loss + self.mu * g_lambda + (self.nu * h_lambda).sum() + 
                   (self.rho/2) * (g_lambda**2 + (torch.relu(h_lambda)**2).sum()))
        else:
            sparsity_loss = 0.1 * (1 - (meta['weights'] > 0.1).float().mean())
            loss = recon_loss + sparsity_loss
      
        if self.use_lagrangian:
            weight_sum = self.lambda_weights.sum().item()
            weight_mean = self.lambda_weights.mean().item()
        else:
            weight_sum = meta['weights'].sum().item()
            weight_mean = meta['weights'].mean().item()
      
        return loss, {
            'recon_loss': recon_loss.item(),
            'total_loss': loss.item(),
            'weight_sum': weight_sum,
            'weight_mean': weight_mean
        }