import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from solvers.base_solver import BasePDESolver
from common.loss_functions import HuberLoss

class SoftmaxExpertSystem(nn.Module):
    def __init__(self, solvers: List[BasePDESolver], input_dim: int, hidden_dim: int = 64, 
                 temperature: float = 1.0, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.solvers = nn.ModuleList(solvers)
        self.n_experts = len(solvers)
        self.input_dim = input_dim
        self.device = device
        self.temperature = temperature
      
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_experts)
        )
      
        self.huber = HuberLoss(delta=1.0)
        self.register_buffer('usage_count', torch.zeros(self.n_experts))
  
    def forward(self, x: torch.Tensor, return_all: bool = False) -> Tuple[torch.Tensor, Dict]:
        logits = self.router(x) / self.temperature
        weights = F.softmax(logits, dim=-1)
      
        solver_outputs = []
        for solver in self.solvers:
            output = solver(x)
            solver_outputs.append(output)
      
        solver_outputs = torch.stack(solver_outputs, dim=1)
      
        with torch.no_grad():
            self.usage_count += weights.sum(dim=0)
      
        combined = (solver_outputs * weights.unsqueeze(-1)).sum(dim=1)
      
        metadata = {
            'weights': weights,
            'regime_weights': weights,
            'usage_count': self.usage_count.clone()
        }
      
        if return_all:
            metadata['solver_outputs'] = solver_outputs
      
        return combined, metadata
  
    def compute_loss(self, x: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        combined, metadata = self.forward(x, return_all=True)
        solver_outputs = metadata['solver_outputs']
      
        recon_loss = self.huber(solver_outputs, target.expand_as(solver_outputs))
        usage_prob = metadata['usage_count'] / (metadata['usage_count'].sum() + 1e-6)
        target_prob = torch.ones_like(usage_prob) / self.n_experts
        balance_loss = F.kl_div(usage_prob.log(), target_prob, reduction='sum')
      
        total_loss = recon_loss + 0.1 * balance_loss
      
        metadata.update({
            'recon_loss': recon_loss,
            'balance_loss': balance_loss
        })
      
        return total_loss, metadata