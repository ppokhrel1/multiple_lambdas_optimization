import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from common.base_classes import PhysicsRegime

class AdaptiveRouter(nn.Module):
    def __init__(self, domain_size: int, n_solvers: int, hidden_dim: int = 64):
        super().__init__()
        self.router_net = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(64 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_solvers),
        )
  
    def forward(self, grid: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        logits = self.router_net(grid)
        weights = F.softmax(logits, dim=-1)
        return weights, {
            'logits': logits,
            'weights': weights
        }

class LagrangianExpertRouter(nn.Module):
    def __init__(self, n_experts: int, input_dim: int, hidden_dim: int = 64, rho: float = 1.0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.n_experts = n_experts
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rho = rho
        self.device = device
      
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
      
        self.regime_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(PhysicsRegime))
        )
      
        self.lambda_weights = nn.Parameter(torch.ones(n_experts) / n_experts)
        self.mu = nn.Parameter(torch.zeros(1))
        self.nu = nn.Parameter(torch.zeros(n_experts))
      
        self.register_buffer('usage_count', torch.zeros(n_experts))
        self.register_buffer('regime_count', torch.zeros(len(PhysicsRegime)))
        self.solver_characteristics = None
  
    def get_regime_weights(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(x)
        regime_logits = self.regime_net(features)
        return F.softmax(regime_logits, dim=-1)
  
    def compute_lagrangian_loss(self, expert_outputs: torch.Tensor, target: torch.Tensor, regime_weights: torch.Tensor):
        epsilon = 1e-8
        expert_losses = []
        for output in expert_outputs.unbind(dim=1):
            loss = F.mse_loss(output, target, reduction='none')
            loss = torch.mean(loss, dim=-1)
            expert_losses.append(loss)
        expert_losses = torch.stack(expert_losses)
      
        g_lambda = 1 - self.lambda_weights.sum()
        h_lambda = -self.lambda_weights
      
        usage_prob = self.usage_count / (self.usage_count.sum() + epsilon)
        target_prob = torch.ones_like(usage_prob) / self.n_experts
        utilization_loss = F.kl_div(
            (usage_prob + epsilon).log(),
            target_prob,
            reduction='sum'
        )
      
        regime_expert_weights = torch.zeros(self.n_experts, device=self.device)
        for i, solver_char in enumerate(self.solver_characteristics):
            regime_idx = list(PhysicsRegime).index(solver_char.optimal_regime)
            regime_expert_weights[i] = regime_weights[:, regime_idx].mean()
      
        regime_expert_weights = F.softmax(regime_expert_weights, dim=0)
        weighted_loss = torch.sum(
            self.lambda_weights * torch.mean(expert_losses, dim=1) *
            (1 + regime_expert_weights)
        )
      
        scale = 0.1
        lagrangian = weighted_loss + \
                    scale * (0.1 * utilization_loss + \
                            self.mu * g_lambda + \
                            (self.nu * h_lambda).sum() + \
                            (self.rho/2) * (g_lambda**2 + (torch.relu(h_lambda)**2).sum()))
      
        return lagrangian, {
            'g_lambda': g_lambda,
            'h_lambda': h_lambda,
            'weighted_loss': weighted_loss.item(),
            'utilization_loss': utilization_loss.item(),
            'expert_losses': torch.mean(expert_losses, dim=1),
            'regime_weights': regime_weights
        }
  
    def forward(self, x: torch.Tensor, return_regime: bool = False) -> Tuple[torch.Tensor, Dict]:
        batch_size = x.shape[0]
        regime_weights = self.get_regime_weights(x)
        weights = F.softmax(self.lambda_weights, dim=0)
        weights = weights.view(1, -1).expand(batch_size, -1)
      
        with torch.no_grad():
            self.usage_count += weights.sum(dim=0)
            self.regime_count += regime_weights.sum(dim=0)
      
        metadata = {
            'regime_weights': regime_weights,
            'usage_count': self.usage_count.clone(),
            'regime_count': self.regime_count.clone()
        }
      
        if return_regime:
            regime_indices = regime_weights.argmax(dim=-1)
            regimes = [PhysicsRegime(list(PhysicsRegime)[i])
                      for i in regime_indices.cpu().numpy()]
            metadata['regimes'] = regimes
      
        return weights, metadata