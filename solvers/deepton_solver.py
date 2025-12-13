import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_solver import BasePDESolver
from common.base_classes import SolverCharacteristics, PhysicsRegime

class DeepONetSolver(BasePDESolver):
    def __init__(self, domain_size: int, hidden_dim: int = 64, branch_dim: int = 40):
        super().__init__(domain_size, hidden_dim)
        self.branch_dim = branch_dim
      
        self.branch_net = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, branch_dim, 1)
        )
      
        self.trunk_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, branch_dim)
        )
      
        self.projection = nn.Conv2d(branch_dim, 3, 1)
  
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        batch_size = grid.shape[0]
        branch_features = self.branch_net(grid)
      
        x = torch.linspace(-1, 1, self.domain_size, device=grid.device)
        y = torch.linspace(-1, 1, self.domain_size, device=grid.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([xx, yy], dim=0)
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1, -1)
      
        coords_flat = coords.permute(0, 2, 3, 1).reshape(-1, 2)
        trunk_out = self.trunk_net(coords_flat)
        trunk_features = trunk_out.view(batch_size, self.domain_size, self.domain_size, self.branch_dim)
        trunk_features = trunk_features.permute(0, 3, 1, 2)
      
        output = branch_features * trunk_features
        output = self.projection(output)
        return output