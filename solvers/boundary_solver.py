import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_solver import BasePDESolver
from common.base_classes import SolverCharacteristics, PhysicsRegime

class BoundaryAwareSolver(BasePDESolver):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__(input_dim, hidden_dim)
      
        self.boundary_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
      
        self.solver_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
      
        self.characteristics = SolverCharacteristics(
            name="Boundary",
            optimal_regime=PhysicsRegime.BOUNDARY,
            computational_cost=1.5,
            accuracy=0.85
        )
  
    def compute_boundary_distance(self, x: torch.Tensor) -> torch.Tensor:
        left_dist = x - (-1)
        right_dist = 1 - x
        return torch.min(left_dist, right_dist)
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        boundary_dist = self.compute_boundary_distance(x)
        x_with_dist = torch.stack([x, boundary_dist], dim=-1)
        x_flat = x_with_dist.view(-1, 2)
        boundary_features = self.boundary_net(x_flat)
        output = self.solver_net(boundary_features)
        return output.view(batch_size, -1)