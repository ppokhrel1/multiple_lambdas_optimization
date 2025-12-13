import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_solver import BasePDESolver
from common.base_classes import SolverCharacteristics, PhysicsRegime

class ShockCapturingSolver(BasePDESolver):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__(input_dim, hidden_dim)
      
        layers = []
        layers.append(nn.Linear(1, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
      
        self.shock_detector = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
      
        self.characteristics = SolverCharacteristics(
            name="WENO",
            optimal_regime=PhysicsRegime.SHOCK,
            computational_cost=2.0,
            accuracy=0.8
        )
  
    def compute_gradients(self, x: torch.Tensor) -> torch.Tensor:
        dx = x[:, 1:] - x[:, :-1]
        dx = F.pad(dx, (1, 0), mode='replicate')
        d2x = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
        d2x = F.pad(d2x, (1, 1), mode='replicate')
        return torch.stack([x, dx, d2x], dim=-1)
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_flat = x.view(-1, 1)
        base_output = self.net(x_flat).view(batch_size, -1)
        grads = self.compute_gradients(x)
        shock_indicators = self.shock_detector(grads)
        output = base_output * shock_indicators.squeeze(-1)
        return output

class WENOSolver(BasePDESolver):
    def __init__(self, domain_size: int, hidden_dim: int = 64):
        super().__init__(domain_size, hidden_dim)
      
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 3, 1)
        )
      
        self.shock_detector = nn.Sequential(
            nn.Conv2d(4*2, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )
  
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        u = self.net(grid)
        u_x, u_y = self.compute_gradients(u)
        shock_input = torch.cat([grid, u_x, u_y], dim=1)
        shock_weights = self.shock_detector(shock_input)
        return u * shock_weights