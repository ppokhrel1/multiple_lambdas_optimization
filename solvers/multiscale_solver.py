import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .base_solver import BasePDESolver
from common.base_classes import SolverCharacteristics, PhysicsRegime

class MultiscaleSolver(BasePDESolver):
    def __init__(self, input_dim: int, scales: List[int] = [1, 2, 4, 8]):
        super().__init__(input_dim)
        self.scales = scales
      
        self.scale_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1)
            ) for _ in scales
        ])
      
        self.combine_net = nn.Sequential(
            nn.Linear(len(scales), self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
      
        self.characteristics = SolverCharacteristics(
            name="Multiscale",
            optimal_regime=PhysicsRegime.TURBULENT,
            computational_cost=3.0,
            accuracy=0.75
        )
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        multi_scale_outputs = []
        for scale, net in zip(self.scales, self.scale_nets):
            if scale > 1:
                x_scaled = F.avg_pool1d(
                    x.unsqueeze(1),
                    kernel_size=scale,
                    stride=1,
                    padding=scale//2
                ).squeeze(1)
                if x_scaled.shape[-1] != self.input_dim:
                    x_scaled = F.interpolate(
                        x_scaled.unsqueeze(1),
                        size=self.input_dim,
                        mode='linear',
                        align_corners=True
                    ).squeeze(1)
            else:
                x_scaled = x
          
            x_flat = x_scaled.view(-1, 1)
            output = net(x_flat).view(batch_size, -1)
            if output.shape[-1] != self.input_dim:
                output = F.interpolate(
                    output.unsqueeze(1),
                    size=self.input_dim,
                    mode='linear',
                    align_corners=True
                ).squeeze(1)
            multi_scale_outputs.append(output)
      
        combined = torch.stack(multi_scale_outputs, dim=-1)
        output = self.combine_net(combined).squeeze(-1)
        return output

class MultiResolutionSolver(BasePDESolver):
    def __init__(self, domain_size: int, scales: List[int] = [1, 2, 4]):
        super().__init__(domain_size)
        self.scales = scales
      
        self.scale_nets = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, self.hidden_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.hidden_dim, 1, 1)
            ) for _ in scales
        ])
      
        self.fusion_net = nn.Sequential(
            nn.Conv2d(len(scales), self.hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, 3, 1)
        )
  
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        multi_scale_outputs = []
        for scale, net in zip(self.scales, self.scale_nets):
            if scale > 1:
                x = F.avg_pool2d(grid, scale)
                out = net(x)
                out = F.interpolate(out, size=(self.domain_size, self.domain_size),
                                  mode='bilinear', align_corners=True)
            else:
                out = net(grid)
            multi_scale_outputs.append(out)
      
        combined = torch.cat(multi_scale_outputs, dim=1)
        return self.fusion_net(combined)