import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_solver import BasePDESolver
from common.base_classes import SolverCharacteristics, PhysicsRegime, SpectralConv1d, SpectralConv2d

class FourierBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.conv = SpectralConv2d(in_channels, out_channels, modes1, modes2)
        self.w = nn.Conv2d(in_channels, out_channels, 1)
      
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        x2 = self.w(x)
        return x1 + x2

class FourierNeuralOperator(BasePDESolver):
    def __init__(self, input_dim: int, modes: int = 16, width: int = 64):
        super().__init__(input_dim)
        self.modes = modes
        self.width = width
      
        self.fc0 = nn.Linear(1, width)
        self.conv0 = SpectralConv1d(width, width, modes)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)
      
        self.characteristics = SolverCharacteristics(
            name="FNO",
            optimal_regime=PhysicsRegime.SMOOTH,
            computational_cost=1.0,
            accuracy=0.9
        )
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(-1, 1)
        x = self.fc0(x)
        x = x.view(batch_size, self.input_dim, self.width)
        x = x.transpose(1, 2)
      
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
      
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
      
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
      
        x = x.transpose(1, 2)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.squeeze(-1)

class FourierPDESolver(BasePDESolver):
    def __init__(self, domain_size: int, mode1: int = 16, mode2: int = 9, width: int = 16):
        super().__init__(domain_size)
        self.mode1 = mode1
        self.model2 = mode2
        self.width = width
      
        self.lift = nn.Sequential(
            nn.Conv2d(2, width, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 1)
        )
      
        self.fourier_layers = nn.ModuleList([
            FourierBlock2D(width, width, mode1, mode2) for _ in range(4)
        ])
      
        self.project = nn.Sequential(
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, 3, 1)
        )
  
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        x = self.lift(grid)
        for layer in self.fourier_layers:
            x = layer(x)
            x = F.gelu(x)
        x = self.project(x)
        return x