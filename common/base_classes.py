from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple
import torch.nn as nn
import torch

class PhysicsRegime(Enum):
    SMOOTH = 'smooth'
    SHOCK = 'shock'
    BOUNDARY = 'boundary'
    TURBULENT = 'turbulent'

@dataclass
class SolverCharacteristics:
    name: str
    optimal_regime: PhysicsRegime
    computational_cost: float
    accuracy: float

class BasePDESolver(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.characteristics = None
  
    def forward(self, x):
        raise NotImplementedError

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
      
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
      
        x_ft = torch.fft.rfft(x)
      
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,
                          device=x.device, dtype=torch.cfloat)
      
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box",
            x_ft[:, :, :self.modes],
            self.weights
        )
      
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
      
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft2(x)
      
        out_ft = torch.zeros(
            x.shape[0], self.out_channels, x.shape[-2], x.shape[-1]//2 + 1,
            device=x.device, dtype=torch.cfloat
        )
        
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1
        )
      
        x = torch.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))
        return x