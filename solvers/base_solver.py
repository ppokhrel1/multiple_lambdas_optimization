import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Tuple
from common.base_classes import BasePDESolver

class BasePDESolver(nn.Module):
    def __init__(self, domain_size: int, hidden_dim: int = 64):
        super().__init__()
        self.domain_size = domain_size
        self.hidden_dim = hidden_dim
  
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
  
    def compute_gradients(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u_x = (u[:, :, 1:, :] - u[:, :, :-1, :]) * self.domain_size
        u_y = (u[:, :, :, 1:] - u[:, :, :, :-1]) * self.domain_size
        u_x = F.pad(u_x, (0, 0, 0, 1))
        u_y = F.pad(u_y, (0, 1, 0, 0))
        return u_x, u_y