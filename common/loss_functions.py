import torch.nn as nn
import torch

class HuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
  
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        abs_diff = diff.abs()
        quadratic = torch.min(abs_diff, torch.tensor(self.delta))
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic.pow(2) + self.delta * linear
        return loss.mean()