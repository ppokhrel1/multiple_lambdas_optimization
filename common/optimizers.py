import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class TwoTimeScaleLagrangianOptimizer:
    def __init__(self, model: nn.Module, eta_theta: float = 1e-4, eta_lambda: float = 1e-3, clipgrad: float = 1.0):
        self.model = model
        self.eta_theta = eta_theta
        self.eta_lambda = eta_lambda
        self.clipgrad = clipgrad
      
        self.theta_params = []
        self.lambda_params = []
        self.dual_params = []
      
        for name, param in model.named_parameters():
            if 'lambda_weights' in name:
                self.lambda_params.append(param)
            elif 'mu' in name or 'nu' in name:
                self.dual_params.append(param)
            else:
                self.theta_params.append(param)
      
        self.theta_optimizer = torch.optim.Adam(self.theta_params, lr=eta_theta)
        self.lambda_optimizer = torch.optim.Adam(self.lambda_params, lr=eta_lambda)
  
    def zero_grad(self):
        self.theta_optimizer.zero_grad()
        self.lambda_optimizer.zero_grad()
        for p in self.dual_params:
            if p.grad is not None:
                p.grad.zero_()
  
    def step(self, loss_dict: Dict[str, torch.Tensor]):
        self.theta_optimizer.step()
        self.lambda_optimizer.step()
      
        with torch.no_grad():
            self.model.lambda_weights.data = self.project_simplex(
                self.model.lambda_weights.data
            )
            noise = 0.01 * torch.randn_like(self.model.lambda_weights)
            self.model.lambda_weights.data.add_(noise)
            self.model.lambda_weights.data = self.project_simplex(
                self.model.lambda_weights.data
            )
  
    def state_dict(self):
        return {
            'theta_optimizer': self.theta_optimizer.state_dict(),
            'lambda_optimizer': self.lambda_optimizer.state_dict(),
            'eta_theta': self.eta_theta,
            'eta_lambda': self.eta_lambda,
            'clipgrad': self.clipgrad
        }
  
    def load_state_dict(self, state_dict):
        self.theta_optimizer.load_state_dict(state_dict['theta_optimizer'])
        self.lambda_optimizer.load_state_dict(state_dict['lambda_optimizer'])
        self.eta_theta = state_dict['eta_theta']
        self.eta_lambda = state_dict['eta_lambda']
        self.clipgrad = state_dict['clipgrad']

    @staticmethod
    def project_simplex(v: torch.Tensor) -> torch.Tensor:
        v_sorted, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(v_sorted, dim=0)
        rho = (cssv - 1) / torch.arange(1, len(v) + 1, device=v.device)
        rho_star = rho[torch.where(v_sorted > rho)[0][-1]]
        return torch.maximum(v - rho_star, torch.zeros_like(v))

class TwoTimeScaleOptimizer:
    def __init__(self, model: nn.Module, lr_theta: float = 1e-3, lr_lambda: float = 1e-4):
        self.model = model
        self.lr_theta = lr_theta
        self.lr_lambda = lr_lambda
      
        self.theta_params = [
            p for n, p in model.named_parameters()
            if not any(x in n for x in ['lambda_weights', 'mu', 'nu'])
        ]
      
        self.theta_optimizer = torch.optim.Adam(self.theta_params, lr=lr_theta)
        if model.use_lagrangian:
            self.lambda_optimizer = torch.optim.Adam(
                [model.lambda_weights], lr=lr_lambda
            )
  
    def step(self, loss_dict: Dict[str, torch.Tensor]):
        self.theta_optimizer.step()
      
        if self.model.use_lagrangian:
            self.lambda_optimizer.step()
            with torch.no_grad():
                self.model.lambda_weights.data = self.project_simplex(
                    self.model.lambda_weights.data
                )
  
    @staticmethod
    def project_simplex(v: torch.Tensor) -> torch.Tensor:
        v_sorted, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(v_sorted, dim=0)
        rho = (cssv - 1) / torch.arange(1, len(v) + 1, device=v.device)
        rho_star = rho[torch.where(v_sorted > rho)[0][-1]]
        return torch.maximum(v - rho_star, torch.zeros_like(v))