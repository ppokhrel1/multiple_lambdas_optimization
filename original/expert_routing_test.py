import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import time
import os


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


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


# Basic Enums and Classes
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


# Base Solver Class
class BasePDESolver(nn.Module):
   def __init__(self, input_dim: int, hidden_dim: int = 64):
       super().__init__()
       self.input_dim = input_dim
       self.hidden_dim = hidden_dim
       self.characteristics = None
  
   def forward(self, x: torch.Tensor) -> torch.Tensor:
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
       # x shape: [batch, channels, width]
       batchsize = x.shape[0]
      
       # Compute Fourier coefficients
       x_ft = torch.fft.rfft(x)
      
       # Initialize output array
       out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,
                          device=x.device, dtype=torch.cfloat)
      
       # Multiply relevant Fourier modes
       out_ft[:, :, :self.modes] = torch.einsum(
           "bix,iox->box",
           x_ft[:, :, :self.modes],
           self.weights
       )
      
       # Return to physical space
       x = torch.fft.irfft(out_ft, n=x.size(-1))
       return x


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
       # x shape: [batch_size, input_dim]
       batch_size = x.shape[0]
      
       # Transform to feature space
       # First reshape to [batch_size * input_dim, 1]
       x = x.view(-1, 1)
       x = self.fc0(x)
       # Reshape to [batch_size, input_dim, width]
       x = x.view(batch_size, self.input_dim, self.width)
       # Transpose to [batch_size, width, input_dim]
       x = x.transpose(1, 2)
      
       # Fourier layers
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
      
       # Transform back
       x = x.transpose(1, 2)
       x = self.fc1(x)
       x = F.gelu(x)
       x = self.fc2(x)
      
       return x.squeeze(-1)




# WENO-like Shock Capturing Solver
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
      
       # Shock detection network
       self.shock_detector = nn.Sequential(
           nn.Linear(3, hidden_dim),  # Takes local gradient info
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
       # Compute first and second derivatives using finite differences
       dx = x[:, 1:] - x[:, :-1]
       dx = F.pad(dx, (1, 0), mode='replicate')
      
       d2x = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
       d2x = F.pad(d2x, (1, 1), mode='replicate')
      
       return torch.stack([x, dx, d2x], dim=-1)
  
   def forward(self, x: torch.Tensor) -> torch.Tensor:
       batch_size = x.shape[0]
      
       # Process through main network
       x_flat = x.view(-1, 1)
       base_output = self.net(x_flat).view(batch_size, -1)
      
       # Compute shock indicators
       grads = self.compute_gradients(x)
       shock_indicators = self.shock_detector(grads)
      
       # Apply shock handling
       output = base_output * shock_indicators.squeeze(-1)
      
       return output


# Boundary-Aware Solver
class BoundaryAwareSolver(BasePDESolver):
   def __init__(self, input_dim: int, hidden_dim: int = 64):
       super().__init__(input_dim, hidden_dim)
      
       # Boundary feature extractor
       self.boundary_net = nn.Sequential(
           nn.Linear(2, hidden_dim),  # Input and distance to boundary
           nn.ReLU(),
           nn.Linear(hidden_dim, hidden_dim)
       )
      
       # Main solver network
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
       # Compute distance to domain boundaries [-1, 1]
       left_dist = x - (-1)
       right_dist = 1 - x
       return torch.min(left_dist, right_dist)
  
   def forward(self, x: torch.Tensor) -> torch.Tensor:
       batch_size = x.shape[0]
      
       # Compute boundary distances
       boundary_dist = self.compute_boundary_distance(x)
      
       # Combine input with boundary information
       x_with_dist = torch.stack([x, boundary_dist], dim=-1)
       x_flat = x_with_dist.view(-1, 2)
      
       # Process through networks
       boundary_features = self.boundary_net(x_flat)
       output = self.solver_net(boundary_features)
      
       return output.view(batch_size, -1)


# Multiscale Solver
class MultiscaleSolver(BasePDESolver):
   def __init__(self, input_dim: int, scales: List[int] = [1, 2, 4, 8]):
       super().__init__(input_dim)
       self.scales = scales
      
       # Create networks for each scale
       self.scale_nets = nn.ModuleList([
           nn.Sequential(
               nn.Linear(1, self.hidden_dim),
               nn.ReLU(),
               nn.Linear(self.hidden_dim, self.hidden_dim),
               nn.ReLU(),
               nn.Linear(self.hidden_dim, 1)
           ) for _ in scales
       ])
      
       # Scale combination network
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
      
       # Process at each scale
       multi_scale_outputs = []
       for scale, net in zip(self.scales, self.scale_nets):
           # Apply scaling through average pooling
           if scale > 1:
               # Ensure output size matches input size using interpolation
               x_scaled = F.avg_pool1d(
                   x.unsqueeze(1),
                   kernel_size=scale,
                   stride=1,
                   padding=scale//2
               ).squeeze(1)
              
               # Interpolate back to original size if needed
               if x_scaled.shape[-1] != self.input_dim:
                   x_scaled = F.interpolate(
                       x_scaled.unsqueeze(1),
                       size=self.input_dim,
                       mode='linear',
                       align_corners=True
                   ).squeeze(1)
           else:
               x_scaled = x
          
           # Process through network
           x_flat = x_scaled.view(-1, 1)
           output = net(x_flat).view(batch_size, -1)
          
           # Ensure output has correct size
           if output.shape[-1] != self.input_dim:
               output = F.interpolate(
                   output.unsqueeze(1),
                   size=self.input_dim,
                   mode='linear',
                   align_corners=True
               ).squeeze(1)
          
           multi_scale_outputs.append(output)
      
       # Combine outputs from different scales
       combined = torch.stack(multi_scale_outputs, dim=-1)
       output = self.combine_net(combined).squeeze(-1)
      
       return output


class SoftmaxExpertSystem(nn.Module):
   """Expert system using softmax-based routing"""
   def __init__(
       self,
       solvers: List[BasePDESolver],
       input_dim: int,
       hidden_dim: int = 64,
       temperature: float = 1.0,
       device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
   ):
       super().__init__()
       self.solvers = nn.ModuleList(solvers)
       self.n_experts = len(solvers)
       self.input_dim = input_dim
       self.device = device
       self.temperature = temperature
      
       # Router network
       self.router = nn.Sequential(
           nn.Linear(input_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, self.n_experts)
       )
      
       self.huber = HuberLoss(delta=1.0)
       # Expert usage tracking
       self.register_buffer('usage_count', torch.zeros(self.n_experts))
  
   def forward(
       self,
       x: torch.Tensor,
       return_all: bool = False
   ) -> Tuple[torch.Tensor, Dict]:
       # Get routing weights
       logits = self.router(x) / self.temperature
       weights = F.softmax(logits, dim=-1)
      
       # Get solutions from all solvers
       solver_outputs = []
       for solver in self.solvers:
           output = solver(x)
           solver_outputs.append(output)
      
       solver_outputs = torch.stack(solver_outputs, dim=1)
      
       # Update usage statistics
       with torch.no_grad():
           self.usage_count += weights.sum(dim=0)
      
       # Weighted combination
       combined = (solver_outputs * weights.unsqueeze(-1)).sum(dim=1)
      
       metadata = {
           'weights': weights,
           'regime_weights': weights,  # For compatibility with Lagrangian system
           'usage_count': self.usage_count.clone()
       }
      
       if return_all:
           metadata['solver_outputs'] = solver_outputs
      
       return combined, metadata
  
   def compute_loss(
       self,
       x: torch.Tensor,
       target: torch.Tensor
   ) -> Tuple[torch.Tensor, Dict]:
       # Forward pass
       combined, metadata = self.forward(x, return_all=True)
       solver_outputs = metadata['solver_outputs']
      
       # Reconstruction loss
       recon_loss = self.huber(combined, target)
      
       # Individual solver losses for load balancing
       solver_losses = []
       for i in range(self.n_experts):
           loss = self.huber(solver_outputs[:, i], target)
           solver_losses.append(loss)
       solver_losses = torch.stack(solver_losses)
      
       # Load balancing loss
       usage_prob = metadata['usage_count'] / (metadata['usage_count'].sum() + 1e-6)
       target_prob = torch.ones_like(usage_prob) / self.n_experts
       balance_loss = F.kl_div(
           usage_prob.log(), target_prob, reduction='sum'
       )
      
       # Total loss
       total_loss = recon_loss + 0.1 * balance_loss
      
       metadata.update({
           'recon_loss': recon_loss.item(),
           'balance_loss': balance_loss.item()
       })
      
       return total_loss, metadata




# New ADMM Expert System
class ADMMExpertSystem(nn.Module):
   """Expert system using ADMM-based optimization"""
   def __init__(
       self,
       solvers: List[BasePDESolver],
       input_dim: int,
       hidden_dim: int = 64,
       rho: float = 1.0,
       device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
   ):
       super().__init__()
       self.solvers = nn.ModuleList(solvers)
       self.n_experts = len(solvers)
       self.input_dim = input_dim
       self.device = device
       self.rho = rho
      
       # Router network
       self.router = nn.Sequential(
           nn.Linear(input_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, self.n_experts)
       )
      
       # ADMM variables
       self.z = nn.Parameter(torch.ones(self.n_experts) / self.n_experts)  # Auxiliary variable
       self.u = nn.Parameter(torch.zeros(self.n_experts))  # Dual variable
      
       self.huber = HuberLoss(delta=1.0)
       # Expert usage tracking
       self.register_buffer('usage_count', torch.zeros(self.n_experts))
  
   def forward(
       self,
       x: torch.Tensor,
       return_all: bool = False
   ) -> Tuple[torch.Tensor, Dict]:
       # Get routing weights
       logits = self.router(x)
       weights = F.softmax(logits, dim=-1)
      
       # Get solutions from all solvers
       solver_outputs = []
       for solver in self.solvers:
           output = solver(x)
           solver_outputs.append(output)
      
       solver_outputs = torch.stack(solver_outputs, dim=1)
      
       # Update usage statistics
       with torch.no_grad():
           self.usage_count += weights.sum(dim=0)
      
       # Weighted combination
       combined = (solver_outputs * weights.unsqueeze(-1)).sum(dim=1)
      
       metadata = {
           'weights': weights,
           'regime_weights': weights,
           'usage_count': self.usage_count.clone(),
           'z': self.z.detach(),
           'u': self.u.detach()
       }
      
       if return_all:
           metadata['solver_outputs'] = solver_outputs
      
       return combined, metadata
  
   def compute_loss(
       self,
       x: torch.Tensor,
       target: torch.Tensor
   ) -> Tuple[torch.Tensor, Dict]:
       # Forward pass
       combined, metadata = self.forward(x, return_all=True)
       solver_outputs = metadata['solver_outputs']
      
       # Reconstruction loss
       recon_loss = self.huber(combined, target)
      
       # ADMM loss components
       weights_mean = metadata['weights'].mean(dim=0)  # Average over batch
       
       # Primal residual
       r = weights_mean - self.z
       
       # Augmented Lagrangian term
       admm_loss = 0.5 * self.rho * torch.norm(r) ** 2 + torch.dot(self.u, r)
       
       # Simplex constraint penalty (z should be on probability simplex)
       simplex_loss = torch.relu(-self.z).sum() + torch.relu(self.z.sum() - 1) ** 2
       
       # Load balancing regularization
       usage_prob = metadata['usage_count'] / (metadata['usage_count'].sum() + 1e-6)
       target_prob = torch.ones_like(usage_prob) / self.n_experts
       balance_loss = F.kl_div(usage_prob.log(), target_prob, reduction='sum')
       
       # Total loss
       total_loss = recon_loss + admm_loss + 0.1 * simplex_loss + 0.1 * balance_loss
      
       metadata.update({
           'recon_loss': recon_loss.item(),
           'admm_loss': admm_loss.item(),
           'simplex_loss': simplex_loss.item(),
           'balance_loss': balance_loss.item(),
           'primal_residual': torch.norm(r).item()
       })
      
       return total_loss, metadata
  
   def update_admm_variables(self):
       """Update ADMM variables (z and u) - should be called after optimizer step"""
       with torch.no_grad():
           # Update z (project onto simplex)
           weights = self.router(torch.randn(1, self.input_dim, device=self.device))
           weights_mean = F.softmax(weights, dim=-1).mean(dim=0)
           
           # z-update with projection to simplex
           z_new = weights_mean + self.u / self.rho
           self.z.data = self._project_simplex(z_new)
           
           # u-update
           self.u.data += self.rho * (weights_mean - self.z)
   
   @staticmethod
   def _project_simplex(v: torch.Tensor) -> torch.Tensor:
       """Project onto probability simplex"""
       v_sorted, _ = torch.sort(v, descending=True)
       cssv = torch.cumsum(v_sorted, dim=0)
       rho = (cssv - 1) / torch.arange(1, len(v) + 1, device=v.device)
       rho_star = rho[torch.where(v_sorted > rho)[0][-1]]
       return torch.maximum(v - rho_star, torch.zeros_like(v))


# New Single Time Scale Lagrangian System
class SingleTimeScaleLagrangianExpertSystem(nn.Module):
   """Complete system using single time scale Lagrangian optimization"""
   def __init__(
       self,
       solvers: List[BasePDESolver],
       input_dim: int,
       hidden_dim: int = 64,
       rho: float = 1.0,
       device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
   ):
       super().__init__()
       self.solvers = nn.ModuleList(solvers)
       self.n_experts = len(solvers)
       self.input_dim = input_dim
       self.device = device
       self.rho = rho
      
       # Store solver characteristics
       self.solver_characteristics = [
           solver.characteristics for solver in solvers
       ]
      
       # Initialize router
       self.router = nn.Sequential(
           nn.Linear(input_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, self.n_experts)
       )
      
       # Lagrangian parameters - all optimized together
       self.lambda_weights = nn.Parameter(torch.ones(self.n_experts) / self.n_experts)
       self.mu = nn.Parameter(torch.zeros(1))
       self.nu = nn.Parameter(torch.zeros(self.n_experts))

       self.huber = HuberLoss(delta=1.0)
       # Expert usage tracking
       self.register_buffer('usage_count', torch.zeros(self.n_experts))
  
   def forward(
       self,
       x: torch.Tensor,
       return_all: bool = False
   ) -> Tuple[torch.Tensor, Dict]:
       # Get routing weights from router network
       router_logits = self.router(x)
       router_weights = F.softmax(router_logits, dim=-1)
      
       # Combine with Lagrangian weights
       combined_weights = F.softmax(self.lambda_weights, dim=0).unsqueeze(0) * router_weights
       combined_weights = combined_weights / (combined_weights.sum(dim=-1, keepdim=True) + 1e-8)
      
       # Get solutions from all solvers
       solver_outputs = []
       for solver in self.solvers:
           output = solver(x)
           solver_outputs.append(output)
      
       solver_outputs = torch.stack(solver_outputs, dim=1)
      
       # Update usage statistics
       with torch.no_grad():
           self.usage_count += combined_weights.sum(dim=0)
      
       # Weighted combination
       combined = (solver_outputs * combined_weights.unsqueeze(-1)).sum(dim=1)
      
       metadata = {
           'weights': combined_weights,
           'router_weights': router_weights,
           'lambda_weights': self.lambda_weights,
           'usage_count': self.usage_count.clone(),
           'solver_outputs': solver_outputs if return_all else None
       }
      
       if return_all:
           metadata['solver_outputs'] = solver_outputs
      
       return combined, metadata
  
   def compute_loss(
       self,
       x: torch.Tensor,
       target: torch.Tensor
   ) -> Tuple[torch.Tensor, Dict]:
       # Forward pass with all outputs
       combined, metadata = self.forward(x, return_all=True)
       solver_outputs = metadata['solver_outputs']
      
       # Reconstruction loss
       recon_loss = self.huber(combined, target)
      
       # Compute individual solver losses
       solver_losses = []
       for i in range(self.n_experts):
           loss = self.huber(solver_outputs[:, i], target)
           solver_losses.append(loss)
       solver_losses = torch.stack(solver_losses)
      
       # Lagrangian constraints
       g_lambda = 1 - self.lambda_weights.sum()
       h_lambda = -self.lambda_weights
      
       # Single time scale Lagrangian loss
       lagrangian = recon_loss + \
                   self.mu * g_lambda + \
                   (self.nu * h_lambda).sum() + \
                   (self.rho/2) * (g_lambda**2 + (torch.relu(h_lambda)**2).sum())
      
       # Add entropy regularization for exploration
       entropy = -(F.softmax(self.lambda_weights, dim=0) *
                  F.log_softmax(self.lambda_weights, dim=0)).sum()
       lagrangian = lagrangian - 0.01 * entropy
      
       # Load balancing
       usage_prob = metadata['usage_count'] / (metadata['usage_count'].sum() + 1e-6)
       target_prob = torch.ones_like(usage_prob) / self.n_experts
       balance_loss = F.kl_div(usage_prob.log(), target_prob, reduction='sum')
       lagrangian = lagrangian + 0.1 * balance_loss
      
       metadata.update({
           'g_lambda': g_lambda,
           'h_lambda': h_lambda,
           'recon_loss': recon_loss.item(),
           'balance_loss': balance_loss.item(),
           'solver_losses': solver_losses.detach()
       })
      
       return lagrangian, metadata


# Router Implementations
class LagrangianExpertRouter(nn.Module):
   def __init__(
       self,
       n_experts: int,
       input_dim: int,
       hidden_dim: int = 64,
       rho: float = 1.0,
       device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
   ):
       super().__init__()
       self.n_experts = n_experts
       self.input_dim = input_dim
       self.hidden_dim = hidden_dim
       self.rho = rho
       self.device = device
      
       # Feature extraction network
       self.feature_net = nn.Sequential(
           nn.Linear(input_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, hidden_dim)
       )
      
       # Regime classification network
       self.regime_net = nn.Sequential(
           nn.Linear(hidden_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, len(PhysicsRegime))
       )
      
       # Lagrangian parameters
       self.lambda_weights = nn.Parameter(torch.ones(n_experts) / n_experts)
       self.mu = nn.Parameter(torch.zeros(1))
       self.nu = nn.Parameter(torch.zeros(n_experts))
      
       # Expert usage tracking
       self.register_buffer('usage_count', torch.zeros(n_experts))
       self.register_buffer('regime_count', torch.zeros(len(PhysicsRegime)))
      
       self.solver_characteristics = None  # Set by LagrangianExpertSystem
  
   def get_regime_weights(self, x: torch.Tensor) -> torch.Tensor:
       features = self.feature_net(x)
       regime_logits = self.regime_net(features)
       return F.softmax(regime_logits, dim=-1)
  
   def compute_lagrangian_loss(
   self,
   expert_outputs: torch.Tensor,
   target: torch.Tensor,
   regime_weights: torch.Tensor
) -> Tuple[torch.Tensor, Dict]:
       # Individual expert losses with epsilon for numerical stability
       epsilon = 1e-8
       expert_losses = []
       for output in expert_outputs.unbind(dim=1):
           loss = F.mse_loss(output, target, reduction='none')
           loss = torch.mean(loss, dim=-1)
           expert_losses.append(loss)
       expert_losses = torch.stack(expert_losses)  # [n_experts, batch_size]
      
       # Constraint terms
       g_lambda = 1 - self.lambda_weights.sum()
       h_lambda = -self.lambda_weights
      
       # Expert utilization loss
       usage_prob = self.usage_count / (self.usage_count.sum() + epsilon)
       target_prob = torch.ones_like(usage_prob) / self.n_experts
       utilization_loss = F.kl_div(
           (usage_prob + epsilon).log(),
           target_prob,
           reduction='sum'
       )
      
       # Regime-based weighting with stability
       regime_expert_weights = torch.zeros(
           self.n_experts, device=self.device
       )
       for i, solver_char in enumerate(self.solver_characteristics):
           regime_idx = list(PhysicsRegime).index(solver_char.optimal_regime)
           regime_expert_weights[i] = regime_weights[:, regime_idx].mean()
      
       # Normalized weights for stability
       regime_expert_weights = F.softmax(regime_expert_weights, dim=0)
      
       # Combined loss with stability
       weighted_loss = torch.sum(
           self.lambda_weights * torch.mean(expert_losses, dim=1) *
           (1 + regime_expert_weights)
       )
      
       # Scaled Lagrangian terms
       scale = 0.1  # Reduce the impact of constraint terms
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
      
       # Get regime weights
       regime_weights = self.get_regime_weights(x)
      
       # Get combination weights
       weights = F.softmax(self.lambda_weights, dim=0)
       weights = weights.view(1, -1).expand(batch_size, -1)
      
       # Update usage statistics
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

class SingleLearningRateLagrangianOptimizer:
    """
    Implements the Lagrangian optimization using a single torch.optim.Adam 
    optimizer and a single learning rate for all parameters (theta and lambda).
    """
    def __init__(
        self,
        model: nn.Module,
        eta: float = 1e-3,  # Single learning rate for all parameters
        clipgrad: float = 1.0
    ):
        self.model = model
        self.eta = eta
        self.clipgrad = clipgrad
       
        # 🧪 Collect all parameters managed by the optimizer
        trainable_params: List[nn.Parameter] = []
        dual_params: List[nn.Parameter] = []
       
        for name, param in model.named_parameters():
            if 'mu' in name or 'nu' in name:
                # These are usually updated manually/separately, not by Adam
                dual_params.append(param)
            else:
                # Include theta and lambda_weights in the single group
                trainable_params.append(param)
        
        self.dual_params = dual_params # Store dual params for zero_grad
       
        # 🚀 Initialize a single optimizer with a single learning rate (eta)
        self.optimizer = torch.optim.Adam(trainable_params, lr=eta)
        
        # Check if lambda_weights exists (needed for step/projection)
        if not hasattr(self.model, 'lambda_weights'):
             raise AttributeError("Model must have an attribute named 'lambda_weights' for projection.")
 
    def zero_grad(self):
        """Zeroes the gradients for all parameters managed by the optimizer and the dual params."""
        self.optimizer.zero_grad()
        # Manually zero gradients for dual parameters if they aren't included in the optimizer
        for p in self.dual_params:
            if p.grad is not None:
                p.grad.zero_()
   
    def step(self):
        """Performs a single optimization step."""
        
        # Clip gradients (applied globally before step)
        torch.nn.utils.clip_grad_norm_(
             self.optimizer.param_groups[0]['params'], # Only one group now
             max_norm=self.clipgrad
        )
        
        # Update model and lambda parameters using the single learning rate
        self.optimizer.step()
       
        # 🔧 Apply projection and noise injection only to lambda_weights (manual step)
        with torch.no_grad():
            
            # Project lambda weights onto simplex
            self.model.lambda_weights.data = self.project_simplex(
                self.model.lambda_weights.data
            )
            # Add small noise to break symmetry
            noise = 0.01 * torch.randn_like(self.model.lambda_weights)
            self.model.lambda_weights.data.add_(noise)
            self.model.lambda_weights.data = self.project_simplex(
                self.model.lambda_weights.data
            )
            
    def state_dict(self):
        """Returns the state of the optimizer and its config."""
        return {
            'optimizer': self.optimizer.state_dict(),
            'eta': self.eta,
            'clipgrad': self.clipgrad
        }
   
    def load_state_dict(self, state_dict):
        """Loads the optimizer state."""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        # Re-initialize configuration parameters
        self.eta = state_dict['eta']
        self.clipgrad = state_dict['clipgrad']
 
    @staticmethod
    def project_simplex(v: torch.Tensor) -> torch.Tensor:
        """Project onto probability simplex (L1 norm <= 1, v >= 0)"""
        if v.ndim > 1:
             raise ValueError("Input tensor for project_simplex must be 1D.")
             
        v_sorted, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(v_sorted, dim=0)
        rho = (cssv - 1) / torch.arange(1, len(v) + 1, device=v.device, dtype=v.dtype)
        
        valid_indices = torch.where(v_sorted > rho)[0]
        if valid_indices.numel() == 0:
            return torch.maximum(v, torch.zeros_like(v))
            
        rho_star = rho[valid_indices[-1]]
        
        return torch.maximum(v - rho_star, torch.zeros_like(v))
    
class TwoTimeScaleLagrangianOptimizer:
   def __init__(
       self,
       model: nn.Module,
       eta_theta: float = 1e-4,  # Reduced learning rate
       eta_lambda: float = 1e-3,  # Reduced learning rate
       clipgrad: float = 1.0
   ):
       self.model = model
       self.eta_theta = eta_theta
       self.eta_lambda = eta_lambda
       self.clipgrad = clipgrad
      
       # Separate parameters
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
      
       # Initialize optimizers with gradient clipping
       self.theta_optimizer = torch.optim.Adam(self.theta_params, lr=eta_theta)
       self.lambda_optimizer = torch.optim.Adam(self.lambda_params, lr=eta_lambda)
  
   def zero_grad(self):
       self.theta_optimizer.zero_grad()
       self.lambda_optimizer.zero_grad()
       for p in self.dual_params:
           if p.grad is not None:
               p.grad.zero_()
  
   def step(self):
       """Fixed step method - no arguments needed"""
       # Update model parameters
       self.theta_optimizer.step()
      
       # Update lambda weights with momentum and projection
       self.lambda_optimizer.step()
      
       with torch.no_grad():
           # Project lambda weights onto simplex
           self.model.lambda_weights.data = self.project_simplex(
               self.model.lambda_weights.data
           )
          
           # Add small noise to break symmetry
           noise = 0.01 * torch.randn_like(self.model.lambda_weights)
           self.model.lambda_weights.data.add_(noise)
           self.model.lambda_weights.data = self.project_simplex(
               self.model.lambda_weights.data
           )


  
   def state_dict(self):
       """Returns the state of the optimizer"""
       return {
           'theta_optimizer': self.theta_optimizer.state_dict(),
           'lambda_optimizer': self.lambda_optimizer.state_dict(),
           'eta_theta': self.eta_theta,
           'eta_lambda': self.eta_lambda,
           'clipgrad': self.clipgrad
       }
  
   def load_state_dict(self, state_dict):
       """Loads the optimizer state"""
       self.theta_optimizer.load_state_dict(state_dict['theta_optimizer'])
       self.lambda_optimizer.load_state_dict(state_dict['lambda_optimizer'])
       self.eta_theta = state_dict['eta_theta']
       self.eta_lambda = state_dict['eta_lambda']
       self.clipgrad = state_dict['clipgrad']


   @staticmethod
   def project_simplex(v: torch.Tensor) -> torch.Tensor:
       """Project onto probability simplex"""
       v_sorted, _ = torch.sort(v, descending=True)
       cssv = torch.cumsum(v_sorted, dim=0)
       rho = (cssv - 1) / torch.arange(1, len(v) + 1, device=v.device)
       rho_star = rho[torch.where(v_sorted > rho)[0][-1]]
       return torch.maximum(v - rho_star, torch.zeros_like(v))





class PDEDataset(Dataset):
   def __init__(self, n_samples: int, input_dim: int, noise_level: float = 0.1, seed: int = 42):
       torch.manual_seed(seed)
       self.n_samples = n_samples
       self.input_dim = input_dim
      
       # Generate data
       x = torch.linspace(-1, 1, input_dim)
       samples_per_regime = n_samples // len(PhysicsRegime)
      
       self.data = []
       for regime in PhysicsRegime:
           # Generate regime-specific data
           if regime == PhysicsRegime.SMOOTH:
               u = torch.sin(2 * np.pi * x)
           elif regime == PhysicsRegime.SHOCK:
               u = torch.tanh(20 * x)
           elif regime == PhysicsRegime.BOUNDARY:
               u = torch.exp(-50 * x**2)
           else:  # TURBULENT
               u = torch.zeros_like(x)
               for k in range(1, 6):
                   u += torch.sin(k * np.pi * x) / k
               u += 0.2 * torch.randn_like(u)
          
           # Generate samples
           for _ in range(samples_per_regime):
               # Add noise and perturbations
               perturbed = u + noise_level * torch.randn_like(u)
               self.data.append({
                   'x': x.clone(),  # [input_dim]
                   'u': perturbed,  # [input_dim]
                   'regime_idx': torch.tensor(list(PhysicsRegime).index(regime))
               })
  
   def __len__(self):
       return len(self.data)
  
   def __getitem__(self, idx):
       item = self.data[idx]
       return {
           'x': item['x'],  # [input_dim]
           'u': item['u'],  # [input_dim]
           'regime_idx': item['regime_idx']
       }


class NavierStokes1DDataset(Dataset):
   def __init__(self, n_samples: int, input_dim: int, noise_level: float = 0.01, seed: int = 42):
       torch.manual_seed(seed)
       self.n_samples = n_samples
       self.input_dim = input_dim
      
       # Physical parameters
       self.dt = 0.001
       self.dx = 2.0 / input_dim
       self.Re = 100  # Reynolds number
       self.nu = 1.0 / self.Re
      
       # Generate samples
       self.data = []
       samples_per_regime = n_samples // len(PhysicsRegime)
      
       for regime in PhysicsRegime:
           for _ in range(samples_per_regime):
               # Generate initial condition based on regime
               u0 = self.generate_initial_condition(regime)
              
               try:
                   # Solve Burgers equation with stability checks
                   u_final = self.solve_burgers(u0)
                  
                   # Check for NaN or inf values
                   if torch.isnan(u_final).any() or torch.isinf(u_final).any():
                       print(f"Warning: NaN or inf detected in solution for {regime}")
                       continue
                  
                   # Add noise with bounds checking
                   noise = noise_level * torch.randn_like(u_final)
                   u_final = u_final + noise
                  
                   # Clip values to prevent extremes
                   u_final = torch.clamp(u_final, min=-10.0, max=10.0)
                  
                   self.data.append({
                       'x': u0.clone(),  # Initial condition
                       'u': u_final,     # Final state
                       'regime_idx': torch.tensor(list(PhysicsRegime).index(regime))
                   })
                  
               except Exception as e:
                   print(f"Error generating sample for {regime}: {str(e)}")
                   continue
      
       # Verify dataset integrity
       self.verify_dataset()


   def generate_initial_condition(self, regime: PhysicsRegime) -> torch.Tensor:
       """Generate initial conditions based on physics regime"""
       x = torch.linspace(-1, 1, self.input_dim)
      
       if regime == PhysicsRegime.SMOOTH:
           # Smooth sinusoidal initial condition
           u0 = torch.sin(2 * np.pi * x) + 0.5 * torch.sin(4 * np.pi * x)
      
       elif regime == PhysicsRegime.SHOCK:
           # Step function that will develop into shock
           u0 = torch.zeros_like(x)
           u0[x < 0] = 1.0
           u0[x >= 0] = -1.0
           # Smooth the discontinuity slightly
           u0 = torch.nn.functional.conv1d(
               u0.view(1, 1, -1),
               torch.ones(1, 1, 5) / 5,
               padding=2
           ).view(-1)
      
       elif regime == PhysicsRegime.BOUNDARY:
           # Gaussian pulse near boundaries
           u0 = (torch.exp(-50 * (x + 0.8)**2) +
                 torch.exp(-50 * (x - 0.8)**2))
      
       else:  # TURBULENT
           # Superposition of waves with random phases
           u0 = torch.zeros_like(x)
           for k in range(1, 6):
               phase = 2 * np.pi * torch.rand(1)
               u0 += torch.sin(k * np.pi * x + phase) / k
           # Add small random perturbations
           u0 += 0.1 * torch.randn_like(x)
      
       # Normalize to prevent extreme values
       u0 = u0 / (torch.max(torch.abs(u0)) + 1e-8)
      
       return u0


   def verify_dataset(self):
       """Verify dataset integrity and print statistics"""
       x_values = []
       u_values = []
      
       for item in self.data:
           x_values.extend(item['x'].numpy())
           u_values.extend(item['u'].numpy())
      
       x_array = np.array(x_values)
       u_array = np.array(u_values)
      
       print(f"\nDataset Statistics:")
       print(f"Number of samples: {len(self.data)}")
       print(f"X range: [{np.min(x_array):.4f}, {np.max(x_array):.4f}]")
       print(f"U range: [{np.min(u_array):.4f}, {np.max(u_array):.4f}]")
       print(f"X mean: {np.mean(x_array):.4f}, std: {np.std(x_array):.4f}")
       print(f"U mean: {np.mean(u_array):.4f}, std: {np.std(u_array):.4f}")
      
       # Check for NaN or inf values
       if np.isnan(x_array).any() or np.isnan(u_array).any():
           print("Warning: NaN values detected in dataset")
       if np.isinf(x_array).any() or np.isinf(u_array).any():
           print("Warning: Infinite values detected in dataset")
  
   def solve_burgers(self, u: torch.Tensor, n_steps: int = 100) -> torch.Tensor:
       """Solve 1D Burgers equation with stability checks"""
       u_current = u.clone()
      
       try:
           for step in range(n_steps):
               # Spatial derivatives with periodic boundary conditions
               # First derivative (advection term)
               du_dx = torch.zeros_like(u_current)
               du_dx[1:-1] = (u_current[2:] - u_current[:-2]) / (2 * self.dx)
               du_dx[0] = (u_current[1] - u_current[-1]) / (2 * self.dx)
               du_dx[-1] = (u_current[0] - u_current[-2]) / (2 * self.dx)
              
               # Second derivative (diffusion term)
               d2u_dx2 = torch.zeros_like(u_current)
               d2u_dx2[1:-1] = (u_current[2:] - 2*u_current[1:-1] + u_current[:-2]) / (self.dx**2)
               d2u_dx2[0] = (u_current[1] - 2*u_current[0] + u_current[-1]) / (self.dx**2)
               d2u_dx2[-1] = (u_current[0] - 2*u_current[-1] + u_current[-2]) / (self.dx**2)
              
               # Compute update with stability check
               update = self.dt * (-u_current * du_dx + self.nu * d2u_dx2)
              
               if torch.isnan(update).any() or torch.isinf(update).any():
                   print(f"Numerical instability detected at step {step}")
                   return u_current
              
               # Update with CFL condition check
               max_velocity = torch.max(torch.abs(u_current))
               cfl = max_velocity * self.dt / self.dx
               if cfl > 1.0:
                   print(f"CFL condition violated at step {step}: {cfl:.4f}")
                   self.dt = self.dt * 0.5  # Reduce time step
                   continue
              
               u_current = u_current + update
              
               # Add periodic boundary condition enforcement
               u_current[0] = u_current[-2]
               u_current[-1] = u_current[1]
              
           return u_current
          
       except Exception as e:
           print(f"Error in solve_burgers: {str(e)}")
           return torch.zeros_like(u)
  
   def __len__(self):
       return len(self.data)
  
   def __getitem__(self, idx):
       item = self.data[idx]
      
       # Final verification before returning
       if torch.isnan(item['x']).any() or torch.isnan(item['u']).any():
           print(f"Warning: NaN detected in item {idx}")
           # Return a valid dummy sample instead
           return {
               'x': torch.zeros_like(item['x']),
               'u': torch.zeros_like(item['u']),
               'regime_idx': item['regime_idx']
           }
          
       return item




class Trainer:
   def __init__(
       self,
       model: nn.Module,
       train_loader: DataLoader,
       val_loader: DataLoader,
       learning_rate: float = 1e-4,
       delta: float = 1.0,
       device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
   ):
       self.model = model.to(device)
       self.train_loader = train_loader
       self.val_loader = val_loader
       self.device = device
       self.huber = HuberLoss(delta=delta)
      
       if isinstance(model, LagrangianExpertSystem):
           self.optimizer = TwoTimeScaleLagrangianOptimizer(
               model,
               eta_theta=learning_rate,
               eta_lambda=learning_rate * 0.1
           )
       else:
           self.optimizer = torch.optim.Adam(
               model.parameters(),
               lr=learning_rate
           )
      
       self.metrics = defaultdict(list)


   def train_epoch(self) -> Dict[str, float]:
       """Train for one epoch"""
       self.model.train()
       epoch_metrics = defaultdict(list)
      
       for batch in self.train_loader:
           metrics = self.train_step(batch)
          
           # Skip if NaN encountered
           if np.isnan(metrics['loss']):
               continue
              
           for k, v in metrics.items():
               epoch_metrics[k].append(v)
      
       # Compute mean metrics for epoch
       return {k: np.mean(v) for k, v in epoch_metrics.items()}


   def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
       x = batch['x'].to(self.device)
       u = batch['u'].to(self.device)
       regime_idx = batch['regime_idx'].to(self.device)
      
       self.optimizer.zero_grad()
      
       if isinstance(self.model, (LagrangianExpertSystem, SingleTimeScaleLagrangianExpertSystem)):
           loss, metadata = self.model.compute_loss(x, u)
          
           if torch.isnan(loss):
               return {
                   'loss': float('nan'),
                   'constraint_violation': float('nan'),
                   'min_weight': float('nan')
               }
          
           loss.backward()
          
           # Fixed: Call step without arguments
           self.optimizer.step()
          
           metrics = {
               'loss': loss.item(),
               'constraint_violation': metadata.get('g_lambda', torch.tensor(0.0)).abs().item(),
               'min_weight': metadata['weights'].min().item()
           }
          
           if 'huber_loss' in metadata:
               metrics['huber_loss'] = metadata['huber_loss']
           elif 'recon_loss' in metadata:
               metrics['huber_loss'] = metadata['recon_loss']
          
           return metrics
          
       elif isinstance(self.model, ADMMExpertSystem):
           loss, metadata = self.model.compute_loss(x, u)
          
           if torch.isnan(loss):
               return {
                   'loss': float('nan'),
                   'primal_residual': float('nan')
               }
          
           loss.backward()
           torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
           self.optimizer.step()
           
           # Update ADMM variables after optimizer step
           self.model.update_admm_variables()
          
           return {
               'loss': loss.item(),
               'huber_loss': metadata['recon_loss'],
               'primal_residual': metadata['primal_residual'],
               'admm_loss': metadata['admm_loss']
           }
          
       else:
           output, metadata = self.model(x)
           loss = self.huber(output, u)
          
           if torch.isnan(loss):
               return {
                   'loss': float('nan'),
                   'regime_accuracy': 0.0
               }
          
           loss.backward()
           torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
           self.optimizer.step()
          
           return {
               'loss': loss.item(),
               'huber_loss': loss.item(),
               'regime_accuracy': (
                   metadata['regime_weights'].argmax(dim=1) == regime_idx
               ).float().mean().item()
           }


   def validate(self) -> Dict[str, float]:
       self.model.eval()
       val_metrics = defaultdict(list)
      
       with torch.no_grad():
           for batch in self.val_loader:
               x = batch['x'].to(self.device)
               u = batch['u'].to(self.device)
               regime_idx = batch['regime_idx'].to(self.device)
              
               try:
                   if isinstance(self.model, (LagrangianExpertSystem, SingleTimeScaleLagrangianExpertSystem)):
                       output, metadata = self.model(x)
                       huber_loss = self.huber(output, u)
                       loss, loss_metadata = self.model.compute_loss(x, u)
                      
                       metrics = {
                           'loss': loss.item(),
                           'huber_loss': huber_loss.item(),
                           'constraint_violation': loss_metadata.get('g_lambda', torch.tensor(0.0)).abs().item()
                       }
                   elif isinstance(self.model, ADMMExpertSystem):
                       output, metadata = self.model(x)
                       huber_loss = self.huber(output, u)
                       loss, loss_metadata = self.model.compute_loss(x, u)
                      
                       metrics = {
                           'loss': loss.item(),
                           'huber_loss': huber_loss.item(),
                           'primal_residual': loss_metadata.get('primal_residual', 0.0)
                       }
                   else:
                       output, metadata = self.model(x)
                       huber_loss = self.huber(output, u)
                      
                       metrics = {
                           'loss': huber_loss.item(),
                           'huber_loss': huber_loss.item(),
                           'regime_accuracy': (
                               metadata['regime_weights'].argmax(dim=1) == regime_idx
                           ).float().mean().item()
                       }
                  
                   for k, v in metrics.items():
                       val_metrics[k].append(v)
                      
               except RuntimeError as e:
                   print(f"Error in validation: {str(e)}")
                   continue
      
       return {k: np.mean(v) for k, v in val_metrics.items()}




class LagrangianExpertSystem(nn.Module):
   """Complete system combining multiple solvers using Lagrangian formulation"""
   def __init__(
       self,
       solvers: List[BasePDESolver],
       input_dim: int,
       hidden_dim: int = 64,
       rho: float = 1.0,
       device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
   ):
       super().__init__()
       self.solvers = nn.ModuleList(solvers)
       self.n_experts = len(solvers)
       self.input_dim = input_dim
       self.device = device
       self.rho = rho
      
       # Store solver characteristics
       self.solver_characteristics = [
           solver.characteristics for solver in solvers
       ]
      
       # Initialize router
       self.router = LagrangianExpertRouter(
           n_experts=self.n_experts,
           input_dim=input_dim,
           hidden_dim=hidden_dim,
           rho=rho,
           device=device
       )
       self.router.solver_characteristics = self.solver_characteristics
      
       # Initialize Lagrangian parameters
       self.lambda_weights = nn.Parameter(torch.ones(self.n_experts) / self.n_experts)
       self.mu = nn.Parameter(torch.zeros(1))
       self.nu = nn.Parameter(torch.zeros(self.n_experts))


       self.huber = HuberLoss(delta=1.0)
  
   def forward(
       self,
       x: torch.Tensor,
       return_all: bool = False
   ) -> Tuple[torch.Tensor, Dict]:
       # Get routing weights and metadata
       weights, router_metadata = self.router(x, return_regime=True)
      
       # Get solutions from all solvers
       solver_outputs = []
       for solver in self.solvers:
           output = solver(x)
           solver_outputs.append(output)
      
       solver_outputs = torch.stack(solver_outputs, dim=1)  # [batch_size, n_solvers, input_dim]
      
       # Use raw lambda weights (without softmax) for training signal
       combined_weights = self.lambda_weights.view(1, -1, 1).expand(x.size(0), -1, x.size(1))
      
       # Weighted combination
       combined = (solver_outputs * combined_weights).sum(dim=1)
      
       metadata = {
           'weights': self.lambda_weights,  # Use raw weights
           'router_metadata': router_metadata,
           'solver_outputs': solver_outputs
       }
      
       return combined, metadata
  
   def compute_loss(
       self,
       x: torch.Tensor,
       target: torch.Tensor
   ) -> Tuple[torch.Tensor, Dict]:
       # Forward pass with all outputs
       _, metadata = self.forward(x, return_all=True)
      
       # Get regime weights
       regime_weights = metadata['router_metadata']['regime_weights']
       solver_outputs = metadata['solver_outputs']
      
       # Compute source-specific losses
       source_losses = []
       for i in range(self.n_experts):
           loss = F.mse_loss(solver_outputs[:, i], target)
           source_losses.append(loss)
       source_losses = torch.stack(source_losses)
      
       # Weighted reconstruction loss using raw lambda weights
       recon_loss = (self.lambda_weights * source_losses).sum()
      
       # Strong constraint penalties
       constraint_scale = 10.0  # Increase constraint penalty
       g_lambda = 1 - self.lambda_weights.sum()
       h_lambda = -self.lambda_weights
      
       # Augmented Lagrangian with stronger penalties
       lagrangian = recon_loss + \
                   constraint_scale * (
                       self.mu * g_lambda + \
                       (self.nu * h_lambda).sum() + \
                       (self.rho/2) * (g_lambda**2 + (torch.relu(h_lambda)**2).sum())
                   )
      
       # Add entropy regularization to encourage exploration
       entropy = -(F.softmax(self.lambda_weights, dim=0) *
                  F.log_softmax(self.lambda_weights, dim=0)).sum()
       lagrangian = lagrangian - 0.01 * entropy  # Small entropy term
      
       metadata = {
           'g_lambda': g_lambda,
           'h_lambda': h_lambda,
           'recon_loss': recon_loss.item(),
           'weights': self.lambda_weights,
           'regime_weights': regime_weights,
           'source_losses': source_losses
       }
      
       return lagrangian, metadata


def plot_solver_outputs(
   models: Dict[str, nn.Module],
   dataset: NavierStokes1DDataset,
   epoch: int,
   save_dir: str = 'plots',
   num_samples: int = 4
):
   """Plot solver outputs comparing all approaches"""
   os.makedirs(save_dir, exist_ok=True)
   
   for model_name, model in models.items():
       if hasattr(model, 'eval'):
           model.eval()
  
   # Create figure with columns for each method and num_samples rows
   n_methods = len(models)
   fig, axes = plt.subplots(num_samples, n_methods, figsize=(5*n_methods, 5*num_samples))
   
   if num_samples == 1:
       axes = axes.reshape(1, -1)
  
   # Color scheme
   colors = {
       'Initial': 'gray',
       'True': 'black',
       'FNO': 'blue',
       'WENO': 'red',
       'Boundary': 'green',
       'Multiscale': 'purple',
       'Combined': 'orange'
   }
  
   with torch.no_grad():
       # Get random samples
       indices = np.random.randint(len(dataset), size=num_samples)
       x_coords = torch.linspace(0, 1, dataset.input_dim)
      
       for i, idx in enumerate(indices):
           sample = dataset[idx]
           x = sample['x'].to(next(iter(models.values())).device).unsqueeze(0)
           u = sample['u']
          
           # Process each model
           for j, (model_name, model) in enumerate(models.items()):
               output, metadata = model(x, return_all=True)
               ax = axes[i, j] if n_methods > 1 else axes[i]
              
               # Plot initial condition
               ax.plot(x_coords, x[0].cpu(), '-',
                      color=colors['Initial'], label='Initial', alpha=0.5)
              
               # Plot true solution
               ax.plot(x_coords, u.cpu(), '-',
                      color=colors['True'], label='True', linewidth=2)
              
               # Plot individual solver outputs
               if 'solver_outputs' in metadata:
                   solver_outputs = metadata['solver_outputs'][0]
                   weights = metadata['weights']
                  
                   # Ensure weights is 1D
                   if isinstance(weights, torch.Tensor):
                       weights = weights.detach().cpu()
                       if weights.dim() > 1:
                           weights = weights.mean(dim=0)  # Average over batch
                  
                   solver_names = ['FNO', 'WENO', 'Boundary', 'Multiscale']
                   for k, (solver_output, name) in enumerate(zip(solver_outputs, solver_names)):
                       weight_value = weights[k].item() if isinstance(weights, torch.Tensor) and len(weights) > k else 0.25
                       ax.plot(x_coords, solver_output.cpu(), '--',
                              color=colors[name], alpha=0.5,
                              label=f'{name} (w={weight_value:.2f})')
              
               # Plot combined solution
               if output.dim() > 1:
                   combined_output = output[0]
               else:
                   combined_output = output
               ax.plot(x_coords, combined_output.cpu(), '-',
                      color=colors['Combined'], label='Combined', linewidth=2)
              
               # Add regime information if available
               regime_info = ""
               if hasattr(metadata, 'get') and metadata.get('regimes') is not None:
                   regime = metadata['regimes'][0]
                   regime_info = f" - Regime: {regime}"
              
               ax.set_title(f'{model_name} - Sample {i+1}{regime_info}')
               ax.set_xlabel('Spatial Position')
               ax.set_ylabel('Loss')
               ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
               ax.grid(True)
  
   plt.tight_layout()
   plt.savefig(f'{save_dir}/all_methods_outputs_epoch_{epoch}.png',
               bbox_inches='tight', dpi=300)
   plt.close()




def plot_training_metrics(
   metrics: Dict[str, Dict],
   epoch: int,
   save_dir: str = 'plots'
):
   """Plot training metrics comparing all approaches"""
   os.makedirs(save_dir, exist_ok=True)
  
   fig, axes = plt.subplots(2, 2, figsize=(15, 15))
  
   # Color scheme for better visualization
   colors = {
       'lagrangian': {'train': 'royalblue', 'val': 'lightblue'},
       'softmax': {'train': 'salmon', 'val': 'lightcoral'},
       'admm': {'train': 'green', 'val': 'lightgreen'},
       'single_time_scale': {'train': 'purple', 'val': 'lavender'}
   }
  
   # Loss plot
   ax = axes[0, 0]
   for model_type in metrics.keys():
       if f'train_loss' in metrics[model_type]:
           ax.plot(metrics[model_type]['train_loss'],
                  label=f'{model_type.capitalize()} Train',
                  color=colors.get(model_type, {}).get('train', 'black'),
                  linestyle='-')
           ax.plot(metrics[model_type]['val_loss'],
                  label=f'{model_type.capitalize()} Val',
                  color=colors.get(model_type, {}).get('val', 'gray'),
                  linestyle='--')
   ax.set_title('Loss Comparison')
   ax.set_xlabel('Epoch')
   ax.set_ylabel('Loss')
   ax.legend()
   ax.grid(True)
  
   # Huber Loss plot
   ax = axes[0, 1]
   for model_type in metrics.keys():
       if f'train_huber_loss' in metrics[model_type]:
           ax.plot(metrics[model_type]['train_huber_loss'],
                  label=f'{model_type.capitalize()} Train',
                  color=colors.get(model_type, {}).get('train', 'black'),
                  linestyle='-')
           ax.plot(metrics[model_type]['val_huber_loss'],
                  label=f'{model_type.capitalize()} Val',
                  color=colors.get(model_type, {}).get('val', 'gray'),
                  linestyle='--')
   ax.set_title('Huber Loss Comparison')
   ax.set_xlabel('Epoch')
   ax.set_ylabel('Huber Loss')
   ax.legend()
   ax.grid(True)
  
   # Constraint/Balance plot (log scale)
   ax = axes[1, 0]
   for model_type in metrics.keys():
       if 'constraint_violation' in metrics[model_type]:
           ax.semilogy(metrics[model_type]['constraint_violation'],
                      label=f'{model_type.capitalize()} Constraint',
                      color=colors.get(model_type, {}).get('train', 'black'))
       if 'primal_residual' in metrics[model_type]:
           ax.semilogy(metrics[model_type]['primal_residual'],
                      label=f'{model_type.capitalize()} ADMM Residual',
                      color=colors.get(model_type, {}).get('train', 'black'),
                      linestyle=':')
   ax.set_title('Constraint/Residual Metrics')
   ax.set_xlabel('Epoch')
   ax.set_ylabel('Value (log scale)')
   ax.legend()
   ax.grid(True)
  
   # Weight stability plot
   ax = axes[1, 1]
   for model_type in metrics.keys():
       if 'min_weight' in metrics[model_type]:
           ax.plot(metrics[model_type]['min_weight'],
                  label=f'{model_type.capitalize()} Min Weight',
                  color=colors.get(model_type, {}).get('train', 'black'))
   ax.set_title('Minimum Weight Evolution')
   ax.set_xlabel('Epoch')
   ax.set_ylabel('Minimum Weight')
   ax.legend()
   ax.grid(True)
  
   # Add timestamp and epoch information
   plt.figtext(0.99, 0.01, f'Epoch {epoch}',
               ha='right', va='bottom', fontsize=8)
   plt.figtext(0.01, 0.01, time.strftime("%Y-%m-%d %H:%M:%S"),
               ha='left', va='bottom', fontsize=8)
  
   plt.tight_layout()
   plt.savefig(f'{save_dir}/all_methods_metrics_epoch_{epoch}.png',
               bbox_inches='tight', dpi=300)
   plt.close()




def main():
   # Parameters
   class Config:
       n_samples = 2000
       input_dim = 64
       hidden_dim = 128
       batch_size = 16  # Reduced batch size
       n_epochs = 500
       lr = 1e-4  # Reduced learning rate
       rho = 0.1  # Reduced rho
       temperature = 2.0  # Increased temperature for softer softmax
       device = 'cuda' if torch.cuda.is_available() else 'cpu'
       save_dir = 'results'
  
   cfg = Config()
  
   # Create save directories
   os.makedirs(f'{cfg.save_dir}/lagrangian', exist_ok=True)
   os.makedirs(f'{cfg.save_dir}/softmax', exist_ok=True)
   os.makedirs(f'{cfg.save_dir}/admm', exist_ok=True)
   os.makedirs(f'{cfg.save_dir}/single_time_scale', exist_ok=True)
  
   # Create datasets
   train_dataset = NavierStokes1DDataset(
       n_samples=cfg.n_samples,
       input_dim=cfg.input_dim
   )
  
   val_dataset = NavierStokes1DDataset(
       n_samples=cfg.n_samples // 5,
       input_dim=cfg.input_dim,
       seed=43
   )
  
   train_loader = DataLoader(
       train_dataset,
       batch_size=cfg.batch_size,
       shuffle=True
   )
  
   val_loader = DataLoader(
       val_dataset,
       batch_size=cfg.batch_size
   )
  
   # Initialize solvers
   def create_solvers():
       return [
           FourierNeuralOperator(
               input_dim=cfg.input_dim,
               modes=16,
               width=32
           ),
           ShockCapturingSolver(
               input_dim=cfg.input_dim,
               hidden_dim=cfg.hidden_dim
           ),
           BoundaryAwareSolver(
               input_dim=cfg.input_dim,
               hidden_dim=cfg.hidden_dim
           ),
           MultiscaleSolver(
               input_dim=cfg.input_dim,
               scales=[1, 2, 4, 8]
           )
       ]
  
   # Create all models
   lagrangian_model = LagrangianExpertSystem(
       solvers=create_solvers(),
       input_dim=cfg.input_dim,
       hidden_dim=cfg.hidden_dim,
       rho=cfg.rho,
       device=cfg.device
   ).to(cfg.device)
  
   softmax_model = SoftmaxExpertSystem(
       solvers=create_solvers(),
       input_dim=cfg.input_dim,
       hidden_dim=cfg.hidden_dim,
       temperature=cfg.temperature,
       device=cfg.device
   ).to(cfg.device)
   
   # New models
   admm_model = ADMMExpertSystem(
       solvers=create_solvers(),
       input_dim=cfg.input_dim,
       hidden_dim=cfg.hidden_dim,
       rho=cfg.rho,
       device=cfg.device
   ).to(cfg.device)
   
   single_time_scale_model = SingleTimeScaleLagrangianExpertSystem(
       solvers=create_solvers(),
       input_dim=cfg.input_dim,
       hidden_dim=cfg.hidden_dim,
       rho=cfg.rho,
       device=cfg.device
   ).to(cfg.device)
  
   # Initialize trainers for all models
   models = {
       'lagrangian': lagrangian_model,
       'softmax': softmax_model,
       'admm': admm_model,
       'single_time_scale': single_time_scale_model
   }
   
   trainers = {}
   for name, model in models.items():
       trainers[name] = Trainer(
           model=model,
           train_loader=train_loader,
           val_loader=val_loader,
           learning_rate=cfg.lr
       )
  
   # Training loop
   metrics = {
       'lagrangian': defaultdict(list),
       'softmax': defaultdict(list),
       'admm': defaultdict(list),
       'single_time_scale': defaultdict(list)
   }
   best_val_loss = {
       'lagrangian': float('inf'),
       'softmax': float('inf'),
       'admm': float('inf'),
       'single_time_scale': float('inf')
   }
  
   try:
       for epoch in range(cfg.n_epochs):
           print(f"\nEpoch {epoch+1}/{cfg.n_epochs}")
          
           # Train all models
           for model_name, trainer in trainers.items():
               print(f"\n{model_name.capitalize()} System:")
               train_metrics = trainer.train_epoch()
               val_metrics = trainer.validate()
              
               for k, v in train_metrics.items():
                   print(f"Train {k}: {v}")
                   metrics[model_name][f'train_{k}'].append(v)
               for k, v in val_metrics.items():
                   print(f"Val {k}: {v}")
                   metrics[model_name][f'val_{k}'].append(v)
                   
               # Save best models
               if val_metrics['loss'] < best_val_loss[model_name]:
                   best_val_loss[model_name] = val_metrics['loss']
                   torch.save({
                       'epoch': epoch,
                       'model_state_dict': models[model_name].state_dict(),
                       'val_loss': val_metrics['loss']
                   }, f'{cfg.save_dir}/{model_name}/best_model.pth')
          
           # Plot results periodically
           if (epoch + 1) % 10 == 0:
               plot_solver_outputs(
                   models=models,
                   dataset=val_dataset,
                   epoch=epoch + 1,
                   save_dir=f'{cfg.save_dir}'
               )
               
               plot_training_metrics(
                   metrics,
                   epoch + 1,
                   f'{cfg.save_dir}'
               )
              
               # Plot comparison
               plot_comparison(
                   metrics,
                   epoch + 1,
                   cfg.save_dir
               )
  
   except KeyboardInterrupt:
       print("\nTraining interrupted by user")
  
   finally:
       # Save final models and metrics
       for model_name, model in models.items():
           torch.save({
               'model_state_dict': model.state_dict(),
               'metrics': metrics[model_name]
           }, f'{cfg.save_dir}/{model_name}/final_model.pth')
      
       # Plot final results
       plot_comparison(metrics, cfg.n_epochs, cfg.save_dir)
      
       print("\nTraining completed. Results saved in", cfg.save_dir)


def plot_comparison(metrics: Dict, epoch: int, save_dir: str):
   """Plot comparison between all approaches"""
   fig, axes = plt.subplots(2, 2, figsize=(15, 15))
  
   # Training loss comparison
   if 'train_loss' in metrics['lagrangian']:
       for model_type in metrics.keys():
           if 'train_loss' in metrics[model_type]:
               axes[0, 0].plot(metrics[model_type]['train_loss'],
                              label=model_type.capitalize())
       axes[0, 0].set_title('Training Loss Comparison')
       axes[0, 0].set_xlabel('Epoch')
       axes[0, 0].set_ylabel('Loss')
       axes[0, 0].legend()
       axes[0, 0].grid(True)
  
   # Validation loss comparison
   if 'val_loss' in metrics['lagrangian']:
       for model_type in metrics.keys():
           if 'val_loss' in metrics[model_type]:
               axes[0, 1].plot(metrics[model_type]['val_loss'],
                              label=model_type.capitalize())
       axes[0, 1].set_title('Validation Loss Comparison')
       axes[0, 1].set_xlabel('Epoch')
       axes[0, 1].set_ylabel('Loss')
       axes[0, 1].legend()
       axes[0, 1].grid(True)
  
   # Weight evolution
   if 'min_weight' in metrics['lagrangian']:
       for model_type in metrics.keys():
           if 'min_weight' in metrics[model_type]:
               axes[1, 0].plot(metrics[model_type]['min_weight'],
                              label=model_type.capitalize())
       axes[1, 0].set_title('Minimum Weight Evolution')
       axes[1, 0].set_xlabel('Epoch')
       axes[1, 0].set_ylabel('Minimum Weight')
       axes[1, 0].legend()
       axes[1, 0].grid(True)
  
   # Constraint violation / ADMM residual
   ax = axes[1, 1]
   for model_type in metrics.keys():
       if 'constraint_violation' in metrics[model_type]:
           ax.semilogy(metrics[model_type]['constraint_violation'],
                      label=f'{model_type.capitalize()} Constraint')
       if 'primal_residual' in metrics[model_type]:
           ax.semilogy(metrics[model_type]['primal_residual'],
                      label=f'{model_type.capitalize()} ADMM Residual',
                      linestyle='--')
   if 'constraint_violation' in metrics['lagrangian'] or 'primal_residual' in metrics['admm']:
       ax.set_title('Constraint/Residual Metrics')
       ax.set_xlabel('Epoch')
       ax.set_ylabel('Value (log scale)')
       ax.legend()
       ax.grid(True)
  
   plt.tight_layout()
   plt.savefig(f'{save_dir}/all_methods_comparison_epoch_{epoch}.png')
   plt.close()




if __name__ == "__main__":
   main()