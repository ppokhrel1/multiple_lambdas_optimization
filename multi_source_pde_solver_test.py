import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import os
from matplotlib.lines import Line2D




from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class NavierStokesDataset(Dataset):
   def __init__(
       self,
       n_samples: int,
       domain_size: int = 64,
       dt: float = 0.001,
       n_steps: int = 100,
       Re: float = 100,
       noise_level: float = 0.01,
       seed: int = 42
   ):
       torch.manual_seed(seed)
       self.domain_size = domain_size
       self.samples = []
      
       # Physical parameters
       self.dt = dt
       self.dx = 2.0 / domain_size
       self.Re = Re
       self.nu = 1.0 / Re
      
       for i in range(n_samples):
           # Generate initial conditions
           u, v, w = self.generate_initial_conditions()
          
           # Solve Navier-Stokes
           solution = self.solve_navier_stokes(u, v, w, n_steps)  # [n_steps, 3, H, W]
          
           # Add noise
           solution = solution + noise_level * torch.randn_like(solution)
          
           # Create grid
           x = torch.linspace(-1, 1, domain_size)
           y = torch.linspace(-1, 1, domain_size)
           grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
           grid = torch.stack([grid_x, grid_y], dim=0)  # [2, H, W]
          
           self.samples.append({
               'grid': grid,
               'solution': solution[-1],  # Only keep final state [3, H, W]
               'parameters': {'Re': self.Re}
           })
   def generate_initial_conditions(self):
       """Generate initial velocity and vorticity fields"""
       # Initialize velocity components and vorticity
       u = torch.zeros((self.domain_size, self.domain_size))  # x-velocity
       v = torch.zeros((self.domain_size, self.domain_size))  # y-velocity
       w = torch.zeros((self.domain_size, self.domain_size))  # vorticity
      
       # Add initial perturbation (e.g., lid-driven cavity)
       u[-1, :] = 1.0  # Top lid moving right with u = 1
      
       # Compute initial vorticity from velocity field
       w = self.compute_vorticity(u, v)
      
       return u, v, w
  
   def compute_vorticity(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
       """Compute vorticity from velocity components"""
       # Compute spatial derivatives
       du_dy = (u[1:, :] - u[:-1, :]) / self.dx
       dv_dx = (v[:, 1:] - v[:, :-1]) / self.dx
      
       # Pad derivatives to match original size
       du_dy = torch.nn.functional.pad(du_dy, (0, 0, 0, 1))
       dv_dx = torch.nn.functional.pad(dv_dx, (0, 1, 0, 0))
      
       # Vorticity = dv/dx - du/dy
       return dv_dx - du_dy
  
   def solve_poisson(self, f: torch.Tensor, boundary_conditions: str = 'dirichlet') -> torch.Tensor:
       """Solve Poisson equation ∇²ψ = f using iterative method"""
       psi = torch.zeros_like(f)
       dx2 = self.dx * self.dx
       error = 1.0
       tolerance = 1e-4
       max_iter = 1000
       iter_count = 0
      
       while error > tolerance and iter_count < max_iter:
           psi_old = psi.clone()
          
           # Jacobi iteration
           psi[1:-1, 1:-1] = 0.25 * (
               psi_old[1:-1, 2:] + psi_old[1:-1, :-2] +
               psi_old[2:, 1:-1] + psi_old[:-2, 1:-1] -
               dx2 * f[1:-1, 1:-1]
           )
          
           # Apply boundary conditions
           if boundary_conditions == 'dirichlet':
               psi[0, :] = 0  # Bottom
               psi[-1, :] = 0  # Top
               psi[:, 0] = 0  # Left
               psi[:, -1] = 0  # Right
          
           error = torch.max(torch.abs(psi - psi_old)).item()
           iter_count += 1
      
       return psi
  
   def compute_velocity_from_stream(self, psi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
       """Compute velocity components from stream function"""
       u = torch.zeros_like(psi)
       v = torch.zeros_like(psi)
      
       # u = ∂ψ/∂y
       u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * self.dx)
      
       # v = -∂ψ/∂x
       v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * self.dx)
      
       return u, v
  
   def solve_navier_stokes(
       self,
       u: torch.Tensor,
       v: torch.Tensor,
       w: torch.Tensor,
       n_steps: int
   ) -> torch.Tensor:
       """Solve Navier-Stokes equations using stream function-vorticity formulation"""
       solutions = []
      
       for _ in range(n_steps):
           # 1. Solve Poisson equation for stream function
           psi = self.solve_poisson(-w)
          
           # 2. Compute velocity components from stream function
           u, v = self.compute_velocity_from_stream(psi)
          
           # 3. Update vorticity using vorticity transport equation
           w_new = w.clone()
          
           # Compute spatial derivatives of vorticity
           dw_dx = torch.zeros_like(w)
           dw_dy = torch.zeros_like(w)
           d2w_dx2 = torch.zeros_like(w)
           d2w_dy2 = torch.zeros_like(w)
          
           # Central differences for interior points
           dw_dx[1:-1, 1:-1] = (w[1:-1, 2:] - w[1:-1, :-2]) / (2 * self.dx)
           dw_dy[1:-1, 1:-1] = (w[2:, 1:-1] - w[:-2, 1:-1]) / (2 * self.dx)
          
           d2w_dx2[1:-1, 1:-1] = (w[1:-1, 2:] - 2*w[1:-1, 1:-1] + w[1:-1, :-2]) / (self.dx * self.dx)
           d2w_dy2[1:-1, 1:-1] = (w[2:, 1:-1] - 2*w[1:-1, 1:-1] + w[:-2, 1:-1]) / (self.dx * self.dx)
          
           # Update vorticity
           w_new[1:-1, 1:-1] = w[1:-1, 1:-1] + self.dt * (
               -u[1:-1, 1:-1] * dw_dx[1:-1, 1:-1]
               -v[1:-1, 1:-1] * dw_dy[1:-1, 1:-1]
               + self.nu * (d2w_dx2[1:-1, 1:-1] + d2w_dy2[1:-1, 1:-1])
           )
          
           # Apply boundary conditions for vorticity
           w_new[0, :] = -2 * psi[1, :] / (self.dx * self.dx)  # Bottom
           w_new[-1, :] = -2 * psi[-2, :] / (self.dx * self.dx)  # Top
           w_new[:, 0] = -2 * psi[:, 1] / (self.dx * self.dx)  # Left
           w_new[:, -1] = -2 * psi[:, -2] / (self.dx * self.dx)  # Right
          
           # Update vorticity
           w = w_new
          
           # Store solution
           solution = torch.stack([u, v, w], dim=0)
           solutions.append(solution)
      
       return torch.stack(solutions)


   def __len__(self):
       return len(self.samples)


   def __getitem__(self, idx):
       return self.samples[idx]




def plot_flow_field(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, title: str):
   """Plot velocity and vorticity fields"""
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
  
   # Plot velocity magnitude
   velocity_mag = torch.sqrt(u**2 + v**2)
   im1 = ax1.imshow(velocity_mag.numpy(), cmap='viridis')
   ax1.set_title('Velocity Magnitude')
   plt.colorbar(im1, ax=ax1)
  
   # Plot vorticity
   im2 = ax2.imshow(w.numpy(), cmap='RdBu')
   ax2.set_title('Vorticity')
   plt.colorbar(im2, ax=ax2)
  
   fig.suptitle(title)
   plt.tight_layout()
   return fig




class PDEDataset(Dataset):
   """2D PDE Dataset Generator"""
   def __init__(
       self,
       n_samples: int,
       domain_size: int,
       noise_level: float = 0.01,
       seed: int = 42
   ):
       torch.manual_seed(seed)
       self.domain_size = domain_size
       self.samples = []
      
       # Generate spatial grid
       x = torch.linspace(-1, 1, domain_size)
       y = torch.linspace(-1, 1, domain_size)
       self.grid_x, self.grid_y = torch.meshgrid(x, y, indexing='ij')
      
       # Generate samples
       for i in range(n_samples):
           # Different types of solutions for variety
           if i % 3 == 0:
               u = self.generate_smooth_solution()
           elif i % 3 == 1:
               u = self.generate_shock_solution()
           else:
               u = self.generate_turbulent_solution()
          
           # Add noise
           u = u + noise_level * torch.randn_like(u)
          
           # Stack grid coordinates
           grid = torch.stack([self.grid_x, self.grid_y], dim=0)
          
           self.samples.append({
               'grid': grid,  # [2, domain_size, domain_size]
               'solution': u.unsqueeze(0),  # [1, domain_size, domain_size]
               'type': i % 3  # Solution type for analysis
           })
  
   def generate_smooth_solution(self) -> torch.Tensor:
       """Generate smooth solution"""
       return torch.sin(2*np.pi*self.grid_x) * torch.cos(2*np.pi*self.grid_y)
  
   def generate_shock_solution(self) -> torch.Tensor:
       """Generate solution with discontinuities"""
       return torch.tanh(10 * (self.grid_x + self.grid_y))
  
   def generate_turbulent_solution(self) -> torch.Tensor:
       """Generate turbulent-like solution"""
       k1, k2 = torch.randint(1, 5, (2,))
       u = torch.sin(k1*np.pi*self.grid_x) * torch.sin(k2*np.pi*self.grid_y)
       u += 0.2 * torch.randn_like(u)  # Add fluctuations
       return u
  
   def __len__(self):
       return len(self.samples)
  
   def __getitem__(self, idx):
       return self.samples[idx]


class SpectralConv2d(nn.Module):
   """2D Spectral Convolution layer"""
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
       # 2D Fourier Transform
       x_ft = torch.fft.rfft2(x)
      
       # Initialize output array
       out_ft = torch.zeros(
           x.shape[0], self.out_channels, x.shape[-2], x.shape[-1]//2 + 1,
           device=x.device, dtype=torch.cfloat
       )
       # Multiply relevant Fourier modes
       out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
           "bixy,ioxy->boxy",
           x_ft[:, :, :self.modes1, :self.modes2],
           self.weights1
       )
      
       # Return to physical space
       x = torch.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))
       return x


class FourierBlock2D(nn.Module):
   """2D Fourier Layer"""
   def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
       super().__init__()
       #self.modes1 = min(modes1, in_channels)  # Ensure modes1 <= in_channels
       #self.modes2 = min(modes2, out_channels)  # Ensure modes2 <= out_channels
       self.conv = SpectralConv2d(in_channels, out_channels, modes1, modes2)
       self.w = nn.Conv2d(in_channels, out_channels, 1)
      
   def forward(self, x: torch.Tensor) -> torch.Tensor:
       x1 = self.conv(x)
       x2 = self.w(x)
       return x1 + x2


class BasePDESolver(nn.Module):
   """Base class for PDE solvers"""
   def __init__(self, domain_size: int, hidden_dim: int = 64):
       super().__init__()
       self.domain_size = domain_size
       self.hidden_dim = hidden_dim
  
   def forward(self, grid: torch.Tensor) -> torch.Tensor:
       raise NotImplementedError
  
   def compute_gradients(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
       """Compute spatial gradients using finite differences"""
       # First derivatives
       u_x = (u[:, :, 1:, :] - u[:, :, :-1, :]) * self.domain_size
       u_y = (u[:, :, :, 1:] - u[:, :, :, :-1]) * self.domain_size
      
       # Pad to maintain size
       u_x = F.pad(u_x, (0, 0, 0, 1))
       u_y = F.pad(u_y, (0, 1, 0, 0))
      
       return u_x, u_y


class Utils:
   @staticmethod
   def create_directory(path: str):
       """Create directory if it doesn't exist"""
       if not os.path.exists(path):
           os.makedirs(path)
  
   @staticmethod
   def save_plot(fig: plt.Figure, path: str, filename: str):
       """Save plot with proper directory handling"""
       Utils.create_directory(path)
       fig.savefig(os.path.join(path, filename))
       plt.close(fig)
  
   @staticmethod
   def plot_2d_solution(
       true_sol: torch.Tensor,
       pred_sol: torch.Tensor,
       title: str = "Solution Comparison"
   ) -> plt.Figure:
       """Plot true vs predicted 2D solutions"""
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
      
       # Plot true solution
       im1 = ax1.imshow(true_sol.cpu().numpy(), cmap='viridis')
       ax1.set_title("True Solution")
       plt.colorbar(im1, ax=ax1)
      
       # Plot predicted solution
       im2 = ax2.imshow(pred_sol.cpu().numpy(), cmap='viridis')
       ax2.set_title("Predicted Solution")
       plt.colorbar(im2, ax=ax2)
      
       fig.suptitle(title)
       return fig




class FourierPDESolver(BasePDESolver):
   """Fourier-based PDE solver"""
   def __init__(self, domain_size: int, mode1: int = 16, mode2: int = 9, width: int = 16):
       super().__init__(domain_size)
       self.mode1 = mode1
       self.model2 = mode2
       self.width = width
      
       # Lifting network
       self.lift = nn.Sequential(
           nn.Conv2d(2, width, 1),  # 2 input channels for grid_x, grid_y
           nn.GELU(),
           nn.Conv2d(width, width, 1)
       )
      
       # Fourier layers
       self.fourier_layers = nn.ModuleList([
           FourierBlock2D(width, width, mode1, mode2) for _ in range(4)
       ])
      
       # Projection network
       self.project = nn.Sequential(
           nn.Conv2d(width, width, 1),
           nn.GELU(),
           nn.Conv2d(width, 3, 1)
       )
  
   def forward(self, grid: torch.Tensor) -> torch.Tensor:
       x = self.lift(grid)
      
       # Apply Fourier layers
       for layer in self.fourier_layers:
           x = layer(x)
           x = F.gelu(x)
      
       # Project to output
       x = self.project(x)
       return x


class WENOSolver(BasePDESolver):
   """WENO-like solver for shock regions"""
   def __init__(self, domain_size: int, hidden_dim: int = 64):
       super().__init__(domain_size, hidden_dim)
      
       # Main network
       self.net = nn.Sequential(
           nn.Conv2d(2, hidden_dim, 3, padding=1),
           nn.ReLU(),
           nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
           nn.ReLU(),
           nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
           nn.ReLU(),
           nn.Conv2d(hidden_dim, 3, 1)
       )
      
       # Shock detector
       self.shock_detector = nn.Sequential(
           nn.Conv2d(4*2, hidden_dim, 3, padding=1),  # 4 = grid + gradients
           nn.ReLU(),
           nn.Conv2d(hidden_dim, 1, 1),
           nn.Sigmoid()
       )
  
   def forward(self, grid: torch.Tensor) -> torch.Tensor:
       # Base solution
       u = self.net(grid)
      
       # Compute gradients for shock detection
       u_x, u_y = self.compute_gradients(u)
       shock_input = torch.cat([grid, u_x, u_y], dim=1)
       shock_weights = self.shock_detector(shock_input)
      
       # Apply shock handling
       return u * shock_weights


class DeepONetSolver(BasePDESolver):
   """DeepONet-based solver"""
   def __init__(self, domain_size: int, hidden_dim: int = 64, branch_dim: int = 40):
       super().__init__(domain_size, hidden_dim)
       self.branch_dim = branch_dim
      
       # Branch network (processes grid points)
       self.branch_net = nn.Sequential(
           nn.Conv2d(2, hidden_dim, 3, padding=1),
           nn.ReLU(),
           nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
           nn.ReLU(),
           nn.Conv2d(hidden_dim, branch_dim, 1)
       )
      
       # Trunk network (processes spatial coordinates)
       self.trunk_net = nn.Sequential(
           nn.Linear(2, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, branch_dim)
       )
      
       # Final projection layer
       self.projection = nn.Conv2d(branch_dim, 3, 1)
  
   def forward(self, grid: torch.Tensor) -> torch.Tensor:
       batch_size = grid.shape[0]
      
       # Process grid through branch network
       # grid shape: [batch_size, 2, H, W]
       branch_features = self.branch_net(grid)  # [batch_size, branch_dim, H, W]
      
       # Create coordinate grid for trunk network
       x = torch.linspace(-1, 1, self.domain_size, device=grid.device)
       y = torch.linspace(-1, 1, self.domain_size, device=grid.device)
       xx, yy = torch.meshgrid(x, y, indexing='ij')
       coords = torch.stack([xx, yy], dim=0)  # [2, H, W]
       coords = coords.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch_size, 2, H, W]
      
       # Process through trunk network with reshaping
       coords_flat = coords.permute(0, 2, 3, 1).reshape(-1, 2)  # [batch_size*H*W, 2]
       trunk_out = self.trunk_net(coords_flat)  # [batch_size*H*W, branch_dim]
       trunk_features = trunk_out.view(batch_size, self.domain_size, self.domain_size, self.branch_dim)
       trunk_features = trunk_features.permute(0, 3, 1, 2)  # [batch_size, branch_dim, H, W]
      
       # Combine features
       output = branch_features * trunk_features
       output = self.projection(output)  # [batch_size, 1, H, W]
      
       return output


class MultiSolverSystem(nn.Module):
   def forward(self, grid: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
       # Ensure proper input dimensions
       if grid.dim() == 3:
           grid = grid.unsqueeze(0)  # Add batch dimension
      
       # Get solutions from all solvers
       solver_outputs = []
       for solver in self.solvers:
           output = solver(grid)  # [batch_size, 1, H, W]
           # Ensure output has correct shape
           if output.dim() == 3:
               output = output.unsqueeze(1)
           solver_outputs.append(output)
      
       solver_outputs = torch.stack(solver_outputs, dim=1)  # [batch_size, n_solvers, 1, H, W]
       # Get routing weights
       if self.use_lagrangian:
           # Use Lagrangian weights
           weights = self.lambda_weights # F.softmax(self.lambda_weights, dim=0)  # [n_solvers]
           weights = weights.view(1, -1, 1, 1, 1).expand(grid.size(0), -1, 1, 1, 1)
       else:
           # Use router weights
           weights, router_info = self.router(grid)  # [batch_size, n_solvers]
           weights = weights.view(1, -1, 1, 1, 1).expand(grid.size(0), -1, 1, 1, 1) #weights.view(-1, len(self.solvers), 1, 1, 1)
      
       # Combine solutions
       output = (solver_outputs * weights).sum(dim=1)  # [batch_size, 1, H, W]
      
       return output, {
           'solver_outputs': solver_outputs,
           'weights': weights.squeeze(-1).squeeze(-1).squeeze(-1),  # [batch_size, n_solvers]
           'lambda_weights': self.lambda_weights if self.use_lagrangian else None
       }


  
   def compute_loss(
       self,
       grid: torch.Tensor,
       target: torch.Tensor,
       meta: Dict
   ) -> Tuple[torch.Tensor, Dict]:
       # Handle the dimension mismatch
       # target: [batch_size, 3, H, W]
       # solver_outputs: [batch_size, n_solvers, 1, H, W]
      
       # First average over the velocity components to get a single channel
       target_avg = target.mean(dim=1, keepdim=True)  # [batch_size, 1, H, W]
      
       # Now expand to match solver outputs
       target_expanded = target_avg.unsqueeze(1).expand(-1, len(self.solvers), -1, -1, -1)
      
       # Reconstruction loss
       recon_loss = F.mse_loss(meta['solver_outputs'], target_expanded)
      
       if self.use_lagrangian:
           # Lagrangian constraints
           g_lambda = 1 - self.lambda_weights.sum()
           h_lambda = -self.lambda_weights
          
           # Augmented Lagrangian terms
           loss = (recon_loss +
                  self.mu * g_lambda +
                  (self.nu * h_lambda).sum() +
                  (self.rho/2) * (g_lambda**2 + (torch.relu(h_lambda)**2).sum()))
       else:
           # Add sparsity regularization for router
           sparsity_loss = 0.1 * (1 - (meta['weights'] > 0.1).float().mean())
           loss = recon_loss + sparsity_loss
      
       # Compute weight statistics
       if self.use_lagrangian:
           weight_sum = self.lambda_weights.sum().item()
           weight_mean = self.lambda_weights.mean().item()
       else:
           weight_sum = meta['weights'].sum().item()
           weight_mean = meta['weights'].mean().item()
      
       return loss, {
           'recon_loss': recon_loss.item(),
           'total_loss': loss.item(),
           'weight_sum': weight_sum,
           'weight_mean': weight_mean
       }








class MultiResolutionSolver(BasePDESolver):
   """Multi-resolution solver for complex patterns"""
   def __init__(self, domain_size: int, scales: List[int] = [1, 2, 4]):
       super().__init__(domain_size)
       self.scales = scales
      
       # Networks for each scale
       self.scale_nets = nn.ModuleList([
           nn.Sequential(
               nn.Conv2d(2, self.hidden_dim, 3, padding=1),
               nn.ReLU(),
               nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
               nn.ReLU(),
               nn.Conv2d(self.hidden_dim, 1, 1)
           ) for _ in scales
       ])
      
       # Feature fusion network
       self.fusion_net = nn.Sequential(
           nn.Conv2d(len(scales), self.hidden_dim, 1),
           nn.ReLU(),
           nn.Conv2d(self.hidden_dim, 3, 1)
       )
  
   def forward(self, grid: torch.Tensor) -> torch.Tensor:
       # Process at different scales
       multi_scale_outputs = []
      
       for scale, net in zip(self.scales, self.scale_nets):
           if scale > 1:
               # Downsample
               x = F.avg_pool2d(grid, scale)
               # Process
               out = net(x)
               # Upsample back
               out = F.interpolate(out, size=(self.domain_size, self.domain_size),
                                mode='bilinear', align_corners=True)
           else:
               out = net(grid)
          
           multi_scale_outputs.append(out)
      
       # Combine outputs
       combined = torch.cat(multi_scale_outputs, dim=1)
       return self.fusion_net(combined)


class AdaptiveRouter(nn.Module):
   def __init__(self, domain_size: int, n_solvers: int, hidden_dim: int = 64):
       super().__init__()
      
       # Single network that processes grid and outputs weights directly
       self.router_net = nn.Sequential(
           # Feature extraction
           nn.Conv2d(2, hidden_dim, 3, padding=1),
           nn.ReLU(),
           nn.AdaptiveAvgPool2d(8),  # Reduce spatial dimensions
           nn.Flatten(),
          
           # Weight prediction
           nn.Linear(64 * hidden_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, n_solvers),
       )
  
   def forward(self, grid: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
       # Get raw logits
       logits = self.router_net(grid)  # [batch_size, n_solvers]
      
       # Apply softmax to get normalized weights
       weights = F.softmax(logits, dim=-1)  # [batch_size, n_solvers]
       #weights = logits
       return weights, {
           'logits': logits,
           'weights': weights
       }








class MultiSolverSystem(nn.Module):
   """Complete system combining multiple solvers with adaptive routing"""
   def __init__(
       self,
       domain_size: int,
       hidden_dim: int = 64,
       use_lagrangian: bool = True,
       rho: float = 1.0
   ):
       super().__init__()
      
       # Initialize solvers
       self.solvers = nn.ModuleList([
           FourierPDESolver(domain_size),
           WENOSolver(domain_size),
           DeepONetSolver(domain_size),
           MultiResolutionSolver(domain_size)
       ])
      
       # Store parameters
       self.domain_size = domain_size
       self.hidden_dim = hidden_dim
       self.use_lagrangian = use_lagrangian
       self.rho = rho
      
       # Initialize router if not using Lagrangian
       if not use_lagrangian:
           self.router = AdaptiveRouter(
               domain_size=domain_size,
               n_solvers=len(self.solvers),
               hidden_dim=hidden_dim
           )
      
       # Lagrangian parameters
       if use_lagrangian:
           self.lambda_weights = nn.Parameter(torch.randn(len(self.solvers)) * 0.01 + 1.0/len(self.solvers) )
           self.mu = nn.Parameter(torch.zeros(1))
           self.nu = nn.Parameter(torch.zeros(len(self.solvers)))


  
   def forward(self, grid: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
       # Ensure proper input dimensions
       if grid.dim() == 3:
           grid = grid.unsqueeze(0)  # Add batch dimension
      
       # Get solutions from all solvers
       solver_outputs = []
       for solver in self.solvers:
           output = solver(grid)  # Should be [batch_size, 1, H, W]
           solver_outputs.append(output)

       solver_outputs = torch.stack(solver_outputs, dim=1)  # [batch_size, n_solvers, 1, H, W]
      
       # Get routing weights
       if self.use_lagrangian:
           # Use Lagrangian weights
           #weights = F.softmax(self.lambda_weights, dim=0)  # [n_solvers]
           weights = self.lambda_weights
           weights = weights.view(1, -1, 1, 1, 1).expand(grid.size(0), -1, 1, 1, 1)
       else:
           # Use router weights
           weights, router_info = self.router(grid)  # [batch_size, n_solvers]
           weights =  weights.view(-1, len(self.solvers), 1, 1, 1)
      
       # Combine solutions
       output = (solver_outputs * weights).sum(dim=1)  # [batch_size, 1, H, W]
      
       return output, {
           'solver_outputs': solver_outputs,
           'weights': weights.squeeze(-1).squeeze(-1).squeeze(-1),  # [batch_size, n_solvers]
           'lambda_weights': self.lambda_weights if self.use_lagrangian else None
       }
  
   def compute_loss(
   self,
   grid: torch.Tensor,
   target: torch.Tensor,
   meta: Dict
) -> Tuple[torch.Tensor, Dict]:
       # Handle the dimension mismatch
       # target: [batch_size, 3, H, W]
       # solver_outputs: [batch_size, n_solvers, 1, H, W]
      
       # First average over the velocity components to get a single channel
       target_avg = target.mean(dim=1, keepdim=True)  # [batch_size, 1, H, W]
      
       # Now expand to match solver outputs
       target_expanded = target.unsqueeze(1).expand(-1, len(self.solvers), -1, -1, -1)  # [batch_size, n_solvers, 3, H, W]
      
       # Reconstruction loss
       #print(meta)
       #print(meta['solver_outputs'].shape, target_expanded.shape, target_avg.shape)
       recon_loss = F.mse_loss(meta['solver_outputs'], target_expanded)
      
       if self.use_lagrangian:
           # Lagrangian constraints
           g_lambda = 1 - self.lambda_weights.sum()
           h_lambda = -self.lambda_weights
          
           # Augmented Lagrangian terms
           loss = (recon_loss +
               self.mu * g_lambda +
               (self.nu * h_lambda).sum() +
               (self.rho/2) * (g_lambda**2 + (torch.relu(h_lambda)**2).sum()))
       else:
           # Add sparsity regularization for router
           sparsity_loss = 0.1 * (1 - (meta['weights'] > 0.1).float().mean())
           loss = recon_loss + sparsity_loss
      
       # Compute weight statistics
       if self.use_lagrangian:
           weight_sum = self.lambda_weights.sum().item()
           weight_mean = self.lambda_weights.mean().item()
       else:
           weight_sum = meta['weights'].sum().item()
           weight_mean = meta['weights'].mean().item()
      
       return loss, {
           'recon_loss': recon_loss.item(),
           'total_loss': loss.item(),
           'weight_sum': weight_sum,
           'weight_mean': weight_mean
       }




def plot_solver_outputs(
   model: nn.Module,
   dataset: NavierStokesDataset,
   epoch: int,
   save_dir: str = 'plots',
   num_samples: int = 4
):
   """Plot solver outputs with detailed comparison for 2D Navier-Stokes equations"""
   os.makedirs(save_dir, exist_ok=True)
   model.eval()
  
   fig, axes = plt.subplots(num_samples, 3, figsize=(20, 5*num_samples))  # Increased width for legend
   if num_samples == 1:
       axes = axes.reshape(1, -1)
  
   # Color scheme
   colors = {
       'FNO': 'blue',
       'WENO': 'red',
       'DeepONet': 'green',
       'MultiRes': 'purple',
       'Combined': 'orange'
   }
  
   with torch.no_grad():
       indices = np.random.randint(len(dataset), size=num_samples)
      
       for i, idx in enumerate(indices):
           sample = dataset[idx]
           device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
           grid = sample['grid'].to(device).unsqueeze(0)
           target = sample['solution'].to(device)
          
           output, metadata = model(grid)
          
           # Create a list to store legend handles and labels
           legend_elements = []
          
           for j, (title, data) in enumerate([
               ('Velocity-U', target[0]),
               ('Velocity-V', target[1]),
               ('Vorticity', target[2])
           ]):
               ax = axes[i, j]
              
               im = ax.imshow(data.cpu(), cmap='RdBu', aspect='auto')
               plt.colorbar(im, ax=ax)
               ax.set_title(f'{title}')
              
               if 'solver_outputs' in metadata:
                   solver_outputs = metadata['solver_outputs'][0]
                   weights = metadata['weights'][0]
                  
                   for k, name in enumerate(['FNO', 'WENO', 'DeepONet', 'MultiRes']):
                       weight = weights[k].item() if isinstance(weights, torch.Tensor) else weights[k]
                       contour = ax.contour(
                           solver_outputs[k, j].cpu(),
                           levels=5,
                           colors=colors[name],
                           alpha=0.3,
                           linewidths=1
                       )
                       # Store legend info instead of using clabel
                       if j == 0:  # Only add to legend for first column
                           legend_elements.append(
                               Line2D([0], [0], color=colors[name], label=f'{name} (w={weight:.2f})')
                           )
              
               ax.set_xlabel('x')
               ax.set_ylabel('y')
              
               combined_contour = ax.contour(
                   output[0, 0].cpu(),
                   levels=5,
                   colors=colors['Combined'],
                   linewidths=2
               )
               if j == 0:  # Only add to legend for first column
                   legend_elements.append(
                       Line2D([0], [0], color=colors['Combined'], label='Combined')
                   )
          
           # Add legend to the right of the last subplot in each row
           if i == 0:  # Only add legend for first row
               fig.legend(
                   handles=legend_elements,
                   loc='center left',
                   bbox_to_anchor=(1.02, 0.5),
                   fontsize=18
               )
  
   plt.tight_layout()
   plt.savefig(f'{save_dir}/solver_outputs_epoch_{epoch}.png', bbox_inches='tight', dpi=300,
               bbox_extra_artists=(fig.legends), pad_inches=0.5)
   plt.close(fig)




def plot_true_solutions(sample, save_dir, epoch):
    """Plot the true solutions for Velocity-U, Velocity-V, and Vorticity."""
    titles = ['Velocity-U', 'Velocity-V', 'Vorticity']
    target = sample['solution']
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for i, title in enumerate(titles):
        im = axes[i].imshow(target[i].cpu(), cmap='RdBu')
        plt.colorbar(im, ax=axes[i])
        axes[i].set_title(f'True {title}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/true_solutions_epoch_{epoch}.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_predictions(trainer, sample, save_dir, epoch, method_name, colors):
    """Plot predictions for a given trainer (Lagrangian or Softmax) alongside true solutions."""
    titles = ['Velocity-U', 'Velocity-V', 'Vorticity']
    target = sample['solution']
    device = next(trainer.model.parameters()).device
    grid = sample['grid'].to(device).unsqueeze(0)
    
    output = None
    with torch.no_grad():
        output, metadata = trainer.model(grid)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    for i, title in enumerate(titles):
        im = axes[i].imshow(output[0, 0].cpu(), cmap='RdBu', alpha=0.3)
        plt.colorbar(im, ax=axes[i])
        # contour = axes[i].contour(
        #     output[0][0].cpu(),
        #     levels=5,
        #     colors=colors[method_name],
        #     alpha=0.7,
        #     linewidths=2
        # )
        mse = F.mse_loss(output[0, 0], target[i]).item()
        axes[i].set_title(f'{method_name} {title}\nMSE: {mse:.2e}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{method_name.lower()}_predictions_epoch_{epoch}.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_solver_weights(all_outputs, save_dir, epoch, colors):
    """Plot the solver weights for both Lagrangian and Softmax methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    solver_names = ['FNO', 'WENO', 'DeepONet', 'MultiRes']
    x = np.arange(len(solver_names))
    width = 0.35
    
    lag_weights = [all_outputs['Lagrangian'][1]['weights'][0][i].item() for i in range(len(solver_names))]
    soft_weights = [all_outputs['Softmax'][1]['weights'][0][i].item() for i in range(len(solver_names))]
    
    ax.bar(x - width/2, lag_weights, width, label='Lagrangian', color=colors['Lagrangian'])
    ax.bar(x + width/2, soft_weights, width, label='Softmax', color=colors['Softmax'])
    
    ax.set_xlabel('Solvers')
    ax.set_ylabel('Weight Value')
    ax.set_title('Solver Weights Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(solver_names)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/solver_weights_epoch_{epoch}.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_comparison_separate(trainers, dataset, epoch, save_dir):
    """Generate separate plots for true solutions, predictions, and solver weights."""
    # Define colors for different methods
    colors = {
        'FNO': 'blue',
        'WENO': 'red',
        'DeepONet': 'green',
        'MultiRes': 'purple',
        'Combined': 'orange',
        'Lagrangian': 'crimson',
        'Softmax': 'navy'
    }
    
    # Select a random sample from the dataset
    sample_idx = np.random.randint(len(dataset))
    sample = dataset[sample_idx]
    
    # Plot true solutions
    plot_true_solutions(sample, save_dir, epoch)
    
    # Dictionary to store outputs and metadata for weight plotting
    all_outputs = {}
    
    # Plot predictions for each trainer
    for method_name, trainer in trainers.items():
        plot_predictions(trainer, sample, save_dir, epoch, method_name.capitalize(), colors)
        # Store outputs and metadata
        device = next(trainer.model.parameters()).device
        grid = sample['grid'].to(device).unsqueeze(0)
        with torch.no_grad():
            output, metadata = trainer.model(grid)
        all_outputs[method_name.capitalize()] = (output, metadata)
    
    # Plot solver weights
    plot_solver_weights(all_outputs, save_dir, epoch, colors)








class TwoTimeScaleOptimizer:
   """Custom optimizer for Lagrangian system"""
   def __init__(
       self,
       model: MultiSolverSystem,
       lr_theta: float = 1e-3,
       lr_lambda: float = 1e-4
   ):
       self.model = model
       self.lr_theta = lr_theta
       self.lr_lambda = lr_lambda
      
       # Separate parameters
       self.theta_params = [
           p for n, p in model.named_parameters()
           if not any(x in n for x in ['lambda_weights', 'mu', 'nu'])
       ]
      
       # Initialize optimizers
       self.theta_optimizer = torch.optim.Adam(self.theta_params, lr=lr_theta)
       if model.use_lagrangian:
           self.lambda_optimizer = torch.optim.Adam(
               [model.lambda_weights], lr=lr_lambda
           )
  
   def step(self, loss_dict: Dict[str, torch.Tensor]):
       # Update model parameters
       self.theta_optimizer.step()
      
       if self.model.use_lagrangian:
           # Update Lagrangian parameters
           self.lambda_optimizer.step()
          
           # Project weights onto simplex
           with torch.no_grad():
               self.model.lambda_weights.data = self.project_simplex(
                   self.model.lambda_weights.data
               )
  
   @staticmethod
   def project_simplex(v: torch.Tensor) -> torch.Tensor:
       """Project onto probability simplex"""
       v_sorted, _ = torch.sort(v, descending=True)
       cssv = torch.cumsum(v_sorted, dim=0)
       rho = (cssv - 1) / torch.arange(1, len(v) + 1, device=v.device)
       rho_star = rho[torch.where(v_sorted > rho)[0][-1]]
       return torch.maximum(v - rho_star, torch.zeros_like(v))


class Trainer:
   def __init__(
       self,
       model: MultiSolverSystem,
       train_loader: DataLoader,
       val_loader: DataLoader,
       lr_theta: float = 1e-3,
       lr_lambda: float = 1e-4,
       device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
   ):
       self.model = model.to(device)
       self.train_loader = train_loader
       self.val_loader = val_loader
       self.device = device
      
       # Initialize optimizer
       self.optimizer = TwoTimeScaleOptimizer(
           model, lr_theta=lr_theta, lr_lambda=lr_lambda
       )
      
       # Initialize metrics storage
       self.metrics = {
          'train_weights': [],
          'total_loss': [],
          'recon_loss': [],
          'val_total_loss': [],
          'val_recon_loss': [],
          'weight_mean': [],
          'weight_std': [],
          'weights_sum': [],
       }
       self.all_metrics = {}
  
   def train_epoch(self) -> Dict[str, float]:
       self.model.train()
       epoch_metrics = defaultdict(list)
      
       for batch in self.train_loader:
           batch_metrics = self.train_step(batch)
           for k, v in batch_metrics.items():
               epoch_metrics[k].append(v)
       averaged_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
       #averaged_metrics = epoch_metrics

       for k, v in averaged_metrics.items():
           if f'train_{k}' not in self.all_metrics:
               self.all_metrics[f'train_{k}'] = [v]
           else:
               self.all_metrics[f'train_{k}'].append(v)
       #print(averaged_metrics, "\n")
       for k, v in averaged_metrics.items():
           self.metrics[f'{k}'].append(v)
       #print("inside training epoch: ", self.all_metrics)
       return averaged_metrics

   def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
       grid = batch['grid'].to(self.device)
       target = batch['solution'].to(self.device)
      
       # Forward pass
       output, meta = self.model(grid)
      
       # Compute loss
       loss, loss_meta = self.model.compute_loss(grid, target, meta)
      
       # Backward pass
       self.optimizer.theta_optimizer.zero_grad()
       if self.model.use_lagrangian:
           self.optimizer.lambda_optimizer.zero_grad()
      
       loss.backward()
       self.optimizer.step(loss_meta)
      
       # Add additional metrics
       step_metrics = loss_meta.copy()
       # Add weight statistics
       weights = meta['weights']
       step_metrics.update({
           'weight_mean': weights.mean().item(),
           'weight_std': weights.std().item(),
       })
      
       # Add constraint/sparsity metrics
       if self.model.use_lagrangian:
           g_lambda = 1 - self.model.lambda_weights.sum()
           h_lambda = -self.model.lambda_weights
           step_metrics['constraint_violation'] = (abs(g_lambda) + torch.relu(h_lambda).sum()).item()
           weights_to_store = self.model.lambda_weights.detach().cpu().numpy()
       else:
           # Compute sparsity metric for softmax
           step_metrics['sparsity'] = (weights > 0.1).float().mean().item()
           weights_to_store = weights.mean(dim=0).detach().cpu().numpy()

       step_metrics = {k: [v] for k, v in step_metrics.items()}
       self.metrics['train_weights'].append(weights_to_store)

       #print("train weihts at trainer: ", self.metrics['train_weights'])
       return {
          'total_loss': loss_meta['total_loss'],
          'recon_loss': loss_meta['recon_loss'],
          'weight_mean': meta['weights'].mean().item(),
          'weight_std': meta['weights'].std().item(),
          'weights_sum': meta['weights'].sum().item()
       }




  
   def validate(self) -> Dict[str, float]:
       self.model.eval()
       val_metrics = defaultdict(list)
      
       with torch.no_grad():
           for batch in self.val_loader:
               grid = batch['grid'].to(self.device)
               target = batch['solution'].to(self.device)
              
               output, meta = self.model(grid)
               loss, loss_meta = self.model.compute_loss(grid, target, meta)
              
               for k, v in loss_meta.items():
                   val_metrics[k].append(v)
      
       val_metrics_ = {k: np.mean(v) for k, v in val_metrics.items()}
       for k, v in val_metrics_.items():
           if f'val_{k}' not in self.all_metrics:
               self.all_metrics[f'val_{k}'] = [v]
           else:
               self.all_metrics[f'val_{k}'].append(v)

       return val_metrics_

  
def plot_detailed_analysis(trainers, dataset, epoch, save_dir):
   """Plot detailed analysis of weights and predictions"""
   plt.close('all')
  
   # Create figure with subplots
   fig = plt.figure(figsize=(20, 8))
   gs = plt.GridSpec(1, 3)  # Changed to 4 columns
  
   # 1. Lagrangian Weight Evolution (leftmost)
   ax1 = fig.add_subplot(gs[0, 0])
   solver_names = ['FNO', 'WENO', 'DeepONet', 'MultiRes']
   colors = ['blue', 'red', 'green', 'purple']
  
   weights_history = trainers['lagrangian'].all_metrics['train_weight_mean']
   print(trainers['lagrangian'].metrics)
   epochs = range(len(weights_history))
   #epochs = range(1, epoch+1)
   #print("weights history at detailed analysis: ", weights_history[-1].shape)
   #for i, solver_name in enumerate(solver_names):
   ax1.plot(epochs,
           weights_history,
           label='Lagrangian',
           color='red',
           alpha=0.7)
  
   
   #ax1.axhline(y=0.25, color='gray', linestyle=':', alpha=0.5)
  
   # 2. Softmax Weight Evolution (middle-left)
   #ax2 = fig.add_subplot(gs[0, 1])
  
   weights_history = trainers['softmax'].all_metrics['train_weight_mean']
   epochs = range(len(weights_history))
   print("weight history: ", weights_history)
   #for i, solver_name in enumerate(solver_names):
   ax1.plot(epochs,
           weights_history,
           label='Softmax',
           color='blue',
           alpha=0.7)
   print(trainers['softmax'].all_metrics)
   if len(trainers['softmax'].all_metrics['train_weight_std']) > 0:
       ax1.fill_between(
           range(len(trainers['softmax'].all_metrics['train_weight_std'])),
           np.array(trainers['softmax'].all_metrics['train_weight_mean']) -
           np.array(trainers['softmax'].all_metrics['train_weight_std']),
           np.array(trainers['softmax'].all_metrics['train_weight_mean']) +
           np.array(trainers['softmax'].all_metrics['train_weight_std']),
           color='blue', alpha=0.2
       )
       ax1.fill_between(
           range(len(trainers['lagrangian'].all_metrics['train_weight_std'])),
           np.array(trainers['lagrangian'].all_metrics['train_weight_mean']) -
           np.array(trainers['lagrangian'].all_metrics['train_weight_std']),
           np.array(trainers['lagrangian'].all_metrics['train_weight_mean']) +
           np.array(trainers['lagrangian'].all_metrics['train_weight_std']),
           color='red', alpha=0.2
       )

   ax1.set_title('Weight Evolution')
   ax1.set_xlabel('Epoch')
   ax1.set_ylabel('Weight Value')
   ax1.grid(True)
   ax1.legend()
   ax1.set_ylim(0, 0.5)
  
   # 3. Current Weight Distribution (middle-right)
   ax3 = fig.add_subplot(gs[0, 1])
   x = np.arange(len(solver_names))
   width = 0.35
  
   for i, (name, trainer) in enumerate(trainers.items()):
       with torch.no_grad():
           sample = dataset[0]
           grid = sample['grid'].to(trainer.device).unsqueeze(0)
           _, meta = trainer.model(grid)
           weights = meta['weights'][0].cpu().numpy()
          
           ax3.bar(x + i*width, weights, width,
                  label=name, alpha=0.7)
  
   ax3.set_title('Current Weight Distribution')
   ax3.set_xticks(x + width/2)
   ax3.set_xticklabels(solver_names, rotation=45)
   ax3.set_ylabel('Weight Value')
   ax3.legend()
   ax3.grid(True)
   ax3.set_ylim(0, 0.5)
  
   # 4. Loss Evolution (rightmost)
   ax4 = fig.add_subplot(gs[0, 2])
   print(trainer.metrics)
   for name, trainer in trainers.items():
       ax4.plot(trainer.all_metrics['train_total_loss'],
                label=f'{name} Train', alpha=0.7)
       if 'val_total_loss' in trainer.all_metrics:
           ax4.plot(trainer.all_metrics['val_total_loss'],
                   label=f'{name} Val', linestyle='--', alpha=0.7)
  
   ax4.set_title('Loss Evolution')
   ax4.set_xlabel('Epoch')
   ax4.set_ylabel('Loss')
   ax4.set_yscale('log')
   ax4.grid(True)
   ax4.legend()
  
   # Add overall title
   plt.suptitle(f'Training Analysis - Epoch {epoch}', y=1.02, fontsize=18)
  
   # Save figure
   plt.tight_layout()
   os.makedirs(save_dir, exist_ok=True)
   plt.savefig(f'{save_dir}/detailed_analysis_epoch_{epoch}.png',
               bbox_inches='tight', dpi=300)
   plt.close()



def main():
   # Parameters
   domain_size = 32
   n_samples = 1024 
   batch_size = 16
   n_epochs = 400
   device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   save_dir = "results"
   print(f"Using device: {device}")
  
   # # Create datasets
   # train_dataset = PDEDataset(n_samples, domain_size)
   # val_dataset = PDEDataset(n_samples//5, domain_size)


   train_dataset = NavierStokesDataset(
       n_samples=n_samples,
       domain_size=domain_size,
       dt=0.001,
       n_steps=100,
       Re=100
       )
   val_dataset = NavierStokesDataset(
       n_samples=n_samples//5,
       domain_size=domain_size,
       dt=0.001,
       n_steps=100,
       Re=100
       )
   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=batch_size)
  
   # Create models - one with Lagrangian, one without
   models = {
       'lagrangian': MultiSolverSystem(domain_size, use_lagrangian=True),
       'softmax': MultiSolverSystem(domain_size, use_lagrangian=False)
   }
  
   # Create trainers
   trainers = {
       name: Trainer(
           model,
           train_loader,
           val_loader,
           device=device
       ) for name, model in models.items()
   }
  
   # Training loop
   for epoch in range(n_epochs):
       print(f"\nEpoch {epoch+1}/{n_epochs}")
      
       for name, trainer in trainers.items():
           trainer.metrics = defaultdict(list)
           print(f"\n{name.capitalize()} model:")
          
           # Train
           train_metrics = trainer.train_epoch()
           print("Training metrics:")
           for k, v in train_metrics.items():
               print(f"Train {k}: {v}")
          
           # Validate
           val_metrics = trainer.validate()
           print("Validation metrics:")
           for k, v in val_metrics.items():
               print(f"Val {k}: {v}")
          
           # # Store metrics
           # for k, v in train_metrics.items():
           #     trainer.metrics[f'{k}'].append(v)
           # for k, v in val_metrics.items():
           #     trainer.metrics[f'{k}'].append(v)
       print("Validation metrics: ", val_metrics)
       print(train_metrics)
       # Plot comparison every 10 epochs
       if (epoch + 1) % 3 == 0:
           #figure = Utils.plot_comparison(trainers, epoch + 1)
           #Utils.save_plot(figure, 'results', 'solver_outputs_epoch' + str(epoch)+ ".png")
           #print(trainer.metrics)
           plot_comparison_separate(trainers, val_dataset, epoch + 1, save_dir)
           plot_detailed_analysis(trainers, val_dataset, epoch + 1, save_dir)
           for name, trainer in trainers.items():
               plot_solver_outputs(
                   trainer.model,
                   val_dataset,
                   epoch + 1,
                   f'{save_dir}/{name}'
               )
  
   # Plot comparison metrics
   plot_comparison_separate(trainers, epoch + 1, save_dir)
   print("\nTraining completed!")


if __name__ == "__main__":
   main()








