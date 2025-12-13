import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
from .base_classes import PhysicsRegime

class PDEDataset(Dataset):
    def __init__(self, n_samples: int, input_dim: int, noise_level: float = 0.1, seed: int = 42):
        torch.manual_seed(seed)
        self.n_samples = n_samples
        self.input_dim = input_dim
      
        x = torch.linspace(-1, 1, input_dim)
        samples_per_regime = n_samples // len(PhysicsRegime)
      
        self.data = []
        for regime in PhysicsRegime:
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
          
            for _ in range(samples_per_regime):
                perturbed = u + noise_level * torch.randn_like(u)
                self.data.append({
                    'x': x.clone(),
                    'u': perturbed,
                    'regime_idx': torch.tensor(list(PhysicsRegime).index(regime))
                })
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'x': item['x'],
            'u': item['u'],
            'regime_idx': item['regime_idx']
        }

class NavierStokes1DDataset(Dataset):
    def __init__(self, n_samples: int, input_dim: int, noise_level: float = 0.01, seed: int = 42):
        torch.manual_seed(seed)
        self.n_samples = n_samples
        self.input_dim = input_dim
      
        self.dt = 0.001
        self.dx = 2.0 / input_dim
        self.Re = 100
        self.nu = 1.0 / self.Re
      
        self.data = []
        samples_per_regime = n_samples // len(PhysicsRegime)
      
        for regime in PhysicsRegime:
            for _ in range(samples_per_regime):
                u0 = self.generate_initial_condition(regime)
              
                try:
                    u_final = self.solve_burgers(u0)
                  
                    if torch.isnan(u_final).any() or torch.isinf(u_final).any():
                        continue
                  
                    noise = noise_level * torch.randn_like(u_final)
                    u_final = u_final + noise
                    u_final = torch.clamp(u_final, min=-10.0, max=10.0)
                  
                    self.data.append({
                        'x': u0.clone(),
                        'u': u_final,
                        'regime_idx': torch.tensor(list(PhysicsRegime).index(regime))
                    })
                  
                except Exception as e:
                    continue
      
        self.verify_dataset()

    def generate_initial_condition(self, regime: PhysicsRegime) -> torch.Tensor:
        x = torch.linspace(-1, 1, self.input_dim)
      
        if regime == PhysicsRegime.SMOOTH:
            u0 = torch.sin(2 * np.pi * x) + 0.5 * torch.sin(4 * np.pi * x)
        elif regime == PhysicsRegime.SHOCK:
            u0 = torch.zeros_like(x)
            u0[x < 0] = 1.0
            u0[x >= 0] = -1.0
            u0 = F.conv1d(
                u0.view(1, 1, -1),
                torch.ones(1, 1, 5) / 5,
                padding=2
            ).view(-1)
        elif regime == PhysicsRegime.BOUNDARY:
            u0 = (torch.exp(-50 * (x + 0.8)**2) +
                 torch.exp(-50 * (x - 0.8)**2))
        else:  # TURBULENT
            u0 = torch.zeros_like(x)
            for k in range(1, 6):
                phase = 2 * np.pi * torch.rand(1)
                u0 += torch.sin(k * np.pi * x + phase) / k
            u0 += 0.1 * torch.randn_like(x)
      
        u0 = u0 / (torch.max(torch.abs(u0)) + 1e-8)
        return u0

    def verify_dataset(self):
        x_values = []
        u_values = []
      
        for item in self.data:
            x_values.extend(item['x'].numpy())
            u_values.extend(item['u'].numpy())
      
        print(f"\nDataset Statistics:")
        print(f"Number of samples: {len(self.data)}")
        print(f"X range: [{np.min(x_values):.4f}, {np.max(x_values):.4f}]")
        print(f"U range: [{np.min(u_values):.4f}, {np.max(u_values):.4f}]")

    def solve_burgers(self, u: torch.Tensor, n_steps: int = 100) -> torch.Tensor:
        u_current = u.clone()
      
        try:
            for step in range(n_steps):
                du_dx = torch.zeros_like(u_current)
                du_dx[1:-1] = (u_current[2:] - u_current[:-2]) / (2 * self.dx)
                du_dx[0] = (u_current[1] - u_current[-1]) / (2 * self.dx)
                du_dx[-1] = (u_current[0] - u_current[-2]) / (2 * self.dx)
              
                d2u_dx2 = torch.zeros_like(u_current)
                d2u_dx2[1:-1] = (u_current[2:] - 2*u_current[1:-1] + u_current[:-2]) / (self.dx**2)
                d2u_dx2[0] = (u_current[1] - 2*u_current[0] + u_current[-1]) / (self.dx**2)
                d2u_dx2[-1] = (u_current[0] - 2*u_current[-1] + u_current[-2]) / (self.dx**2)
              
                update = self.dt * (-u_current * du_dx + self.nu * d2u_dx2)
              
                if torch.isnan(update).any() or torch.isinf(update).any():
                    return u_current
              
                max_velocity = torch.max(torch.abs(u_current))
                cfl = max_velocity * self.dt / self.dx
                if cfl > 1.0:
                    self.dt = self.dt * 0.5
                    continue
              
                u_current = u_current + update
                u_current[0] = u_current[-2]
                u_current[-1] = u_current[1]
              
            return u_current
          
        except Exception as e:
            return torch.zeros_like(u)
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):
        item = self.data[idx]
        return item

class MultiSourceDataset(Dataset):
    def __init__(self, n_samples: int, input_dim: int, n_sources: int):
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.n_sources = n_sources
      
        self.x = torch.randn(n_samples, input_dim)
        self.y = torch.zeros(n_samples, input_dim)
        for i in range(n_samples):
            if i % 3 == 0:
                self.y[i] = torch.sin(self.x[i])
            elif i % 3 == 1:
                self.y[i] = torch.sign(self.x[i])
            else:
                self.y[i] = self.x[i]**2
              
        self.y += 0.1 * torch.randn_like(self.y)
  
    def __len__(self):
        return self.n_samples
  
    def __getitem__(self, idx):
        return {
            'x': self.x[idx],
            'y': self.y[idx]
        }

class NavierStokesDataset(Dataset):
    def __init__(self, n_samples: int, domain_size: int = 64, dt: float = 0.001, 
                 n_steps: int = 100, Re: float = 100, noise_level: float = 0.01, seed: int = 42):
        torch.manual_seed(seed)
        self.domain_size = domain_size
        self.samples = []
      
        self.dt = dt
        self.dx = 2.0 / domain_size
        self.Re = Re
        self.nu = 1.0 / Re
      
        for i in range(n_samples):
            u, v, w = self.generate_initial_conditions()
            solution = self.solve_navier_stokes(u, v, w, n_steps)
            solution = solution + noise_level * torch.randn_like(solution)
          
            x = torch.linspace(-1, 1, domain_size)
            y = torch.linspace(-1, 1, domain_size)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            grid = torch.stack([grid_x, grid_y], dim=0)
          
            self.samples.append({
                'grid': grid,
                'solution': solution[-1],
                'parameters': {'Re': self.Re}
            })

    def generate_initial_conditions(self):
        u = torch.zeros((self.domain_size, self.domain_size))
        v = torch.zeros((self.domain_size, self.domain_size))
        w = torch.zeros((self.domain_size, self.domain_size))
        u[-1, :] = 1.0
        w = self.compute_vorticity(u, v)
        return u, v, w
  
    def compute_vorticity(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        du_dy = (u[1:, :] - u[:-1, :]) / self.dx
        dv_dx = (v[:, 1:] - v[:, :-1]) / self.dx
        du_dy = F.pad(du_dy, (0, 0, 0, 1))
        dv_dx = F.pad(dv_dx, (0, 1, 0, 0))
        return dv_dx - du_dy
  
    def solve_poisson(self, f: torch.Tensor, boundary_conditions: str = 'dirichlet') -> torch.Tensor:
        psi = torch.zeros_like(f)
        dx2 = self.dx * self.dx
        error = 1.0
        tolerance = 1e-4
        max_iter = 1000
        iter_count = 0
      
        while error > tolerance and iter_count < max_iter:
            psi_old = psi.clone()
            psi[1:-1, 1:-1] = 0.25 * (
                psi_old[1:-1, 2:] + psi_old[1:-1, :-2] +
                psi_old[2:, 1:-1] + psi_old[:-2, 1:-1] -
                dx2 * f[1:-1, 1:-1]
            )
          
            if boundary_conditions == 'dirichlet':
                psi[0, :] = 0
                psi[-1, :] = 0
                psi[:, 0] = 0
                psi[:, -1] = 0
          
            error = torch.max(torch.abs(psi - psi_old)).item()
            iter_count += 1
        return psi
  
    def compute_velocity_from_stream(self, psi: torch.Tensor):
        u = torch.zeros_like(psi)
        v = torch.zeros_like(psi)
        u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * self.dx)
        v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * self.dx)
        return u, v
  
    def solve_navier_stokes(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, n_steps: int):
        solutions = []
      
        for _ in range(n_steps):
            psi = self.solve_poisson(-w)
            u, v = self.compute_velocity_from_stream(psi)
            w_new = w.clone()
          
            dw_dx = torch.zeros_like(w)
            dw_dy = torch.zeros_like(w)
            d2w_dx2 = torch.zeros_like(w)
            d2w_dy2 = torch.zeros_like(w)
          
            dw_dx[1:-1, 1:-1] = (w[1:-1, 2:] - w[1:-1, :-2]) / (2 * self.dx)
            dw_dy[1:-1, 1:-1] = (w[2:, 1:-1] - w[:-2, 1:-1]) / (2 * self.dx)
            d2w_dx2[1:-1, 1:-1] = (w[1:-1, 2:] - 2*w[1:-1, 1:-1] + w[1:-1, :-2]) / (self.dx * self.dx)
            d2w_dy2[1:-1, 1:-1] = (w[2:, 1:-1] - 2*w[1:-1, 1:-1] + w[:-2, 1:-1]) / (self.dx * self.dx)
          
            w_new[1:-1, 1:-1] = w[1:-1, 1:-1] + self.dt * (
                -u[1:-1, 1:-1] * dw_dx[1:-1, 1:-1]
                -v[1:-1, 1:-1] * dw_dy[1:-1, 1:-1]
                + self.nu * (d2w_dx2[1:-1, 1:-1] + d2w_dy2[1:-1, 1:-1])
            )
          
            w_new[0, :] = -2 * psi[1, :] / (self.dx * self.dx)
            w_new[-1, :] = -2 * psi[-2, :] / (self.dx * self.dx)
            w_new[:, 0] = -2 * psi[:, 1] / (self.dx * self.dx)
            w_new[:, -1] = -2 * psi[:, -2] / (self.dx * self.dx)
            w = w_new
            solution = torch.stack([u, v, w], dim=0)
            solutions.append(solution)
        return torch.stack(solutions)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]