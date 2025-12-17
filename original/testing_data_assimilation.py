import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from dataclasses import dataclass
import os
import gc
from scipy import ndimage

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

def huber_loss(pred, target, delta=1.0):
    """Huber loss function for 2D"""
    diff = pred - target
    abs_diff = torch.abs(diff)
    quadratic = torch.min(abs_diff, torch.tensor(delta, device=pred.device))
    linear = abs_diff - quadratic
    return 0.5 * quadratic.pow(2) + delta * linear

class BasePDEModel2D(nn.Module):
    """Base class for 2D PDE models"""
    def __init__(
        self,
        grid_size: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
    
    def forward(
        self,
        x: torch.Tensor,  # Initial state: [batch, 2, H, W]
        measurements: torch.Tensor  # Measurements: [batch, n_sources, 2, H, W]
    ) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError
    
    def compute_loss(
        self,
        x: torch.Tensor,
        measurements: torch.Tensor,
        true_solution: Optional[torch.Tensor] = None,
        is_training: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError

class ResidualBlock2D(nn.Module):
    """Basic 2D residual block"""
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return x + self.layers(x)

class MultiStepPredictionModule2D(nn.Module):
    def __init__(
        self,
        grid_size: int,
        hidden_dim: int,
        n_steps: int,
        dt: float = 0.001,
        dx: float = None,
        nu: float = 0.01
    ):
        super().__init__()
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        self.dt = dt
        self.dx = dx if dx is not None else 2.0 / grid_size
        self.nu = nu
        
        # Enhanced 2D prediction network
        self.prediction_net = nn.Sequential(
            nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            ResidualBlock2D(hidden_dim, hidden_dim),
            ResidualBlock2D(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1)
        )
                
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.prediction_net(x), None

class MultiSourceNavierStokes2DDataset(Dataset):
    def __init__(self, n_samples: int, grid_size: int, n_sources: int, n_timesteps: int = 50, 
                 noise_levels: Optional[List[float]] = None, bias_levels: Optional[List[float]] = None, 
                 seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.n_sources = n_sources
        self.n_timesteps = n_timesteps
        
        # Physical parameters
        self.dt = 0.001
        self.dx = 2.0 / grid_size
        self.Re = 50
        self.nu = 1.0 / self.Re
        self.max_velocity = 1e3
        self.eps = 1e-8
        
        # Configuration parameters
        self.config = {
            'cfl_safety': 0.8,
            'filter_threshold': 1e2,
            'measurement_max': 1e6,
            'max_iter': 100,
            'tolerance': 1e-6
        }
        
        # Generate 2D input grid
        x = torch.linspace(-1, 1, grid_size)
        y = torch.linspace(-1, 1, grid_size)
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
        
        # Set measurement characteristics with bounds
        if noise_levels is None:
            noise_levels = [min(0.02 * (i + 1), 0.1) for i in range(n_sources)]
        if bias_levels is None:
            bias_levels = [min(0.03 * (i - n_sources/2), 0.15) for i in range(n_sources)]
        
        self.noise_levels = noise_levels
        self.bias_levels = bias_levels
        
        # Initialize storage
        self.solutions = []  # Will store [u_component, v_component]
        self.measurements = []
        self.states = []
        
        # Generate samples
        self._generate_samples()
        
        # Convert to tensors
        self.states = torch.stack(self.states)
        self.solutions = torch.stack(self.solutions)
        self.measurements = torch.stack(self.measurements)
        
        # Verify stability
        self.verify_dataset_stability()
        
        # Create 2D grid for all samples
        X_expanded = self.X.unsqueeze(0).unsqueeze(0).repeat(n_samples, 1, 1, 1)
        Y_expanded = self.Y.unsqueeze(0).unsqueeze(0).repeat(n_samples, 1, 1, 1)
        self.grid = torch.cat([X_expanded, Y_expanded], dim=1)

    def _generate_samples(self):
        """Generate samples for each physics regime in 2D"""
        samples_per_regime = self.n_samples // len(PhysicsRegime)
        
        for regime in PhysicsRegime:
            print(f"Generating {samples_per_regime} samples for {regime} in 2D")
            for i in range(samples_per_regime):
                try:
                    phase_shift = 2 * np.pi * i / samples_per_regime
                    initial_state_u, initial_state_v = self.generate_initial_condition(regime, phase_shift)
                    
                    # Stack u and v components
                    initial_state = torch.stack([initial_state_u, initial_state_v], dim=0)
                    
                    if torch.isnan(initial_state).any() or torch.isinf(initial_state).any():
                        raise ValueError("Unstable initial condition detected")
                    
                    self.states.append(initial_state)
                    
                    solution_sequence = self.solve_navier_stokes_sequence_2d(initial_state_u, initial_state_v)
                    if torch.isnan(solution_sequence).any() or torch.isinf(solution_sequence).any():
                        raise ValueError("Unstable solution detected")
                    
                    self.solutions.append(solution_sequence)
                    
                    source_measurements = self.generate_source_measurements_2d(solution_sequence)
                    source_measurements = self.process_measurements(source_measurements)
                    self.measurements.append(source_measurements)
                    
                except Exception as e:
                    print(f"Error generating sample {i} for {regime}: {str(e)}")
                    if len(self.solutions) > 0:
                        idx = np.random.randint(len(self.solutions))
                        self.states.append(self.states[idx].clone())
                        self.solutions.append(self.solutions[idx].clone())
                        self.measurements.append(self.measurements[idx].clone())
                    else:
                        raise RuntimeError("Unable to generate initial stable solutions")
    
    def solve_navier_stokes_sequence_2d(self, u0: torch.Tensor, v0: torch.Tensor, n_steps: Optional[int] = None) -> torch.Tensor:
        """Solve 2D Burgers' equation using pseudo-spectral method"""
        if n_steps is None:
            n_steps = self.n_timesteps
            
        # Convert to numpy for FFT
        u_np = u0.numpy()
        v_np = v0.numpy()
        N = self.grid_size
        
        # Wavenumbers in 2D
        kx = 2 * np.pi * np.fft.fftfreq(N, d=self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(N, d=self.dx)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K2 = KX**2 + KY**2
        
        # Initial conditions in Fourier space
        u_hat = np.fft.fft2(u_np)
        v_hat = np.fft.fft2(v_np)
        
        solutions_u = [u0.clone()]
        solutions_v = [v0.clone()]
        
        for t in range(n_steps):
            # Convert to physical space
            u = np.real(np.fft.ifft2(u_hat))
            v = np.real(np.fft.ifft2(v_hat))
            
            # Nonlinear terms
            uu_hat = np.fft.fft2(u * u)
            uv_hat = np.fft.fft2(u * v)
            vv_hat = np.fft.fft2(v * v)
            
            # 2D Burgers' equations in Fourier space
            rhs_u = -1j * (KX * uu_hat + KY * uv_hat) - self.nu * K2 * u_hat
            rhs_v = -1j * (KX * uv_hat + KY * vv_hat) - self.nu * K2 * v_hat
            
            # Semi-implicit scheme for stability
            u_hat_new = (u_hat + self.dt * rhs_u) / (1 + self.dt * self.nu * K2)
            v_hat_new = (v_hat + self.dt * rhs_v) / (1 + self.dt * self.nu * K2)
            
            # Dealising
            cutoff = N // 2
            u_hat_new[cutoff:, :] = 0
            u_hat_new[:, cutoff:] = 0
            v_hat_new[cutoff:, :] = 0
            v_hat_new[:, cutoff:] = 0
            
            u_hat = u_hat_new
            v_hat = v_hat_new
            
            # Convert back and store
            u_next = torch.from_numpy(np.real(np.fft.ifft2(u_hat))).float()
            v_next = torch.from_numpy(np.real(np.fft.ifft2(v_hat))).float()
            
            solutions_u.append(u_next)
            solutions_v.append(v_next)
            
            if self.check_instability_2d(u_next, v_next):
                raise ValueError(f"Solution became unstable at step {t}")
        
        # Stack solutions: [timesteps, 2, H, W]
        solutions_u_stacked = torch.stack(solutions_u)
        solutions_v_stacked = torch.stack(solutions_v)
        return torch.stack([solutions_u_stacked, solutions_v_stacked], dim=1)

    def compute_stable_derivative_2d(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial derivatives with periodic boundary conditions in 2D"""
        # Use finite differences with periodic padding
        u_padded = F.pad(u.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='circular').squeeze()
        
        # Central differences
        du_dx = (u_padded[2:, 1:-1] - u_padded[:-2, 1:-1]) / (2 * self.dx)
        du_dy = (u_padded[1:-1, 2:] - u_padded[1:-1, :-2]) / (2 * self.dx)
        
        return torch.clamp(du_dx, -self.max_velocity/self.dx, self.max_velocity/self.dx), \
               torch.clamp(du_dy, -self.max_velocity/self.dx, self.max_velocity/self.dx)

    def spatial_filter_2d(self, field: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """2D spatial filtering for stability"""
        original_shape = field.shape
        if field.dim() == 2:
            field = field.unsqueeze(0).unsqueeze(0)
        elif field.dim() == 3:
            field = field.unsqueeze(1)
        
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=field.device) / (kernel_size * kernel_size)
        padding = kernel_size // 2
        field_padded = F.pad(field, (padding, padding, padding, padding), mode='circular')
        filtered = F.conv2d(field_padded, kernel)
        
        if len(original_shape) == 2:
            filtered = filtered[0, 0]
        elif len(original_shape) == 3:
            filtered = filtered[:, 0]
            
        return filtered

    def check_instability_2d(self, u: torch.Tensor, v: torch.Tensor) -> bool:
        """Check for numerical instabilities in 2D"""
        return (torch.isnan(u).any() or torch.isinf(u).any() or 
                torch.isnan(v).any() or torch.isinf(v).any() or
                torch.max(torch.abs(u)) > self.max_velocity or
                torch.max(torch.abs(v)) > self.max_velocity)

    def process_measurements(self, measurements: torch.Tensor) -> torch.Tensor:
        """Process measurements to handle invalid values (same as 1D)"""
        measurements = torch.where(torch.isinf(measurements), 
                                 torch.tensor(float('nan')), measurements)
        
        measurements = torch.clamp(measurements, 
                                 -self.config['measurement_max'], 
                                 self.config['measurement_max'])
        
        nan_mask = torch.isnan(measurements)
        if nan_mask.any():
            measurements = self.interpolate_nan_values_2d(measurements)
            
        return measurements

    def interpolate_nan_values_2d(self, data: torch.Tensor) -> torch.Tensor:
        """Interpolate NaN values in 2D data"""
        data_numpy = data.numpy()
        for i in range(data_numpy.shape[0]):
            for j in range(data_numpy.shape[1]):
                # Use nearest neighbor interpolation for 2D
                from scipy import interpolate
                mask = np.isnan(data_numpy[i, j])
                if mask.any() and not mask.all():
                    x = np.arange(data_numpy.shape[2])
                    y = np.arange(data_numpy.shape[3])
                    X, Y = np.meshgrid(x, y, indexing='ij')
                    
                    # Points with valid data
                    valid_points = ~mask
                    if valid_points.sum() > 3:  # Need enough points for interpolation
                        interp = interpolate.griddata(
                            (X[valid_points], Y[valid_points]),
                            data_numpy[i, j][valid_points],
                            (X, Y),
                            method='nearest'
                        )
                        data_numpy[i, j] = interp
        return torch.from_numpy(data_numpy)

    def generate_initial_condition(self, regime: PhysicsRegime, phase_shift: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate 2D initial conditions for u and v components"""
        x = self.X
        y = self.Y
        phase_shift_tensor = torch.tensor(phase_shift)
        
        try:
            if regime == PhysicsRegime.SMOOTH:
                u0, v0 = self._generate_smooth_initial_condition_2d(x, y, phase_shift_tensor)
            elif regime == PhysicsRegime.SHOCK:
                u0, v0 = self._generate_shock_initial_condition_2d(x, y, phase_shift_tensor)
            elif regime == PhysicsRegime.BOUNDARY:
                u0, v0 = self._generate_boundary_initial_condition_2d(x, y, phase_shift_tensor)
            else:
                u0, v0 = self._generate_turbulent_initial_condition_2d(x, y, phase_shift_tensor)
            
            u0 = self._normalize_and_stabilize_2d(u0)
            v0 = self._normalize_and_stabilize_2d(v0)
            
            return u0, v0
            
        except Exception as e:
            print(f"Error generating 2D initial condition for {regime}: {str(e)}")
            # Fallback to simple vortex
            u0 = torch.sin(np.pi * x) * torch.cos(np.pi * y)
            v0 = -torch.cos(np.pi * x) * torch.sin(np.pi * y)
            return u0, v0

    def _generate_smooth_initial_condition_2d(self, x: torch.Tensor, y: torch.Tensor, phase_shift: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Smooth 2D flow with vortices"""
        # Stream function for smooth flow
        psi = (torch.sin(2 * np.pi * x + phase_shift) * torch.cos(2 * np.pi * y) +
               0.5 * torch.sin(4 * np.pi * x + 2*phase_shift) * torch.cos(4 * np.pi * y))
        
        # Velocity from stream function: u = ∂ψ/∂y, v = -∂ψ/∂x
        psi_padded = F.pad(psi.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='circular').squeeze()
        u0 = (psi_padded[1:-1, 2:] - psi_padded[1:-1, :-2]) / (2 * self.dx)
        v0 = -(psi_padded[2:, 1:-1] - psi_padded[:-2, 1:-1]) / (2 * self.dx)
        
        # Add Gaussian vortex
        gaussian = torch.exp(-10 * ((x - 0.3 * torch.sin(phase_shift))**2 + (y - 0.3 * torch.cos(phase_shift))**2))
        u0 = u0 + 0.2 * gaussian * torch.cos(np.pi * y)
        v0 = v0 + 0.2 * gaussian * torch.sin(np.pi * x)
        
        return u0, v0

    def _generate_shock_initial_condition_2d(self, x: torch.Tensor, y: torch.Tensor, phase_shift: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shock-like discontinuity in 2D"""
        shift1 = 0.4 * torch.sin(phase_shift)
        shift2 = 0.4 * torch.cos(phase_shift)
        
        # Create step function in x-direction for u
        u0 = torch.zeros_like(x)
        u0[x < shift1] = 0.8
        u0[x >= shift1] = -0.4
        
        # Create step function in y-direction for v
        v0 = torch.zeros_like(y)
        v0[y < shift2] = 0.5
        v0[y >= shift2] = -0.3
        
        # Smooth with convolution
        kernel = torch.ones(1, 1, 5, 5) / 25
        u0_padded = F.pad(u0.unsqueeze(0).unsqueeze(0), (2, 2, 2, 2), mode='circular')
        v0_padded = F.pad(v0.unsqueeze(0).unsqueeze(0), (2, 2, 2, 2), mode='circular')
        
        u0 = F.conv2d(u0_padded, kernel).squeeze()
        v0 = F.conv2d(v0_padded, kernel).squeeze()
        
        # Add oscillations
        u0 = u0 + 0.05 * torch.sin(8 * np.pi * x) * torch.cos(4 * np.pi * y)
        v0 = v0 + 0.05 * torch.cos(4 * np.pi * x) * torch.sin(8 * np.pi * y)
        
        return u0, v0

    def _generate_boundary_initial_condition_2d(self, x: torch.Tensor, y: torch.Tensor, phase_shift: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Boundary layers in 2D"""
        shift1 = 0.2 * torch.sin(phase_shift)
        shift2 = 0.2 * torch.cos(phase_shift)
        
        # Boundary layers near edges
        u0 = (torch.exp(-10 * (x + 0.6 + shift1)**2) + 
              torch.exp(-15 * (x - 0.6 + shift2)**2) +
              0.3 * torch.exp(-20 * (x + shift1)**2))
        
        v0 = (torch.exp(-10 * (y + 0.6 + shift1)**2) + 
              torch.exp(-15 * (y - 0.6 + shift2)**2) +
              0.3 * torch.exp(-20 * (y + shift1)**2))
        
        # Add sinusoidal variation
        u0 = u0 + 0.1 * torch.sin(4 * np.pi * x + phase_shift) * torch.cos(2 * np.pi * y)
        v0 = v0 + 0.1 * torch.cos(2 * np.pi * x) * torch.sin(4 * np.pi * y + phase_shift)
        
        return u0, v0

    def _generate_turbulent_initial_condition_2d(self, x: torch.Tensor, y: torch.Tensor, phase_shift: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Turbulent-like flow with multiple modes"""
        u0 = torch.zeros_like(x)
        v0 = torch.zeros_like(y)
        
        # Multiple Fourier modes
        for kx in range(1, 5):
            for ky in range(1, 5):
                phase = phase_shift + torch.tensor(2 * np.pi * (kx + ky) / 8)
                amplitude = 0.5 / (kx**0.5 + ky**0.5)
                
                u0 += amplitude * torch.sin(kx * np.pi * x + phase) * torch.cos(ky * np.pi * y)
                v0 += amplitude * torch.cos(kx * np.pi * x) * torch.sin(ky * np.pi * y + phase)
        
        # Add random Gaussian vortices
        for _ in range(3):
            center_x = 0.5 * torch.sin(phase_shift + torch.tensor(np.random.rand() * np.pi))
            center_y = 0.5 * torch.cos(phase_shift + torch.tensor(np.random.rand() * np.pi))
            width = 0.1 + 0.05 * np.random.rand()
            amplitude = 0.1 + 0.05 * np.random.rand()
            
            vortex = amplitude * torch.exp(-((x - center_x)**2 + (y - center_y)**2) / width**2)
            u0 += vortex * torch.cos(np.pi * y)
            v0 += vortex * torch.sin(np.pi * x)
        
        # Add small random noise
        u0 += 0.02 * torch.randn_like(x)
        v0 += 0.02 * torch.randn_like(y)
        
        return u0, v0

    def _normalize_and_stabilize_2d(self, u0: torch.Tensor) -> torch.Tensor:
        """Normalize and stabilize 2D field"""
        u0 = torch.where(torch.isnan(u0) | torch.isinf(u0), 
                        torch.zeros_like(u0), u0)
        
        max_val = torch.max(torch.abs(u0)) + self.eps
        u0 = u0 / max_val
        
        if torch.max(torch.abs(u0)) > self.config['filter_threshold']:
            u0 = self.spatial_filter_2d(u0)
        
        return u0

    def generate_source_measurements_2d(self, solution: torch.Tensor) -> torch.Tensor:
        """Generate corrupted measurements from 2D solution"""
        measurements = []
        
        for source_idx in range(self.n_sources):
            try:
                # solution shape: [timesteps, 2, H, W]
                biased = solution + self.bias_levels[source_idx]
                noisy = biased + self.noise_levels[source_idx] * torch.randn_like(solution)
                
                if source_idx % 3 == 0:
                    # Apply spatial filtering
                    filtered = torch.zeros_like(noisy)
                    for t in range(noisy.shape[0]):
                        for c in range(noisy.shape[1]):
                            filtered[t, c] = self.spatial_filter_2d(noisy[t, c], kernel_size=5)
                    noisy = filtered
                
                noisy = self._add_missing_data_2d(noisy, source_idx)
                measurements.append(noisy)
                
            except Exception as e:
                print(f"Error generating 2D measurements for source {source_idx}: {str(e)}")
                measurements.append(solution.clone())

        measurements = torch.stack(measurements)  # [n_sources, timesteps, 2, H, W]
        return self.process_measurements(measurements)

    def _add_missing_data_2d(self, data: torch.Tensor, source_idx: int) -> torch.Tensor:
        """Add missing data (NaN values) in 2D"""
        missing_prob = 0.005 * (source_idx + 1)
        mask = torch.rand_like(data) > missing_prob
        return torch.where(mask, data, torch.tensor(float('nan')))

    def verify_dataset_stability(self):
        """Verify overall dataset stability (same as 1D)"""
        if torch.isnan(self.solutions).any() or torch.isinf(self.solutions).any():
            raise ValueError("Dataset contains unstable solutions")
        if torch.isnan(self.measurements).any() or torch.isinf(self.measurements).any():
            raise ValueError("Dataset contains unstable measurements")

    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset"""
        return {
            'x': self.states[idx],  # [2, H, W]
            'measurements': self.measurements[idx],  # [n_sources, timesteps, 2, H, W]
            'true_solution': self.solutions[idx],  # [timesteps, 2, H, W]
        }

class EnhancedMultiSourceBase2D(BasePDEModel2D):
    def __init__(
        self,
        n_sources: int,
        grid_size: int,
        hidden_dim: int = 128,
        n_prediction_steps: int = 5,
        dt: float = 0.001,
    ):
        super().__init__(grid_size, hidden_dim)
        self.n_sources = n_sources
        self.n_prediction_steps = n_prediction_steps
        self.dt = dt
        
        # Physical parameters
        self.dx = 2.0 / grid_size
        self.Re = 50
        self.viscosity = 1.0 / self.Re
        
        # 2D prediction network
        self.prediction_module = MultiStepPredictionModule2D(
            grid_size=grid_size,
            hidden_dim=hidden_dim,
            n_steps=n_prediction_steps,
            dt=dt,
            dx=self.dx,
            nu=self.viscosity
        )

    def compute_physics_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-informed loss based on 2D Burgers' equation
        predictions: [batch, timesteps, 2, H, W] (u and v components)
        """
        total_residual = 0.0
        L = 2
        
        for t in range(predictions.shape[1] - 1):
            u = predictions[:, t, 0]  # u-component: [batch, H, W]
            v = predictions[:, t, 1]  # v-component: [batch, H, W]
            
            U_scale = torch.max(torch.abs(torch.cat([u, v])))
            nu = (U_scale * 2.0) / self.Re
            
            u_next = predictions[:, t + 1, 0]
            v_next = predictions[:, t + 1, 1]
            
            # Compute spatial derivatives with periodic boundary
            def gradient_periodic(field):
                # Pad with circular boundary
                padded = F.pad(field.unsqueeze(1), (1, 1, 1, 1), mode='circular').squeeze(1)
                dx = (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]) / (2 * self.dx)
                dy = (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]) / (2 * self.dx)
                return dx, dy
            
            du_dx, du_dy = gradient_periodic(u)
            dv_dx, dv_dy = gradient_periodic(v)
            
            # Second derivatives
            du_dx2, _ = gradient_periodic(du_dx)
            _, du_dy2 = gradient_periodic(du_dy)
            dv_dx2, _ = gradient_periodic(dv_dx)
            _, dv_dy2 = gradient_periodic(dv_dy)
            
            # Time derivatives
            du_dt = (u_next - u) / self.dt
            dv_dt = (v_next - v) / self.dt
            
            # 2D Burgers' equation residuals
            with torch.no_grad():
                # Shock detection based on velocity gradient magnitude
                vel_grad_mag = torch.sqrt(du_dx**2 + du_dy**2 + dv_dx**2 + dv_dy**2)
                shock_mask = (vel_grad_mag > 0.5 * U_scale / L)
                shock_factor = torch.clamp(vel_grad_mag / (U_scale / L + 1e-6), 0, 3)
            
            effective_viscosity = nu * (1 + shock_factor)
            
            residual_u = du_dt + u * du_dx + v * du_dy - effective_viscosity * (du_dx2 + du_dy2)
            residual_v = dv_dt + u * dv_dx + v * dv_dy - effective_viscosity * (dv_dx2 + dv_dy2)
            
            # CFL-based weighting
            vel_mag = torch.sqrt(u**2 + v**2)
            cfl = (vel_mag * self.dt / self.dx).max(dim=(-1, -2), keepdim=True).values
            weight = torch.exp(-5.0 * cfl)
            
            total_residual += (weight * (residual_u.pow(2) + residual_v.pow(2))).mean()
        
        physics_loss = total_residual / (predictions.shape[1] - 1 + 1e-5)
        
        # Add mass conservation for each component
        mass_u_initial = torch.trapz(torch.trapz(predictions[:, 0, 0], dx=self.dx, dim=-1), dx=self.dx, dim=-1)
        mass_u_final = torch.trapz(torch.trapz(predictions[:, -1, 0], dx=self.dx, dim=-1), dx=self.dx, dim=-1)
        mass_v_initial = torch.trapz(torch.trapz(predictions[:, 0, 1], dx=self.dx, dim=-1), dx=self.dx, dim=-1)
        mass_v_final = torch.trapz(torch.trapz(predictions[:, -1, 1], dx=self.dx, dim=-1), dx=self.dx, dim=-1)
        
        mass_conservation = ((mass_u_final - mass_u_initial).pow(2) + 
                            (mass_v_final - mass_v_initial).pow(2)).mean()
        
        total_loss = physics_loss + 0.1 * mass_conservation
        
        return total_loss

    def compute_measurement_loss(
        self, 
        predictions: torch.Tensor, 
        analysis_state: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss between predictions and analysis state in 2D
        predictions: [batch, timesteps, 2, H, W]
        analysis_state: [batch, timesteps, 2, H, W] (target)
        """
        meas_loss = 0.0
        
        # Compare each predicted timestep with analysis state at that timestep
        for t in range(predictions.shape[1]):
            # Mask out NaN values from analysis state
            valid_mask = ~torch.isnan(analysis_state[:, t])
            if valid_mask.any():
                pred_masked = predictions[:, t][valid_mask]
                target_masked = analysis_state[:, t][valid_mask]
                loss = huber_loss(pred_masked, target_masked).mean()
                meas_loss += loss
        
        return meas_loss / predictions.shape[1]


class SoftmaxMultiSourceIntegration2D(EnhancedMultiSourceBase2D):
    def __init__(self, *args, temperature: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.lambda_weights = nn.Parameter(torch.randn(self.n_sources) * 0.01 + 1.0/self.n_sources)
        
    def get_weights(self):
        return F.softmax(self.lambda_weights / self.temperature, dim=0)
    
    def get_analysis_state(self, measurements: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Compute analysis state for each timestep in 2D"""
        weights = self.get_weights()
        
        # measurements: [batch, n_sources, timesteps, 2, H, W]
        # weights: [n_sources] -> reshape to [1, n_sources, 1, 1, 1, 1]
        weights_reshaped = weights.view(1, -1, 1, 1, 1, 1).expand_as(measurements)
        
        # Weighted sum over sources: [batch, timesteps, 2, H, W]
        weighted_measurements = measurements * weights_reshaped
        analysis_states = weighted_measurements.sum(dim=1)
        
        # Process NaN values after fusion
        analysis_states = self.process_fused_measurements(analysis_states)
        
        return analysis_states, weights, {
            'analysis_state': analysis_states,
            'weights': weights,
        }
    
    def process_fused_measurements(self, analysis_states: torch.Tensor) -> torch.Tensor:
        """Interpolate NaN values in the fused 2D analysis state"""
        batch_size, timesteps, channels, H, W = analysis_states.shape
        
        for b in range(batch_size):
            for t in range(timesteps):
                for c in range(channels):
                    # Get single channel at single timestep
                    state_tc = analysis_states[b, t, c]
                    
                    # Find valid measurements
                    valid_mask = ~torch.isnan(state_tc)
                    
                    # If we have some valid measurements, interpolate
                    if valid_mask.any() and not valid_mask.all():
                        # Use 2D interpolation
                        from scipy.interpolate import griddata
                        
                        valid_coords = torch.stack(torch.where(valid_mask), dim=1).cpu().numpy()
                        valid_values = state_tc[valid_mask].cpu().numpy()
                        
                        if len(valid_coords) > 3:
                            # Create grid for interpolation
                            x_coords, y_coords = torch.meshgrid(
                                torch.arange(H), torch.arange(W), indexing='ij'
                            )
                            grid_coords = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1).cpu().numpy()
                            
                            # Interpolate
                            interpolated = griddata(valid_coords, valid_values, grid_coords, method='nearest')
                            interpolated_reshaped = interpolated.reshape(H, W)
                            analysis_states[b, t, c] = torch.from_numpy(interpolated_reshaped).to(state_tc.device)
                    elif not valid_mask.any():
                        # No valid measurements at all, set to zero
                        analysis_states[b, t, c] = 0.0
        
        return analysis_states
    
    def forward(self, x: torch.Tensor, measurements: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Recursive data assimilation and prediction for 2D
        1. Compute analysis state at all timesteps from measurements
        2. Use analysis_state[:, 0] as initial condition
        3. Predict forward recursively
        """
        # Get analysis state for ALL timesteps (target for predictions)
        analysis_state, weights, meta = self.get_analysis_state(measurements)
        
        # Start autoregressive prediction from initial analysis state
        predictions_list = [analysis_state[:, 0]]  # Initial state: [batch, 2, H, W]
        
        # Predict future timesteps
        for t in range(1, analysis_state.shape[1]):
            current_state = predictions_list[-1]
            next_prediction, _ = self.prediction_module(current_state)
            predictions_list.append(next_prediction)
        
        predictions = torch.stack(predictions_list, dim=1)  # [batch, timesteps, 2, H, W]
        
        # Store analysis state for loss computation
        meta['analysis_state'] = analysis_state
        
        return predictions, meta
    
    def compute_loss(
        self,
        x: torch.Tensor,
        measurements: torch.Tensor,
        true_solution: Optional[torch.Tensor] = None,
        is_training: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        predictions, meta = self.forward(x, measurements)
        analysis_state = meta['analysis_state']
        weights = meta['weights']
        meas_loss = self.compute_measurement_loss(predictions, analysis_state, weights)
        
        # Physics loss on predictions (ensures PDE consistency)
        physics_loss = self.compute_physics_loss(predictions) if is_training else torch.tensor(0.0)
        
        # Balance losses
        total_loss = meas_loss + 0.01 * physics_loss
        
        return total_loss, {
            'loss': total_loss.item(),
            'meas_loss': meas_loss.item(),
            'physics_loss': physics_loss.item() if is_training else 0.0,
            'weights': weights,
        }


class SingleTimeScaleLagrangianOptimizer2D(EnhancedMultiSourceBase2D):
    """Single time scale Lagrangian optimizer for 2D"""
    
    def __init__(
        self,
        n_sources: int,
        grid_size: int,
        hidden_dim: int = 128,
        n_prediction_steps: int = 5,
        dt: float = 0.001,
        rho: float = 0.005,
    ):
        super().__init__(
            n_sources=n_sources,
            grid_size=grid_size,
            hidden_dim=hidden_dim,
            n_prediction_steps=n_prediction_steps,
            dt=dt
        )
        self.rho = rho
        
        # Initialize Lagrangian parameters
        self.lambda_weights = nn.Parameter(torch.randn(n_sources) * 0.01 + 1.0/n_sources)
        self.mu = nn.Parameter(torch.zeros(1))
        self.nu = nn.Parameter(torch.zeros(n_sources))
        
    def compute_lagrangian_weights(self) -> torch.Tensor:
        """Compute weights using Lagrangian formulation with non-negativity constraint"""
        weights = F.relu(self.lambda_weights) 
        sum_weights = weights.sum()
        return weights / (sum_weights + 1e-10)

    def forward(self, x: torch.Tensor, measurements: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        analysis_state, meta = self.get_analysis_state(measurements)
        predictions_list = [analysis_state[:, 0]]
        
        for t in range(1, analysis_state.shape[1]):
            current_state = predictions_list[-1]
            next_prediction, _ = self.prediction_module(current_state)
            predictions_list.append(next_prediction)
        
        predictions = torch.stack(predictions_list, dim=1)
        return predictions, meta
    
    def get_analysis_state(self, measurements):
        weights = self.compute_lagrangian_weights()
        weights_reshaped = weights.view(1, -1, 1, 1, 1, 1).expand_as(measurements)
        weighted_measurements = measurements * weights_reshaped
        analysis_states = weighted_measurements.sum(dim=1)
        
        return analysis_states, {
            'analysis_state': analysis_states,
            'weights': weights,
        }

    def compute_constraint_losses(self, weights: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute Lagrangian constraint losses (same as 1D)"""
        g = weights.sum() - 1.0
        h = -weights
        
        equality_term = self.mu * g + (self.rho/2) * g.pow(2)
        inequality_term = (self.nu * h + (self.rho/2) * h.pow(2)).sum()
        
        constraint_loss = equality_term + inequality_term
        
        return constraint_loss, {
            'equality_violation': torch.abs(g).item(),
            'inequality_violation': torch.relu(h).sum().item(),
            'constraint_loss': constraint_loss.item()
        }
    
    def compute_loss(
        self,
        x: torch.Tensor,
        measurements: torch.Tensor,
        true_solution: Optional[torch.Tensor] = None,
        is_training: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        predictions, meta = self.forward(x, measurements)
        analysis_state = meta['analysis_state']
        weights = meta['weights']
        meas_loss = self.compute_measurement_loss(predictions, analysis_state, weights)

        # Physics loss
        physics_loss = self.compute_physics_loss(predictions) if is_training else torch.tensor(0.0)
        
        # Constraint losses
        constraint_loss, constraint_dict = self.compute_constraint_losses(weights) if is_training else (torch.tensor(0.0), {})
        
        # Total loss
        total_loss = meas_loss + 0.01 * physics_loss + 0.001 * constraint_loss
        
        return total_loss, {
            'loss': total_loss.item(),
            'meas_loss': meas_loss.item(),
            'physics_loss': physics_loss.item() if is_training else 0.0,
            'weights': weights,
            **(constraint_dict if is_training else {})
        }


class TwoTimeScaleLagrangianOptimizer2D(EnhancedMultiSourceBase2D):
    def __init__(
        self,
        n_sources: int,
        grid_size: int,
        hidden_dim: int = 128,
        n_prediction_steps: int = 5,
        dt: float = 0.001,
        rho: float = 0.1,
        multiplier_lr: float = 0.1,
        multiplier_update_frequency: int = 1,
        constraint_weight: float = 0.1
    ):
        super().__init__(
            n_sources=n_sources,
            grid_size=grid_size,
            hidden_dim=hidden_dim,
            n_prediction_steps=n_prediction_steps,
            dt=dt
        )
        self.rho = rho
        self.multiplier_lr = multiplier_lr
        self.multiplier_update_frequency = multiplier_update_frequency
        self.constraint_weight = constraint_weight
        self.multiplier_update_counter = 0
        
        # Better initialization for Lagrangian parameters
        self.lambda_weights = nn.Parameter(torch.ones(n_sources) * 1.0/n_sources)
        self.mu = nn.Parameter(torch.zeros(1))
        self.nu = nn.Parameter(torch.zeros(n_sources))
        
        # Add momentum for multiplier updates
        self.register_buffer('mu_momentum', torch.zeros(1))
        self.register_buffer('nu_momentum', torch.zeros(n_sources))
        
    def compute_lagrangian_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute weights with better numerical stability (same as 1D)"""
        weights = F.softplus(self.lambda_weights)
        sum_weights = weights.sum()
        entropy = -torch.sum(weights * torch.log(weights + 1e-10))
        return weights / (sum_weights + 1e-10), entropy

    def compute_constraint_losses(self, weights: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute improved constraint losses with better gradient flow (same as 1D)"""
        g = weights.sum() - 1.0
        h = torch.relu(0.01 - weights)
        
        equality_term = self.mu * g + (self.rho/2) * g.pow(2)
        inequality_term = (self.nu * h + (self.rho/2) * h.pow(2)).sum()
        
        constraint_loss = equality_term + inequality_term
        
        barrier_term = -0.01 * torch.log(weights + 1e-10).sum()
        
        return constraint_loss + barrier_term, {
            'equality_violation': torch.abs(g).item(),
            'inequality_violation': h.sum().item(),
            'constraint_loss': constraint_loss.item(),
            'barrier_term': barrier_term.item()
        }

    def update_multipliers(self, measurements: torch.Tensor):
        """Improved multiplier update with momentum and adaptive steps (same as 1D)"""
        if self.multiplier_update_counter % self.multiplier_update_frequency == 0:
            with torch.no_grad():
                weights, _ = self.compute_lagrangian_weights()
                
                g = weights.sum() - 1.0
                h = torch.relu(0.01 - weights)
                
                mu_update = self.multiplier_lr * self.rho * g
                self.mu_momentum = 0.9 * self.mu_momentum + mu_update
                self.mu.data += self.mu_momentum
                
                nu_update = self.multiplier_lr * self.rho * h
                self.nu_momentum = 0.9 * self.nu_momentum + nu_update
                self.nu.data = torch.clamp(self.nu.data + self.nu_momentum, min=0, max=10.0)
                
                if torch.abs(g) > 0.1:
                    self.rho = min(self.rho * 1.1, 1.0)
                elif torch.abs(g) < 0.01:
                    self.rho = max(self.rho * 0.9, 0.01)

        self.multiplier_update_counter += 1

    def compute_loss(
        self,
        x: torch.Tensor,
        measurements: torch.Tensor,
        true_solution: Optional[torch.Tensor] = None,
        is_training: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        predictions, meta = self.forward(x, measurements)
        analysis_state = meta['analysis_state']
        weights, entropy = self.compute_lagrangian_weights()
        meas_loss = self.compute_measurement_loss(predictions, analysis_state, weights)
        
        # Physics loss
        physics_loss = self.compute_physics_loss(predictions) if is_training else torch.tensor(0.0)
        
        # Constraint losses
        if is_training:
            constraint_loss, constraint_dict = self.compute_constraint_losses(weights)
            entropy_penalty = -0.001 * entropy
        else:
            constraint_loss = torch.tensor(0.0)
            constraint_dict = {}
            entropy_penalty = torch.tensor(0.0)
        
        # Balanced total loss
        total_loss = (
            meas_loss + 
            0.01 * physics_loss + 
            self.constraint_weight * constraint_loss +
            entropy_penalty
        )
        
        return total_loss, {
            'loss': total_loss.item(),
            'meas_loss': meas_loss.item(),
            'physics_loss': physics_loss.item() if is_training else 0.0,
            'constraint_loss': constraint_loss.item() if is_training else 0.0,
            'weights': weights,
            'entropy': entropy.item(),
            'rho': self.rho,
            **constraint_dict
        }

    def forward(self, x: torch.Tensor, measurements: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Fixed forward pass with proper weight computation"""
        analysis_state, meta = self.get_analysis_state(measurements)
        predictions_list = [analysis_state[:, 0]]
        
        for t in range(1, analysis_state.shape[1]):
            current_state = predictions_list[-1]
            next_prediction, _ = self.prediction_module(current_state)
            predictions_list.append(next_prediction)
        
        predictions = torch.stack(predictions_list, dim=1)
        
        # Update meta with current weights and entropy
        weights, entropy = self.compute_lagrangian_weights()
        meta.update({
            'weights': weights,
            'entropy': entropy.item(),
            'rho': self.rho
        })
        
        return predictions, meta
    
    def get_analysis_state(self, measurements):
        weights, entropy = self.compute_lagrangian_weights()
        weights_reshaped = weights.view(1, -1, 1, 1, 1, 1).expand_as(measurements)
        weighted_measurements = measurements * weights_reshaped
        analysis_states = weighted_measurements.sum(dim=1)
        
        return analysis_states, {
            'analysis_state': analysis_states,
            'weights': weights,
            'entropy': entropy.item()
        }


class ADMMOptimizer2D(EnhancedMultiSourceBase2D):
    """ADMM-based optimizer for 2D multi-source integration"""
    
    def __init__(
        self,
        n_sources: int,
        grid_size: int,
        hidden_dim: int = 128,
        n_prediction_steps: int = 5,
        dt: float = 0.001,
        rho: float = 0.1,
        admm_iterations: int = 3,
    ):
        super().__init__(
            n_sources=n_sources,
            grid_size=grid_size,
            hidden_dim=hidden_dim,
            n_prediction_steps=n_prediction_steps,
            dt=dt
        )
        self.rho = rho
        self.admm_iterations = admm_iterations
        
        # ADMM variables
        self.lambda_weights = nn.Parameter(torch.randn(n_sources) * 0.01 + 1.0/n_sources)
        self.z = torch.nn.Parameter(torch.ones(n_sources) / n_sources)
        self.register_buffer('u_dual', torch.zeros(n_sources))
        
    def compute_lagrangian_weights(self) -> torch.Tensor:
        """ADMM uses z as the constraint-satisfying variable (same as 1D)"""
        return F.softmax(self.z, dim=0)
    
    def forward(self, x: torch.Tensor, measurements: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        analysis_state, meta = self.get_analysis_state(measurements)
        predictions_list = [analysis_state[:, 0]]
        
        for t in range(1, analysis_state.shape[1]):
            current_state = predictions_list[-1]
            next_prediction, _ = self.prediction_module(current_state)
            predictions_list.append(next_prediction)
        
        predictions = torch.stack(predictions_list, dim=1)
        return predictions, meta
    
    def get_analysis_state(self, measurements):
        weights = self.compute_lagrangian_weights()
        weights_reshaped = weights.view(1, -1, 1, 1, 1, 1).expand_as(measurements)
        weighted_measurements = measurements * weights_reshaped
        analysis_states = weighted_measurements.sum(dim=1)
        
        return analysis_states, {
            'analysis_state': analysis_states,
            'weights': weights,
            'lambda': self.lambda_weights,
            'z': self.z,
            'u': self.u_dual
        }
    
    def admm_update(self):
        """Perform ADMM updates for lambda, z, and u (same as 1D)"""
        lambda_prev = self.lambda_weights.detach().clone()
        
        with torch.no_grad():
            v = self.lambda_weights.detach() + self.u_dual
            z_new = F.relu(v)
            z_sum = z_new.sum()
            if z_sum > 0:
                self.z.data = z_new / z_sum
            else:
                self.z.data = torch.ones_like(z_new) / len(z_new)
        
        self.u_dual = self.u_dual + (self.lambda_weights.detach() - self.z.detach())
        
        return {
            'lambda_change': torch.norm(self.lambda_weights.detach() - lambda_prev).item(),
            'residual': torch.norm(self.lambda_weights.detach() - self.z.detach()).item(),
        }
    
    def compute_loss(
        self,
        x: torch.Tensor,
        measurements: torch.Tensor,
        true_solution: Optional[torch.Tensor] = None,
        is_training: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        predictions, meta = self.forward(x, measurements)
        analysis_state = meta['analysis_state']
        weights = meta['weights']
        meas_loss = self.compute_measurement_loss(predictions, analysis_state, weights)
        
        # Physics loss
        physics_loss = self.compute_physics_loss(predictions) if is_training else torch.tensor(0.0)
        
        # ADMM augmented Lagrangian term
        residual_term = (self.rho/2) * torch.norm(self.lambda_weights - self.z + self.u_dual).pow(2)
        
        total_loss = meas_loss + 0.01 * physics_loss + 0.001 * residual_term
        
        admm_info = {}
        
        return total_loss, {
            'loss': total_loss.item(),
            'meas_loss': meas_loss.item(),
            'physics_loss': physics_loss.item() if is_training else 0.0,
            'weights': weights,
            'admm_residual': residual_term.item(),
            **admm_info
        }


class FourWayComparativeTrainer2D:
    """Trainer for comparing Softmax, Single-Scale Lagrangian, Two-Scale Lagrangian, and ADMM in 2D"""
    
    def __init__(
        self,
        models: Dict[str, EnhancedMultiSourceBase2D],
        learning_rates: Dict[str, float],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.models = {name: model.to(device) for name, model in models.items()}
        
        # Method labels for plotting
        self.method_labels = {
            'softmax': 'Softmax',
            'lagrangian_single_scale': 'Single-Scale Lagrangian',
            'lagrangian_two_scale': 'Two-Scale Lagrangian',
            'admm': 'ADMM'
        }
        
        # Method colors for plotting
        self.method_colors = {
            'softmax': 'blue',
            'lagrangian_single_scale': 'green',
            'lagrangian_two_scale': 'red',
            'admm': 'purple'
        }
        
        # Initialize optimizers for each model
        self.optimizers = {}
        self.schedulers = {}
        
        for name, model in self.models.items():
            if name == 'lagrangian_two_scale':
                # Two-scale needs separate optimizers
                theta_params = [p for n, p in model.named_parameters() 
                               if not any(x in n for x in ['lambda_weights', 'mu', 'nu'])]
                lambda_params = [model.lambda_weights]
                
                self.optimizers[f'{name}_theta'] = torch.optim.AdamW(
                    theta_params, lr=learning_rates.get(f'{name}_theta', 1e-4))
                self.optimizers[f'{name}_lambda'] = torch.optim.AdamW(
                    lambda_params, lr=learning_rates.get(f'{name}_lambda', 1e-3))
                
                self.schedulers[f'{name}_theta'] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizers[f'{name}_theta'], mode='min', factor=0.5, patience=5)
                self.schedulers[f'{name}_lambda'] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizers[f'{name}_lambda'], mode='min', factor=0.5, patience=5)
            else:
                # Single optimizer for other methods
                self.optimizers[name] = torch.optim.AdamW(
                    model.parameters(),
                    lr=learning_rates.get(name, 1e-4),
                    weight_decay=1e-6
                )
                self.schedulers[name] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizers[name], mode='min', factor=0.5, patience=5)
        
        # Initialize metrics storage
        self.metrics = {name: defaultdict(list) for name in models.keys()}
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """Single training step for all 2D models"""
        x = batch['x'].to(self.device)
        measurements = batch['measurements'].to(self.device)
        true_solution = batch.get('true_solution', None)
        if true_solution is not None:
            true_solution = true_solution.to(self.device)
        
        all_metrics = {}
        
        # Train each model
        for name, model in self.models.items():
            model.train()
            
            if name == 'lagrangian_two_scale':
                # Two-scale optimization
                self.optimizers[f'{name}_theta'].zero_grad()
                self.optimizers[f'{name}_lambda'].zero_grad()
                
                loss, meta = model.compute_loss(x, measurements, true_solution, is_training=True)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.optimizers[f'{name}_theta'].step()
                self.optimizers[f'{name}_lambda'].step()
                model.update_multipliers(measurements)
  
            elif name == 'admm':
                # ADMM optimization
                self.optimizers[name].zero_grad()
                loss, meta = model.compute_loss(x, measurements, true_solution, is_training=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.optimizers[name].step()
                
                # ADMM updates after gradient step
                admm_info = model.admm_update()
                meta.update(admm_info)
            
            else:
                # Single-scale optimization
                self.optimizers[name].zero_grad()
                loss, meta = model.compute_loss(x, measurements, true_solution, is_training=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.optimizers[name].step()
            
            # Store metrics
            all_metrics[name] = {
                'loss': loss.item(),
                'meas_loss': meta.get('meas_loss', 0.0),
                'physics_loss': meta.get('physics_loss', 0.0),
                'weights_mean': meta['weights'].mean().item(),
                'weights_std': meta['weights'].std().item(),
            }
            
            # Add method-specific metrics
            if 'constraint_loss' in meta:
                all_metrics[name]['constraint_loss'] = meta['constraint_loss']
            if 'admm_residual' in meta:
                all_metrics[name]['admm_residual'] = meta['admm_residual']
        
        return all_metrics
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, Dict[str, float]]:
        """Train all 2D models for one epoch"""
        epoch_metrics = {name: defaultdict(list) for name in self.models.keys()}
        
        for batch in dataloader:
            batch_metrics = self.train_step(batch)
            
            for name, metrics in batch_metrics.items():
                for k, v in metrics.items():
                    if v is not None:
                        epoch_metrics[name][k].append(v)
        
        # Average metrics
        return {
            name: {k: float(np.mean(v)) for k, v in metrics.items()}
            for name, metrics in epoch_metrics.items()
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, Dict[str, float]]:
        """Validate all 2D models using measurement consistency"""
        for model in self.models.values():
            model.eval()
        
        val_metrics = {name: defaultdict(list) for name in self.models.keys()}
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch['x'].to(self.device)
                measurements = batch['measurements'].to(self.device)
                true_solution = batch['true_solution'].to(self.device)
                
                for name, model in self.models.items():
                    predictions, meta = model(x, measurements)
                    analysis_state = meta['analysis_state']
                    weights = meta['weights']
                    meas_loss = model.compute_measurement_loss(predictions, analysis_state, weights)
                    val_metrics[name]['meas_loss'].append(meas_loss.item())
                    
                    # Also compute true solution error for analysis
                    true_error = F.mse_loss(predictions, true_solution)
                    val_metrics[name]['true_error'].append(true_error.item())
        
        # Average metrics
        return {
            name: {k: float(np.mean(v)) for k, v in metrics.items()}
            for name, metrics in val_metrics.items()
        }
    
    def update_schedulers(self, val_losses: Dict[str, Dict[str, float]]):
        """Update learning rate schedulers"""
        for name, scheduler in self.schedulers.items():
            if 'lagrangian_two_scale' in name:
                if 'theta' in name or 'lambda' in name:
                    loss_value = val_losses['lagrangian_two_scale']['meas_loss']
                    scheduler.step(loss_value)
            else:
                loss_value = val_losses[name]['meas_loss']
                scheduler.step(loss_value)


def plot_comparative_results_2d(
    trainer: FourWayComparativeTrainer2D,
    dataset: MultiSourceNavierStokes2DDataset,
    epoch: int,
    save_dir: str = 'results'
):
    """Plot detailed 2D comparison between all four approaches"""
    plt.close('all')
    
    fig = plt.figure(figsize=(24, 20))
    gs = plt.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)
    
    # Training Loss (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    for name, metrics in trainer.metrics.items():
        if 'loss' in metrics:
            ax1.plot(metrics['loss'], 
                    label=trainer.method_labels[name], 
                    color=trainer.method_colors[name], 
                    alpha=0.7, linewidth=2)
    ax1.set_title('Training Loss', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Weight Distribution (Top Middle)
    ax2 = fig.add_subplot(gs[0, 1])
    sample_idx = np.random.choice(len(dataset))
    sample = dataset[sample_idx]
    x = sample['x'].to(trainer.device).unsqueeze(0)
    measurements = sample['measurements'].to(trainer.device).unsqueeze(0)
    
    with torch.no_grad():
        weights_data = {}
        for name, model in trainer.models.items():
            _, meta = model(x, measurements)
            weights = meta['weights'].detach().cpu().numpy()
            weights_data[name] = weights
    
    # Plot as grouped bar chart
    n_methods = len(weights_data)
    n_sources = len(next(iter(weights_data.values())))
    x_positions = np.arange(n_sources)
    width = 0.8 / n_methods
    
    for i, (name, weights) in enumerate(weights_data.items()):
        ax2.bar(x_positions + i*width - width*(n_methods-1)/2, 
                weights, width, 
                label=trainer.method_labels[name], 
                color=trainer.method_colors[name], 
                alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax2.set_title('Weight Distribution', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Source Index', fontsize=12)
    ax2.set_ylabel('Weight Value', fontsize=12)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([f'S{i+1}' for i in x_positions])
    
    # Physics Loss (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    for name, metrics in trainer.metrics.items():
        if 'physics_loss' in metrics:
            ax3.plot(metrics['physics_loss'], 
                    label=trainer.method_labels[name], 
                    color=trainer.method_colors[name], 
                    alpha=0.7, linewidth=2)
    ax3.set_title('Physics Loss', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Measurement Loss (Second Row Left)
    ax4 = fig.add_subplot(gs[1, 0])
    for name, metrics in trainer.metrics.items():
        if 'meas_loss' in metrics:
            ax4.plot(metrics['meas_loss'], 
                    label=trainer.method_labels[name], 
                    color=trainer.method_colors[name], 
                    alpha=0.7, linewidth=2)
    ax4.set_title('Measurement Consistency Loss', fontsize=16, fontweight='bold')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # Weight Evolution (Second Row Middle)
    ax5 = fig.add_subplot(gs[1, 1])
    for name, metrics in trainer.metrics.items():
        if 'weights_mean' in metrics:
            ax5.plot(metrics['weights_mean'], 
                    label=trainer.method_labels[name], 
                    color=trainer.method_colors[name], alpha=0.7, linewidth=2)
    ax5.set_title('Weight Evolution', fontsize=16, fontweight='bold')
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Mean Weight', fontsize=12)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Constraint Violations (Second Row Right)
    ax6 = fig.add_subplot(gs[1, 2])
    constraint_models = ['lagrangian_single_scale', 'lagrangian_two_scale', 'admm']
    for name in constraint_models:
        if name in trainer.metrics:
            if 'constraint_loss' in trainer.metrics[name]:
                ax6.plot(trainer.metrics[name]['constraint_loss'], 
                        label=f'{trainer.method_labels[name]} Constraint', 
                        color=trainer.method_colors[name], alpha=0.7, linewidth=2)
            elif 'admm_residual' in trainer.metrics[name]:
                ax6.plot(trainer.metrics[name]['admm_residual'], 
                        label=f'{trainer.method_labels[name]} Residual', 
                        color=trainer.method_colors[name], alpha=0.7, linewidth=2)
    
    ax6.set_title('Constraint Violations', fontsize=16, fontweight='bold')
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('Violation', fontsize=12)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')
    
    # 2D Velocity Field Visualization - Initial Condition (Third Row)
    ax7 = fig.add_subplot(gs[2, 0])
    true_solution = sample['true_solution'].to(trainer.device)
    initial_state = sample['x'].to(trainer.device)
    
    # Plot initial u-component
    im1 = ax7.imshow(initial_state[0, 0].cpu().numpy(), cmap='RdBu', origin='lower', 
                     extent=[-1, 1, -1, 1], aspect='auto')
    ax7.set_title('Initial u(x,y)', fontsize=14, fontweight='bold')
    ax7.set_xlabel('x', fontsize=11)
    ax7.set_ylabel('y', fontsize=11)
    plt.colorbar(im1, ax=ax7, fraction=0.046, pad=0.04)
    
    ax8 = fig.add_subplot(gs[2, 1])
    # Plot initial v-component
    im2 = ax8.imshow(initial_state[0, 1].cpu().numpy(), cmap='RdBu', origin='lower', 
                     extent=[-1, 1, -1, 1], aspect='auto')
    ax8.set_title('Initial v(x,y)', fontsize=14, fontweight='bold')
    ax8.set_xlabel('x', fontsize=11)
    ax8.set_ylabel('y', fontsize=11)
    plt.colorbar(im2, ax=ax8, fraction=0.046, pad=0.04)
    
    # True solution at final timestep (Fourth Row)
    ax9 = fig.add_subplot(gs[3, 0])
    final_true = true_solution[-1, 0].cpu().numpy()
    im3 = ax9.imshow(final_true, cmap='RdBu', origin='lower', 
                     extent=[-1, 1, -1, 1], aspect='auto')
    ax9.set_title('True Solution u(x,y, t=final)', fontsize=14, fontweight='bold')
    ax9.set_xlabel('x', fontsize=11)
    ax9.set_ylabel('y', fontsize=11)
    plt.colorbar(im3, ax=ax9, fraction=0.046, pad=0.04)
    
    # Model predictions comparison (Fourth Row, Middle and Right)
    with torch.no_grad():
        predictions_dict = {}
        for name, model in trainer.models.items():
            predictions, _ = model(x, measurements)
            predictions_dict[name] = predictions[0, -1, 0].cpu().numpy()  # u-component at final time
        
        # Plot each method's prediction
        for idx, (name, pred) in enumerate(predictions_dict.items()):
            if idx < 2:  # Plot first two methods
                ax = fig.add_subplot(gs[2 + idx, 2])
                im = ax.imshow(pred, cmap='RdBu', origin='lower', 
                              extent=[-1, 1, -1, 1], aspect='auto')
                ax.set_title(f'{trainer.method_labels[name]} Prediction', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('x', fontsize=11)
                ax.set_ylabel('y', fontsize=11)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            elif idx < 4:  # Plot next two methods
                ax = fig.add_subplot(gs[idx - 2, 3])
                im = ax.imshow(pred, cmap='RdBu', origin='lower', 
                              extent=[-1, 1, -1, 1], aspect='auto')
                ax.set_title(f'{trainer.method_labels[name]} Prediction', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('x', fontsize=11)
                ax.set_ylabel('y', fontsize=11)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f'2D Multi-Source Integration Comparison - Epoch {epoch}', 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(f'{save_dir}/2d_comparison_epoch_{epoch}.png', 
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    plt.clf()
    gc.collect()


def analyze_source_reliability_2d(
    trainer: FourWayComparativeTrainer2D,
    dataset: MultiSourceNavierStokes2DDataset,
    num_samples: int = 50,
    save_dir: str = 'results'
):
    """Analyze the reliability of different sources in 2D"""
    for model in trainer.models.values():
        model.eval()
    
    source_errors = defaultdict(list)
    source_weights = defaultdict(list)
    
    with torch.no_grad():
        for _ in range(num_samples):
            idx = np.random.randint(len(dataset))
            sample = dataset[idx]
            x = sample['x'].to(trainer.device).unsqueeze(0)
            measurements = sample['measurements'].to(trainer.device).unsqueeze(0)
            true_solution = sample['true_solution'].to(trainer.device)
            
            for name, model in trainer.models.items():
                predictions, meta = model(x, measurements)
                
                for i in range(dataset.n_sources):
                    # Get valid mask for this source [timesteps, 2, H, W]
                    valid_mask = ~torch.isnan(measurements[0, i])
                    
                    if valid_mask.any():
                        # Remove batch dimension from predictions for comparison
                        pred_flat = predictions[0]  # Shape: [timesteps, 2, H, W]
                        
                        # Apply the same mask to both predictions and measurements
                        pred_masked = pred_flat[valid_mask]
                        source_masked = measurements[0, i][valid_mask]
                        
                        # Calculate error only on valid points
                        error = F.mse_loss(pred_masked, source_masked)
                        source_errors[f'{name}_source_{i}'].append(error.item())
                        source_weights[f'{name}_source_{i}'].append(meta['weights'][i].item())
    
    # Plot analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Error distribution
    error_bars = []
    method_names = []
    for name, model in trainer.models.items():
        error_means = []
        for i in range(dataset.n_sources):
            key = f'{name}_source_{i}'
            if key in source_errors and source_errors[key]:
                error_means.append(np.mean(source_errors[key]))
            else:
                error_means.append(0.0)
        
        positions = np.arange(dataset.n_sources) + list(trainer.models.keys()).index(name) * 0.2 - 0.3
        bars = ax1.bar(positions, error_means, 0.2, 
                       label=trainer.method_labels[name], 
                       color=trainer.method_colors[name],
                       alpha=0.7, edgecolor='black', linewidth=0.5)
        error_bars.append(bars)
        method_names.append(trainer.method_labels[name])
    
    ax1.set_title('2D Prediction-Measurement Error by Source', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Source Index', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(np.arange(dataset.n_sources))
    ax1.set_xticklabels([f'S{i+1}' for i in range(dataset.n_sources)])
    
    # Weight distribution
    weight_bars = []
    for name, model in trainer.models.items():
        weight_means = []
        for i in range(dataset.n_sources):
            key = f'{name}_source_{i}'
            if key in source_weights and source_weights[key]:
                weight_means.append(np.mean(source_weights[key]))
            else:
                weight_means.append(0.0)
        
        positions = np.arange(dataset.n_sources) + list(trainer.models.keys()).index(name) * 0.2 - 0.3
        bars = ax2.bar(positions, weight_means, 0.2, 
                       label=trainer.method_labels[name], 
                       color=trainer.method_colors[name],
                       alpha=0.7, edgecolor='black', linewidth=0.5)
        weight_bars.append(bars)
    
    ax2.set_title('2D Average Source Weights', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Source Index', fontsize=12)
    ax2.set_ylabel('Weight', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(np.arange(dataset.n_sources))
    ax2.set_xticklabels([f'S{i+1}' for i in range(dataset.n_sources)])
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/2d_source_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    plt.clf()
    gc.collect()
    
    return source_errors, source_weights


def main_2d():
    """Main 2D training script with four-way comparison"""
    # Parameters for 2D
    n_samples = 512  # Reduced for 2D (more computationally intensive)
    grid_size = 32   # 32x32 grid for 2D (1024 spatial points total)
    n_sources = 3
    batch_size = 16  # Smaller batch size for 2D
    timesteps = 5
    n_epochs = 200   # Fewer epochs for 2D
    save_dir = 'results_2d'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Creating 2D datasets with grid size {grid_size}x{grid_size}...")
    
    # Create 2D datasets
    train_dataset = MultiSourceNavierStokes2DDataset(
        n_samples=n_samples,
        grid_size=grid_size,
        n_sources=n_sources,
        n_timesteps=timesteps,
        noise_levels=[0.02, 0.02, 0.03, 0.04, 0.1],
        bias_levels=[0.0, 0.01, -0.02, 0.03, 0.05]
    )
    
    val_dataset = MultiSourceNavierStokes2DDataset(
        n_samples=n_samples//8,
        grid_size=grid_size,
        n_sources=n_sources,
        n_timesteps=timesteps,
        noise_levels=[0.02, 0.02, 0.03, 0.04, 0.1],
        bias_levels=[0.0, 0.01, -0.02, 0.03, 0.05]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          num_workers=4, pin_memory=True)
    
    print(f"Creating 2D models...")
    
    # Create all four 2D models
    models = {
        'softmax': SoftmaxMultiSourceIntegration2D(
            n_sources=n_sources,
            grid_size=grid_size,
            hidden_dim=256,  # Reduced for 2D
            n_prediction_steps=timesteps,
            temperature=1.0
        ),
        'lagrangian_single_scale': SingleTimeScaleLagrangianOptimizer2D(
            n_sources=n_sources,
            grid_size=grid_size,
            hidden_dim=256,
            n_prediction_steps=timesteps,
            rho=0.005
        ),
        'lagrangian_two_scale': TwoTimeScaleLagrangianOptimizer2D(
            n_sources=n_sources,
            grid_size=grid_size,
            hidden_dim=256,
            n_prediction_steps=timesteps,
            rho=0.1,
            multiplier_lr=0.1,
            multiplier_update_frequency=1,
            constraint_weight=0.1
        ),
        'admm': ADMMOptimizer2D(
            n_sources=n_sources,
            grid_size=grid_size,
            hidden_dim=256,
            n_prediction_steps=timesteps,
            rho=0.1,
            admm_iterations=3
        )
    }
    
    # Learning rates for each model/component
    learning_rates = {
        'softmax': 1e-4,
        'lagrangian_single_scale': 1e-4,
        'lagrangian_two_scale_theta': 1e-4,
        'lagrangian_two_scale_lambda': 1e-3,
        'admm': 1e-4
    }
    
    # Create trainer
    trainer = FourWayComparativeTrainer2D(
        models=models,
        learning_rates=learning_rates,
        device=device
    )
    
    # Training loop
    best_val_loss = {name: float('inf') for name in models.keys()}
    
    try:
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            
            # Train
            train_metrics = trainer.train_epoch(train_loader)
            
            # Validate
            val_metrics = trainer.validate(val_loader)
            
            # Print metrics for all models
            for name in models.keys():
                print(f"\n{trainer.method_labels[name]}:")
                print(f"  Train Loss: {train_metrics[name]['loss']:.7f}")
                print(f"  Meas Loss: {train_metrics[name]['meas_loss']:.7f}")
                print(f"  Physics Loss: {train_metrics[name]['physics_loss']:.7f}")
                print(f"  Val Meas Loss: {val_metrics[name]['meas_loss']:.7f}")
                print(f"  Val True Error: {val_metrics[name]['true_error']:.7f}")
                if 'constraint_loss' in train_metrics[name]:
                    print(f"  Constraint Loss: {train_metrics[name]['constraint_loss']:.7f}")
            
            # Store metrics
            for name in trainer.metrics.keys():
                for k, v in train_metrics[name].items():
                    trainer.metrics[name][k].append(v)
            
            # Update learning rates
            trainer.update_schedulers(val_metrics)
            
            # Plot comparison every 10 epochs
            if (epoch + 1) % 10 == 0:
                plot_comparative_results_2d(
                    trainer,
                    val_dataset,
                    epoch + 1,
                    save_dir
                )
            
            # Save best models
            for name in models.keys():
                if val_metrics[name]['meas_loss'] < best_val_loss[name]:
                    best_val_loss[name] = val_metrics[name]['meas_loss']
                    torch.save({
                        'model_state': trainer.models[name].state_dict(),
                        'val_loss': val_metrics[name]['meas_loss'],
                        'epoch': epoch,
                        'model_type': name
                    }, f'{save_dir}/best_2d_{name}_model.pth')
    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final models...")
    
    finally:
        # Save final models and metrics
        torch.save({
            'model_states': {name: model.state_dict() for name, model in trainer.models.items()},
            'metrics': trainer.metrics,
            'config': {
                'n_samples': n_samples,
                'grid_size': grid_size,
                'n_sources': n_sources,
                'n_epochs': n_epochs
            }
        }, f'{save_dir}/final_2d_models.pth')
        
        # Plot final comparison and analysis
        plot_comparative_results_2d(trainer, val_dataset, n_epochs, save_dir)
        analyze_source_reliability_2d(trainer, val_dataset)
        
        print("\n" + "="*80)
        print("2D TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nResults saved to: {save_dir}/")
        print("\nBest validation losses:")
        for name, loss in best_val_loss.items():
            print(f"  {trainer.method_labels[name]}: {loss:.7f}")
        print("="*80)


if __name__ == "__main__":
    main_2d()