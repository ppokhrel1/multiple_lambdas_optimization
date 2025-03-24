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

def huber_loss(pred, target, delta = 1.0):
    """Huber loss function"""
    diff = pred - target
    abs_diff = torch.abs(diff)
    quadratic = torch.min(abs_diff, torch.tensor(delta, device=pred.device))
    linear = abs_diff - quadratic
    return 0.5 * quadratic.pow(2) + delta * linear

class BasePDEModel(nn.Module):
    """Base class for PDE models"""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
    
    def forward(
        self,
        x: torch.Tensor,
        measurements: torch.Tensor
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

class ResidualBlock(nn.Module):
    """Basic residual block"""
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, in_dim)
        )
    
    def forward(self, x):
        return x + self.layers(x)

class MultiStepPredictionModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_steps: int,
        dt: float = 0.001,
        dx: float = None,
        nu: float = 0.01
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        self.dt = dt
        self.dx = dx if dx is not None else 2.0/input_dim
        self.nu = nu
        
        # Enhanced prediction network architecture
        self.prediction_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),  # Update input size to match concatenation
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout
            
            #ResidualBlock(hidden_dim * 2, hidden_dim * 2),
            #nn.LayerNorm(hidden_dim * 2),
            #nn.Dropout(0.2),  # Add dropout
            
            nn.Linear(hidden_dim * 2, input_dim)  # Output size matches the input dimension
        )
                
        # # Additional physics-aware processing
        # self.physics_net = nn.Sequential(
        #     nn.Linear(input_dim * 4, hidden_dim),  # [state, prediction, du_dx, d2u_dx2]
        #     nn.LayerNorm(hidden_dim),
        #     nn.ReLU(),
        #     ResidualBlock(hidden_dim, hidden_dim),
        #     nn.Linear(hidden_dim, input_dim)
        # )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = x.shape[0]
        predictions = [x]
        current_state = x
        
        return self.prediction_net(x), None

        '''
        for step in range(self.n_steps-1):
            # Compute spatial derivatives
            #du_dx, d2u_dx2 = self.compute_derivatives(current_state)
            
            # Neural network prediction
            network_input = current_state.unsqueeze(0) #torch.cat([current_state, current_state], dim=-1)
            prediction = self.prediction_net(network_input)

            
            next_state = prediction[-1]

            # # Combine neural and physics predictions
            # combined_input = torch.cat([current_state, prediction, du_dx, d2u_dx2], dim=-1)
            # physics_correction = self.physics_net(combined_input)
            
            # # Final update with residual connection
            # next_state = current_state + self.dt * (prediction + physics_correction + physics_update)
            
            # # Apply boundary conditions
            # next_state = F.pad(next_state[..., 1:-1], (1,1), mode='circular')
            
            # # Add noise to prevent collapse to straight line
            # if self.training:
            #    noise = torch.randn_like(next_state) * 1e-6 * self.dt
            #    next_state = next_state + noise
                
            predictions.append(next_state)
            current_state = next_state
        
        predictions = torch.stack(predictions, dim=0)

        return predictions, None
        '''
    
    def compute_derivatives(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial derivatives with improved numerical stability"""
        # Pad for periodic boundary conditions
        u_padded = F.pad(u, (1, 1), mode='circular')
        
        # First derivative with central difference
        du_dx = torch.zeros_like(u)
        du_dx = (u_padded[..., 2:] - u_padded[..., :-2]) / (2 * self.dx)
        
        # Second derivative
        d2u_dx2 = torch.zeros_like(u)
        d2u_dx2 = (u_padded[..., 2:] - 2*u_padded[..., 1:-1] + u_padded[..., :-2]) / (self.dx**2)
        
        return du_dx, d2u_dx2


    


import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from enum import Enum
from scipy.ndimage import gaussian_filter
from typing import Optional, Tuple, List, Dict

class PhysicsRegime(Enum):
    SMOOTH = "smooth"
    SHOCK = "shock"
    BOUNDARY = "boundary"
    TURBULENT = "turbulent"

class MultiSourceNavierStokes1DDataset(Dataset):
    def __init__(self, n_samples: int, input_dim: int, n_sources: int, n_timesteps: int = 50, 
                 noise_levels: Optional[List[float]] = None, bias_levels: Optional[List[float]] = None, 
                 seed: int = 42):
        """
        Initialize the dataset with multiple sources of Navier-Stokes solutions.
        
        Args:
            n_samples: Number of samples to generate
            input_dim: Spatial dimension
            n_sources: Number of measurement sources
            n_timesteps: Number of time steps
            noise_levels: List of noise levels for each source
            bias_levels: List of bias levels for each source
            seed: Random seed
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.n_sources = n_sources
        self.n_timesteps = n_timesteps
        
        # Physical parameters
        self.dt = 0.001  # Smaller timestep for stability
        self.dx = 2.0 / input_dim
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
        
        # Generate input grid
        self.x = torch.linspace(-1, 1, input_dim)
        
        # Set measurement characteristics with bounds
        if noise_levels is None:
            noise_levels = [min(0.02 * (i + 1), 0.1) for i in range(n_sources)]
        if bias_levels is None:
            bias_levels = [min(0.03 * (i - n_sources/2), 0.15) for i in range(n_sources)]
        
        self.noise_levels = noise_levels
        self.bias_levels = bias_levels
        
        # Initialize storage
        self.solutions = []
        self.measurements = []
        self.states = []
        
        # Generate samples for each regime
        self._generate_samples()
        
        # Convert to tensors with stability checks
        self.states = torch.stack(self.states)
        self.solutions = torch.stack(self.solutions)
        self.measurements = torch.stack(self.measurements)
        
        # Additional stability verification
        self.verify_dataset_stability()
        
        # Repeat x grid for all samples
        self.x = self.x.repeat(n_samples, 1)

    def _generate_samples(self):
        """Generate samples for each physics regime"""
        samples_per_regime = self.n_samples // len(PhysicsRegime)
        
        for regime in PhysicsRegime:
            print(f"Generating {samples_per_regime} samples for {regime}")
            for i in range(samples_per_regime):
                try:
                    # Generate initial state with stability check
                    phase_shift = 2 * np.pi * i / samples_per_regime
                    initial_state = self.generate_initial_condition(regime, phase_shift)
                    
                    # Verify initial condition stability
                    if torch.isnan(initial_state).any() or torch.isinf(initial_state).any():
                        raise ValueError("Unstable initial condition detected")
                    
                    self.states.append(initial_state)
                    
                    # Solve with stability checks
                    solution_sequence = self.solve_navier_stokes_sequence(initial_state)
                    if torch.isnan(solution_sequence).any() or torch.isinf(solution_sequence).any():
                        raise ValueError("Unstable solution detected")
                    
                    self.solutions.append(solution_sequence)
                    
                    # Generate and process measurements
                    source_measurements = self.generate_source_measurements(solution_sequence)
                    source_measurements = self.process_measurements(source_measurements)
                    self.measurements.append(source_measurements)
                    
                except Exception as e:
                    print(f"Error generating sample {i} for {regime}: {str(e)}")
                    # Generate a replacement sample from existing stable solutions
                    if len(self.solutions) > 0:
                        idx = np.random.randint(len(self.solutions))
                        self.states.append(self.states[idx].clone())
                        self.solutions.append(self.solutions[idx].clone())
                        self.measurements.append(self.measurements[idx].clone())
                    else:
                        raise RuntimeError("Unable to generate initial stable solutions")
    def solve_navier_stokes_sequence(self, u0: torch.Tensor, n_steps: Optional[int] = None) -> torch.Tensor:

        if n_steps is None:
            n_steps = self.n_timesteps
            
        solutions = [u0]
        u = u0.clone()
        
        try:
            for t in range(n_steps):
                # Check CFL condition
                if not self.check_cfl_condition(u):
                    raise ValueError("CFL condition violated")
                
                # Compute spatial derivatives with stability
                du_dx = self.compute_stable_derivative(u, self.dx)
                d2u_dx2 = self.compute_stable_second_derivative(u, self.dx)
                
                # Time update with limiting
                du_dt = -u * du_dx + self.nu * d2u_dx2
                du_dt = torch.clamp(du_dt, -self.max_velocity, self.max_velocity)
                
                u = u + self.dt * du_dt
                
                # Apply spatial filtering if needed
                if torch.max(torch.abs(u)) > self.config['filter_threshold']:
                    u = self.spatial_filter(u)
                
                # Periodic boundary conditions
                u = self.apply_boundary_conditions(u)
                
                solutions.append(u.clone())
                
                # Check for instabilities
                if self.check_instability(u):
                    raise ValueError(f"Solution became unstable at step {t}")
                
        except Exception as e:
            print(f"Error in solution: {str(e)}")
            # Return partial solution if available
            if len(solutions) > 1:
                return torch.stack(solutions[:len(solutions)])
            raise
        
        return torch.stack(solutions)

    def check_cfl_condition(self, u: torch.Tensor) -> bool:
        """Check CFL condition for stability"""
        cfl = torch.max(torch.abs(u)) * self.dt / self.dx
        return cfl <= self.config['cfl_safety']

    def compute_stable_derivative(self, u: torch.Tensor, dx: float) -> torch.Tensor:
        """Compute spatial derivative with stability considerations"""
        du_dx = torch.zeros_like(u)
        du_dx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
        du_dx[0] = (u[1] - u[-1]) / (2 * dx)
        du_dx[-1] = (u[0] - u[-2]) / (2 * dx)
        return torch.clamp(du_dx, -self.max_velocity/dx, self.max_velocity/dx)

    def compute_stable_second_derivative(self, u: torch.Tensor, dx: float) -> torch.Tensor:
        """Compute second spatial derivative with stability"""
        d2u_dx2 = torch.zeros_like(u)
        d2u_dx2[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx**2)
        d2u_dx2[0] = (u[1] - 2*u[0] + u[-1]) / (dx**2)
        d2u_dx2[-1] = (u[0] - 2*u[-1] + u[-2]) / (dx**2)
        return torch.clamp(d2u_dx2, -self.max_velocity/(dx**2), self.max_velocity/(dx**2))

    def spatial_filter(self, field: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        # Ensure field is 3D [batch, channel, spatial]
        original_shape = field.shape
        if field.dim() == 1:
            field = field.unsqueeze(0).unsqueeze(0)
        elif field.dim() == 2:
            field = field.unsqueeze(1)
        
        # Create and normalize the kernel
        kernel = torch.ones(1, 1, kernel_size, device=field.device) / kernel_size
        
        # Reflect padding to avoid border effects
        padding = kernel_size // 2
        field_padded = torch.nn.functional.pad(field, (padding, padding), mode='reflect')
        
        # Apply convolution
        filtered = torch.nn.functional.conv1d(field_padded, kernel)
        
        # Restore original shape
        if len(original_shape) == 1:
            filtered = filtered[0, 0]
        elif len(original_shape) == 2:
            filtered = filtered[:, 0]
            
        return filtered

    def apply_boundary_conditions(self, u: torch.Tensor) -> torch.Tensor:
        """Apply periodic boundary conditions"""
        u_new = u.clone()
        u_new[0] = u[-2]
        u_new[-1] = u[1]
        return u_new

    def check_instability(self, u: torch.Tensor) -> bool:
        """Check for numerical instabilities"""
        return torch.isnan(u).any() or torch.isinf(u).any() or \
               torch.max(torch.abs(u)) > self.max_velocity

    def process_measurements(self, measurements: torch.Tensor) -> torch.Tensor:
        """Process measurements to handle invalid values"""
        # Replace inf values with NaN
        measurements = torch.where(torch.isinf(measurements), 
                                 torch.tensor(float('nan')), measurements)
        
        # Clip extreme values
        measurements = torch.clamp(measurements, 
                                 -self.config['measurement_max'], 
                                 self.config['measurement_max'])
        
        # Handle NaN values through interpolation
        nan_mask = torch.isnan(measurements)
        if nan_mask.any():
            measurements = self.interpolate_nan_values(measurements)
            
        return measurements

    def interpolate_nan_values(self, data: torch.Tensor) -> torch.Tensor:
        """Interpolate NaN values in the data"""
        data_numpy = data.numpy()
        for i in range(data_numpy.shape[0]):
            mask = np.isnan(data_numpy[i])
            data_numpy[i, mask] = np.interp(
                np.flatnonzero(mask),
                np.flatnonzero(~mask),
                data_numpy[i, ~mask]
            )
        return torch.from_numpy(data_numpy)

    def generate_initial_condition(self, regime: PhysicsRegime, phase_shift: float = 0.0) -> torch.Tensor:
        
        x = torch.linspace(-1, 1, self.input_dim)
        phase_shift_tensor = torch.tensor(phase_shift)
        
        try:
            if regime == PhysicsRegime.SMOOTH:
                u0 = self._generate_smooth_initial_condition(x, phase_shift_tensor)
            elif regime == PhysicsRegime.SHOCK:
                u0 = self._generate_shock_initial_condition(x, phase_shift_tensor)
            elif regime == PhysicsRegime.BOUNDARY:
                u0 = self._generate_boundary_initial_condition(x, phase_shift_tensor)
            else:  # TURBULENT
                u0 = self._generate_turbulent_initial_condition(x, phase_shift_tensor)
            
            # Normalize and ensure stability
            u0 = self._normalize_and_stabilize(u0)
            
            return u0
            
        except Exception as e:
            print(f"Error generating initial condition for {regime}: {str(e)}")
            # Return a stable fallback condition
            return torch.sin(np.pi * x)

    def _generate_smooth_initial_condition(self, x: torch.Tensor, phase_shift: torch.Tensor) -> torch.Tensor:
        """Generate smooth initial condition"""
        # Multiple frequency components with varying amplitudes
        u0 = (torch.sin(2 * np.pi * x + phase_shift) + 
              0.5 * torch.sin(4 * np.pi * x + 2*phase_shift) +
              0.25 * torch.sin(6 * np.pi * x + 3*phase_shift))
        
        # Add localized features
        gaussian = torch.exp(-10 * (x - 0.3 * torch.sin(phase_shift))**2)
        u0 = u0 + 0.2 * gaussian
        
        return u0

    def _generate_shock_initial_condition(self, x: torch.Tensor, phase_shift: torch.Tensor) -> torch.Tensor:
        """Generate shock-like initial condition"""
        shift1 = 0.4 * torch.sin(phase_shift)
        shift2 = 0.4 * torch.sin(phase_shift + torch.tensor(np.pi/3))
        
        u0 = torch.zeros_like(x)
        u0[x < shift1] = 0.8
        u0[x >= shift1] = -0.4
        u0[x < shift2] = 0.5
        
        # Smooth the shocks
        u0 = F.conv1d(
            u0.view(1, 1, -1),
            torch.ones(1, 1, 3) / 3,
            padding=1
        ).view(-1)
        
        # Add oscillations
        u0 = u0 + 0.05 * torch.sin(8 * np.pi * x)
        
        return u0

    def _generate_boundary_initial_condition(self, x: torch.Tensor, phase_shift: torch.Tensor) -> torch.Tensor:
        """Generate boundary layer initial condition"""
        shift1 = 0.2 * torch.sin(phase_shift)
        shift2 = 0.2 * torch.sin(phase_shift + torch.tensor(np.pi/2))
        
        u0 = (torch.exp(-10 * (x + 0.6 + shift1)**2) + 
              torch.exp(-15 * (x - 0.6 + shift2)**2) +
              0.3 * torch.exp(-20 * (x + shift1)**2))
        
        # Add wave-like features
        u0 = u0 + 0.1 * torch.sin(4 * np.pi * x + phase_shift)
        
        return u0

    def _generate_turbulent_initial_condition(self, x: torch.Tensor, phase_shift: torch.Tensor) -> torch.Tensor:
        """Generate turbulent initial condition"""
        u0 = torch.zeros_like(x)
        
        # Multiple scales with random phases
        for k in range(1, 8):
            phase = phase_shift + torch.tensor(2 * np.pi * k / 5)
            amplitude = 0.7 / (k**0.5)
            u0 += amplitude * torch.sin(k * np.pi * x + phase)
        
        # Add localized turbulent features
        for _ in range(3):
            center = 0.5 * torch.sin(phase_shift + torch.tensor(np.random.rand() * np.pi))
            width = 0.1 + 0.05 * np.random.rand()
            amplitude = 0.1 + 1e-4 * np.random.rand()
            u0 += amplitude * torch.exp(-(x - center)**2 / width**2)
        
        # Add small scale noise
        u0 += 0.02 * torch.randn_like(x)
        
        return u0

    def _normalize_and_stabilize(self, u0: torch.Tensor) -> torch.Tensor:
        """Normalize and stabilize the initial condition"""
        # Remove any NaN or Inf values
        u0 = torch.where(torch.isnan(u0) | torch.isinf(u0), 
                        torch.zeros_like(u0), u0)
        
        # Normalize
        u0 = u0 / (torch.max(torch.abs(u0)) + self.eps)
        
        # Apply spatial filtering if needed
        if torch.max(torch.abs(u0)) > self.config['filter_threshold']:
            u0 = self.spatial_filter(u0)
        
        return u0

    def generate_source_measurements(self, solution: torch.Tensor) -> torch.Tensor:
        measurements = []
        
        for source_idx in range(self.n_sources):
            try:
                # Add bias
                biased = solution + self.bias_levels[source_idx]
                
                # Add noise
                noisy = biased + self.noise_levels[source_idx] * torch.randn_like(solution)
                
                # Add source-specific characteristics
                if source_idx % 3 == 0:
                    # Apply filtering timestep by timestep
                    filtered = torch.zeros_like(noisy)
                    for t in range(noisy.shape[0]):
                        filtered[t] = self.spatial_filter(noisy[t], kernel_size=5)
                    noisy = filtered
                elif source_idx % 3 == 1:
                    # Add sparse spikes
                    #noisy = self._add_sparse_spikes(noisy)
                    pass
                # Add missing data
                noisy = self._add_missing_data(noisy, source_idx)
                
                measurements.append(noisy)
                
            except Exception as e:
                print(f"Error generating measurements for source {source_idx}: {str(e)}")
                # Fallback to clean measurements
                measurements.append(solution.clone())

        measurements = torch.stack(measurements)
        return self.process_measurements(measurements)

    def _add_sparse_spikes(self, data: torch.Tensor, spike_prob: float = 0.005,
                          spike_magnitude: float = 1.5) -> torch.Tensor:
        """Add sparse spikes to the data"""
        spike_mask = torch.rand_like(data) < spike_prob
        return torch.where(spike_mask, spike_magnitude * data, data)

    def _add_missing_data(self, data: torch.Tensor, source_idx: int) -> torch.Tensor:
        """Add missing data (NaN values)"""
        missing_prob = 0.005 * (source_idx + 1)
        mask = torch.rand_like(data) > missing_prob
        return torch.where(mask, data, torch.tensor(float('nan')))

    def verify_dataset_stability(self):
        """Verify overall dataset stability"""
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
            'x': self.states[idx],
            'measurements': self.measurements[idx],
            'true_solution': self.solutions[idx],
        }


class EnhancedMultiSourceBase(BasePDEModel):
    def __init__(
        self,
        n_sources: int,
        input_dim: int,
        hidden_dim: int = 128,
        n_prediction_steps: int = 5,
        dt: float = 0.001,
    ):
        super().__init__(input_dim, hidden_dim)
        self.n_sources = n_sources
        self.n_prediction_steps = n_prediction_steps
        self.dt = dt
        
        # Physical parameters
        self.dx = 2.0 / input_dim
        self.Re = 50
        self.viscosity = 1.0 / self.Re
        
        
        # Single prediction network - Fix input_dim to match your spatial dimension
        self.prediction_module = MultiStepPredictionModule(
            input_dim=input_dim,  # Changed from 96 to input_dim
            hidden_dim=hidden_dim,
            n_steps=n_prediction_steps,
            dt=dt,
            dx=self.dx,
            nu=self.viscosity
        )


    def compute_physics_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-informed loss based on Burgers' equation
        predictions: [batch, timesteps, spatial_points] - Model predictions
        """
        total_residual = 0.0
        
        # For each timestep except the last one
        for t in range(predictions.shape[1] - 1):
            current_state = predictions[:, t]  # Current predicted state
            next_state_pred = predictions[:, t + 1]  # Next predicted state
            
            # Compute spatial derivatives for current state
            du_dx, d2u_dx2 = self.compute_derivatives(current_state)
            
            # Physics consistency check (Burgers equation)
            du_dt = (next_state_pred - current_state) / self.dt
            physics_residual = (
                du_dt + 
                current_state * du_dx - 
                self.viscosity * d2u_dx2
            ).pow(2).mean()
            
            total_residual += physics_residual
        
        # Average over timesteps
        physics_loss = total_residual / (predictions.shape[1] - 1 + 1e-5)
        
        # Add conservation laws
        mass_initial = torch.trapz(predictions[:, 0], dx=self.dx, dim=-1)
        mass_final = torch.trapz(predictions[:, -1], dx=self.dx, dim=-1)
        mass_conservation = (mass_final - mass_initial).pow(2).mean()
        
        energy_initial = torch.trapz(predictions[:, 0]**2, dx=self.dx, dim=-1)
        energy_final = torch.trapz(predictions[:, -1]**2, dx=self.dx, dim=-1)
        energy_conservation = (energy_final - energy_initial).pow(2).mean()
        
        # Scale the losses
        total_loss = (
            physics_loss + 
            0.1 * mass_conservation + 
            0.1 * energy_conservation
        )
        
        return  total_loss




    def compute_derivatives(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial derivatives with periodic boundary conditions"""
        # Pad for periodic boundary conditions
        u_padded = F.pad(u, (1, 1), mode='circular')
        
        # First derivative
        du_dx = (u_padded[:, 2:] - u_padded[:, :-2]) / (2 * self.dx)
        
        # Second derivative
        d2u_dx2 = (u_padded[:, 2:] - 2*u_padded[:, 1:-1] + u_padded[:, :-2]) / (self.dx**2)
        
        return du_dx, d2u_dx2


    def compute_measurement_loss(
        self, 
        predictions: torch.Tensor, 
        measurements: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute measurement consistency loss
        predictions: [batch, n_sources, timesteps, input_dim]
        measurements: [batch, n_sources, timesteps, input_dim]
        weights: [n_sources]
        """
        meas_loss = 0.0
        valid_samples = 0
        
        for i in range(self.n_sources):
            for t in range(min(predictions.shape[2], measurements.shape[2])):
                valid_mask = ~torch.isnan(measurements[:, i, t])
                if valid_mask.any():
                    loss = huber_loss(
                        predictions[:, i, t][valid_mask],
                        measurements[:, i, t][valid_mask]
                    ).mean()
                    meas_loss += weights[i] * loss
                    valid_samples += 1
        
        return meas_loss / max(valid_samples, 1)


    def compute_source_reliability(self, measurements: torch.Tensor) -> torch.Tensor:
        """
        Compute reliability scores based on measurement consistency
        measurements: [batch, n_sources, timesteps, input_dim]
        """
        # Spatial consistency
        mean_state = measurements.mean(dim=1, keepdim=True)  # [batch, 1, timesteps, input_dim]
        spatial_deviation = (measurements - mean_state).abs().mean(dim=-1)  # [batch, n_sources, timesteps]
        
        # Temporal consistency
        temporal_diff = (measurements[:, :, 1:] - measurements[:, :, :-1]).abs().mean(dim=-1)  # [batch, n_sources, timesteps-1]
        
        # Physics consistency (simplified)
        physics_residual = torch.zeros(self.n_sources, device=measurements.device)
        for i in range(self.n_sources):
            valid_mask = ~torch.isnan(measurements[:, i])
            if valid_mask.any():
                source_pred = measurements[:, i][valid_mask].reshape(-1, self.input_dim)
                du_dx = (source_pred[:, 2:] - source_pred[:, :-2]) / (2 * self.dx)
                physics_residual[i] = du_dx.abs().mean()
        
        # Combine metrics (lower is better)
        reliability = -(
            spatial_deviation.mean(dim=(0, 2)) +  # Average over batch and time
            temporal_diff.mean(dim=(0, 2)) +      # Average over batch and time
            physics_residual
        )
        
        return reliability


class SoftmaxMultiSourceIntegration(EnhancedMultiSourceBase):
    def __init__(self, *args, temperature: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.lambda_weights = nn.Parameter(torch.randn(self.n_sources) * 0.01 + 1.0/self.n_sources)
        self.register_buffer('current_weights', self.lambda_weights)
        
    def forward(self, x: torch.Tensor,  t: int = 0) -> Tuple[torch.Tensor, Dict]:
        # Generate prediction from analysis state
        #predictions, _ = self.prediction_module(x)  # [batch, pred_steps, spatial_points]
        #print(predictions.shape, x.shape)
        predictions, _ = self.prediction_module(x)
        
        return predictions, {
            'weights': self.lambda_weights,
        }

    def get_analysis_state(self, measurements):
        # Get reliability scores and weights
        weights = F.softmax(self.lambda_weights, dim=0)
        
        # Directly combine measurements using weights
        analysis_states = []  # List to store each timestep's analysis state
        
        weights_reshaped = weights.view(1, -1, 1, 1).expand(measurements.shape[0], measurements.shape[1], measurements.shape[2], measurements.shape[3])

        # Multiply measurements by the reshaped weights
        weighted_measurements = measurements * weights_reshaped  # Shape: [batch_size, 3, 51, 32]

        # Sum across the source dimension (the second dimension, which has size 3)
        analysis_states = weighted_measurements.sum(dim=1)  # Shape: [batch_size, 51, 32]


        # Stack along time dimension to match x shape
        # analysis_states = torch.stack(analysis_states, dim=1)  # [batch, timesteps, spatial_points]
        
        return analysis_states, {
            'analysis_state': analysis_states,
            'weights': self.lambda_weights,
        }


    def compute_loss(
        self,
        x: torch.Tensor,
        measurements: torch.Tensor,
        true_solution: Optional[torch.Tensor] = None,
        is_training: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        analysis_state, _ = self.get_analysis_state(measurements)
        
        # Ensure the batch sizes match
        batch_size = analysis_state.shape[0]
        
        # Initialize predictions list with the initial state
        predictions_list = [analysis_state[:, 0]]
        
        # Loop over timesteps to make predictions
        for t in range(analysis_state.shape[1] - 1):
            # Get the current state (last prediction)
            current_state = predictions_list[-1]
            
            # Make prediction for the next timestep
            next_prediction, _ = self.prediction_module(current_state)
            #print(next_prediction.shape)
            
            # Append the prediction to the list
            predictions_list.append(next_prediction)  # Take the last prediction step
        
        # Stack predictions along the timestep dimension
        predictions = torch.stack(predictions_list, dim=1)
        
        # Compute loss between predictions and analysis state
        loss = F.l1_loss(predictions, analysis_state)
        
        # Compute physics loss if training
        physics_loss = self.compute_physics_loss(predictions) if is_training else torch.tensor(0.0)
        
        # Combine losses
        total_loss = loss + 1e-4 * physics_loss
        
        return total_loss, {
            'loss': total_loss.item(),
            'pred_loss': loss.item(),
            'physics_loss': physics_loss.item() if is_training else 0.0,
            'weights': F.softmax(self.lambda_weights, dim=0),
            'reconstruction_loss': loss.item(),
        }


class TwoTimeScaleLagrangianOptimizer(EnhancedMultiSourceBase):
    def __init__(
        self,
        n_sources: int,
        input_dim: int,
        hidden_dim: int = 128,
        n_prediction_steps: int = 5,
        dt: float = 0.001,
        rho: float = 0.1,
        multiplier_lr: float = 0.01,
        multiplier_update_frequency: int = 5
    ):
        super().__init__(
            n_sources=n_sources,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_prediction_steps=n_prediction_steps,
            dt=dt
        )
        self.rho = rho
        self.multiplier_lr = multiplier_lr
        self.multiplier_update_frequency = multiplier_update_frequency
        self.multiplier_update_counter = 0
        
        # Initialize Lagrangian parameters
        self.lambda_weights = nn.Parameter(torch.randn(n_sources) * 0.01 + 1.0/n_sources)
        self.mu = nn.Parameter(torch.zeros(1))
        self.nu = nn.Parameter(torch.zeros(n_sources))
        self.register_buffer('current_weights', self.lambda_weights)
        
    def compute_lagrangian_weights(self) -> torch.Tensor:
        """Compute weights using Lagrangian formulation"""
        # Combine base weights with reliability scores
        weights = self.lambda_weights

        # Apply equality constraint
        sum_weights = weights.sum()
        weights = weights / (sum_weights + 1e-8)  # Normalize to sum to 1
        
        # Apply non-negativity constraint
        #weights = F.relu(weights)  # Ensure non-negative

        
        return weights
    
    def get_analysis_state(self, measurements):
        # Get reliability scores and weights

        weights = self.compute_lagrangian_weights()
        
        # Directly combine measurements using weights
        analysis_states = []  # List to store each timestep's analysis state

        #print(measurements.shape)
        weights_reshaped = weights.view(1, -1, 1, 1).expand(measurements.shape[0], measurements.shape[1], measurements.shape[2], measurements.shape[3])

        # Multiply measurements by the reshaped weights
        weighted_measurements = measurements * weights_reshaped  # Shape: [batch_size, 3, 51, 32]

        # Sum across the source dimension (the second dimension, which has size 3)
        analysis_states = weighted_measurements.sum(dim=1)  # Shape: [batch_size, 51, 32]


        # Stack along time dimension to match x shape
        # analysis_states = torch.stack(analysis_states, dim=1)  # [batch, timesteps, spatial_points]
        
        return analysis_states, {
            'analysis_state': analysis_states,
            'weights': self.lambda_weights,
        }


    def forward(self, x: torch.Tensor, t: int = 0) -> Tuple[torch.Tensor, Dict]:
        batch_size = x.shape[0]
        # Generate prediction from analysis state

        # Generate predictions from analysis state
        predictions, _ = self.prediction_module(x)  # [batch, pred_steps, spatial_points]
        
        return predictions, {
            'predictions': predictions,
            'weights': self.lambda_weights,
        }

    def compute_constraint_losses(self, weights: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute Lagrangian constraint losses"""
        # Equality constraint: sum to 1
        g = weights.sum() - 1.0
        
        # Inequality constraint: non-negativity
        h = -weights
        
        # Constraint violations
        equality_violation = torch.abs(g)
        inequality_violation = torch.relu(h).sum()
        
        # Augmented Lagrangian terms
        equality_term = self.mu * g + (self.rho/2) * g.pow(2)
        inequality_term = (self.nu * h + (self.rho/2) * h.pow(2)).sum()
        
        constraint_loss = equality_term + inequality_term
        
        return constraint_loss, {
            'equality_violation': equality_violation.item(),
            'inequality_violation': inequality_violation.item(),
            'constraint_loss': constraint_loss.item()
        }

    def update_multipliers(self, weights: torch.Tensor):
        """Two-timescale update for Lagrange multipliers"""
        self.multiplier_update_counter += 1
        
        if self.multiplier_update_counter % self.multiplier_update_frequency == 0:
            with torch.no_grad():
                # Update for equality constraint
                g = weights.sum() - 1.0
                self.mu.data += self.multiplier_lr * self.rho * g
                
                # Update for inequality constraints
                h = -weights
                self.nu.data += self.multiplier_lr * self.rho * torch.relu(h)
    
    def compute_loss(
        self,
        x: torch.Tensor,
        measurements: torch.Tensor,
        true_solution: Optional[torch.Tensor] = None,
        is_training: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        analysis_state, _ = self.get_analysis_state(measurements)

        predictions, metadata = self.forward(analysis_state)
        
        weights = self.compute_lagrangian_weights()
        
        analysis_state, _ = self.get_analysis_state(measurements)
        
        # Ensure the batch sizes match
        batch_size = analysis_state.shape[0]
        
        # Initialize predictions list with the initial state
        predictions_list = [analysis_state[:, 0]]
        
        # Loop over timesteps to make predictions
        for t in range(analysis_state.shape[1] - 1):
            # Get the current state (last prediction)
            current_state = predictions_list[-1]
            
            # Make prediction for the next timestep
            next_prediction, _ = self.prediction_module(current_state)
            
            # Append the prediction to the list
            predictions_list.append(next_prediction)  # Take the last prediction step
        
        # Stack predictions along the timestep dimension
        predictions = torch.stack(predictions_list, dim=1)
        
        # Compute loss between predictions and analysis state
        loss = F.l1_loss(predictions, analysis_state)
        
        valid_samples = analysis_state.shape[1]
        if valid_samples > 0:
            loss = loss / valid_samples
        
        physics_loss = 0
        predictions_list = predictions
        if is_training:
            physics_loss += self.compute_physics_loss(predictions_list)
            
            # Constraint losses
            constraint_loss, constraint_dict = self.compute_constraint_losses(weights)
            
            # Lagrangian multiplier terms
            equality_constraint = torch.abs(weights.sum() - 1.0)
            inequality_constraint = torch.relu(-weights).sum()
            
            multiplier_loss = (
                self.mu * equality_constraint +
                (self.nu * inequality_constraint).sum() +
                0.5 * self.rho * (equality_constraint**2 + inequality_constraint.pow(2).sum())
            )
            
            # Total loss
            total_loss = (
                loss +
                1e-4 * physics_loss +
                constraint_loss +
                multiplier_loss
            )
        else:
            total_loss = loss
            physics_loss = torch.tensor(0.0, device=x.device)
            constraint_dict = {}
        
        return total_loss, {
            'loss': total_loss.item(),
            'pred_loss': loss.item(),
            'physics_loss': physics_loss.item() if is_training else 0.0,
            'weights': weights,
            'reconstruction_loss': loss.item(),
            **(constraint_dict if is_training else {})
        }

    def get_source_importance(self) -> Dict[str, torch.Tensor]:
        """Get source importance metrics"""
        with torch.no_grad():
            weights = F.softmax(self.lambda_weights, dim=0)
            return {
                'base_weights': self.lambda_weights.detach(),
                'normalized_weights': weights,
                'multipliers': self.nu.detach(),
                'equality_multiplier': self.mu.detach()
            }
            
    def reset_multiplier_state(self):
        """Reset the multiplier update counter"""
        self.multiplier_update_counter = 0

    def adjust_timescales(self, new_multiplier_lr: float = None, new_update_frequency: int = None):
        """Adjust the timescale parameters"""
        if new_multiplier_lr is not None:
            self.multiplier_lr = new_multiplier_lr
        if new_update_frequency is not None:
            self.multiplier_update_frequency = new_update_frequency


class ComparativeTrainer:
    """Trainer for comparing Softmax and Lagrangian approaches"""
    def __init__(
        self,
        softmax_model: SoftmaxMultiSourceIntegration,
        lagrangian_model: TwoTimeScaleLagrangianOptimizer,
        lr_softmax: float = 1e-4,
        lr_theta: float = 1e-4,
        lr_lambda: float = 1e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.softmax_model = softmax_model.to(device)
        self.lagrangian_model = lagrangian_model.to(device)
        self.device = device
        
        # Initialize optimizers
        self.softmax_optimizer = torch.optim.AdamW(
            softmax_model.parameters(),
            lr=lr_softmax,
            weight_decay=1e-6
        )
        
        # Separate optimizers for Lagrangian model components
        self.theta_optimizer = torch.optim.AdamW(
            [p for n, p in lagrangian_model.named_parameters()
             if not any(x in n for x in ['lambda_weights', 'mu', 'nu'])],
            lr=lr_theta,
            weight_decay=1e-6
        )
        
        self.lambda_optimizer = torch.optim.AdamW(
            [lagrangian_model.lambda_weights],
            lr=lr_lambda,
            weight_decay=1e-6
        )
        
        self.multiplier_optimizer = torch.optim.Adam(
            [lagrangian_model.mu, lagrangian_model.nu],
            lr=lr_lambda * 0.1
        )
        
        # Learning rate schedulers
        self.schedulers = {
            'softmax': torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.softmax_optimizer, mode='min', factor=0.5, patience=5
            ),
            'theta': torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.theta_optimizer, mode='min', factor=0.5, patience=5
            ),
            'lambda': torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.lambda_optimizer, mode='min', factor=0.5, patience=5
            )
        }
        
        # Initialize metric storage
        self.metrics = {
            'softmax': defaultdict(list),
            'lagrangian': defaultdict(list)
        }
    
    def train_epoch(
        self,
        dataloader: DataLoader
    ) -> Dict[str, Dict[str, float]]:
        self.softmax_model.train()
        self.lagrangian_model.train()
        
        epoch_metrics = {
            'softmax': defaultdict(list),
            'lagrangian': defaultdict(list)
        }
        
        for batch in dataloader:
            batch_metrics = self.train_step(batch)
            
            for model_type in ['softmax', 'lagrangian']:
                for k, v in batch_metrics[model_type].items():
                    if v is not None:
                        epoch_metrics[model_type][k].append(v)
        
        return {
            k: {m: float(np.mean(metrics)) for m, metrics in metrics_list.items()}
            for k, metrics_list in epoch_metrics.items()
        }


    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        x = batch['x'].to(self.device)
        measurements = batch['measurements'].to(self.device)
        true_solution = batch.get('true_solution', None)
        if true_solution is not None:
            true_solution = true_solution.to(self.device)

        # Train Softmax model
        self.softmax_optimizer.zero_grad()
        loss_soft, meta_soft = self.softmax_model.compute_loss(
            x, measurements, true_solution, is_training=True
        )
        loss_soft.backward()
        torch.nn.utils.clip_grad_norm_(self.softmax_model.parameters(), max_norm=1.0)
        self.softmax_optimizer.step()

        # Train Lagrangian model
        self.theta_optimizer.zero_grad()
        self.lambda_optimizer.zero_grad()
        self.multiplier_optimizer.zero_grad()
        
        loss_lag, meta_lag = self.lagrangian_model.compute_loss(
            x, measurements, true_solution, is_training=True
        )
        loss_lag.backward()
        torch.nn.utils.clip_grad_norm_(self.lagrangian_model.parameters(), max_norm=1.0)
        self.theta_optimizer.step()
        self.lambda_optimizer.step()
        self.multiplier_optimizer.step()

        # Calculate true losses
        true_loss_soft = 0.0
        true_loss_lag = 0.0
        
        with torch.no_grad():
            # Get initial states
            analysis_state_soft, _ = self.softmax_model.get_analysis_state(measurements)
            analysis_state_lag, _ = self.lagrangian_model.get_analysis_state(measurements)

            # Autoregressive predictions for both models
            soft_preds = [analysis_state_soft[:, 0]]
            lag_preds = [analysis_state_lag[:, 0]]
            
            current_soft = analysis_state_soft[:, 0]
            current_lag = analysis_state_lag[:, 0]
            
            for t in range(true_solution.shape[1] - 1):
                # Softmax prediction
                pred_soft, _ = self.softmax_model(current_soft)
                soft_preds.append(pred_soft)
                current_soft = pred_soft
                
                # Lagrangian prediction
                pred_lag, _ = self.lagrangian_model(current_lag)
                lag_preds.append(pred_lag)
                current_lag = pred_lag

            # Stack predictions [batch, timesteps, features]
            soft_preds = torch.stack(soft_preds, dim=1)
            lag_preds = torch.stack(lag_preds, dim=1)
            
            # Ensure matching dimensions
            if true_solution is not None:
                true_loss_soft = F.mse_loss(
                    soft_preds[:, :true_solution.shape[1]], 
                    true_solution[:, :soft_preds.shape[1]]
                )
                true_loss_lag = F.mse_loss(
                    lag_preds[:, :true_solution.shape[1]], 
                    true_solution[:, :lag_preds.shape[1]]
                )

        return {
            'softmax': {
                'loss': loss_soft.item(),
                'physics_loss': meta_soft.get('physics_loss', 0.0),
                'pred_loss': meta_soft.get('pred_loss', 0.0),
                'weights_mean': meta_soft['weights'].mean().item(),
                'weights_std': meta_soft['weights'].std().item(),
                'true_loss': true_loss_soft.item() if true_solution is not None else 0.0
            },
            'lagrangian': {
                'loss': loss_lag.item(),
                'physics_loss': meta_lag.get('physics_loss', 0.0),
                'pred_loss': meta_lag.get('pred_loss', 0.0),
                'constraint_loss': meta_lag.get('constraint_loss', 0.0),
                'weights_mean': meta_lag['weights'].mean().item(),
                'weights_std': meta_lag['weights'].std().item(),
                'true_loss': true_loss_lag.item() if true_solution is not None else 0.0
            }
        }

    
    def validate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, Dict[str, float]]:
        self.softmax_model.eval()
        self.lagrangian_model.eval()
        
        val_metrics = {
            'softmax': defaultdict(list),
            'lagrangian': defaultdict(list)
        }
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch['x'].to(self.device)
                measurements = batch['measurements'].to(self.device)
                true_solution = batch['true_solution'].to(self.device)
                
                # Get predictions for all timesteps
                soft_preds, lag_preds = [], []
                
                # Softmax model predictions
                current_state = x
                for t in range(true_solution.shape[1]):
                    pred, _ = self.softmax_model(current_state)
                    soft_preds.append(pred)
                    current_state = pred  # Autoregressive update
                
                # Lagrangian model predictions
                current_state = x
                for t in range(true_solution.shape[1]):
                    pred, _ = self.lagrangian_model(current_state)
                    lag_preds.append(pred)
                    current_state = pred  # Autoregressive update
                
                # Stack predictions [batch, timesteps, features]
                soft_preds = torch.stack(soft_preds, dim=1)
                lag_preds = torch.stack(lag_preds, dim=1)
                
                # Calculate losses per timestep
                soft_loss = F.mse_loss(soft_preds, true_solution)
                lag_loss = F.mse_loss(lag_preds, true_solution)
                
                val_metrics['softmax']['loss'].append(soft_loss.item())
                val_metrics['lagrangian']['loss'].append(lag_loss.item())
        
                
              
            
        # Average the metrics
        return {
            k: {m: float(np.mean(v)) for m, v in metrics.items() if len(v) > 0}
            for k, metrics in val_metrics.items()
        }



def plot_comparative_results(
    trainer: ComparativeTrainer,
    dataset: MultiSourceNavierStokes1DDataset,
    epoch: int,
    save_dir: str = 'results'
):
    """Plot detailed comparison between approaches"""
    plt.close('all')
    
    fig = plt.figure(figsize=(20, 12))
    gs = plt.GridSpec(2, 3, figure=fig)
    
    # Training Loss (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(trainer.metrics['softmax']['loss'],
             label='Softmax', color='blue', alpha=0.7)
    ax1.plot(trainer.metrics['lagrangian']['loss'],
             label='Lagrangian', color='red', alpha=0.7)
    ax1.set_title('Training Loss', fontsize=12)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    

    # Weight Distribution (Top Middle)
    ax2 = fig.add_subplot(gs[0, 1])
    sample_idx = np.random.choice(len(dataset))
    sample_idx = 15
    sample = dataset[sample_idx]

    x = sample['x'].to(trainer.device).unsqueeze(0)
    measurements = sample['measurements'].to(trainer.device).unsqueeze(0)
    true_solution = sample['true_solution'].to(trainer.device)
    with torch.no_grad():
        output_soft, metrics_soft = trainer.softmax_model(x, measurements)
        output_lag, metrics_lag = trainer.lagrangian_model(x, measurements)
    
        weights_soft = [a for a in metrics_soft['weights'].detach().cpu().numpy()]
        weights_lag = [w for w in metrics_lag['weights'].detach().cpu().numpy()]
        
        ax2.bar(np.arange(len(weights_soft)) - 0.2, 
                weights_soft, 0.4, 
                label='Softmax', color='blue', alpha=0.7)
        ax2.bar(np.arange(len(weights_lag)) + 0.2, 
                weights_lag, 0.4,
                label='Lagrangian', color='red', alpha=0.7)
    ax2.set_title('Weight Distribution', fontsize=12)
    ax2.set_xlabel('Source Index')
    ax2.set_ylabel('Weight Value')
    ax2.legend()
    ax2.grid(True)
    
    # Physics Loss (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(trainer.metrics['softmax']['physics_loss'],
             label='Softmax', color='blue', alpha=0.7)
    ax3.plot(trainer.metrics['lagrangian']['physics_loss'],
             label='Lagrangian', color='red', alpha=0.7)
    ax3.set_title('Physics Loss', fontsize=12)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True)
    
    # Time Evolution Plot (Bottom Left)
    ax4 = fig.add_subplot(gs[1, 0])
    
    line_colors = {
        'true': 'black',
        'softmax': 'blue',
        'lagrangian': 'red'
    }
    source_colors = ['purple', 'orange', 'brown', 'pink', 'gray']
    
    # Get random sample
    sample_idx = np.random.choice(len(dataset))
    sample = dataset[sample_idx]
    x = sample['x'].to(trainer.device).unsqueeze(0)
    measurements = sample['measurements'].to(trainer.device).unsqueeze(0)
    true_solution = sample['true_solution'].to(trainer.device)
    
    # Get predictions for all timesteps
    with torch.no_grad():
        predictions_soft = []
        predictions_lag = []
        current_soft = x.clone()  # Keep full spatial dimension [1, 64]
        current_lag = x.clone()   # [batch_size=1, input_dim=64]
        
        for t in range(true_solution.shape[0]):  # For each timestep
            # Get full predictions
            pred_soft, _ = trainer.softmax_model(current_soft, measurements)
            pred_lag, _ = trainer.lagrangian_model(current_lag, measurements)
            
            # Store full predictions
            predictions_soft.append(pred_soft)
            predictions_lag.append(pred_lag)
            
            # Update current state with full prediction
            current_soft = pred_soft.detach()
            current_lag = pred_lag.detach()

        # Stack predictions [batch=1, timesteps, spatial=64]
        predictions_soft = torch.stack(predictions_soft, dim=1)
        predictions_lag = torch.stack(predictions_lag, dim=1)

    
    # Choose spatial point for visualization
    spatial_point = x.shape[-1] // 2
    time_steps = np.arange(true_solution.shape[0])
    print(predictions_lag.shape, predictions_soft.shape, true_solution.shape)
    #print(true_solution.shape, predictions_soft.shape, predictions_lag.shape)
    print(predictions_soft[0, spatial_point].cpu().numpy())
    print(predictions_lag[0, spatial_point].cpu().numpy())
    # Plot true solution
    print(predictions_lag.shape, predictions_soft.shape)
    true_line = ax4.plot(time_steps, 
                        true_solution[:, spatial_point].cpu().numpy(),
                        color=line_colors['true'], linestyle='-',
                        alpha=0.7, linewidth=2, label='True')[0]
    
    # Plot model predictions over time
    soft_line = ax4.plot(time_steps, 
                        predictions_soft[0, :, spatial_point].cpu().numpy(), 
                        color=line_colors['softmax'], linestyle='--',
                        alpha=0.7, linewidth=2, label='Softmax')[0]
    
    lag_line = ax4.plot(time_steps, 
                       predictions_lag[0, :, spatial_point].cpu().numpy(), 
                       color=line_colors['lagrangian'], linestyle=':',
                       alpha=0.7, linewidth=2, label='Lagrangian')[0]
    
    # Plot measurements
    legend_handles = [true_line, soft_line, lag_line]
    legend_labels = ['True', 'Softmax', 'Lagrangian']
    
    for j in range(dataset.n_sources):
        valid_mask = ~torch.isnan(measurements[0, j, :, spatial_point])
        if valid_mask.any():
            scatter = ax4.scatter(
                time_steps[valid_mask.cpu()],
                measurements[0, j, valid_mask, spatial_point].cpu().numpy(),
                alpha=0.3, s=20, color=source_colors[j],
                label=f'Source {j+1}'
            )
            legend_handles.append(scatter)
            legend_labels.append(f'Source {j+1}')
    
    # Set y-axis limits
    y_values = np.concatenate([
        true_solution[spatial_point].cpu().numpy(),
        predictions_soft[0, :, spatial_point].cpu().numpy(),
        predictions_lag[0, :, spatial_point].cpu().numpy(),
        #measurements[0, :, :, spatial_point].cpu().numpy()[~torch.isnan(measurements[0, :, :, spatial_point])]
    ])
    y_values = y_values[~np.isnan(y_values) & ~np.isinf(y_values)]
    
    y_min, y_max = np.nanmin(y_values), np.nanmax(y_values)
    y_range = y_max - y_min
    y_min = y_min - 0.1 * y_range
    y_max = y_max + 0.1 * y_range
    
    ax4.set_ylim([y_min, y_max])
    ax4.ticklabel_format(style='plain')
    ax4.set_title(f'Time Evolution at x={spatial_point}', fontsize=12)
    ax4.legend(legend_handles, legend_labels,
              bbox_to_anchor=(1.05, 1),
              loc='upper left',
              borderaxespad=0.)
    ax4.grid(True)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Value')
    
    # Weight Evolution (Bottom Middle)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(trainer.metrics['softmax']['weights_mean'],
             label='Softmax', color='blue', alpha=0.7)
    ax5.plot(trainer.metrics['lagrangian']['weights_mean'],
             label='Lagrangian', color='red', alpha=0.7)
    
    if len(trainer.metrics['softmax']['weights_std']) > 0:
        ax5.fill_between(
            range(len(trainer.metrics['softmax']['weights_std'])),
            np.array(trainer.metrics['softmax']['weights_mean']) -
            np.array(trainer.metrics['softmax']['weights_std']),
            np.array(trainer.metrics['softmax']['weights_mean']) +
            np.array(trainer.metrics['softmax']['weights_std']),
            color='blue', alpha=0.2
        )
        ax5.fill_between(
            range(len(trainer.metrics['lagrangian']['weights_std'])),
            np.array(trainer.metrics['lagrangian']['weights_mean']) -
            np.array(trainer.metrics['lagrangian']['weights_std']),
            np.array(trainer.metrics['lagrangian']['weights_mean']) +
            np.array(trainer.metrics['lagrangian']['weights_std']),
            color='red', alpha=0.2
        )
    
    ax5.set_title('Weight Evolution', fontsize=12)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Weight Value')
    ax5.legend()
    ax5.grid(True)
    
    # True Error (Bottom Right)
    if 'true_error' in trainer.metrics['softmax']:
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(trainer.metrics['softmax']['true_error'],
                 label='Softmax', color='blue', alpha=0.7)
        ax6.plot(trainer.metrics['lagrangian']['true_error'],
                 label='Lagrangian', color='red', alpha=0.7)
        ax6.set_title('True Solution Error', fontsize=12)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Error')
        ax6.legend()
        ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_epoch_{epoch}.png', 
                bbox_inches='tight',
                dpi=300)
    
    plt.close(fig)
    plt.clf()
    gc.collect()

def plot_prediction_comparison(
    trainer: ComparativeTrainer,
    dataset: MultiSourceNavierStokes1DDataset,
    epoch: int,
    save_dir: str = 'results'
):
    """Plot predictions vs true solution at different spatial points"""
    plt.close('all')
    
    # Create 2x2 grid of plots for different spatial locations
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))  # Increased width for legend
    axes = axes.flatten()
    
    # Select random sample from dataset
    sample_idx = np.random.choice(len(dataset))
    sample = dataset[sample_idx]
    
    # Get data and move to device
    x = sample['x'].to(trainer.device).unsqueeze(0)
    measurements = sample['measurements'].to(trainer.device).unsqueeze(0)
    true_solution = sample['true_solution'].to(trainer.device)
    
    # Get model predictions
    output_soft, output_lag = [], []
    with torch.no_grad():
        output_soft, meta_soft = trainer.softmax_model(x, measurements)
        output_lag, meta_lag = trainer.lagrangian_model(x, measurements)
    
    # Get predictions for all timesteps
    with torch.no_grad():
        predictions_soft = []
        predictions_lag = []
        current_soft = x.clone()  # [1, 64]
        current_lag = x.clone()
        
        for t in range(true_solution.shape[0]):
            pred_soft, _ = trainer.softmax_model(current_soft, measurements)
            pred_lag, _ = trainer.lagrangian_model(current_lag, measurements)
            
            predictions_soft.append(pred_soft)
            predictions_lag.append(pred_lag)
            
            current_soft = pred_soft
            current_lag = pred_lag
        
        # Stack predictions [1, timesteps, 64]
        predictions_soft = torch.stack(predictions_soft, dim=1)
        predictions_lag = torch.stack(predictions_lag, dim=1)

    # Get dimensions
    n_timesteps = true_solution.shape[0]
    time_steps = np.arange(n_timesteps)
    
    # Select spatial points evenly distributed in domain
    spatial_points = [8, 16, 24, 30]
    
    # Define colors for sources
    source_colors = plt.cm.Set3(np.linspace(0, 1, dataset.n_sources))
    # Plot at each spatial point
    for idx, spatial_point in enumerate(spatial_points):
        ax = axes[idx]
    
        
        # Plot true solution
        ax.plot(time_steps, 
                true_solution[:, spatial_point].cpu().numpy(),
                'k-', label='True Solution', linewidth=2)
        
        # Plot raw measurements from each source
        for j in range(dataset.n_sources):
            valid_mask = ~torch.isnan(measurements[0, j, :, spatial_point])
            if valid_mask.any():
                ax.scatter(time_steps[valid_mask.cpu()],
                          measurements[0, j, valid_mask, spatial_point].cpu().numpy(),
                          color=source_colors[j], alpha=0.3, s=30,
                          label=f'Source {j+1}')

        # Plot model outputs
        ax.plot(time_steps, 
                predictions_soft[0, :, spatial_point].cpu().numpy(),
                '--', color='blue', label='Softmax', alpha=0.8, linewidth=2)
        ax.plot(time_steps, 
                predictions_lag[0, :, spatial_point].cpu().numpy(),
                '--', color='red', label='Lagrangian', alpha=0.8, linewidth=2)
        
        # Format weights for display
        weights_soft = [f"{w:.3f}" for w in meta_soft['weights'].detach().cpu().numpy()]
        weights_lag = [f"{w:.3f}" for w in meta_lag['weights'].detach().cpu().numpy()]
        
        # Create title with weights
        title = f'Spatial Point x = {spatial_point/dataset.input_dim:.2f}\n'
        subtitle = (f'Weights (Softmax): {weights_soft}\n'
                   f'Weights (Lagrangian): {weights_lag}')
        
        ax.set_title(title + subtitle, fontsize=10, pad=10)
        ax.set_xlabel('Time Steps', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Move legend outside of plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                 fontsize=9, frameon=True, fancybox=True, shadow=True)
        # Set y-limits
        y_data = [true_solution[spatial_point].detach().cpu().numpy(),
                 predictions_soft[0, :, spatial_point].detach().cpu().numpy(),
                 predictions_lag[0, :, spatial_point].detach().cpu().numpy()]
        
        # Add measurement data
        for j in range(dataset.n_sources):
            valid_mask = ~torch.isnan(measurements[0, j, :, spatial_point])
            if valid_mask.any():
                y_data.append(measurements[0, j, valid_mask, spatial_point].detach().cpu().numpy())
        
        y_data = np.concatenate([d for d in y_data if len(d) > 0])
        y_range = np.ptp(y_data)
        y_mean = np.mean(y_data)
        ax.set_ylim([y_mean - 1.5*y_range/2, y_mean + 1.5*y_range/2])
        
        # Add minor grid
        ax.grid(True, which='minor', alpha=0.15)
        ax.grid(True, which='major', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Add main title with padding
    fig.suptitle(f'Solution Evolution at Different Spatial Points (Epoch {epoch})', 
                 y=1.02, fontsize=14, fontweight='bold')
    
    # Adjust spacing for legends
    plt.subplots_adjust(right=0.85)
    
    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/spatial_evolution_epoch_{epoch}.png',
                bbox_inches='tight', dpi=300, pad_inches=0.2)
    plt.close()


def main():
    """Main training script"""
    # Parameters
    n_samples = 2048 # multiple of batch size
    input_dim = 64
    n_sources = 3
    batch_size = 64
    timesteps = 100
    n_epochs = 500
    save_dir = 'results'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create datasets
    train_dataset = MultiSourceNavierStokes1DDataset(
        n_samples=n_samples,
        input_dim=input_dim,
        n_sources=n_sources,
        n_timesteps = timesteps,
        noise_levels=[0.02, 0.02, 0.03, 0.04, 0.1],  # Reduced noise levels
        bias_levels=[0.0, 0.01, -0.02, 0.03, 0.05]    # Reduced bias levels
    )
    
    val_dataset = MultiSourceNavierStokes1DDataset(
        n_samples=n_samples//8,
        input_dim=input_dim,
        n_sources=n_sources,
        n_timesteps=timesteps,
        noise_levels=[0.02, 0.02, 0.03, 0.04, 0.1],  # Reduced noise levels
        bias_levels=[0.0, 0.01, -0.02, 0.03, 0.05]    # Reduced bias levels
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          num_workers=4)
    
    # Create models
    softmax_model = SoftmaxMultiSourceIntegration(
        n_sources=n_sources,
        input_dim=input_dim,
        hidden_dim=128,
        n_prediction_steps=batch_size   )
    
    lagrangian_model = TwoTimeScaleLagrangianOptimizer(
        n_sources=n_sources,
        input_dim=input_dim,
        hidden_dim=128,
        n_prediction_steps=batch_size,
    )
    
    # Create trainer
    trainer = ComparativeTrainer(
        softmax_model=softmax_model,
        lagrangian_model=lagrangian_model,
        lr_softmax=1e-4,
        lr_theta=1e-3,
        lr_lambda=1e-5,
        device=device
    )
    
    # Training loop
    best_val_loss = {'softmax': float('inf'), 'lagrangian': float('inf')}
    
    try:
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            
            # Train
            train_metrics = trainer.train_epoch(train_loader)
            
            # Validate
            val_metrics = trainer.validate(val_loader)
            
            # Print metrics
            print("\nSoftmax Model:")
            for k, v in train_metrics['softmax'].items():
                print(f"Train {k}: {v:.7f}")
            print(f"Val loss: {val_metrics['softmax']['loss']:.7f}")
            
            print("\nLagrangian Model:")
            for k, v in train_metrics['lagrangian'].items():
                print(f"Train {k}: {v:.7f}")
            print(f"Val loss: {val_metrics['lagrangian']['loss']:.7f}")
            
            # Store metrics
            for model_type in ['softmax', 'lagrangian']:
                for k, v in train_metrics[model_type].items():
                    trainer.metrics[model_type][k].append(v)
            
            # Update learning rates
            for scheduler_name, scheduler in trainer.schedulers.items():
                if scheduler_name == 'softmax':
                    scheduler.step(val_metrics['softmax']['loss'])
                else:
                    scheduler.step(val_metrics['lagrangian']['loss'])
            
            # Plot comparison every 10 epochs
            if (epoch + 1) % 5 == 0:
                plot_comparative_results(
                    trainer,
                    val_dataset,
                    epoch + 1,
                    save_dir
                )
                plot_prediction_comparison(
                    trainer,
                    val_dataset,
                    epoch + 1,
                    save_dir
                )
            
            # Save best models
            for model_type in ['softmax', 'lagrangian']:
                if val_metrics[model_type]['loss'] < best_val_loss[model_type]:
                    best_val_loss[model_type] = val_metrics[model_type]['loss']
                    torch.save({
                        'model_state': (
                            trainer.softmax_model.state_dict()
                            if model_type == 'softmax'
                            else trainer.lagrangian_model.state_dict()
                        ),
                        'val_loss': val_metrics[model_type]['loss'],
                        'epoch': epoch
                    }, f'{save_dir}/best_{model_type}_model.pth')
    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final models...")
    
    finally:
        # Save final models and metrics
        torch.save({
            'softmax_state': trainer.softmax_model.state_dict(),
            'lagrangian_state': trainer.lagrangian_model.state_dict(),
            'metrics': trainer.metrics
        }, f'{save_dir}/final_models.pth')
        
        # Plot final comparison
        plot_comparative_results(trainer, val_dataset, n_epochs, save_dir)
        print("\nTraining completed. Final models and plots saved.")



def analyze_source_reliability(
    trainer: ComparativeTrainer,
    dataset: MultiSourceNavierStokes1DDataset,
    num_samples: int = 100
):
    """Analyze the reliability of different sources"""
    trainer.softmax_model.eval()
    trainer.lagrangian_model.eval()
    
    source_errors = defaultdict(list)
    source_weights = defaultdict(list)
    
    with torch.no_grad():
        for _ in range(num_samples):
            idx = np.random.randint(len(dataset))
            sample = dataset[idx]
            x = sample['x'].to(trainer.device).unsqueeze(0)
            measurements = sample['measurements'].to(trainer.device).unsqueeze(0)
            true_solution = sample['true_solution'].to(trainer.device)
            
            # Get model outputs
            _, meta_soft = trainer.softmax_model(x, measurements)
            _, meta_lag = trainer.lagrangian_model(x, measurements)
            
            # Analyze each source
            for i in range(dataset.n_sources):
                valid_mask = ~torch.isnan(measurements[0, i])
                if valid_mask.any():
                    # Compute errors
                    source_data = measurements[0, i][valid_mask]
                    true_data = true_solution[valid_mask]
                    error = F.mse_loss(source_data, true_data)
                    
                    source_errors[f'source_{i}'].append(error.item())
                    
                    # Store weights
                    source_weights[f'softmax_{i}'].append(
                        meta_soft['weights'][i].item()
                    )
                    source_weights[f'lagrangian_{i}'].append(
                        meta_lag['weights'][i].item()
                    )
    
    # Plot analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Error distribution
    error_means = [np.mean(errors) for errors in source_errors.values()]
    error_stds = [np.std(errors) for errors in source_errors.values()]
    
    ax1.bar(range(len(source_errors)),
            error_means,
            yerr=error_stds,
            alpha=0.7)
    ax1.set_title('Source Error Distribution')
    ax1.set_xlabel('Source Index')
    ax1.set_ylabel('MSE')
    
    # Weight distribution
    positions = np.arange(dataset.n_sources)
    width = 0.35
    
    softmax_means = [np.mean(source_weights[f'softmax_{i}'])
                    for i in range(dataset.n_sources)]
    lagrangian_means = [np.mean(source_weights[f'lagrangian_{i}'])
                       for i in range(dataset.n_sources)]
    
    ax2.bar(positions - width/2, softmax_means,
            width, label='Softmax', alpha=0.7)
    ax2.bar(positions + width/2, lagrangian_means,
            width, label='Lagrangian', alpha=0.7)
    
    ax2.set_title('Average Source Weights')
    ax2.set_xlabel('Source Index')
    ax2.set_ylabel('Weight')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('source_analysis.png')
    plt.close()
    plt.clf()
    gc.collect()
    
    return source_errors, source_weights

if __name__ == "__main__":
    main()



