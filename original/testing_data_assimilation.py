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
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, input_dim)
        )
                
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.prediction_net(x), None

class MultiSourceNavierStokes1DDataset(Dataset):
    def __init__(self, n_samples: int, input_dim: int, n_sources: int, n_timesteps: int = 50, 
                 noise_levels: Optional[List[float]] = None, bias_levels: Optional[List[float]] = None, 
                 seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.n_sources = n_sources
        self.n_timesteps = n_timesteps
        
        # Physical parameters
        self.dt = 0.001
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
        
        # Generate samples
        self._generate_samples()
        
        # Convert to tensors
        self.states = torch.stack(self.states)
        self.solutions = torch.stack(self.solutions)
        self.measurements = torch.stack(self.measurements)
        
        # Verify stability
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
                    phase_shift = 2 * np.pi * i / samples_per_regime
                    initial_state = self.generate_initial_condition(regime, phase_shift)
                    
                    if torch.isnan(initial_state).any() or torch.isinf(initial_state).any():
                        raise ValueError("Unstable initial condition detected")
                    
                    self.states.append(initial_state)
                    
                    solution_sequence = self.solve_navier_stokes_sequence(initial_state)
                    if torch.isnan(solution_sequence).any() or torch.isinf(solution_sequence).any():
                        raise ValueError("Unstable solution detected")
                    
                    self.solutions.append(solution_sequence)
                    
                    source_measurements = self.generate_source_measurements(solution_sequence)
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
    
    def solve_navier_stokes_sequence(self, u0: torch.Tensor, n_steps: Optional[int] = None) -> torch.Tensor:
        if n_steps is None:
            n_steps = self.n_timesteps
            
        solutions = [u0]
        u = u0.clone()
        
        try:
            for t in range(n_steps):
                if not self.check_cfl_condition(u):
                    raise ValueError("CFL condition violated")
                
                du_dx = self.compute_stable_derivative(u, self.dx)
                d2u_dx2 = self.compute_stable_second_derivative(u, self.dx)
                
                du_dt = -u * du_dx + self.nu * d2u_dx2
                du_dt = torch.clamp(du_dt, -self.max_velocity, self.max_velocity)
                
                u = u + self.dt * du_dt
                
                if torch.max(torch.abs(u)) > self.config['filter_threshold']:
                    u = self.spatial_filter(u)
                
                u = self.apply_boundary_conditions(u)
                solutions.append(u.clone())
                
                if self.check_instability(u):
                    raise ValueError(f"Solution became unstable at step {t}")
                
        except Exception as e:
            print(f"Error in solution: {str(e)}")
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
        original_shape = field.shape
        if field.dim() == 1:
            field = field.unsqueeze(0).unsqueeze(0)
        elif field.dim() == 2:
            field = field.unsqueeze(1)
        
        kernel = torch.ones(1, 1, kernel_size, device=field.device) / kernel_size
        padding = kernel_size // 2
        field_padded = torch.nn.functional.pad(field, (padding, padding), mode='circular')
        filtered = torch.nn.functional.conv1d(field_padded, kernel)
        
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
        measurements = torch.where(torch.isinf(measurements), 
                                 torch.tensor(float('nan')), measurements)
        
        measurements = torch.clamp(measurements, 
                                 -self.config['measurement_max'], 
                                 self.config['measurement_max'])
        
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
            else:
                u0 = self._generate_turbulent_initial_condition(x, phase_shift_tensor)
            
            u0 = self._normalize_and_stabilize(u0)
            return u0
            
        except Exception as e:
            print(f"Error generating initial condition for {regime}: {str(e)}")
            return torch.sin(np.pi * x)

    def _generate_smooth_initial_condition(self, x: torch.Tensor, phase_shift: torch.Tensor) -> torch.Tensor:
        u0 = (torch.sin(2 * np.pi * x + phase_shift) + 
              0.5 * torch.sin(4 * np.pi * x + 2*phase_shift) +
              0.25 * torch.sin(6 * np.pi * x + 3*phase_shift))
        
        gaussian = torch.exp(-10 * (x - 0.3 * torch.sin(phase_shift))**2)
        u0 = u0 + 0.2 * gaussian
        
        return u0

    def _generate_shock_initial_condition(self, x: torch.Tensor, phase_shift: torch.Tensor) -> torch.Tensor:
        shift1 = 0.4 * torch.sin(phase_shift)
        shift2 = 0.4 * torch.sin(phase_shift + torch.tensor(np.pi/3))
        
        u0 = torch.zeros_like(x)
        u0[x < shift1] = 0.8
        u0[x >= shift1] = -0.4
        u0[x < shift2] = 0.5
        
        u0 = F.conv1d(
            u0.view(1, 1, -1),
            torch.ones(1, 1, 3) / 3,
            padding=1
        ).view(-1)
        
        u0 = u0 + 0.05 * torch.sin(8 * np.pi * x)
        
        return u0

    def _generate_boundary_initial_condition(self, x: torch.Tensor, phase_shift: torch.Tensor) -> torch.Tensor:
        shift1 = 0.2 * torch.sin(phase_shift)
        shift2 = 0.2 * torch.sin(phase_shift + torch.tensor(np.pi/2))
        
        u0 = (torch.exp(-10 * (x + 0.6 + shift1)**2) + 
              torch.exp(-15 * (x - 0.6 + shift2)**2) +
              0.3 * torch.exp(-20 * (x + shift1)**2))
        
        u0 = u0 + 0.1 * torch.sin(4 * np.pi * x + phase_shift)
        
        return u0

    def _generate_turbulent_initial_condition(self, x: torch.Tensor, phase_shift: torch.Tensor) -> torch.Tensor:
        u0 = torch.zeros_like(x)
        
        for k in range(1, 8):
            phase = phase_shift + torch.tensor(2 * np.pi * k / 5)
            amplitude = 0.7 / (k**0.5)
            u0 += amplitude * torch.sin(k * np.pi * x + phase)
        
        for _ in range(3):
            center = 0.5 * torch.sin(phase_shift + torch.tensor(np.random.rand() * np.pi))
            width = 0.1 + 0.05 * np.random.rand()
            amplitude = 0.1 + 1e-4 * np.random.rand()
            u0 += amplitude * torch.exp(-(x - center)**2 / width**2)
        
        u0 += 0.02 * torch.randn_like(x)
        
        return u0

    def _normalize_and_stabilize(self, u0: torch.Tensor) -> torch.Tensor:
        u0 = torch.where(torch.isnan(u0) | torch.isinf(u0), 
                        torch.zeros_like(u0), u0)
        
        u0 = u0 / (torch.max(torch.abs(u0)) + self.eps)
        
        if torch.max(torch.abs(u0)) > self.config['filter_threshold']:
            u0 = self.spatial_filter(u0)
        
        return u0

    def generate_source_measurements(self, solution: torch.Tensor) -> torch.Tensor:
        measurements = []
        
        for source_idx in range(self.n_sources):
            try:
                biased = solution + self.bias_levels[source_idx]
                noisy = biased + self.noise_levels[source_idx] * torch.randn_like(solution)
                
                if source_idx % 3 == 0:
                    filtered = torch.zeros_like(noisy)
                    for t in range(noisy.shape[0]):
                        filtered[t] = self.spatial_filter(noisy[t], kernel_size=5)
                    noisy = filtered
                
                noisy = self._add_missing_data(noisy, source_idx)
                measurements.append(noisy)
                
            except Exception as e:
                print(f"Error generating measurements for source {source_idx}: {str(e)}")
                measurements.append(solution.clone())

        measurements = torch.stack(measurements)
        return self.process_measurements(measurements)

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
        
        # Single prediction network
        self.prediction_module = MultiStepPredictionModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_steps=n_prediction_steps,
            dt=dt,
            dx=self.dx,
            nu=self.viscosity
        )

    def compute_physics_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-informed loss based on Burgers' equation
        predictions: [batch, timesteps, spatial_points]
        """
        total_residual = 0.0
        L = 2
        
        for t in range(predictions.shape[1] - 1):
            u = predictions[:, t]
            U_scale = torch.max(torch.abs(u))
            nu = (U_scale * 2.0) / self.Re

            u_next = predictions[:, t + 1]
            
            u_prev = torch.roll(u, shifts=1, dims=-1)
            u_next_space = torch.roll(u, shifts=-1, dims=-1)
            
            du_dx = (u_next_space - u_prev) / (2 * self.dx)
            d2u_dx2 = (u_next_space - 2*u + u_prev) / (self.dx**2)
            du_dt = (u_next - u) / self.dt
            
            with torch.no_grad():
                shock_mask = (torch.abs(du_dx) > 0.5*U_scale/L)
                shock_factor = torch.clamp(torch.abs(du_dx)/(U_scale/L + 1e-6), 0, 3)
            
            effective_viscosity = nu * (1 + shock_factor)
            residual = du_dt + u*du_dx - effective_viscosity*d2u_dx2
            
            cfl = (torch.abs(u) * self.dt / self.dx).max(dim=-1, keepdim=True).values
            weight = torch.exp(-5.0 * cfl)
            
            total_residual += (weight * residual.pow(2)).mean()
        
        physics_loss = total_residual / (predictions.shape[1] - 1 + 1e-5)
        
        # Add conservation law
        mass_initial = torch.trapz(predictions[:, 0], dx=self.dx, dim=-1)
        mass_final = torch.trapz(predictions[:, -1], dx=self.dx, dim=-1)
        mass_conservation = (mass_final - mass_initial).pow(2).mean()
        
        total_loss = physics_loss + 0.1 * mass_conservation
        
        return total_loss

    def compute_measurement_loss(
        self, 
        predictions: torch.Tensor, 
        analysis_state: torch.Tensor,  # CHANGED: target is now analysis state
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss between predictions and ANALYSIS STATE (fused measurements)
        predictions: [batch, timesteps, spatial_points]
        analysis_state: [batch, timesteps, spatial_points] (target)
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


class SoftmaxMultiSourceIntegration(EnhancedMultiSourceBase):
    def __init__(self, *args, temperature: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.lambda_weights = nn.Parameter(torch.randn(self.n_sources) * 0.01 + 1.0/self.n_sources)
        
    def get_weights(self):
        return F.softmax(self.lambda_weights / self.temperature, dim=0)
        
    def forward(self, x: torch.Tensor, measurements: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Pure data assimilation forward pass"""
        # Get analysis state from measurements
        analysis_state, meta = self.get_analysis_state(measurements)
        
        # Start autoregressive prediction from initial analysis state
        predictions_list = [analysis_state[:, 0]]
        
        # Predict future timesteps
        for t in range(1, analysis_state.shape[1]):
            current_state = predictions_list[-1]
            next_prediction, _ = self.prediction_module(current_state)
            predictions_list.append(next_prediction)
        
        predictions = torch.stack(predictions_list, dim=1)  # [batch, timesteps, spatial]
        return predictions, meta

    def get_analysis_state(self, measurements: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Compute analysis state for each timestep"""
        weights = self.get_weights()
        
        # measurements: [batch, n_sources, timesteps, spatial]
        # weights: [n_sources] -> reshape to [1, n_sources, 1, 1]
        weights_reshaped = weights.view(1, -1, 1, 1).expand_as(measurements)
        
        # Weighted sum over sources: [batch, timesteps, spatial]
        weighted_measurements = measurements * weights_reshaped
        analysis_states = weighted_measurements.sum(dim=1)
        
        # IMPORTANT: Process NaN values after fusion
        analysis_states = self.process_fused_measurements(analysis_states)
        
        return analysis_states, weights, {
            'analysis_state': analysis_states,
            'weights': weights,
        }
    def process_fused_measurements(self, analysis_states: torch.Tensor) -> torch.Tensor:
        """Interpolate NaN values in the fused analysis state"""
        batch_size, timesteps, spatial = analysis_states.shape
        
        for b in range(batch_size):
            for t in range(timesteps):
                # Get single timestep
                state_t = analysis_states[b, t]
                
                # Find valid measurements
                valid_mask = ~torch.isnan(state_t)
                
                # If we have some valid measurements, interpolate
                if valid_mask.any() and not valid_mask.all():
                    # Use scipy or simple interpolation
                    from scipy.interpolate import interp1d
                    
                    valid_x = torch.arange(spatial)[valid_mask].cpu().numpy()
                    valid_y = state_t[valid_mask].cpu().numpy()
                    
                    if len(valid_x) > 1:
                        # Interpolate
                        f = interp1d(valid_x, valid_y, kind='linear', fill_value='extrapolate')
                        all_x = torch.arange(spatial).cpu().numpy()
                        interpolated = f(all_x)
                        analysis_states[b, t] = torch.from_numpy(interpolated).to(state_t.device)
                    elif len(valid_x) == 1:
                        # Fill with constant
                        analysis_states[b, t] = state_t[valid_mask].item()
                elif not valid_mask.any():
                    # No valid measurements at all, set to zero
                    analysis_states[b, t] = 0.0
        
        return analysis_states
    

    def forward(self, x: torch.Tensor, measurements: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Recursive data assimilation and prediction
        1. Compute analysis state at all timesteps from measurements
        2. Use analysis_state[:, 0] as initial condition
        3. Predict forward recursively
        """
        # Get analysis state for ALL timesteps (target for predictions)
        analysis_state, weights, meta = self.get_analysis_state(measurements)
        
        # Start autoregressive prediction from initial analysis state
        predictions_list = [analysis_state[:, 0]]
        
        # Predict future timesteps
        for t in range(1, analysis_state.shape[1]):
            current_state = predictions_list[-1]
            next_prediction, _ = self.prediction_module(current_state)
            predictions_list.append(next_prediction)
        
        predictions = torch.stack(predictions_list, dim=1)  # [batch, timesteps, spatial]
        
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
        analysis_state = meta['analysis_state']  # Extract fused analysis state
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

class SingleTimeScaleLagrangianOptimizer(EnhancedMultiSourceBase):
    """Single time scale Lagrangian optimizer - updates all parameters simultaneously"""
    
    def __init__(
        self,
        n_sources: int,
        input_dim: int,
        hidden_dim: int = 128,
        n_prediction_steps: int = 5,
        dt: float = 0.001,
        rho: float = 0.005,
    ):
        super().__init__(
            n_sources=n_sources,
            input_dim=input_dim,
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
        """Compute weights using Lagrangian formulation with non-negativity constraint via ReLU."""
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
        weights_reshaped = weights.view(1, -1, 1, 1).expand_as(measurements)
        weighted_measurements = measurements * weights_reshaped
        analysis_states = weighted_measurements.sum(dim=1)
        
        return analysis_states, {
            'analysis_state': analysis_states,
            'weights': weights,
        }

    def compute_constraint_losses(self, weights: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute Lagrangian constraint losses"""
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
        analysis_state = meta['analysis_state']  # Extract fused analysis state
        weights = meta['weights']
        meas_loss = self.compute_measurement_loss(predictions, analysis_state, weights)

        # Physics loss
        physics_loss = self.compute_physics_loss(predictions) if is_training else torch.tensor(0.0)
        
        # Constraint losses (using normalized weights)
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

class TwoTimeScaleLagrangianOptimizer(EnhancedMultiSourceBase):
    def __init__(
        self,
        n_sources: int,
        input_dim: int,
        hidden_dim: int = 128,
        n_prediction_steps: int = 5,
        dt: float = 0.001,
        rho: float = 0.1,  # Increased from 0.005
        multiplier_lr: float = 0.1,  # Increased from 0.01
        multiplier_update_frequency: int = 1,  # More frequent updates
        constraint_weight: float = 0.1  # New: weight for constraint loss
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
        self.constraint_weight = constraint_weight
        self.multiplier_update_counter = 0
        
        # Better initialization for Lagrangian parameters
        self.lambda_weights = nn.Parameter(torch.ones(n_sources) * 1.0/n_sources)
        self.mu = nn.Parameter(torch.zeros(1))
        self.nu = nn.Parameter(torch.zeros(n_sources))
        
        # Add momentum for multiplier updates
        self.register_buffer('mu_momentum', torch.zeros(1))
        self.register_buffer('nu_momentum', torch.zeros(n_sources))
        
    def compute_lagrangian_weights(self) -> torch.Tensor:
        """Compute weights with better numerical stability"""
        weights = F.softplus(self.lambda_weights)  # Use softplus instead of ReLU
        sum_weights = weights.sum()
        # Add entropy regularization for better exploration
        entropy = -torch.sum(weights * torch.log(weights + 1e-10))
        return weights / (sum_weights + 1e-10), entropy

    def compute_constraint_losses(self, weights: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute improved constraint losses with better gradient flow"""
        # Equality constraint: sum(weights) = 1
        g = weights.sum() - 1.0
        
        # Inequality constraints: weights >= 0 (already satisfied by softplus)
        # But we still penalize near-zero weights to avoid collapse
        h = torch.relu(0.01 - weights)  # Penalize weights below 0.01
        
        # Augmented Lagrangian terms
        equality_term = self.mu * g + (self.rho/2) * g.pow(2)
        inequality_term = (self.nu * h + (self.rho/2) * h.pow(2)).sum()
        
        constraint_loss = equality_term + inequality_term
        
        # Add barrier term for strict feasibility
        barrier_term = -0.01 * torch.log(weights + 1e-10).sum()
        
        return constraint_loss + barrier_term, {
            'equality_violation': torch.abs(g).item(),
            'inequality_violation': h.sum().item(),
            'constraint_loss': constraint_loss.item(),
            'barrier_term': barrier_term.item()
        }

    def update_multipliers(self, measurements: torch.Tensor):
        """Improved multiplier update with momentum and adaptive steps"""
        if self.multiplier_update_counter % self.multiplier_update_frequency == 0:
            with torch.no_grad():
                weights, _ = self.compute_lagrangian_weights()
                
                # Compute constraint violations
                g = weights.sum() - 1.0
                h = torch.relu(0.01 - weights)
                
                # Update with momentum
                mu_update = self.multiplier_lr * self.rho * g
                self.mu_momentum = 0.9 * self.mu_momentum + mu_update
                self.mu.data += self.mu_momentum
                
                # Update inequality multipliers with clipping
                nu_update = self.multiplier_lr * self.rho * h
                self.nu_momentum = 0.9 * self.nu_momentum + nu_update
                self.nu.data = torch.clamp(self.nu.data + self.nu_momentum, min=0, max=10.0)
                
                # Adaptive rho adjustment
                if torch.abs(g) > 0.1:  # Large constraint violation
                    self.rho = min(self.rho * 1.1, 1.0)
                elif torch.abs(g) < 0.01:  # Small constraint violation
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
        analysis_state = meta['analysis_state']  # Extract fused analysis state
        weights, entropy = self.compute_lagrangian_weights()
        meas_loss = self.compute_measurement_loss(predictions, analysis_state, weights)
        # Physics loss
        physics_loss = self.compute_physics_loss(predictions) if is_training else torch.tensor(0.0)
        
        # Constraint losses
        if is_training:
            constraint_loss, constraint_dict = self.compute_constraint_losses(weights)
            # Add entropy regularization to prevent weight collapse
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
        weights_reshaped = weights.view(1, -1, 1, 1).expand_as(measurements)
        weighted_measurements = measurements * weights_reshaped
        analysis_states = weighted_measurements.sum(dim=1)
        
        return analysis_states, {
            'analysis_state': analysis_states,
            'weights': weights,
            'entropy': entropy.item()
        }
    

class ADMMOptimizer(EnhancedMultiSourceBase):
    """ADMM-based optimizer for multi-source integration"""
    
    def __init__(
        self,
        n_sources: int,
        input_dim: int,
        hidden_dim: int = 128,
        n_prediction_steps: int = 5,
        dt: float = 0.001,
        rho: float = 0.1,
        admm_iterations: int = 3,
    ):
        super().__init__(
            n_sources=n_sources,
            input_dim=input_dim,
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
        """ADMM uses z as the constraint-satisfying variable"""
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
        weights_reshaped = weights.view(1, -1, 1, 1).expand_as(measurements)
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
        """Perform ADMM updates for lambda, z, and u"""
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
        analysis_state = meta['analysis_state']  # Extract fused analysis state
        weights = meta['weights']
        meas_loss = self.compute_measurement_loss(predictions, analysis_state, weights) 
        # Physics loss
        physics_loss = self.compute_physics_loss(predictions) if is_training else torch.tensor(0.0)
        
        # ADMM augmented Lagrangian term
        # Compute on raw variables, not normalized weights
        residual_term = (self.rho/2) * torch.norm(self.lambda_weights - self.z + self.u_dual).pow(2)
        
        total_loss = meas_loss + 0.01 * physics_loss + 0.001 * residual_term
        
        # ADMM updates happen AFTER backward pass in trainer
        admm_info = {}
        
        return total_loss, {
            'loss': total_loss.item(),
            'meas_loss': meas_loss.item(),
            'physics_loss': physics_loss.item() if is_training else 0.0,
            'weights': weights,
            'admm_residual': residual_term.item(),
            **admm_info
        }

class FourWayComparativeTrainer:
    """Trainer for comparing Softmax, Single-Scale Lagrangian, Two-Scale Lagrangian, and ADMM"""
    
    def __init__(
        self,
        models: Dict[str, EnhancedMultiSourceBase],
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
        """Single training step for all models"""
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
        """Train all models for one epoch"""
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
        """Validate all models using measurement consistency"""
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
                    analysis_state = meta['analysis_state']  # Extract fused analysis state
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


def plot_comparative_results(
    trainer: FourWayComparativeTrainer,
    dataset: MultiSourceNavierStokes1DDataset,
    epoch: int,
    save_dir: str = 'results'
):
    """Plot detailed comparison between all four approaches"""
    plt.close('all')
    
    fig = plt.figure(figsize=(20, 16))
    gs = plt.GridSpec(3, 3, figure=fig)
    
    # Training Loss (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    for name, metrics in trainer.metrics.items():
        if 'loss' in metrics:
            ax1.plot(metrics['loss'], 
                    label=trainer.method_labels[name], 
                    color=trainer.method_colors[name], 
                    alpha=0.7)
    ax1.set_title('Training Loss', fontsize=20)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Weight Distribution (Top Middle)
    ax2 = fig.add_subplot(gs[0, 1])
    sample_idx = np.random.choice(len(dataset))
    sample = dataset[sample_idx]
    x = sample['x'].to(trainer.device).unsqueeze(0)
    measurements = sample['measurements'].to(trainer.device).unsqueeze(0)
    
    with torch.no_grad():
        weights_data = {}
        for name, model in trainer.models.items():
            _, meta = model(x, measurements)  # Fixed: removed double unsqueeze
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
                alpha=0.7)
    
    ax2.set_title('Weight Distribution', fontsize=20)
    ax2.set_xlabel('Source Index')
    ax2.set_ylabel('Weight Value')
    ax2.legend()
    ax2.grid(True)
    
    # Physics Loss (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    for name, metrics in trainer.metrics.items():
        if 'physics_loss' in metrics:
            ax3.plot(metrics['physics_loss'], 
                    label=trainer.method_labels[name], 
                    color=trainer.method_colors[name], 
                    alpha=0.7)
    ax3.set_title('Physics Loss', fontsize=20)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True)
    
    # Measurement Loss (Middle Row Left)
    ax4 = fig.add_subplot(gs[1, 0])
    for name, metrics in trainer.metrics.items():
        if 'meas_loss' in metrics:
            ax4.plot(metrics['meas_loss'], 
                    label=trainer.method_labels[name], 
                    color=trainer.method_colors[name], 
                    alpha=0.7)
    ax4.set_title('Measurement Consistency Loss', fontsize=20)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True)
    
    # Weight Evolution (Middle Row Middle)
    ax5 = fig.add_subplot(gs[1, 1])
    for name, metrics in trainer.metrics.items():
        if 'weights_mean' in metrics:
            ax5.plot(metrics['weights_mean'], 
                    label=trainer.method_labels[name], 
                    color=trainer.method_colors[name], alpha=0.7)
    ax5.set_title('Weight Evolution', fontsize=20)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Mean Weight')
    ax5.legend()
    ax5.grid(True)
    
    # Constraint Violations (Middle Row Right) - NEW
    ax6 = fig.add_subplot(gs[1, 2])
    constraint_models = ['lagrangian_single_scale', 'lagrangian_two_scale', 'admm']
    for name in constraint_models:
        if name in trainer.metrics:
            if 'constraint_loss' in trainer.metrics[name]:
                ax6.plot(trainer.metrics[name]['constraint_loss'], 
                        label=f'{trainer.method_labels[name]} Constraint', 
                        color=trainer.method_colors[name], alpha=0.7)
            elif 'admm_residual' in trainer.metrics[name]:
                ax6.plot(trainer.metrics[name]['admm_residual'], 
                        label=f'{trainer.method_labels[name]} Residual', 
                        color=trainer.method_colors[name], alpha=0.7)
    
    ax6.set_title('Constraint Violations', fontsize=20)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Violation')
    ax6.legend()
    ax6.grid(True)
    
    # Spatial Evolution Plot (Bottom Row)
    ax7 = fig.add_subplot(gs[2, :])
    true_solution = sample['true_solution'].to(trainer.device)
    measurements_plot = sample['measurements'].to(trainer.device)
    
    with torch.no_grad():
        spatial_point = x.shape[-1] // 2  # This is 32 for input_dim=64
        time_steps = np.arange(true_solution.shape[0])
        ax7.plot(time_steps, 
                true_solution[:, spatial_point].cpu().numpy(),
                'k-', label='True Solution', linewidth=2, alpha=0.8)
        
        # Plot raw measurements
        source_colors = plt.cm.Set3(np.linspace(0, 1, dataset.n_sources))
        for j in range(dataset.n_sources):
            valid_mask = ~torch.isnan(measurements_plot[j, :, spatial_point])
            if valid_mask.any():
                ax7.scatter(time_steps[valid_mask.cpu()],
                          measurements_plot[j, valid_mask, spatial_point].cpu().numpy(),
                          color=source_colors[j], alpha=0.3, s=30,
                          label=f'Source {j+1}')
        
        # Plot model predictions
        for name, model in trainer.models.items():
            # Fixed: removed double unsqueeze
            predictions, _ = model(x, measurements)
            ax7.plot(time_steps, 
                    predictions[0, :, spatial_point].cpu().numpy(),
                    '--', color=trainer.method_colors[name],
                    label=trainer.method_labels[name], 
                    alpha=0.8, linewidth=2)
    
    ax7.set_title(f'Prediction vs Measurements at x={spatial_point/dataset.input_dim:.2f}', fontsize=20)
    ax7.set_xlabel('Time Step')
    ax7.set_ylabel('Value')
    ax7.legend()
    ax7.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_epoch_{epoch}.png', 
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    plt.clf()
    gc.collect()

def analyze_source_reliability(
    trainer: FourWayComparativeTrainer,
    dataset: MultiSourceNavierStokes1DDataset,
    num_samples: int = 100,
    save_dir: str = 'results'
):
    """Analyze the reliability of different sources"""
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
                    # Get valid mask for this source [timesteps, spatial_points]
                    valid_mask = ~torch.isnan(measurements[0, i])
                    
                    if valid_mask.any():
                        # Remove batch dimension from predictions for comparison
                        pred_flat = predictions[0]  # Shape: [timesteps, spatial_points]
                        
                        # Apply the same mask to both predictions and measurements
                        pred_masked = pred_flat[valid_mask]
                        source_masked = measurements[0, i][valid_mask]
                        
                        # Calculate error only on valid points
                        error = F.mse_loss(pred_masked, source_masked)
                        source_errors[f'{name}_source_{i}'].append(error.item())
                        source_weights[f'{name}_source_{i}'].append(meta['weights'][i].item())
    
    # Plot analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Error distribution
    for name, model in trainer.models.items():
        error_means = []
        for i in range(dataset.n_sources):
            key = f'{name}_source_{i}'
            if key in source_errors and source_errors[key]:
                error_means.append(np.mean(source_errors[key]))
            else:
                error_means.append(0.0)
        
        positions = np.arange(dataset.n_sources) + list(trainer.models.keys()).index(name) * 0.2 - 0.3
        ax1.bar(positions, error_means, 0.2, label=trainer.method_labels[name], alpha=0.7)
    
    ax1.set_title('Prediction-Measurement Error by Source')
    ax1.set_xlabel('Source Index')
    ax1.set_ylabel('MSE')
    ax1.legend()
    
    # Weight distribution
    for name, model in trainer.models.items():
        weight_means = []
        for i in range(dataset.n_sources):
            key = f'{name}_source_{i}'
            if key in source_weights and source_weights[key]:
                weight_means.append(np.mean(source_weights[key]))
            else:
                weight_means.append(0.0)
        
        positions = np.arange(dataset.n_sources) + list(trainer.models.keys()).index(name) * 0.2 - 0.3
        ax2.bar(positions, weight_means, 0.2, label=trainer.method_labels[name], alpha=0.7)
    
    ax2.set_title('Average Source Weights')
    ax2.set_xlabel('Source Index')
    ax2.set_ylabel('Weight')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/source_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    plt.clf()
    gc.collect()
    
    return source_errors, source_weights


def main():
    """Main training script with four-way comparison"""
    # Parameters
    n_samples = 1024
    input_dim = 64
    n_sources = 3
    batch_size = 64
    timesteps = 5
    n_epochs = 500
    save_dir = 'results'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create datasets
    train_dataset = MultiSourceNavierStokes1DDataset(
        n_samples=n_samples,
        input_dim=input_dim,
        n_sources=n_sources,
        n_timesteps=timesteps,
        noise_levels=[0.02, 0.02, 0.03, 0.04, 0.1],
        bias_levels=[0.0, 0.01, -0.02, 0.03, 0.05]
    )
    
    val_dataset = MultiSourceNavierStokes1DDataset(
        n_samples=n_samples//8,
        input_dim=input_dim,
        n_sources=n_sources,
        n_timesteps=timesteps,
        noise_levels=[0.02, 0.02, 0.03, 0.04, 0.1],
        bias_levels=[0.0, 0.01, -0.02, 0.03, 0.05]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          num_workers=4)
    
    # Create all four models
    models = {
        'softmax': SoftmaxMultiSourceIntegration(
            n_sources=n_sources,
            input_dim=input_dim,
            hidden_dim=512,
            n_prediction_steps=timesteps
        ),
        'lagrangian_single_scale': SingleTimeScaleLagrangianOptimizer(
            n_sources=n_sources,
            input_dim=input_dim,
            hidden_dim=512,
            n_prediction_steps=timesteps,
            rho=0.005
        ),
        'lagrangian_two_scale': TwoTimeScaleLagrangianOptimizer(
            n_sources=n_sources,
            input_dim=input_dim,
            hidden_dim=512,
            n_prediction_steps=timesteps,
            rho=0.1,  # Increased
            multiplier_lr=0.1,  # Increased
            multiplier_update_frequency=1,  # More frequent
            constraint_weight=0.1  # New parameter
        ),
        'admm': ADMMOptimizer(
            n_sources=n_sources,
            input_dim=input_dim,
            hidden_dim=512,
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
    trainer = FourWayComparativeTrainer(
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
            
            # Store metrics
            for name in trainer.metrics.keys():
                for k, v in train_metrics[name].items():
                    trainer.metrics[name][k].append(v)
            
            # Update learning rates
            trainer.update_schedulers(val_metrics)
            
            # Plot comparison every 10 epochs
            if (epoch + 1) % 10 == 0:
                plot_comparative_results(
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
                    }, f'{save_dir}/best_{name}_model.pth')
    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final models...")
    
    finally:
        # Save final models and metrics
        torch.save({
            'model_states': {name: model.state_dict() for name, model in trainer.models.items()},
            'metrics': trainer.metrics,
            'config': {
                'n_samples': n_samples,
                'input_dim': input_dim,
                'n_sources': n_sources,
                'n_epochs': n_epochs
            }
        }, f'{save_dir}/final_models.pth')
        
        # Plot final comparison and analysis
        plot_comparative_results(trainer, val_dataset, n_epochs, save_dir)
        analyze_source_reliability(trainer, val_dataset)
        print("\nTraining completed. Final models and plots saved.")

if __name__ == "__main__":
    main()