import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional, Any
import warnings
from enum import Enum
from collections import defaultdict
import os
import gc
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ==================== Enums and Base Classes ====================

class PhysicsRegime(Enum):
    """Physics regimes for different solution characteristics"""
    SMOOTH = 'smooth'
    SHOCK = 'shock'
    BOUNDARY_LAYER = 'boundary_layer'
    TURBULENT = 'turbulent'

class BaseExpertRoutingModel(nn.Module):
    """Base class for expert routing models"""
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                    physics_params: Dict[str, float] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError

def huber_loss(pred, target, delta=1.0):
    return F.huber_loss(pred, target, reduction='none', delta=delta)

# ==================== Enhanced Burgers Dataset ====================

class EnhancedBurgersDataset(Dataset):
    """Enhanced dataset for 1D Burgers' equation with realistic physics regimes"""
    
    def __init__(self, n_samples: int = 2000, grid_size: int = 256, t_final: float = 0.5,
                 re: float = 1000, noise_std: float = 0.02, seed: int = 42,
                 include_regimes: bool = True):
        super().__init__()
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.t_final = t_final
        self.nu = 1.0 / re
        self.noise_std = noise_std
        self.include_regimes = include_regimes
        
        # Physical grid
        self.x = np.linspace(-1, 1, grid_size)
        self.dx = self.x[1] - self.x[0]
        
        np.random.seed(seed)
        self.data = self._generate_enhanced_data()
        
    def _classify_solution_regime(self, u0: np.ndarray, u_solution: np.ndarray) -> PhysicsRegime:
        """Classify the physics regime based on solution characteristics"""
        # Compute gradients
        u_grad = np.gradient(u_solution, self.dx)
        u_grad_norm = np.abs(u_grad).max()
        
        # Compute smoothness metric
        smoothness = np.mean(np.abs(np.gradient(u_grad, self.dx)))
        
        # Check for shocks (steep gradients)
        if u_grad_norm > 10.0:  # Threshold for shocks
            return PhysicsRegime.SHOCK
        
        # Check for boundary layers (steep gradients near boundaries)
        boundary_grad_norm = np.max(np.abs(u_grad[:10])) + np.max(np.abs(u_grad[-10:]))
        if boundary_grad_norm > 5.0:
            return PhysicsRegime.BOUNDARY_LAYER
        
        # Check for turbulence (high frequency components)
        u_fft = np.abs(np.fft.rfft(u_solution))
        high_freq_energy = np.sum(u_fft[20:]) / np.sum(u_fft)
        if high_freq_energy > 0.3:
            return PhysicsRegime.TURBULENT
        
        # Otherwise smooth
        return PhysicsRegime.SMOOTH
    
    def _generate_initial_condition(self, regime: Optional[PhysicsRegime] = None) -> np.ndarray:
        """Generate initial condition for specific regime"""
        if regime is None:
            regime = np.random.choice(list(PhysicsRegime))
        
        if regime == PhysicsRegime.SMOOTH:
            # Smooth sinusoidal functions
            k = np.random.uniform(1, 2)
            phase = np.random.uniform(0, 2 * np.pi)
            u0 = np.sin(k * np.pi * self.x + phase)
            
        elif regime == PhysicsRegime.SHOCK:
            # Shock-forming initial condition
            shock_position = np.random.uniform(-0.5, 0.5)
            width = np.random.uniform(0.05, 0.1)
            u0 = np.tanh((self.x - shock_position) / width)
            
        elif regime == PhysicsRegime.BOUNDARY_LAYER:
            # Boundary layer initial condition
            boundary_layer_width = np.random.uniform(0.05, 0.15)
            u0 = np.exp(-((self.x + 0.8) / boundary_layer_width)**2) - np.exp(-((self.x - 0.8) / boundary_layer_width)**2)
            
        else:  # TURBULENT
            # Multiple frequency components
            u0 = np.zeros_like(self.x)
            for k in range(1, 6):
                amplitude = np.random.uniform(0.2, 0.5) / k
                phase = np.random.uniform(0, 2 * np.pi)
                u0 += amplitude * np.sin(k * np.pi * self.x + phase)
            
            # Add some noise
            u0 += np.random.normal(0, 0.1, size=self.grid_size)
        
        # Normalize
        u0 = u0 / (np.max(np.abs(u0)) + 1e-8)
        return u0
    
    def _solve_burgers_enhanced(self, u0: np.ndarray) -> np.ndarray:
        """Solve Burgers' equation with spectral method and proper scaling"""
        N = len(u0)
        k = 2 * np.pi * np.fft.fftfreq(N, d=self.dx)
        
        # Time stepping with adaptive CFL
        dt = 0.001
        n_steps = int(self.t_final / dt)
        
        u_hat = np.fft.fft(u0)
        
        for step in range(n_steps):
            # Physical space for nonlinear term
            u = np.real(np.fft.ifft(u_hat))
            
            # Dealias: zero out high frequencies
            u_hat_dealiased = u_hat.copy()
            u_hat_dealiased[N//3:] = 0
            u_hat_dealiased[-N//3:] = 0
            
            # Nonlinear term in Fourier space
            u_sq = 0.5 * np.real(np.fft.ifft(u_hat_dealiased))**2
            u_sq_hat = np.fft.fft(u_sq)
            
            # Burgers equation in Fourier space
            rhs = -1j * k * u_sq_hat - self.nu * k**2 * u_hat
            
            # RK2 time integration
            u_hat_temp = u_hat + 0.5 * dt * rhs
            u_temp = np.real(np.fft.ifft(u_hat_temp))
            
            u_sq_temp = 0.5 * u_temp**2
            u_sq_hat_temp = np.fft.fft(u_sq_temp)
            rhs_temp = -1j * k * u_sq_hat_temp - self.nu * k**2 * u_hat_temp
            
            u_hat = u_hat + dt * rhs_temp
            
            # Stability check
            if np.any(np.isnan(u_hat)) or np.any(np.isinf(u_hat)):
                return u0
        
        u_final = np.real(np.fft.ifft(u_hat))
        return u_final
    
    def _generate_enhanced_data(self) -> List[Dict]:
        """Generate enhanced dataset with regime classification"""
        data = []
        
        for i in range(self.n_samples):
            if self.include_regimes:
                # Sample regime
                regime = np.random.choice(list(PhysicsRegime))
                u0 = self._generate_initial_condition(regime)
            else:
                # Legacy mode
                ic_type = np.random.choice(['sinusoidal', 'gaussian', 'step'])
                
                if ic_type == 'sinusoidal':
                    k = np.random.uniform(1, 3)
                    phase = np.random.uniform(0, 2*np.pi)
                    u0 = np.sin(k * np.pi * self.x + phase)
                    
                elif ic_type == 'gaussian':
                    center = np.random.uniform(-0.5, 0.5)
                    width = np.random.uniform(5, 15)
                    u0 = np.exp(-width * (self.x - center)**2)
                    
                else:  # step
                    step_pos = np.random.uniform(-0.5, 0.5)
                    u0 = np.zeros_like(self.x)
                    u0[self.x > step_pos] = 1.0
                    u0 = ndimage.gaussian_filter1d(u0, sigma=3)
                
                u0 = u0 / (np.max(np.abs(u0)) + 1e-8)
                regime = None
            
            # Solve equation
            u_solution = self._solve_burgers_enhanced(u0)
            
            # Classify regime if not already specified
            if not self.include_regimes or regime is None:
                regime = self._classify_solution_regime(u0, u_solution)
            
            # Add noise
            noise = np.random.normal(0, self.noise_std, size=self.grid_size)
            u_noisy = u_solution + noise
            
            data.append({
                'u0': u0.astype(np.float32),
                'u_solution': u_solution.astype(np.float32),
                'u_noisy': u_noisy.astype(np.float32),
                'regime': regime.value if isinstance(regime, PhysicsRegime) else 'unknown'
            })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'u0': torch.FloatTensor(item['u0']),
            'u_solution': torch.FloatTensor(item['u_solution']),
            'u_noisy': torch.FloatTensor(item['u_noisy']),
            'regime': item['regime']
        }

# ==================== Expert Solvers (Enhanced) ====================

class ExpertSolver(nn.Module):
    """Individual expert solver with enhanced architecture"""
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, 
                 expert_type: str = 'fourier', dropout_rate: float = 0.1):
        super().__init__()
        self.expert_type = expert_type
        self.input_dim = input_dim
        
        if expert_type == 'fourier':
            # Fourier-based expert with residual connections
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )
            
            # Fourier processing layer
            self.fourier_transform = nn.Linear(hidden_dim, hidden_dim)
            
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, input_dim)
            )
            
        elif expert_type == 'spectral':
            # Spectral method expert with 1D convolutions
            self.layers = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=5, padding=2, padding_mode='circular'),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Conv1d(64, 128, kernel_size=5, padding=2, padding_mode='circular'),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Conv1d(128, 64, kernel_size=5, padding=2, padding_mode='circular'),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 1, kernel_size=5, padding=2, padding_mode='circular')
            )
            
        elif expert_type == 'finite_difference':
            # Finite difference expert with residual blocks
            self.initial_conv = nn.Conv1d(1, 64, kernel_size=3, padding=1, padding_mode='circular')
            
            self.res_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(64, 64, kernel_size=3, padding=1, padding_mode='circular'),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Conv1d(64, 64, kernel_size=3, padding=1, padding_mode='circular'),
                    nn.BatchNorm1d(64)
                ) for _ in range(3)
            ])
            
            self.final_conv = nn.Conv1d(64, 1, kernel_size=3, padding=1, padding_mode='circular')
            
        else:  # 'neural'
            # Dense neural network expert
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, input_dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.expert_type == 'fourier':
            # Process through encoder
            encoded = self.encoder(x)
            
            # Apply Fourier-like transformation (learned linear transform)
            freq_domain = self.fourier_transform(encoded)
            freq_domain = torch.fft.ifft(torch.fft.fft(freq_domain, dim=-1).real, dim=-1).real
            
            # Decode
            return self.decoder(freq_domain + encoded)  # Residual connection
            
        elif self.expert_type in ['spectral', 'finite_difference']:
            # Add channel dimension
            x = x.unsqueeze(1)
            
            if self.expert_type == 'finite_difference':
                x = self.initial_conv(x)
                for res_block in self.res_blocks:
                    residual = x
                    x = res_block(x)
                    x = F.relu(x + residual)  # Residual connection
                x = self.final_conv(x)
            else:
                x = self.layers(x)
                
            return x.squeeze(1)
        else:
            return self.layers(x)

# ==================== FNO Baseline (Enhanced) ====================

class EnhancedFNOBaseline(nn.Module):
    """Enhanced Fourier Neural Operator with Physics Loss and corrected LayerNorm"""
    
    def __init__(self, modes: int = 16, width: int = 64, grid_size: int = 256):
        super().__init__()
        self.modes = modes
        self.width = width
        self.grid_size = grid_size
        
        self.fc0 = nn.Sequential(
            nn.Linear(1, width),
            nn.LayerNorm(width),
            nn.ReLU()
        )
        
        self.fourier_layers = nn.ModuleList([EnhancedFourierLayer(width, width, modes=modes) for _ in range(4)])
        self.conv_layers = nn.ModuleList([nn.Conv1d(width, width, kernel_size=1) for _ in range(4)])
        
        # We use GroupNorm(1, width) because it acts like LayerNorm but handles (B, C, L) shapes
        self.norms = nn.ModuleList([nn.GroupNorm(1, width) for _ in range(4)])
        
        self.fc1 = nn.Sequential(
            nn.Linear(width, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)          
        x = self.fc0(x)              
        x = x.permute(0, 2, 1)       
        
        for i in range(4):
            residual = x
            x_fourier = self.fourier_layers[i](x)
            x_conv = self.conv_layers[i](x)
            x = x_fourier + x_conv + residual
            x = F.gelu(self.norms[i](x)) # Norm handles (B, C, L) correctly now
        
        x = x.permute(0, 2, 1)       
        x = self.fc1(x)              
        return x.squeeze(-1)

    def compute_physics_loss(self, predictions: torch.Tensor, initial_conditions: torch.Tensor,
                       nu: float = 0.01, dx: float = 0.0078, t_final: float = 0.5) -> torch.Tensor:
        """Compute physics-constrained loss with matched dimensions"""
        if predictions.dim() == 3: 
            predictions = predictions.squeeze(-1)
        
        # Normalize for stability
        predictions_norm = predictions / (torch.max(torch.abs(predictions), dim=-1, keepdim=True)[0] + 1e-8)
        
        # 1. Spatial Derivatives (Periodic/Circular)
        # Using circular padding keeps the output size at 256
        u_padded = F.pad(predictions_norm, (1, 1), mode='circular')
        u_x = (u_padded[:, 2:] - u_padded[:, :-2]) / (2 * dx)
        
        u_xx_padded = F.pad(u_x, (1, 1), mode='circular')
        u_xx = (u_xx_padded[:, 2:] - u_xx_padded[:, :-2]) / (2 * dx) # Central diff for 2nd deriv
        
        # 2. Temporal Derivative
        u_t = (predictions_norm - initial_conditions) / t_final
        
        # 3. Burgers' Equation Residual: f = u_t + u*u_x - nu*u_xx
        # REMOVED the [:, 1:-1] slicing to match size 256
        residual = u_t + predictions_norm * u_x - nu * u_xx
        
        # Loss components
        physics_loss = torch.mean(residual**2)
        
        # Conservation loss (Mass balance)
        mass_initial = torch.trapz(initial_conditions, dx=dx, dim=-1)
        mass_final = torch.trapz(predictions_norm, dx=dx, dim=-1)
        conservation_loss = torch.mean((mass_final - mass_initial)**2)
        
        return physics_loss + 0.01 * conservation_loss

class EnhancedFourierLayer(nn.Module):
    """Enhanced Fourier layer with better initialization"""
    
    def __init__(self, in_channels: int, out_channels: int, modes: int = 16):
        super().__init__()
        self.modes = modes
        
        # Learnable Fourier weights
        scale = 1 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes)
        )
        self.weights_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        
        # FFT
        x_ft = torch.fft.rfft(x, dim=-1, norm='ortho')
        
        # Multiply relevant Fourier modes
        out_ft_real = torch.zeros(B, self.weights_real.shape[1], x_ft.shape[-1], 
                                 device=x.device, dtype=torch.float32)
        out_ft_imag = torch.zeros(B, self.weights_imag.shape[1], x_ft.shape[-1],
                                 device=x.device, dtype=torch.float32)
        
        modes = min(self.modes, x_ft.shape[-1])
        
        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        x_ft_real = x_ft.real[:, :, :modes]
        x_ft_imag = x_ft.imag[:, :, :modes]
        
        # Real part: ac - bd
        out_ft_real[:, :, :modes] = (
            torch.einsum("bix,iox->box", x_ft_real, self.weights_real[:, :, :modes]) -
            torch.einsum("bix,iox->box", x_ft_imag, self.weights_imag[:, :, :modes])
        )
        
        # Imaginary part: ad + bc
        out_ft_imag[:, :, :modes] = (
            torch.einsum("bix,iox->box", x_ft_real, self.weights_imag[:, :, :modes]) +
            torch.einsum("bix,iox->box", x_ft_imag, self.weights_real[:, :, :modes])
        )
        
        # Combine
        out_ft = torch.complex(out_ft_real, out_ft_imag)
        
        # Inverse FFT
        return torch.fft.irfft(out_ft, n=L, dim=-1, norm='ortho')

# ==================== Base Router (Enhanced) ====================

class BaseRouter(nn.Module):
    """Base class for expert routers with enhanced physics loss"""
    
    def __init__(self, n_experts: int = 4, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.n_experts = n_experts
        self.input_dim = input_dim
        
        # Enhanced router network
        self.router_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_experts)
        )
    
    def compute_physics_loss(self, predictions: torch.Tensor, initial_conditions: torch.Tensor,
                       nu: float = 0.01, dx: float = 0.0078, t_final: float = 0.5) -> torch.Tensor:
        """Compute physics-constrained loss with matched dimensions"""
        if predictions.dim() == 3: 
            predictions = predictions.squeeze(-1)
        
        # Normalize for stability
        predictions_norm = predictions / (torch.max(torch.abs(predictions), dim=-1, keepdim=True)[0] + 1e-8)
        
        # 1. Spatial Derivatives (Periodic/Circular)
        # Using circular padding keeps the output size at 256
        u_padded = F.pad(predictions_norm, (1, 1), mode='circular')
        u_x = (u_padded[:, 2:] - u_padded[:, :-2]) / (2 * dx)
        
        u_xx_padded = F.pad(u_x, (1, 1), mode='circular')
        u_xx = (u_xx_padded[:, 2:] - u_xx_padded[:, :-2]) / (2 * dx) # Central diff for 2nd deriv
        
        # 2. Temporal Derivative
        u_t = (predictions_norm - initial_conditions) / t_final
        
        # 3. Burgers' Equation Residual: f = u_t + u*u_x - nu*u_xx
        # REMOVED the [:, 1:-1] slicing to match size 256
        residual = u_t + predictions_norm * u_x - nu * u_xx
        
        # Loss components
        physics_loss = torch.mean(residual**2)
        
        # Conservation loss (Mass balance)
        mass_initial = torch.trapz(initial_conditions, dx=dx, dim=-1)
        mass_final = torch.trapz(predictions_norm, dx=dx, dim=-1)
        conservation_loss = torch.mean((mass_final - mass_initial)**2)
        
        return physics_loss + 0.01 * conservation_loss

# ==================== Softmax Router (Enhanced) ====================

class SoftmaxRouter(BaseRouter):
    """Softmax-based expert routing with temperature scaling"""
    
    def __init__(self, n_experts: int = 4, input_dim: int = 256, hidden_dim: int = 128,
                 temperature: float = 1.0, entropy_weight: float = 0.5):
        super().__init__(n_experts, input_dim, hidden_dim)
        self.temperature = temperature
        self.entropy_weight = entropy_weight
    
    def forward(self, x: torch.Tensor, experts: List[nn.Module], return_weights: bool = False):
        # Compute logits
        logits = self.router_network(x)
        
        # Temperature-scaled softmax
        weights = F.softmax(logits / self.temperature, dim=-1)
        
        # Compute entropy for regularization
        entropy = -torch.sum(weights * torch.log(weights + 1e-10), dim=-1).mean()
        
        # Get expert outputs
        expert_outputs = []
        for expert in experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Combine with weights
        output = torch.sum(expert_outputs * weights.unsqueeze(-1), dim=1)
        
        if return_weights:
            return output, weights, entropy
        return output

# ==================== Lagrangian Routers (Enhanced) ====================

class LagrangianSingleTimeRouter(BaseRouter):
    """Lagrangian routing with single time scale optimization"""
    
    def __init__(self, n_experts: int = 4, input_dim: int = 256, hidden_dim: int = 128,
                 rho: float = 0.1, constraint_type: str = 'simplex'):
        super().__init__(n_experts, input_dim, hidden_dim)
        self.rho = rho
        self.constraint_type = constraint_type
        
        # Lagrange multipliers
        self.lambda_multipliers = nn.Parameter(torch.zeros(n_experts))
        
        # Adaptive rho
        self.rho_min = 0.01
        self.rho_max = 1.0
        
    def forward(self, x: torch.Tensor, experts: List[nn.Module], return_weights: bool = False):
        logits = self.router_network(x)
        
        if self.constraint_type == 'simplex':
            weights = self._simplex_projection(logits)
        else:
            weights = F.softmax(logits, dim=-1)
        
        # Apply Lagrangian adjustment
        weights = weights + self.lambda_multipliers.unsqueeze(0)
        weights = F.softmax(weights, dim=-1)
        
        # Get expert outputs
        expert_outputs = []
        for expert in experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Combine
        output = torch.sum(expert_outputs * weights.unsqueeze(-1), dim=1)
        
        # Compute constraint violation
        constraint_violation = self._compute_constraints(weights)
        
        if return_weights:
            return output, weights, constraint_violation
        return output
    
    def _simplex_projection(self, logits: torch.Tensor) -> torch.Tensor:
        # 1. Handle potential NaNs in logits before sorting
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
        
        u, _ = torch.sort(logits, descending=True, dim=-1)
        cssv = torch.cumsum(u, dim=-1)
        ind = torch.arange(1, logits.shape[-1] + 1, device=logits.device)
        cond = u - (cssv - 1.0) / ind > 0
        
        # 2. Safety clamp for rho
        rho = torch.sum(cond, dim=-1, keepdim=True)
        rho = torch.clamp(rho, min=1) 
        
        # 3. Use long() for indexing to avoid device-side asserts
        theta = (torch.gather(cssv, -1, (rho - 1).long()) - 1.0) / rho
        return F.relu(logits - theta)
    
    def _compute_constraints(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute constraint violations"""
        # Equality constraint: sum to 1
        sum_constraint = torch.abs(weights.sum(dim=-1) - 1.0).mean()
        
        # Inequality constraints: non-negativity
        nonneg_constraint = F.relu(-weights).mean()
        
        return sum_constraint + nonneg_constraint
    
    def update_multipliers(self, constraint_violation: torch.Tensor):
        """Update Lagrange multipliers with adaptive rho"""
        with torch.no_grad():
            self.lambda_multipliers += self.rho * constraint_violation
            
            # Adaptive rho adjustment
            if constraint_violation > 0.1:
                self.rho = min(self.rho * 1.1, self.rho_max)
            elif constraint_violation < 0.01:
                self.rho = max(self.rho * 0.9, self.rho_min)

class LagrangianTwoTimeRouter(LagrangianSingleTimeRouter):
    """Lagrangian routing with two time scale optimization"""
    
    def __init__(self, n_experts: int = 4, input_dim: int = 256, hidden_dim: int = 128,
                 rho: float = 0.1, fast_lr: float = 0.01, slow_lr: float = 0.001):
        super().__init__(n_experts, input_dim, hidden_dim, rho)
        self.fast_lr = fast_lr
        self.slow_lr = slow_lr
        
        # Regime classifier for adaptive routing
        self.regime_classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 4)  # 4 regimes
        )
        
        # Separate optimizers will be created in trainer
    
    def forward(self, x: torch.Tensor, experts: List[nn.Module], return_weights: bool = False):
        # Get regime classification
        regime_logits = self.regime_classifier(x)
        regime_weights = F.softmax(regime_logits, dim=-1).mean(dim=0)
        
        # Get base logits
        logits = self.router_network(x)
        
        # Regime-adaptive adjustment
        adapted_logits = logits + torch.log(regime_weights.unsqueeze(0) + 1e-10)
        
        # Project onto simplex
        weights = self._simplex_projection(adapted_logits)
        
        # Apply Lagrangian adjustment (fast time scale)
        weights = weights + self.lambda_multipliers.unsqueeze(0) * self.fast_lr
        weights = F.softmax(weights, dim=-1)
        
        # Get expert outputs
        expert_outputs = []
        for expert in experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Combine
        output = torch.sum(expert_outputs * weights.unsqueeze(-1), dim=1)
        
        # Compute constraint violation
        constraint_violation = self._compute_constraints(weights)
        
        if return_weights:
            return output, weights, constraint_violation
        return output

# ==================== ADMM Router (Enhanced) ====================

class ADMMRouter(BaseRouter):
    """ADMM-based expert routing"""
    
    def __init__(self, n_experts: int = 4, input_dim: int = 256, hidden_dim: int = 128,
                 rho: float = 0.1, admm_iterations: int = 3):
        super().__init__(n_experts, input_dim, hidden_dim)
        self.rho = rho
        self.admm_iterations = admm_iterations
        
        # ADMM variables
        self.register_buffer('z', torch.ones(n_experts) / n_experts)
        self.register_buffer('u', torch.zeros(n_experts))
        
    def forward(self, x: torch.Tensor, experts: List[nn.Module], return_weights: bool = False):
        logits = self.router_network(x)
        
        # Perform ADMM updates
        weights = self._admm_update(logits)
        
        # Get expert outputs
        expert_outputs = []
        for expert in experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Combine
        output = torch.sum(expert_outputs * weights.unsqueeze(-1), dim=1)
        
        if return_weights:
            return output, weights, torch.tensor(0.0, device=x.device)
        return output
    
    def _admm_update(self, logits: torch.Tensor) -> torch.Tensor:
        """Perform one ADMM update step"""
        # x-update: softmax of logits
        x = F.softmax(logits, dim=-1)
        
        # z-update: projection onto simplex
        v = x.mean(dim=0) + self.u
        z_new = self._project_simplex(v.unsqueeze(0)).squeeze(0)
        
        # u-update
        u_new = self.u + (x.mean(dim=0) - z_new)
        
        # Update variables
        self.z.data = z_new
        self.u.data = u_new
        
        return x
    
    def _project_simplex(self, v: torch.Tensor) -> torch.Tensor:
        device = v.device
        v_sorted, _ = torch.sort(v, descending=True, dim=-1)
        cssv = torch.cumsum(v_sorted, dim=-1)
        
        rho = torch.arange(1, v.shape[-1] + 1, device=device).float()
        cond = v_sorted - (cssv - 1.0) / rho > 0
        
        # SAFE INDEXING: ensure rho_index is at least 0
        rho_index = torch.sum(cond, dim=-1, keepdim=True) - 1
        rho_index = torch.clamp(rho_index, min=0).long() # Ensure long type and non-negative
        
        theta = (torch.gather(cssv, -1, rho_index) - 1.0) / (rho_index.float() + 1.0)
        
        return F.relu(v - theta)

# ==================== Expert Routing System (Enhanced) ====================
class EnhancedExpertRoutingSystem(nn.Module):
    """Enhanced system for expert routing with improved training"""
    
    def __init__(self, routing_method: str = 'softmax', n_experts: int = 4,
                 input_dim: int = 256, hidden_dim: int = 128, dropout_rate: float = 0.1,
                 physics_weight: float = 1e-3, entropy_weight: float = 0.5,  # Changed weights
                 constraint_weight: float = 0.01, device: str = 'cpu'):  # Changed constraint weight
        super().__init__()
        self.routing_method = routing_method
        self.n_experts = n_experts
        self.input_dim = input_dim
        self.physics_weight = physics_weight
        self.entropy_weight = entropy_weight
        self.constraint_weight = constraint_weight
        self.device = device
        
        # Initialize experts with enhanced architectures
        expert_types = ['fourier', 'spectral', 'finite_difference', 'neural'][:n_experts]
        self.experts = nn.ModuleList([
            ExpertSolver(input_dim, hidden_dim, expert_type=et, dropout_rate=dropout_rate)
            for et in expert_types
        ])
        
        # Initialize router
        if routing_method == 'softmax':
            self.router = SoftmaxRouter(n_experts, input_dim, hidden_dim)
        elif routing_method == 'lagrangian_single':
            self.router = LagrangianSingleTimeRouter(n_experts, input_dim, hidden_dim)
        elif routing_method == 'lagrangian_two':
            self.router = LagrangianTwoTimeRouter(n_experts, input_dim, hidden_dim)
        elif routing_method == 'admm':
            self.router = ADMMRouter(n_experts, input_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown routing method: {routing_method}")
        
        # Move to device
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass returning predictions and metadata"""
        if self.routing_method == 'softmax':
            predictions, weights, entropy = self.router(x, self.experts, return_weights=True)
            metadata = {'weights': weights, 'entropy': entropy}
        elif self.routing_method in ['lagrangian_single', 'lagrangian_two']:
            predictions, weights, constraint_violation = self.router(x, self.experts, return_weights=True)
            metadata = {'weights': weights, 'constraint_violation': constraint_violation}
        elif self.routing_method == 'admm':
            predictions, weights, _ = self.router(x, self.experts, return_weights=True)
            metadata = {'weights': weights}
        else:
            predictions = self.router(x, self.experts)
            metadata = {}
        
        return predictions, metadata
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    initial_conditions: torch.Tensor, metadata: Dict) -> Tuple[torch.Tensor, Dict]:
        """Compute all losses"""
        # Reconstruction loss (Huber for robustness)
        recon_loss = huber_loss(predictions, targets, delta=1.0).mean()
        
        # Physics loss - increased weight
        dx = 2.0 / self.input_dim
        physics_loss = self.router.compute_physics_loss(
            predictions, initial_conditions, nu=0.01, dx=dx, t_final=0.5
        )
        
        # Total loss with balanced weights
        total_loss = recon_loss + self.physics_weight * physics_loss
        
        # Add regularization terms based on routing method
        if self.routing_method == 'softmax' and 'entropy' in metadata:
            # Add entropy regularization (not subtract)
            total_loss = total_loss + self.entropy_weight * metadata['entropy']  # Changed to addition
        elif self.routing_method in ['lagrangian_single', 'lagrangian_two'] and 'constraint_violation' in metadata:
            total_loss = total_loss + self.constraint_weight * metadata['constraint_violation']
        
        # Collect metrics
        metrics = {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'physics_loss': physics_loss.item(),
            'weights_mean': metadata.get('weights', torch.zeros(self.n_experts)).mean().item(),
            'weights_std': metadata.get('weights', torch.zeros(self.n_experts)).std().item()
        }
        
        # Method-specific metrics
        if 'entropy' in metadata:
            metrics['entropy'] = metadata['entropy'].item()
        if 'constraint_violation' in metadata:
            metrics['constraint_violation'] = metadata['constraint_violation'].item()
        
        return total_loss, metrics

# ==================== Weight Tracking Class ====================

class WeightTracker:
    """Track and analyze expert weights over training"""
    
    def __init__(self, n_experts: int = 4, expert_names: List[str] = None):
        self.n_experts = n_experts
        self.expert_names = expert_names or [f'Expert_{i}' for i in range(n_experts)]
        
        # Storage for weight statistics
        self.epoch_weights = []  # List of weight arrays per epoch
        self.weight_means = []   # Mean weights per expert per epoch
        self.weight_stds = []    # Std of weights per expert per epoch
        self.weight_entropies = []  # Entropy of weight distribution per epoch
        
        # Specialization metrics
        self.dominant_expert_counts = []  # Count of times each expert is dominant
        self.weight_sparsity = []  # How sparse are the weight distributions
        
        # Per-regime statistics (if available)
        self.regime_weights = defaultdict(list)
    
    def update(self, weights: torch.Tensor, regimes: Optional[List[str]] = None):
        """Update tracker with new batch of weights"""
        weights_np = weights.detach().cpu().numpy()
        
        # Store epoch weights
        self.epoch_weights.append(weights_np)
        
        # Compute statistics
        batch_mean = weights_np.mean(axis=0)  # Mean per expert across batch
        batch_std = weights_np.std(axis=0)    # Std per expert across batch
        
        self.weight_means.append(batch_mean)
        self.weight_stds.append(batch_std)
        
        # Compute entropy (measure of confidence/specialization)
        # Higher entropy = more uniform = less confident/specialized
        batch_entropy = -np.sum(batch_mean * np.log(batch_mean + 1e-10))
        self.weight_entropies.append(batch_entropy)
        
        # Compute sparsity (percentage of weights below threshold)
        sparsity_threshold = 0.2  # Consider weights below this as "inactive"
        sparsity = np.mean(weights_np < sparsity_threshold)
        self.weight_sparsity.append(sparsity)
        
        # Track dominant expert
        dominant_experts = np.argmax(weights_np, axis=1)
        expert_counts = np.bincount(dominant_experts, minlength=self.n_experts)
        self.dominant_expert_counts.append(expert_counts)
        
        # Track per-regime weights if regimes provided
        if regimes is not None:
            for i, regime in enumerate(regimes):
                if i < len(weights_np):
                    self.regime_weights[regime].append(weights_np[i])
    
    def get_summary(self, epoch: int) -> Dict[str, Any]:
        """Get summary statistics for current epoch"""
        if not self.weight_means:
            return {}
        
        current_means = self.weight_means[-1]
        current_stds = self.weight_stds[-1]
        
        # Find most and least used experts
        most_used_idx = np.argmax(current_means)
        least_used_idx = np.argmin(current_means)
        
        # Specialization score (0-1, higher = more specialized)
        # Based on how concentrated weights are
        max_weight = np.max(current_means)
        min_weight = np.min(current_means)
        specialization = (max_weight - min_weight) / (max_weight + 1e-10)
        
        return {
            'epoch': epoch,
            'expert_means': dict(zip(self.expert_names, current_means)),
            'expert_stds': dict(zip(self.expert_names, current_stds)),
            'entropy': self.weight_entropies[-1],
            'sparsity': self.weight_sparsity[-1],
            'specialization': specialization,
            'most_used_expert': {
                'name': self.expert_names[most_used_idx],
                'weight': current_means[most_used_idx],
                'confidence': 1.0 - current_stds[most_used_idx] / (current_means[most_used_idx] + 1e-10)
            },
            'least_used_expert': {
                'name': self.expert_names[least_used_idx],
                'weight': current_means[least_used_idx]
            },
            'dominant_counts': dict(zip(self.expert_names, self.dominant_expert_counts[-1]))
        }

# ==================== Enhanced Comparative Trainer with Weight Tracking ====================

class EnhancedExpertRoutingComparativeTrainer:
    """Enhanced trainer for comparing different routing methods with weight tracking"""
    
    def __init__(self, models: Dict[str, nn.Module], learning_rates: Dict[str, float],
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {name: model.to(device) for name, model in models.items()}
        
        # Method labels
        self.method_labels = {
            'fno': 'FNO Baseline',
            'softmax': 'Softmax Routing',
            'lagrangian_single': 'Single-Scale Lagrangian',
            'lagrangian_two': 'Two-Scale Lagrangian',
            'admm': 'ADMM Routing'
        }
        
        # Initialize weight trackers for routing methods
        self.weight_trackers = {}
        expert_names = ['Fourier', 'Spectral', 'FiniteDiff', 'Neural']
        
        for name, model in self.models.items():
            if name != 'fno':
                self.weight_trackers[name] = WeightTracker(
                    n_experts=model.n_experts,
                    expert_names=expert_names[:model.n_experts]
                )
        
        # Initialize optimizers and schedulers
        self.optimizers = {}
        self.schedulers = {}
        
        for name, model in self.models.items():
            if name == 'lagrangian_two':
                # Separate optimizers for two-scale Lagrangian
                theta_params = [p for n, p in model.named_parameters()
                              if not any(x in n for x in ['lambda_multipliers'])]
                lambda_params = [p for n, p in model.named_parameters()
                               if 'lambda_multipliers' in n]
                
                self.optimizers[f'{name}_theta'] = torch.optim.AdamW(
                    theta_params, lr=learning_rates.get(f'{name}_theta', 1e-4))
                self.optimizers[f'{name}_lambda'] = torch.optim.AdamW(
                    lambda_params, lr=learning_rates.get(f'{name}_lambda', 1e-3))
                
                self.schedulers[f'{name}_theta'] = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizers[f'{name}_theta'], gamma=0.95)
                self.schedulers[f'{name}_lambda'] = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizers[f'{name}_lambda'], gamma=0.95)
            else:
                # Single optimizer for other methods
                self.optimizers[name] = torch.optim.AdamW(
                    model.parameters(),
                    lr=learning_rates.get(name, 1e-4),
                    weight_decay=1e-6
                )
                self.schedulers[name] = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizers[name], gamma=0.95)
        
        # Metrics storage
        self.metrics = {name: defaultdict(list) for name in models.keys()}
        self.val_metrics = {name: defaultdict(list) for name in models.keys()}
    
    def train_step(self, batch: Dict) -> Dict[str, Dict[str, float]]:
        """Single training step for all models"""
        u0 = batch['u0'].to(self.device)
        u_solution = batch['u_solution'].to(self.device)
        u_noisy = batch['u_noisy'].to(self.device)
        regimes = batch.get('regime', None)
        
        all_metrics = {}
        
        for name, model in self.models.items():
            model.train()
            
            if name == 'fno':
                # FNO baseline
                self.optimizers[name].zero_grad()
                predictions = model(u_noisy)
                
                # Compute losses for FNO
                recon_loss = huber_loss(predictions, u_solution).mean()
                
                # Physics loss
                dx = 2.0 / model.grid_size
                physics_loss = model.compute_physics_loss(predictions, u_noisy, nu=0.01, dx=dx, t_final=0.5)
                
                total_loss = recon_loss + 0.001 * physics_loss
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.optimizers[name].step()
                
                all_metrics[name] = {
                    'loss': total_loss.item(),
                    'recon_loss': recon_loss.item(),
                    'physics_loss': physics_loss.item()
                }
                
            else:
                # Expert routing methods
                if name == 'lagrangian_two':
                    # Two-scale optimization
                    self.optimizers[f'{name}_theta'].zero_grad()
                    self.optimizers[f'{name}_lambda'].zero_grad()
                    
                    predictions, metadata = model(u_noisy)
                    total_loss, metrics = model.compute_loss(
                        predictions, u_solution, u_noisy, metadata
                    )
                    
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    self.optimizers[f'{name}_theta'].step()
                    self.optimizers[f'{name}_lambda'].step()
                    
                    # Update Lagrange multipliers
                    if hasattr(model.router, 'update_multipliers'):
                        model.router.update_multipliers(metadata.get('constraint_violation', torch.tensor(0.0)))
                
                else:
                    # Single-scale optimization
                    self.optimizers[name].zero_grad()
                    
                    predictions, metadata = model(u_noisy)
                    total_loss, metrics = model.compute_loss(
                        predictions, u_solution, u_noisy, metadata
                    )
                    
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    self.optimizers[name].step()
                
                # Track weights and confidence
                if 'weights' in metadata:
                    weights = metadata['weights']
                    self.weight_trackers[name].update(weights, regimes)
                    
                    # Add weight statistics to metrics
                    weight_mean = weights.mean(dim=0).detach().cpu().numpy()
                    weight_std = weights.std(dim=0).detach().cpu().numpy()
                    
                    for i in range(model.n_experts):
                        metrics[f'weight_exp_{i}_mean'] = float(weight_mean[i])
                        metrics[f'weight_exp_{i}_std'] = float(weight_std[i])
                    
                    # Compute confidence metrics
                    weight_entropy = -torch.sum(weights.mean(dim=0) * torch.log(weights.mean(dim=0) + 1e-10))
                    weight_sparsity = torch.mean((weights < 0.2).float())
                    
                    metrics['weight_entropy'] = weight_entropy.item()
                    metrics['weight_sparsity'] = weight_sparsity.item()
                    metrics['weight_max'] = weights.max().item()
                    metrics['weight_min'] = weights.min().item()
                
                all_metrics[name] = metrics
        
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
        result_metrics = {}
        for name, metrics in epoch_metrics.items():
            result_metrics[name] = {k: float(np.mean(v)) for k, v in metrics.items()}
        
        # Store in history
        for name, metrics in result_metrics.items():
            for k, v in metrics.items():
                self.metrics[name][k].append(v)
        
        return result_metrics
    
    def validate(self, dataloader: DataLoader) -> Dict[str, Dict[str, float]]:
        """Validate all models"""
        for model in self.models.values():
            model.eval()
        
        val_metrics = {name: defaultdict(list) for name in self.models.keys()}
        
        with torch.no_grad():
            for batch in dataloader:
                u0 = batch['u0'].to(self.device)
                u_solution = batch['u_solution'].to(self.device)
                u_noisy = batch['u_noisy'].to(self.device)
                regimes = batch.get('regime', None)
                
                for name, model in self.models.items():
                    if name == 'fno':
                        predictions = model(u_noisy)
                        recon_loss = huber_loss(predictions, u_solution).mean()
                        dx = 2.0 / model.grid_size
                        physics_loss = model.compute_physics_loss(predictions, u_noisy, nu=0.01, dx=dx, t_final=0.5)
                        
                        total_loss = recon_loss + 0.001 * physics_loss
                        val_metrics[name]['loss'].append(total_loss.item())
                        val_metrics[name]['recon_loss'].append(recon_loss.item())
                        
                    else:
                        predictions, metadata = model(u_noisy)
                        total_loss, metrics = model.compute_loss(
                            predictions, u_solution, u_noisy, metadata
                        )
                        
                        for k, v in metrics.items():
                            val_metrics[name][k].append(v)
                    
                    # Additional validation metrics
                    if name != 'fno' and 'weights' in metadata:
                        weights = metadata['weights']
                        weight_entropy = -torch.sum(weights.mean(dim=0) * torch.log(weights.mean(dim=0) + 1e-10))
                        val_metrics[name]['weight_entropy'].append(weight_entropy.item())
                        
                        # Expert utilization
                        dominant_expert = torch.argmax(weights, dim=1)
                        for i in range(model.n_experts):
                            utilization = torch.mean((dominant_expert == i).float())
                            val_metrics[name][f'utilization_exp_{i}'].append(utilization.item())
        
        # Average and store
        result_metrics = {}
        for name, metrics in val_metrics.items():
            result_metrics[name] = {k: float(np.mean(v)) for k, v in metrics.items()}
            for k, v in result_metrics[name].items():
                self.val_metrics[name][k].append(v)
        
        return result_metrics
    
    def update_schedulers(self, val_losses: Dict[str, Dict[str, float]]):
        """Update learning rate schedulers"""
        for name, scheduler in self.schedulers.items():
            if 'lagrangian_two' in name:
                # Use appropriate loss for two-scale methods
                if 'theta' in name:
                    loss_value = val_losses['lagrangian_two'].get('loss', 1.0)
                else:
                    loss_value = val_losses['lagrangian_two'].get('constraint_violation', 1.0)
            else:
                base_name = name.replace('_theta', '').replace('_lambda', '')
                loss_value = val_losses.get(base_name, {}).get('loss', 1.0)
            
            scheduler.step()
    
    def print_weight_summary(self, epoch: int):
        """Print detailed weight and confidence summary"""
        print("\n" + "="*60)
        print(f"EXPERT WEIGHTS AND CONFIDENCE - Epoch {epoch}")
        print("="*60)
        
        for name, tracker in self.weight_trackers.items():
            summary = tracker.get_summary(epoch)
            if not summary:
                continue
            
            print(f"\n{self.method_labels[name]} ({name}):")
            print("-" * 40)
            
            # Print expert weights
            print("Expert Weights (mean ± std):")
            for exp_name, mean in summary['expert_means'].items():
                std = summary['expert_stds'][exp_name]
                confidence = 1.0 - (std / (mean + 1e-10))
                print(f"  {exp_name:12s}: {mean:.4f} ± {std:.4f} (conf: {confidence:.3f})")
            
            # Print confidence metrics
            print(f"\nConfidence Metrics:")
            print(f"  Entropy:          {summary['entropy']:.4f}")
            print(f"  Sparsity:         {summary['sparsity']:.4f}")
            print(f"  Specialization:   {summary['specialization']:.4f}")
            
            # Print dominant expert
            print(f"\nMost Used Expert:")
            print(f"  {summary['most_used_expert']['name']}: "
                  f"weight={summary['most_used_expert']['weight']:.4f}, "
                  f"confidence={summary['most_used_expert']['confidence']:.3f}")
            
            # Print dominant counts
            print(f"\nDominant Expert Counts:")
            for exp_name, count in summary['dominant_counts'].items():
                print(f"  {exp_name:12s}: {count}")
        
        print("\n" + "="*60)

# ==================== Main Training Function ====================

def train_comparative_expert_routing_with_weights():
    """Main training function for comparative expert routing with weight tracking"""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Parameters
    n_samples = 5000
    grid_size = 64
    batch_size = 32
    n_epochs = 500
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("ENHANCED EXPERT ROUTING WITH WEIGHT TRACKING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Grid size: {grid_size}")
    print(f"Number of samples: {n_samples}")
    print("="*60)
    
    # Create dataset
    print("Creating enhanced Burgers dataset...")
    dataset = EnhancedBurgersDataset(
        n_samples=n_samples,
        grid_size=grid_size,
        t_final=0.5,
        re=100,
        noise_std=0.02,
        include_regimes=True
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    regimes = [dataset[i]['regime'] for i in range(len(dataset))]
    train_indices, val_indices = train_test_split(
        range(len(dataset)), 
        test_size=0.2, 
        stratify=regimes,
        random_state=42
    )

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create models
    print("Creating models...")
    models = {
        'fno': EnhancedFNOBaseline(modes=16, width=64, grid_size=grid_size),
        'softmax': EnhancedExpertRoutingSystem(
            routing_method='softmax',
            n_experts=4,
            input_dim=grid_size,
            hidden_dim=256,
            device=device
        ),
        'lagrangian_single': EnhancedExpertRoutingSystem(
            routing_method='lagrangian_single',
            n_experts=4,
            input_dim=grid_size,
            hidden_dim=256,
            device=device
        ),
        'lagrangian_two': EnhancedExpertRoutingSystem(
            routing_method='lagrangian_two',
            n_experts=4,
            input_dim=grid_size,
            hidden_dim=256,
            device=device
        ),
        'admm': EnhancedExpertRoutingSystem(
            routing_method='admm',
            n_experts=4,
            input_dim=grid_size,
            hidden_dim=256,
            device=device
        )
    }
    
    # Learning rates
    learning_rates = {
        'fno': 1e-3,
        'softmax': 1e-3,
        'lagrangian_single': 1e-3,
        'lagrangian_two_theta': 1e-3,
        'lagrangian_two_lambda': 5e-4,
        'admm': 1e-3
    }
    
    # Create trainer
    trainer = EnhancedExpertRoutingComparativeTrainer(
        models=models,
        learning_rates=learning_rates,
        device=device
    )
    
    # Training loop
    print("\nStarting training...")
    print("="*60)
    
    best_val_loss = {name: float('inf') for name in models.keys()}
    
    for epoch in range(n_epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        
        # Print progress with weights
        trainer.print_weight_summary(epoch + 1)
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{n_epochs} Performance:")
        print("-" * 50)
        for name in models.keys():
            print(f"{trainer.method_labels[name]:25s} | "
                  f"Train Loss: {train_metrics[name].get('loss', 0):.4e} | "
                  f"Val Loss: {val_metrics[name].get('loss', 0):.4e} | "
                  f"Physics: {train_metrics[name].get('physics_loss', 0):.4e} | "
                  f"Train Recon Loss: {train_metrics[name].get('recon_loss'):.4e} | "
                  f"Val Recon Loss: {val_metrics[name].get('recon_loss'):.4e}")
        
        # Update schedulers
        trainer.update_schedulers(val_metrics)
        
        # Save best models
        for name in models.keys():
            if val_metrics[name].get('loss', float('inf')) < best_val_loss[name]:
                best_val_loss[name] = val_metrics[name]['loss']
                checkpoint = {
                    'model_state': trainer.models[name].state_dict(),
                    'val_loss': best_val_loss[name],
                    'epoch': epoch,
                    'metrics': trainer.metrics[name]
                }
                
                # Add weight tracker data for routing methods
                if name != 'fno' and name in trainer.weight_trackers:
                    checkpoint['weight_tracker'] = {
                        'weight_means': trainer.weight_trackers[name].weight_means,
                        'weight_entropies': trainer.weight_trackers[name].weight_entropies,
                        'weight_sparsity': trainer.weight_trackers[name].weight_sparsity
                    }
                
                torch.save(checkpoint, f'expert_routing_results/best_{name}_model.pth')
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} completed.")
        print(f"Best validation losses so far:")
        for name in models.keys():
            print(f"  {trainer.method_labels[name]}: {best_val_loss[name]:.4e}")
    
    # Final evaluation and summary
    print("\n" + "="*60)
    print("FINAL RESULTS AND WEIGHT ANALYSIS")
    print("="*60)
    
    # Load best models and evaluate
    for name in models.keys():
        checkpoint = torch.load(f'expert_routing_results/best_{name}_model.pth', 
                               map_location=device,
                               weights_only=False)
        trainer.models[name].load_state_dict(checkpoint['model_state'])
        
        # Final validation
        trainer.models[name].eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                u0 = batch['u0'].to(device)
                u_solution = batch['u_solution'].to(device)
                
                if name == 'fno':
                    predictions = trainer.models[name](u0)
                    loss = huber_loss(predictions, u_solution).mean()
                else:
                    predictions, _ = trainer.models[name](u0)
                    loss = huber_loss(predictions, u_solution).mean()
                
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        print(f"\n{trainer.method_labels[name]:25s}:")
        print(f"  Best Val Loss: {checkpoint['val_loss']:.4e}")
        print(f"  Final Val Loss: {avg_val_loss:.4e}")
        
        # Print final weight statistics for routing methods
        if name != 'fno' and name in trainer.weight_trackers:
            tracker = trainer.weight_trackers[name]
            summary = tracker.get_summary(n_epochs)
            if summary:
                print(f"  Final Expert Weights:")
                for exp_name, mean in summary['expert_means'].items():
                    std = summary['expert_stds'][exp_name]
                    confidence = 1.0 - (std / (mean + 1e-10))
                    print(f"    {exp_name}: {mean:.4f} ± {std:.4f} (conf: {confidence:.3f})")
                print(f"  Final Entropy: {summary['entropy']:.4f}")
                print(f"  Final Specialization: {summary['specialization']:.4f}")
    
    # Save comprehensive results
    final_results = {
        'metrics': trainer.metrics,
        'val_metrics': trainer.val_metrics,
        'best_val_loss': best_val_loss,
        'weight_trackers': {}
    }
    
    # Save weight tracker data
    for name, tracker in trainer.weight_trackers.items():
        final_results['weight_trackers'][name] = {
            'weight_means': tracker.weight_means,
            'weight_stds': tracker.weight_stds,
            'weight_entropies': tracker.weight_entropies,
            'weight_sparsity': tracker.weight_sparsity,
            'dominant_expert_counts': tracker.dominant_expert_counts,
            'expert_names': tracker.expert_names
        }
    
    torch.save(final_results, 'expert_routing_results/final_results_detailed.pth')
    
    # Print comprehensive weight analysis
    print("\n" + "="*60)
    print("COMPREHENSIVE WEIGHT ANALYSIS")
    print("="*60)
    
    for name in ['softmax', 'lagrangian_single', 'lagrangian_two', 'admm']:
        if name in trainer.weight_trackers:
            tracker = trainer.weight_trackers[name]
            summary = tracker.get_summary(n_epochs)
            if summary:
                print(f"\n{name.upper()} Routing Method:")
                print("-" * 40)
                
                # Weight evolution from first to last epoch
                if len(tracker.weight_means) > 1:
                    first_means = tracker.weight_means[0]
                    last_means = tracker.weight_means[-1]
                    
                    print(f"  Weight Evolution (Epoch 1 → Epoch {n_epochs}):")
                    for i, exp_name in enumerate(tracker.expert_names):
                        change = last_means[i] - first_means[i]
                        percent_change = (change / (first_means[i] + 1e-10)) * 100
                        print(f"    {exp_name:12s}: {first_means[i]:.4f} → {last_means[i]:.4f} "
                              f"({change:+.4f}, {percent_change:+.1f}%)")
                
                # Specialization analysis
                print(f"\n  Specialization Metrics:")
                print(f"    Entropy (final): {summary['entropy']:.4f}")
                print(f"    Sparsity (final): {summary['sparsity']:.4f}")
                print(f"    Specialization Score: {summary['specialization']:.4f}")
                
                # Most used expert
                print(f"\n  Most Used Expert:")
                print(f"    {summary['most_used_expert']['name']} "
                      f"(weight: {summary['most_used_expert']['weight']:.4f}, "
                      f"confidence: {summary['most_used_expert']['confidence']:.3f})")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("Results saved in 'expert_routing_results/' directory")
    print("="*60)
    
    return trainer, final_results

# ==================== Main Execution ====================

if __name__ == "__main__":
    # Create results directory
    os.makedirs('expert_routing_results', exist_ok=True)
    
    # Run training with weight tracking
    trainer, final_results = train_comparative_expert_routing_with_weights()
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL TRAINING SUMMARY")
    print("="*60)
    
    # Print best validation losses
    print("\nBest Validation Losses Across All Methods:")
    print("-" * 40)
    
    routing_methods = ['softmax', 'lagrangian_single', 'lagrangian_two', 'admm']
    
    # Get best performance for each method
    for name in trainer.models.keys():
        if name in trainer.metrics and 'loss' in trainer.metrics[name]:
            best_train_loss = min(trainer.metrics[name]['loss'])
            best_val_loss = min(trainer.val_metrics[name].get('loss', [float('inf')]))
            
            print(f"{trainer.method_labels[name]:25s}:")
            print(f"  Best Train Loss: {best_train_loss:.4e}")
            print(f"  Best Val Loss:   {best_val_loss:.4e}")
            
            if name in routing_methods:
                # Print weight statistics for routing methods
                print(f"  Final Weight Distribution:")
                if name in trainer.weight_trackers:
                    tracker = trainer.weight_trackers[name]
                    if tracker.weight_means:
                        final_weights = tracker.weight_means[-1]
                        for i, exp_name in enumerate(tracker.expert_names):
                            print(f"    {exp_name}: {final_weights[i]:.4f}")
    
    print("\n" + "="*60)
    print("Expert Routing Training Complete!")
    print("="*60)