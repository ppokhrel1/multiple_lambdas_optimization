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

warnings.filterwarnings('ignore')

# ==================== Enums and Base Classes ====================

class PhysicsRegime2D(Enum):
    """Physics regimes for 2D flow characteristics"""
    VORTEX = 'vortex'
    SHEAR_LAYER = 'shear_layer'
    BOUNDARY_LAYER = 'boundary_layer'
    TURBULENT = 'turbulent'
    SHOCK = 'shock'

class BaseExpertRoutingModel2D(nn.Module):
    """Base class for 2D expert routing models"""
    def __init__(self, grid_size: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    physics_params: Dict[str, float] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError

def huber_loss(pred, target, delta=1.0):
    """Huber loss for robust training"""
    diff = pred - target
    abs_diff = torch.abs(diff)
    quadratic = torch.min(abs_diff, torch.tensor(delta, device=pred.device))
    linear = abs_diff - quadratic
    return 0.5 * quadratic.pow(2) + delta * linear

# ==================== Enhanced 2D Burgers Dataset ====================

class EnhancedBurgersDataset2D(Dataset):
    """Enhanced dataset for 2D Burgers' equation with realistic physics regimes"""
    
    def __init__(self, n_samples: int = 1000, grid_size: int = 64, t_final: float = 0.5,
                 re: float = 100, noise_std: float = 0.02, seed: int = 42,
                 include_regimes: bool = True):
        super().__init__()
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.t_final = t_final
        self.nu = 1.0 / re
        self.noise_std = noise_std
        self.include_regimes = include_regimes
        
        # Create 2D grid
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        self.X, self.Y = np.meshgrid(x, y)
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        
        np.random.seed(seed)
        self.data = self._generate_enhanced_data()
        
    def _classify_solution_regime(self, u: np.ndarray, v: np.ndarray) -> PhysicsRegime2D:
        """Classify the physics regime based on flow characteristics"""
        # Compute velocity magnitude
        vel_magnitude = np.sqrt(u**2 + v**2)
        
        # Compute vorticity
        uy, ux = np.gradient(u, self.dy, self.dx)
        vy, vx = np.gradient(v, self.dy, self.dx)
        vorticity = vx - uy
        
        # Compute strain rate
        strain = np.sqrt((ux - vy)**2 + (uy + vx)**2)
        
        # Check for shocks (high strain rate)
        if np.max(strain) > 5.0:
            return PhysicsRegime2D.SHOCK
        
        # Check for vortices (high vorticity concentration)
        vorticity_norm = np.abs(vorticity) / (np.max(np.abs(vorticity)) + 1e-8)
        if np.mean(vorticity_norm > 0.5) > 0.1:  # More than 10% high vorticity
            return PhysicsRegime2D.VORTEX
        
        # Check for shear layers (high velocity gradient in one direction)
        if np.max(np.abs(ux)) > 2.0 or np.max(np.abs(vy)) > 2.0:
            return PhysicsRegime2D.SHEAR_LAYER
        
        # Check for boundary layers (high gradients near boundaries)
        boundary_grad = np.max(np.abs(u[:, :5])) + np.max(np.abs(u[:, -5:])) + \
                       np.max(np.abs(v[:5, :])) + np.max(np.abs(v[-5:, :]))
        if boundary_grad > 3.0:
            return PhysicsRegime2D.BOUNDARY_LAYER
        
        # Otherwise turbulent
        return PhysicsRegime2D.TURBULENT
    
    def _generate_initial_condition(self, regime: Optional[PhysicsRegime2D] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate initial condition for specific regime"""
        if regime is None:
            regime = np.random.choice(list(PhysicsRegime2D))
        
        if regime == PhysicsRegime2D.VORTEX:
            # Taylor-Green vortices
            u0 = np.sin(np.pi * self.X) * np.cos(np.pi * self.Y)
            v0 = -np.cos(np.pi * self.X) * np.sin(np.pi * self.Y)
            
        elif regime == PhysicsRegime2D.SHEAR_LAYER:
            # Shear layer with perturbation
            u0 = np.tanh(10 * self.Y)
            v0 = 0.05 * np.sin(2 * np.pi * self.X) * np.exp(-(self.Y**2) / 0.1)
            
        elif regime == PhysicsRegime2D.BOUNDARY_LAYER:
            # Boundary layer flow
            u0 = 1.0 - np.exp(-np.abs(self.Y) / 0.1)
            v0 = 0.01 * np.sin(2 * np.pi * self.X) * (1 - np.exp(-np.abs(self.Y) / 0.1))
            
        elif regime == PhysicsRegime2D.TURBULENT:
            # Random vorticity field
            psi = np.random.randn(self.grid_size, self.grid_size)
            psi = ndimage.gaussian_filter(psi, sigma=2)
            u0 = np.gradient(psi, self.dx, axis=1)
            v0 = -np.gradient(psi, self.dy, axis=0)
            
        else:  # SHOCK
            # Shock formation
            u0 = np.where(self.X < 0, 1.0, -0.5)
            v0 = 0.1 * np.sin(2 * np.pi * self.Y)
            
            # Smooth the shock
            u0 = ndimage.gaussian_filter1d(u0, sigma=1, axis=1)
            v0 = ndimage.gaussian_filter1d(v0, sigma=1, axis=0)
        
        # Normalize
        u0 = u0 / (np.max(np.abs(u0)) + 1e-8)
        v0 = v0 / (np.max(np.abs(v0)) + 1e-8)
        
        return u0, v0
    
    def _solve_burgers_2d_enhanced(self, u0: np.ndarray, v0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Solve 2D Burgers' equation with spectral method and proper scaling"""
        N = self.grid_size
        
        # Wavenumbers
        kx = 2 * np.pi * np.fft.fftfreq(N, d=self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(N, d=self.dy)
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        
        # Initial conditions in Fourier space
        u_hat = np.fft.fft2(u0)
        v_hat = np.fft.fft2(v0)
        
        # Time stepping
        dt = 0.001
        n_steps = int(self.t_final / dt)
        
        for step in range(n_steps):
            # Dealias: zero out high frequencies
            u_hat_dealiased = u_hat.copy()
            v_hat_dealiased = v_hat.copy()
            u_hat_dealiased[N//3:, :] = 0
            u_hat_dealiased[:, N//3:] = 0
            v_hat_dealiased[N//3:, :] = 0
            v_hat_dealiased[:, N//3:] = 0
            
            # Convert to physical space
            u = np.real(np.fft.ifft2(u_hat_dealiased))
            v = np.real(np.fft.ifft2(v_hat_dealiased))
            
            # Nonlinear terms
            uu_hat = np.fft.fft2(u * u)
            uv_hat = np.fft.fft2(u * v)
            vv_hat = np.fft.fft2(v * v)
            
            # Burgers' equations in Fourier space
            rhs_u = -1j * (KX * uu_hat + KY * uv_hat) - self.nu * K2 * u_hat
            rhs_v = -1j * (KX * uv_hat + KY * vv_hat) - self.nu * K2 * v_hat
            
            # RK2 time integration
            u_hat_temp = u_hat + 0.5 * dt * rhs_u
            v_hat_temp = v_hat + 0.5 * dt * rhs_v
            
            # Convert temp to physical space
            u_temp = np.real(np.fft.ifft2(u_hat_temp))
            v_temp = np.real(np.fft.ifft2(v_hat_temp))
            
            # Nonlinear terms with temp
            uu_hat_temp = np.fft.fft2(u_temp * u_temp)
            uv_hat_temp = np.fft.fft2(u_temp * v_temp)
            vv_hat_temp = np.fft.fft2(v_temp * v_temp)
            
            rhs_u_temp = -1j * (KX * uu_hat_temp + KY * uv_hat_temp) - self.nu * K2 * u_hat_temp
            rhs_v_temp = -1j * (KX * uv_hat_temp + KY * vv_hat_temp) - self.nu * K2 * v_hat_temp
            
            u_hat = u_hat + dt * rhs_u_temp
            v_hat = v_hat + dt * rhs_v_temp
            
            # Stability check
            if np.any(np.isnan(u_hat)) or np.any(np.isinf(u_hat)):
                return u0, v0
        
        # Final solution
        u_final = np.real(np.fft.ifft2(u_hat))
        v_final = np.real(np.fft.ifft2(v_hat))
        
        return u_final, v_final
    
    def _generate_enhanced_data(self) -> List[Dict]:
        """Generate enhanced dataset with regime classification"""
        data = []
        
        for i in range(self.n_samples):
            if self.include_regimes:
                # Sample regime
                regime = np.random.choice(list(PhysicsRegime2D))
                u0, v0 = self._generate_initial_condition(regime)
            else:
                # Legacy mode
                ic_type = np.random.choice(['vortex', 'gaussian', 'shear_layer', 'random'])
                
                if ic_type == 'vortex':
                    u0 = np.sin(np.pi * self.X) * np.cos(np.pi * self.Y)
                    v0 = -np.cos(np.pi * self.X) * np.sin(np.pi * self.Y)
                    
                elif ic_type == 'gaussian':
                    centers = [(0.5, 0.5), (-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5)]
                    center = centers[np.random.randint(0, 4)]
                    width = np.random.uniform(5, 15)
                    
                    psi = np.exp(-width * ((self.X - center[0])**2 + (self.Y - center[1])**2))
                    u0 = np.gradient(psi, self.dx, axis=1)
                    v0 = -np.gradient(psi, self.dy, axis=0)
                    
                elif ic_type == 'shear_layer':
                    u0 = np.tanh(10 * self.Y)
                    v0 = 0.1 * np.sin(2 * np.pi * self.X)
                    
                else:  # random
                    psi = np.random.randn(self.grid_size, self.grid_size)
                    psi = ndimage.gaussian_filter(psi, sigma=2)
                    u0 = np.gradient(psi, self.dx, axis=1)
                    v0 = -np.gradient(psi, self.dy, axis=0)
                
                # Normalize
                u0 = u0 / (np.max(np.abs(u0)) + 1e-8)
                v0 = v0 / (np.max(np.abs(v0)) + 1e-8)
                regime = None
            
            # Solve equation
            u_solution, v_solution = self._solve_burgers_2d_enhanced(u0, v0)
            
            # Classify regime if not already specified
            if not self.include_regimes or regime is None:
                regime = self._classify_solution_regime(u_solution, v_solution)
            
            # Add noise
            noise_u = np.random.normal(0, self.noise_std, size=(self.grid_size, self.grid_size))
            noise_v = np.random.normal(0, self.noise_std, size=(self.grid_size, self.grid_size))
            u_noisy = u_solution + noise_u
            v_noisy = v_solution + noise_v
            
            data.append({
                'u0': u0.astype(np.float32),
                'v0': v0.astype(np.float32),
                'u_solution': u_solution.astype(np.float32),
                'v_solution': v_solution.astype(np.float32),
                'u_noisy': u_noisy.astype(np.float32),
                'v_noisy': v_noisy.astype(np.float32),
                'regime': regime.value if isinstance(regime, PhysicsRegime2D) else 'unknown'
            })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Stack u and v components
        initial = np.stack([item['u0'], item['v0']], axis=0)
        solution = np.stack([item['u_solution'], item['v_solution']], axis=0)
        noisy = np.stack([item['u_noisy'], item['v_noisy']], axis=0)
        
        return {
            'u0': torch.FloatTensor(initial),
            'u_solution': torch.FloatTensor(solution),
            'u_noisy': torch.FloatTensor(noisy),
            'regime': item['regime']
        }

# ==================== Expert Solvers 2D (Enhanced) ====================

class ExpertSolver2D(nn.Module):
    """Enhanced 2D expert solver with improved architectures"""
    
    def __init__(self, input_channels: int = 2, hidden_channels: int = 64,
                 expert_type: str = 'fourier', dropout_rate: float = 0.1):
        super().__init__()
        self.expert_type = expert_type
        self.input_channels = input_channels
        
        if expert_type == 'fourier':
            # Fourier-based expert with residual connections
            self.conv1 = nn.Conv2d(input_channels, hidden_channels, 3, padding=1, padding_mode='circular')
            self.bn1 = nn.BatchNorm2d(hidden_channels)
            
            self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, padding_mode='circular')
            self.bn2 = nn.BatchNorm2d(hidden_channels)
            
            self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, padding_mode='circular')
            self.bn3 = nn.BatchNorm2d(hidden_channels)
            
            self.conv4 = nn.Conv2d(hidden_channels, input_channels, 3, padding=1, padding_mode='circular')
            
        elif expert_type == 'spectral':
            # Spectral method expert
            self.layers = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=5, padding=2, padding_mode='circular'),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout2d(dropout_rate),
                nn.Conv2d(32, 64, kernel_size=5, padding=2, padding_mode='circular'),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Dropout2d(dropout_rate),
                nn.Conv2d(64, 32, kernel_size=5, padding=2, padding_mode='circular'),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, input_channels, kernel_size=5, padding=2, padding_mode='circular')
            )
            
        elif expert_type == 'finite_difference':
            # Finite difference expert with residual blocks
            self.initial_conv = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, padding_mode='circular')
            self.initial_bn = nn.BatchNorm2d(64)
            
            self.res_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='circular'),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(dropout_rate),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='circular'),
                    nn.BatchNorm2d(64)
                ) for _ in range(3)
            ])
            
            self.final_conv = nn.Conv2d(64, input_channels, kernel_size=3, padding=1, padding_mode='circular')
            
        else:  # 'neural'
            # Neural network expert with attention
            self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1, padding_mode='circular')
            self.bn1 = nn.BatchNorm2d(64)
            
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1, padding_mode='circular')
            self.bn2 = nn.BatchNorm2d(128)
            
            # Attention mechanism
            self.attention_conv = nn.Conv2d(128, 128, 1)
            self.attention_bn = nn.BatchNorm2d(128)
            
            self.conv3 = nn.Conv2d(128, 64, 3, padding=1, padding_mode='circular')
            self.bn3 = nn.BatchNorm2d(64)
            
            self.conv4 = nn.Conv2d(64, input_channels, 3, padding=1, padding_mode='circular')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.expert_type == 'fourier':
            residual = x
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.conv4(x)
            return x + residual  # Residual connection
            
        elif self.expert_type == 'neural':
            residual = x
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            
            # Attention
            attn = torch.sigmoid(self.attention_bn(self.attention_conv(x)))
            x = x * attn
            
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.conv4(x)
            return x + residual
            
        elif self.expert_type == 'finite_difference':
            x = F.relu(self.initial_bn(self.initial_conv(x)))
            residual = x
            
            for res_block in self.res_blocks:
                x_in = x
                x = res_block(x)
                x = F.relu(x + x_in)  # Residual connection
            
            x = self.final_conv(x)
            return x
            
        else:
            return self.layers(x)

# ==================== Enhanced FNO 2D ====================

class EnhancedFNOBaseline2D(nn.Module):
    """Enhanced Fourier Neural Operator for 2D with better architecture"""
    
    def __init__(self, modes: int = 16, width: int = 64, grid_size: int = 64):
        super().__init__()
        self.modes = modes
        self.width = width
        self.grid_size = grid_size
        
        # Input projection
        self.fc0 = nn.Sequential(
            nn.Linear(2, width),
            nn.LayerNorm(width),
            nn.ReLU()
        )
        
        # Fourier layers with residual connections
        self.fourier_layers = nn.ModuleList([
            EnhancedFourierLayer2D(width, width, modes=modes)
            for _ in range(4)
        ])
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(width, width, kernel_size=1),
                nn.LayerNorm(width),
                nn.ReLU()
            ) for _ in range(4)
        ])
        
        # Output projection
        self.fc1 = nn.Sequential(
            nn.Linear(width, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 2, H, W)
        x = x.permute(0, 2, 3, 1)  # (batch, H, W, 2)
        x = self.fc0(x)  # (batch, H, W, width)
        x = x.permute(0, 3, 1, 2)  # (batch, width, H, W)
        
        # Process through Fourier layers
        for i in range(4):
            residual = x
            
            # Fourier branch
            x_fourier = self.fourier_layers[i](x)
            
            # Convolution branch
            x_conv = self.conv_layers[i](x)
            
            # Combine
            x = x_fourier + x_conv + residual
            x = F.gelu(x)
        
        # Output
        x = x.permute(0, 2, 3, 1)  # (batch, H, W, width)
        x = self.fc1(x)  # (batch, H, W, 2)
        x = x.permute(0, 3, 1, 2)  # (batch, 2, H, W)
        return x
    
    def compute_physics_loss(self, predictions: torch.Tensor, initial_conditions: torch.Tensor,
                       nu: float = 0.01, dx: float = 0.03125, t_final: float = 0.5) -> torch.Tensor:
        """Compute 2D Burgers' equation physics loss (vectorized)"""
        batch_size, _, H, W = predictions.shape
        
        # Normalize predictions
        max_val_pred = torch.amax(torch.abs(predictions), dim=(2, 3), keepdim=True) + 1e-8
        predictions_norm = predictions / max_val_pred
        
        max_val_init = torch.amax(torch.abs(initial_conditions), dim=(2, 3), keepdim=True) + 1e-8
        initial_norm = initial_conditions / max_val_init
        
        # Extract u and v components
        u = predictions_norm[:, 0]  # shape: (batch_size, H, W)
        v = predictions_norm[:, 1]
        u0 = initial_norm[:, 0]
        v0 = initial_norm[:, 1]
        
        # Add channel dimension for padding
        u_with_channel = u.unsqueeze(1)  # shape: (batch_size, 1, H, W)
        v_with_channel = v.unsqueeze(1)
        
        # Pad with circular boundary conditions
        u_padded = F.pad(u_with_channel, (1, 1, 1, 1), mode='circular')
        v_padded = F.pad(v_with_channel, (1, 1, 1, 1), mode='circular')
        
        # First derivatives (central difference)
        u_x = (u_padded[:, :, 1:-1, 2:] - u_padded[:, :, 1:-1, :-2]) / (2 * dx)
        u_y = (u_padded[:, :, 2:, 1:-1] - u_padded[:, :, :-2, 1:-1]) / (2 * dx)
        v_x = (v_padded[:, :, 1:-1, 2:] - v_padded[:, :, 1:-1, :-2]) / (2 * dx)
        v_y = (v_padded[:, :, 2:, 1:-1] - v_padded[:, :, :-2, 1:-1]) / (2 * dx)
        
        # Pad first derivatives for second derivatives
        u_x_padded = F.pad(u_x, (1, 1, 1, 1), mode='circular')
        u_y_padded = F.pad(u_y, (1, 1, 1, 1), mode='circular')
        v_x_padded = F.pad(v_x, (1, 1, 1, 1), mode='circular')
        v_y_padded = F.pad(v_y, (1, 1, 1, 1), mode='circular')
        
        # Second derivatives
        u_xx = (u_x_padded[:, :, 1:-1, 2:] - 2 * u_x + u_x_padded[:, :, 1:-1, :-2]) / (dx**2)
        u_yy = (u_x_padded[:, :, 2:, 1:-1] - 2 * u_y + u_x_padded[:, :, :-2, 1:-1]) / (dx**2)
        v_xx = (v_x_padded[:, :, 1:-1, 2:] - 2 * v_x + v_x_padded[:, :, 1:-1, :-2]) / (dx**2)
        v_yy = (v_x_padded[:, :, 2:, 1:-1] - 2 * v_y + v_x_padded[:, :, :-2, 1:-1]) / (dx**2)
        
        # Temporal derivatives
        u_t = (u - u0) / t_final
        v_t = (v - v0) / t_final
        
        # Remove channel dimension for element-wise multiplication
        u_squeezed = u.unsqueeze(1)
        v_squeezed = v.unsqueeze(1)
        
        # Burgers' equation residuals
        residual_u = u_t.unsqueeze(1) + u_squeezed * u_x + v_squeezed * u_y - nu * (u_xx + u_yy)
        residual_v = v_t.unsqueeze(1) + u_squeezed * v_x + v_squeezed * v_y - nu * (v_xx + v_yy)
        
        # Compute mean squared residual
        total_residual = torch.mean(residual_u**2 + residual_v**2) * (dx**4)
        
        return total_residual

class EnhancedFourierLayer2D(nn.Module):
    """Enhanced 2D Fourier layer with better initialization"""
    
    def __init__(self, in_channels: int, out_channels: int, modes: int = 16):
        super().__init__()
        self.modes = modes
        
        # Learnable Fourier weights
        scale = 1 / (in_channels * out_channels)
        self.weights1_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes)
        )
        self.weights1_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes)
        )
        self.weights2_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes)
        )
        self.weights2_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 2D FFT
        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        
        # Get relevant modes
        modes = min(self.modes, H, W // 2 + 1)
        
        # Prepare output
        out_ft = torch.zeros(B, self.weights1_real.shape[1], H, W // 2 + 1,
                            device=x.device, dtype=torch.complex64)
        
        # Process low frequencies
        x_ft_low = x_ft[:, :, :modes, :modes]
        
        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        x_real = x_ft_low.real
        x_imag = x_ft_low.imag
        
        # First weight set
        w1_real = self.weights1_real[:, :, :modes, :modes]
        w1_imag = self.weights1_imag[:, :, :modes, :modes]
        
        out_real1 = torch.einsum('bixy,ioxy->boxy', x_real, w1_real) - \
                    torch.einsum('bixy,ioxy->boxy', x_imag, w1_imag)
        out_imag1 = torch.einsum('bixy,ioxy->boxy', x_real, w1_imag) + \
                    torch.einsum('bixy,ioxy->boxy', x_imag, w1_real)
        
        # Second weight set (for high frequencies)
        if H > modes:
            x_ft_high = x_ft[:, :, -modes:, :modes]
            x_real_high = x_ft_high.real
            x_imag_high = x_ft_high.imag
            
            w2_real = self.weights2_real[:, :, :modes, :modes]
            w2_imag = self.weights2_imag[:, :, :modes, :modes]
            
            out_real2 = torch.einsum('bixy,ioxy->boxy', x_real_high, w2_real) - \
                        torch.einsum('bixy,ioxy->boxy', x_imag_high, w2_imag)
            out_imag2 = torch.einsum('bixy,ioxy->boxy', x_real_high, w2_imag) + \
                        torch.einsum('bixy,ioxy->boxy', x_imag_high, w2_real)
            
            out_ft[:, :, -modes:, :modes] = torch.complex(out_real2, out_imag2)
        
        out_ft[:, :, :modes, :modes] = torch.complex(out_real1, out_imag1)
        
        # Inverse FFT
        return torch.fft.irfft2(out_ft, s=(H, W), dim=(-2, -1), norm='ortho')

# ==================== Base Router 2D (Enhanced) ====================

class BaseRouter2D(nn.Module):
    """Base class for 2D expert routers with enhanced feature extraction"""
    
    def __init__(self, n_experts: int = 4, grid_size: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.n_experts = n_experts
        self.grid_size = grid_size
        
        # Enhanced feature extractor for 2D
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.router_network = nn.Sequential(
            nn.Linear(128, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_experts)
        )
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from 2D input"""
        features = self.feature_extractor(x)
        return features.view(x.shape[0], -1)
    
    def compute_physics_loss(self, predictions: torch.Tensor, initial_conditions: torch.Tensor,
                           nu: float = 0.01, dx: float = 0.03125, t_final: float = 0.5) -> torch.Tensor:
        """Compute 2D Burgers' equation physics loss with proper scaling"""
        return EnhancedFNOBaseline2D.compute_physics_loss(None, predictions, initial_conditions, nu, dx, t_final)

# ==================== Softmax Router 2D (Enhanced) ====================

class SoftmaxRouter2D(BaseRouter2D):
    """Softmax-based expert routing for 2D with temperature scaling"""
    
    def __init__(self, n_experts: int = 4, grid_size: int = 64, hidden_dim: int = 256,
                 temperature: float = 2.0, entropy_weight: float = 0.5):
        super().__init__(n_experts, grid_size, hidden_dim)
        self.temperature = temperature
        self.entropy_weight = entropy_weight
    
    def forward(self, x: torch.Tensor, experts: List[nn.Module], return_weights: bool = False):
        features = self.extract_features(x)
        logits = self.router_network(features)
        weights = F.softmax(logits / self.temperature, dim=-1)
        
        # Compute entropy for regularization
        entropy = -torch.sum(weights * torch.log(weights + 1e-10), dim=-1).mean()
        
        # Get expert outputs
        expert_outputs = []
        for expert in experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Combine with weights
        output = torch.sum(expert_outputs * weights.view(-1, self.n_experts, 1, 1, 1), dim=1)
        
        if return_weights:
            return output, weights, entropy
        return output

# ==================== Lagrangian Routers 2D (Enhanced) ====================

class LagrangianSingleTimeRouter2D(BaseRouter2D):
    """Lagrangian routing with single time scale optimization for 2D"""
    
    def __init__(self, n_experts: int = 4, grid_size: int = 64, hidden_dim: int = 256,
                 rho: float = 0.1, constraint_type: str = 'simplex'):
        super().__init__(n_experts, grid_size, hidden_dim)
        self.rho = rho
        self.constraint_type = constraint_type
        
        # Lagrange multipliers
        self.lambda_multipliers = nn.Parameter(torch.zeros(n_experts))
        
        # Adaptive rho
        self.rho_min = 0.01
        self.rho_max = 1.0
    
    def forward(self, x: torch.Tensor, experts: List[nn.Module], return_weights: bool = False):
        features = self.extract_features(x)
        logits = self.router_network(features)
        
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
        output = torch.sum(expert_outputs * weights.view(-1, self.n_experts, 1, 1, 1), dim=1)
        
        # Compute constraint violation
        constraint_violation = self._compute_constraints(weights)
        
        if return_weights:
            return output, weights, constraint_violation
        return output
    
    def _simplex_projection(self, logits: torch.Tensor) -> torch.Tensor:
        """Project onto simplex"""
        u, _ = torch.sort(logits, descending=True, dim=-1)
        cssv = torch.cumsum(u, dim=-1)
        
        ind = torch.arange(1, logits.shape[-1] + 1, device=logits.device)
        cond = u - (cssv - 1.0) / ind > 0
        rho = torch.sum(cond, dim=-1, keepdim=True)
        
        theta = (torch.gather(cssv, -1, rho - 1) - 1.0) / rho
        
        return F.relu(logits - theta)
    
    def _compute_constraints(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute constraint violations"""
        sum_constraint = torch.abs(weights.sum(dim=-1) - 1.0).mean()
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

class LagrangianTwoTimeRouter2D(LagrangianSingleTimeRouter2D):
    """Lagrangian routing with two time scale optimization for 2D"""
    
    def __init__(self, n_experts: int = 4, grid_size: int = 64, hidden_dim: int = 256,
                 rho: float = 0.1, fast_lr: float = 0.01, slow_lr: float = 0.001):
        super().__init__(n_experts, grid_size, hidden_dim, rho)
        self.fast_lr = fast_lr
        self.slow_lr = slow_lr
        
        # Regime classifier for 2D - output should match n_experts, not number of regimes
        self.regime_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_experts)  # Changed from 5 to n_experts
        )
    
    def forward(self, x: torch.Tensor, experts: List[nn.Module], return_weights: bool = False):
        features = self.extract_features(x)
        logits = self.router_network(features)
        
        # Regime classification for adaptive routing
        regime_logits = self.regime_classifier(features)
        regime_weights = F.softmax(regime_logits, dim=-1)
        
        # Regime-based weight adaptation - now both have shape [batch_size, n_experts]
        adapted_logits = logits + torch.log(regime_weights + 1e-10)
        
        if self.constraint_type == 'simplex':
            weights = self._simplex_projection(adapted_logits)
        else:
            weights = F.softmax(adapted_logits, dim=-1)
        
        # Apply Lagrangian adjustment
        weights = weights + self.lambda_multipliers.unsqueeze(0) * self.fast_lr
        weights = F.softmax(weights, dim=-1)
        
        # Get expert outputs
        expert_outputs = []
        for expert in experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Combine
        output = torch.sum(expert_outputs * weights.view(-1, self.n_experts, 1, 1, 1), dim=1)
        
        # Compute constraint violation
        constraint_violation = self._compute_constraints(weights)
        
        if return_weights:
            return output, weights, constraint_violation
        return output

# ==================== ADMM Router 2D (Enhanced) ====================

class ADMMRouter2D(BaseRouter2D):
    """ADMM-based expert routing for 2D"""
    
    def __init__(self, n_experts: int = 4, grid_size: int = 64, hidden_dim: int = 256,
                 rho: float = 0.1, admm_iterations: int = 3):
        super().__init__(n_experts, grid_size, hidden_dim)
        self.rho = rho
        self.admm_iterations = admm_iterations
        
        # ADMM variables
        self.register_buffer('z', torch.ones(n_experts) / n_experts)
        self.register_buffer('u_dual', torch.zeros(n_experts))
    
    def forward(self, x: torch.Tensor, experts: List[nn.Module], return_weights: bool = False):
        features = self.extract_features(x)
        logits = self.router_network(features)
        
        # Perform ADMM updates
        weights = self._admm_update(logits)
        
        # Get expert outputs
        expert_outputs = []
        for expert in experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Combine
        output = torch.sum(expert_outputs * weights.view(-1, self.n_experts, 1, 1, 1), dim=1)
        
        if return_weights:
            return output, weights, torch.tensor(0.0, device=x.device)
        return output
    
    def _admm_update(self, logits: torch.Tensor) -> torch.Tensor:
        """Perform ADMM updates"""
        # x-update: softmax of logits
        x = F.softmax(logits, dim=-1)
        
        # z-update: projection onto simplex
        v = x.mean(dim=0) + self.u_dual
        z_new = self._project_simplex(v.unsqueeze(0)).squeeze(0)
        
        # u-update
        u_new = self.u_dual + (x.mean(dim=0) - z_new)
        
        # Update variables
        self.z.data = z_new
        self.u_dual.data = u_new
        
        return x
    
    def _project_simplex(self, v: torch.Tensor) -> torch.Tensor:
        """Project vectors onto simplex"""
        device = v.device
        v_sorted, _ = torch.sort(v, descending=True, dim=-1)
        cssv = torch.cumsum(v_sorted, dim=-1)
        
        rho = torch.arange(1, v.shape[-1] + 1, device=device).float()
        cond = v_sorted - (cssv - 1.0) / rho > 0
        rho_index = torch.sum(cond, dim=-1, keepdim=True) - 1
        
        theta = (torch.gather(cssv, -1, rho_index) - 1.0) / (rho_index + 1.0)
        
        return F.relu(v - theta)

# ==================== Expert Routing System 2D (Enhanced) ====================

class EnhancedExpertRoutingSystem2D(nn.Module):
    """Enhanced system for 2D expert routing with improved training"""
    
    def __init__(self, routing_method: str = 'softmax', n_experts: int = 4,
                 grid_size: int = 64, hidden_dim: int = 128, dropout_rate: float = 0.1,
                 physics_weight: float = 0.01, entropy_weight: float = 0.5,  # Changed weights
                 constraint_weight: float = 0.01, device: str = 'cpu'):  # Changed constraint weight
        super().__init__()
        self.routing_method = routing_method
        self.n_experts = n_experts
        self.grid_size = grid_size
        self.physics_weight = physics_weight
        self.entropy_weight = entropy_weight
        self.constraint_weight = constraint_weight
        self.device = device
        
        # Initialize 2D experts with enhanced architectures
        expert_types = ['fourier', 'spectral', 'finite_difference', 'neural'][:n_experts]
        self.experts = nn.ModuleList([
            ExpertSolver2D(input_channels=2, hidden_channels=64, 
                          expert_type=et, dropout_rate=dropout_rate)
            for et in expert_types
        ])
        
        # Initialize router
        if routing_method == 'softmax':
            self.router = SoftmaxRouter2D(n_experts, grid_size, 256)
        elif routing_method == 'lagrangian_single':
            self.router = LagrangianSingleTimeRouter2D(n_experts, grid_size, 256)
        elif routing_method == 'lagrangian_two':
            self.router = LagrangianTwoTimeRouter2D(n_experts, grid_size, 256)
        elif routing_method == 'admm':
            self.router = ADMMRouter2D(n_experts, grid_size, 256)
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
        """Compute all losses for 2D"""
        # Reconstruction loss (Huber for robustness)
        recon_loss = huber_loss(predictions, targets, delta=1.0).mean()
        
        # Physics loss - increased weight
        dx = 2.0 / self.grid_size
        physics_loss = self.router.compute_physics_loss(
            predictions, initial_conditions, nu=0.01, dx=dx, t_final=0.5
        )
        
        # Total loss with balanced weights
        total_loss = recon_loss + self.physics_weight * physics_loss
        
        # Add regularization terms based on routing method
        if self.routing_method == 'softmax' and 'entropy' in metadata:
            # Entropy regularization (encourage diverse expert usage)
            entropy_reg = -torch.sum(metadata['weights'] * torch.log(metadata['weights'] + 1e-10), dim=-1).mean()
            total_loss = total_loss + self.entropy_weight * entropy_reg
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

# ==================== Weight Tracking Class for 2D ====================

class WeightTracker2D:
    """Track and analyze expert weights over training for 2D models"""
    
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

# ==================== Enhanced Comparative Trainer with Weight Tracking (2D) ====================

class EnhancedExpertRoutingComparativeTrainer2D:
    """Enhanced trainer for comparing 2D routing methods with weight tracking"""
    
    def __init__(self, models: Dict[str, nn.Module], learning_rates: Dict[str, float],
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {name: model.to(device) for name, model in models.items()}
        
        # Method labels
        self.method_labels = {
            'fno_2d': '2D FNO Baseline',
            'softmax_2d': '2D Softmax Routing',
            'lagrangian_single_2d': '2D Single-Scale Lagrangian',
            'lagrangian_two_2d': '2D Two-Scale Lagrangian',
            'admm_2d': '2D ADMM Routing'
        }
        
        # Initialize weight trackers for routing methods
        self.weight_trackers = {}
        expert_names = ['Fourier', 'Spectral', 'FiniteDiff', 'Neural']
        
        for name, model in self.models.items():
            if name != 'fno_2d':
                self.weight_trackers[name] = WeightTracker2D(
                    n_experts=model.n_experts,
                    expert_names=expert_names[:model.n_experts]
                )
        
        # Initialize optimizers and schedulers
        self.optimizers = {}
        self.schedulers = {}
        
        for name, model in self.models.items():
            if name == 'lagrangian_two_2d':
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
        """Single training step for all 2D models"""
        u0 = batch['u0'].to(self.device)
        u_solution = batch['u_solution'].to(self.device)
        u_noisy = batch['u_noisy'].to(self.device)
        regimes = batch.get('regime', None)
        
        all_metrics = {}
        
        for name, model in self.models.items():
            model.train()
            
            if name == 'fno_2d':
                # FNO baseline
                self.optimizers[name].zero_grad()
                predictions = model(u_noisy)
                
                # Compute losses
                recon_loss = huber_loss(predictions, u_solution).mean()
                
                # Physics loss
                dx = 2.0 / model.grid_size
                physics_loss = model.compute_physics_loss(predictions, u_noisy, nu=0.01, dx=dx)
                
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
                if name == 'lagrangian_two_2d':
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
        """Train all 2D models for one epoch"""
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
        """Validate all 2D models"""
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
                    if name == 'fno_2d':
                        predictions = model(u_noisy)
                        recon_loss = huber_loss(predictions, u_solution).mean()
                        
                        val_metrics[name]['loss'].append(recon_loss.item())
                        val_metrics[name]['recon_loss'].append(recon_loss.item())
                        
                    else:
                        predictions, metadata = model(u_noisy)
                        total_loss, metrics = model.compute_loss(
                            predictions, u_solution, u_noisy, metadata
                        )
                        
                        for k, v in metrics.items():
                            val_metrics[name][k].append(v)
                    
                    # Additional validation metrics
                    if name != 'fno_2d' and 'weights' in metadata:
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
            if 'lagrangian_two_2d' in name:
                # Use appropriate loss for two-scale methods
                if 'theta' in name:
                    loss_value = val_losses['lagrangian_two_2d'].get('loss', 1.0)
                else:
                    loss_value = val_losses['lagrangian_two_2d'].get('constraint_violation', 1.0)
            else:
                base_name = name.replace('_theta', '').replace('_lambda', '')
                loss_value = val_losses.get(base_name, {}).get('loss', 1.0)
            
            scheduler.step()
    
    def print_weight_summary(self, epoch: int):
        """Print detailed weight and confidence summary"""
        print("\n" + "="*60)
        print(f"2D EXPERT WEIGHTS AND CONFIDENCE - Epoch {epoch}")
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

# ==================== Main Training Function 2D ====================

def train_comparative_expert_routing_2d_with_weights():
    """Main training function for 2D comparative expert routing with weight tracking"""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Parameters
    n_samples = 1000
    grid_size = 64
    batch_size = 8  # Smaller batch size for 2D
    n_epochs = 500
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("2D EXPERT ROUTING WITH WEIGHT TRACKING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Number of samples: {n_samples}")
    print("="*60)
    
    # Create dataset
    print("Creating enhanced 2D Burgers dataset...")
    dataset = EnhancedBurgersDataset2D(
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
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Create models
    print("Creating 2D models...")
    models = {
        'fno_2d': EnhancedFNOBaseline2D(modes=16, width=64, grid_size=grid_size),
        'softmax_2d': EnhancedExpertRoutingSystem2D(
            routing_method='softmax',
            n_experts=4,
            grid_size=grid_size,
            hidden_dim=256,
            device=device
        ),
        'lagrangian_single_2d': EnhancedExpertRoutingSystem2D(
            routing_method='lagrangian_single',
            n_experts=4,
            grid_size=grid_size,
            hidden_dim=256,
            device=device
        ),
        'lagrangian_two_2d': EnhancedExpertRoutingSystem2D(
            routing_method='lagrangian_two',
            n_experts=4,
            grid_size=grid_size,
            hidden_dim=256,
            device=device
        ),
        'admm_2d': EnhancedExpertRoutingSystem2D(
            routing_method='admm',
            n_experts=4,
            grid_size=grid_size,
            hidden_dim=256,
            device=device
        )
    }
    
    # Learning rates
    learning_rates = {
        'fno_2d': 1e-3,
        'softmax_2d': 1e-3,
        'lagrangian_single_2d': 1e-3,
        'lagrangian_two_2d_theta': 1e-3,
        'lagrangian_two_2d_lambda': 5e-4,
        'admm_2d': 1e-3
    }
    
    # Create trainer
    trainer = EnhancedExpertRoutingComparativeTrainer2D(
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
            print(f"{trainer.method_labels[name]:30s} | "
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
                if name != 'fno_2d' and name in trainer.weight_trackers:
                    checkpoint['weight_tracker'] = {
                        'weight_means': trainer.weight_trackers[name].weight_means,
                        'weight_entropies': trainer.weight_trackers[name].weight_entropies,
                        'weight_sparsity': trainer.weight_trackers[name].weight_sparsity
                    }
                
                torch.save(checkpoint, f'expert_routing_2d_results/best_{name}_model.pth')
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} completed.")
        print(f"Best validation losses so far:")
        for name in models.keys():
            print(f"  {trainer.method_labels[name]}: {best_val_loss[name]:.4e}")
    
    # Final evaluation and summary
    print("\n" + "="*60)
    print("2D FINAL RESULTS AND WEIGHT ANALYSIS")
    print("="*60)
    
    # Load best models and evaluate
    for name in models.keys():
        checkpoint = torch.load(f'expert_routing_2d_results/best_{name}_model.pth', 
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
                
                if name == 'fno_2d':
                    predictions = trainer.models[name](u0)
                    loss = huber_loss(predictions, u_solution).mean()
                else:
                    predictions, _ = trainer.models[name](u0)
                    loss = huber_loss(predictions, u_solution).mean()
                
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        print(f"\n{trainer.method_labels[name]:30s}:")
        print(f"  Best Val Loss: {checkpoint['val_loss']:.4e}")
        print(f"  Final Val Loss: {avg_val_loss:.4e}")
        
        # Print final weight statistics for routing methods
        if name != 'fno_2d' and name in trainer.weight_trackers:
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
    
    torch.save(final_results, 'expert_routing_2d_results/final_results_detailed.pth')
    
    # Print comprehensive weight analysis
    print("\n" + "="*60)
    print("2D COMPREHENSIVE WEIGHT ANALYSIS")
    print("="*60)
    
    for name in ['softmax_2d', 'lagrangian_single_2d', 'lagrangian_two_2d', 'admm_2d']:
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
    print("2D Training completed successfully!")
    print("Results saved in 'expert_routing_2d_results/' directory")
    print("="*60)
    
    return trainer, final_results

# ==================== Weight Analysis Functions for 2D ====================

def analyze_saved_weights_2d(checkpoint_path: str):
    """Analyze weight evolution from saved 2D checkpoint"""
    results = torch.load(checkpoint_path, map_location='cpu')
    
    if 'weight_trackers' not in results:
        print("No weight tracker data found in checkpoint")
        return results
    
    print("="*60)
    print("2D WEIGHT EVOLUTION ANALYSIS")
    print("="*60)
    
    for method_name, tracker_data in results['weight_trackers'].items():
        print(f"\nMethod: {method_name}")
        print("-" * 40)
        
        weight_means = tracker_data['weight_means']
        weight_entropies = tracker_data['weight_entropies']
        expert_names = tracker_data.get('expert_names', [f'Expert_{i}' for i in range(len(weight_means[0]))])
        
        if not weight_means:
            continue
        
        # Analyze trend
        initial_means = np.array(weight_means[0])
        final_means = np.array(weight_means[-1])
        
        print(f"Expert Weights Evolution:")
        for i in range(len(initial_means)):
            diff = final_means[i] - initial_means[i]
            trend = "↑" if diff > 0.1 else "↓" if diff < -0.1 else "→"
            print(f"  {expert_names[i]}: {initial_means[i]:.3f} → {final_means[i]:.3f} {trend}")
        
        # Entropy analysis
        initial_entropy = weight_entropies[0]
        final_entropy = weight_entropies[-1]
        entropy_change = final_entropy - initial_entropy
        
        print(f"\nEntropy Analysis:")
        print(f"  Initial: {initial_entropy:.3f}")
        print(f"  Final:   {final_entropy:.3f}")
        print(f"  Change:  {entropy_change:+.3f}")
        
        if entropy_change < -0.1:
            print("  → Entropy decreased: More specialized!")
        elif entropy_change > 0.1:
            print("  → Entropy increased: More balanced!")
        else:
            print("  → Entropy stable: Consistent behavior")
        
        # Dominant expert analysis
        if 'dominant_expert_counts' in tracker_data and tracker_data['dominant_expert_counts']:
            initial_counts = tracker_data['dominant_expert_counts'][0]
            final_counts = tracker_data['dominant_expert_counts'][-1]
            
            print(f"\nDominant Expert Analysis:")
            print(f"  Initial distribution: {initial_counts}")
            print(f"  Final distribution:   {final_counts}")
            
            if len(final_counts) > 0:
                dominant_idx = np.argmax(final_counts)
                print(f"  Most dominant expert: {expert_names[dominant_idx]} ({final_counts[dominant_idx]} samples)")
    
    return results

def print_weight_history_2d(results, method_name: str, expert_idx: Optional[int] = None):
    """Print detailed weight history for a specific 2D method"""
    if 'weight_trackers' not in results or method_name not in results['weight_trackers']:
        print(f"No weight data found for method: {method_name}")
        return
    
    tracker_data = results['weight_trackers'][method_name]
    weight_means = tracker_data['weight_means']
    weight_stds = tracker_data['weight_stds']
    expert_names = tracker_data.get('expert_names', [f'Expert_{i}' for i in range(len(weight_means[0]))])
    
    print(f"\n{'='*60}")
    print(f"2D WEIGHT HISTORY FOR {method_name.upper()}")
    print(f"{'='*60}")
    
    if expert_idx is not None:
        # Print history for specific expert
        print(f"\nExpert: {expert_names[expert_idx]}")
        print("-" * 40)
        for epoch, means in enumerate(weight_means):
            if epoch % 10 == 0:  # Print every 10 epochs
                std = weight_stds[epoch][expert_idx]
                confidence = 1.0 - (std / (means[expert_idx] + 1e-10))
                print(f"Epoch {epoch:3d}: {means[expert_idx]:.4f} ± {std:.4f} (conf: {confidence:.3f})")
    else:
        # Print summary for all experts
        print("\nFinal Weights:")
        print("-" * 40)
        for i, name in enumerate(expert_names):
            final_mean = weight_means[-1][i]
            final_std = weight_stds[-1][i]
            confidence = 1.0 - (final_std / (final_mean + 1e-10))
            print(f"{name:12s}: {final_mean:.4f} ± {final_std:.4f} (conf: {confidence:.3f})")
        
        # Print evolution
        print(f"\nWeight Evolution (Epoch 0 → {len(weight_means)-1}):")
        print("-" * 40)
        for i, name in enumerate(expert_names):
            initial = weight_means[0][i]
            final = weight_means[-1][i]
            change = final - initial
            pct_change = (change / (initial + 1e-10)) * 100
            print(f"{name:12s}: {initial:.4f} → {final:.4f} ({change:+.4f}, {pct_change:+.1f}%)")

# ==================== Main Execution ====================

if __name__ == "__main__":
    # Create results directory
    os.makedirs('expert_routing_2d_results', exist_ok=True)
    
    # Run training with weight tracking
    trainer, final_results = train_comparative_expert_routing_2d_with_weights()
    
    # After training, analyze the weights
    print("\n" + "="*60)
    print("POST-TRAINING 2D WEIGHT ANALYSIS")
    print("="*60)
    
    # Analyze saved results
    if os.path.exists('expert_routing_2d_results/final_results_detailed.pth'):
        results = analyze_saved_weights_2d('expert_routing_2d_results/final_results_detailed.pth')
        
        # Print detailed weight histories
        for method in ['softmax_2d', 'lagrangian_single_2d', 'lagrangian_two_2d', 'admm_2d']:
            print_weight_history_2d(results, method)
        
        # Compare methods
        print(f"\n{'='*60}")
        print("2D METHOD COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        routing_methods = ['softmax_2d', 'lagrangian_single_2d', 'lagrangian_two_2d', 'admm_2d']
        for method in routing_methods:
            if method in results['weight_trackers']:
                tracker_data = results['weight_trackers'][method]
                weight_means = tracker_data['weight_means']
                weight_entropies = tracker_data['weight_entropies']
                
                if weight_means:
                    final_means = weight_means[-1]
                    specialization = (np.max(final_means) - np.min(final_means)) / (np.max(final_means) + 1e-10)
                    
                    print(f"\n{method.upper()}:")
                    print(f"  Final Entropy: {weight_entropies[-1]:.4f}")
                    print(f"  Specialization: {specialization:.4f}")
                    print(f"  Most Used Expert: {np.argmax(final_means)} ({final_means[np.argmax(final_means)]:.4f})")
    
    # Print final summary
    print("\n" + "="*60)
    print("2D FINAL TRAINING SUMMARY")
    print("="*60)
    
    # Print best validation losses
    print("\nBest Validation Losses Across All 2D Methods:")
    print("-" * 40)
    
    # Get best performance for each method
    for name in trainer.models.keys():
        if name in trainer.metrics and 'loss' in trainer.metrics[name]:
            best_train_loss = min(trainer.metrics[name]['loss'])
            best_val_loss = min(trainer.val_metrics[name].get('loss', [float('inf')]))
            
            print(f"{trainer.method_labels[name]:30s}:")
            print(f"  Best Train Loss: {best_train_loss:.4e}")
            print(f"  Best Val Loss:   {best_val_loss:.4e}")
            
            if name in ['softmax_2d', 'lagrangian_single_2d', 'lagrangian_two_2d', 'admm_2d']:
                # Print weight statistics for routing methods
                print(f"  Final Weight Distribution:")
                if name in trainer.weight_trackers:
                    tracker = trainer.weight_trackers[name]
                    if tracker.weight_means:
                        final_weights = tracker.weight_means[-1]
                        for i, exp_name in enumerate(tracker.expert_names):
                            print(f"    {exp_name}: {final_weights[i]:.4f}")
    
    print("\n" + "="*60)
    print("2D Expert Routing Training Complete!")
    print("="*60)