import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import os


class NavierStokes1DDataset(Dataset):
   def __init__(self, n_samples: int, input_dim: int, n_sources: int):
       self.n_samples = n_samples
       self.input_dim = input_dim
       self.n_sources = n_sources
      
       # Physical parameters
       self.dt = 0.001
       self.dx = 2.0 / input_dim
       self.Re = 100
       self.nu = 1.0 / self.Re
      
       # Generate input data (initial conditions)
       self.x = torch.zeros(n_samples, input_dim)
       for i in range(n_samples):
           self.x[i] = self.generate_initial_condition()
      
       # Generate solution data
       self.y = torch.zeros(n_samples, input_dim)
       for i in range(n_samples):
           # Solve Burgers equation for each initial condition
           solution = self.solve_burgers(self.x[i])
           # Take final state as target
           self.y[i] = solution[-1]
      
       # Add noise
       self.y += 0.1 * torch.randn_like(self.y)
  
   def generate_initial_condition(self) -> torch.Tensor:
       """Generate random initial condition"""
       x = torch.linspace(-1, 1, self.input_dim)
      
       # Different types of initial conditions
       condition_type = torch.randint(0, 3, (1,)).item()
      
       if condition_type == 0:
           # Sine wave
           u = torch.sin(np.pi * x)
       elif condition_type == 1:
           # Gaussian pulse
           u = torch.exp(-10 * x**2)
       else:
           # Smoothed step function
           u = torch.zeros_like(x)
           u[x < 0] = 1.0
           u[x >= 0] = -1.0
           # Smooth the discontinuity
           u = F.conv1d(u.view(1, 1, -1),
                       torch.ones(1, 1, 5)/5,
                       padding=2).squeeze()
      
       return u
  
   def solve_burgers(self, u: torch.Tensor, n_steps: int = 100) -> torch.Tensor:
       """Solve 1D Burgers equation"""
       solutions = [u]
       u_current = u.clone()
      
       for _ in range(n_steps):
           # Compute spatial derivatives
           du_dx = torch.zeros_like(u_current)
           du_dx[1:-1] = (u_current[2:] - u_current[:-2]) / (2 * self.dx)
           du_dx[0] = (u_current[1] - u_current[0]) / self.dx
           du_dx[-1] = (u_current[-1] - u_current[-2]) / self.dx
          
           d2u_dx2 = torch.zeros_like(u_current)
           d2u_dx2[1:-1] = (u_current[2:] - 2*u_current[1:-1] + u_current[:-2]) / (self.dx**2)
           d2u_dx2[0] = (u_current[2] - 2*u_current[1] + u_current[0]) / (self.dx**2)
           d2u_dx2[-1] = (u_current[-1] - 2*u_current[-2] + u_current[-3]) / (self.dx**2)
          
           # Update using Burgers equation
           u_current = u_current + self.dt * (
               -u_current * du_dx +  # Advection term
               self.nu * d2u_dx2    # Diffusion term
           )
          
           # Apply boundary conditions
           u_current[0] = u_current[-2]
           u_current[-1] = u_current[1]
          
           solutions.append(u_current.clone())
      
       return torch.stack(solutions)
  
   def __len__(self):
       return self.n_samples
  
   def __getitem__(self, idx):
       return {
           'x': self.x[idx],
           'y': self.y[idx]
       }




# 1. Data Generation and Dataset
class MultiSourceDataset(Dataset):
   def __init__(self, n_samples: int, input_dim: int, n_sources: int):
       self.n_samples = n_samples
       self.input_dim = input_dim
       self.n_sources = n_sources
      
       # Generate synthetic data
       self.x = torch.randn(n_samples, input_dim)
      
       # Generate ground truth with different patterns
       self.y = torch.zeros(n_samples, input_dim)
       for i in range(n_samples):
           if i % 3 == 0:
               self.y[i] = torch.sin(self.x[i])  # Smooth pattern
           elif i % 3 == 1:
               self.y[i] = torch.sign(self.x[i])  # Discontinuous pattern
           else:
               self.y[i] = self.x[i]**2  # Nonlinear pattern
              
       # Add noise
       self.y += 0.1 * torch.randn_like(self.y)
  
   def __len__(self):
       return self.n_samples
  
   def __getitem__(self, idx):
       return {
           'x': self.x[idx],
           'y': self.y[idx]
       }


# 2. Model Implementation
class LargeScaleSourceIntegration(nn.Module):
   def __init__(
       self,
       n_sources: int,
       input_dim: int,
       hidden_dim: int = 128,
       sparse_topk: int = 10
   ):
       super().__init__()
       self.n_sources = n_sources
       self.sparse_topk = sparse_topk
      
       # Source-specific transforms
       self.source_transforms = nn.ModuleList([
           nn.Sequential(
               nn.Linear(input_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, input_dim)
           ) for _ in range(n_sources)
       ])
      
       # Weight network
       self.weight_network = nn.Sequential(
           nn.Linear(input_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, n_sources)
       )
      
       # Source confidence networks
       self.confidence_nets = nn.ModuleList([
           nn.Sequential(
               nn.Linear(input_dim, 32),
               nn.ReLU(),
               nn.Linear(32, 1),
               nn.Sigmoid()
           ) for _ in range(n_sources)
       ])
      
   def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
       batch_size = x.shape[0]
      
       # Compute weights
       logits = self.weight_network(x)
       weights = F.softmax(logits, dim=-1)
      
       # Get top-k sources
       topk_values, topk_indices = torch.topk(weights, k=self.sparse_topk, dim=-1)
      
       # Process only top-k sources
       selected_outputs = []
       selected_confidences = []
       selected_weights = []
      
       for b in range(batch_size):
           batch_outputs = []
           batch_confidences = []
           batch_weights = []
          
           for idx in topk_indices[b]:
               output = self.source_transforms[idx](x[b:b+1])
               confidence = self.confidence_nets[idx](output)
               weight = weights[b:b+1, idx:idx+1]
              
               batch_outputs.append(output)
               batch_confidences.append(confidence)
               batch_weights.append(weight)
          
           selected_outputs.append(torch.cat(batch_outputs, dim=0))
           selected_confidences.append(torch.cat(batch_confidences, dim=0))
           selected_weights.append(torch.cat(batch_weights, dim=0))
      
       selected_outputs = torch.stack(selected_outputs)  # [batch_size, topk, input_dim]
       selected_confidences = torch.stack(selected_confidences)  # [batch_size, topk, 1]
       selected_weights = torch.stack(selected_weights)  # [batch_size, topk, 1]
      
       # Combine outputs
       combined_weights = selected_weights * selected_confidences
       combined_weights = combined_weights / (combined_weights.sum(dim=1, keepdim=True) + 1e-6)
       output = (selected_outputs * combined_weights).sum(dim=1)
      
       return output, {
           'weights': weights,
           'confidences': selected_confidences.squeeze(-1),
           'sparsity': (weights > 0.01).float().mean()
       }




# 3. Trainer Implementation
class MultiSourceTrainer:
   def __init__(
       self,
       model: LargeScaleSourceIntegration,
       lr: float = 1e-3,
       device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
   ):
       self.model = model.to(device)
       self.device = device
       self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
       self.metrics = defaultdict(list)
      
   def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
       self.model.train()
       epoch_metrics = defaultdict(list)
      
       for batch in dataloader:
           metrics = self.train_step(batch)
           for k, v in metrics.items():
               epoch_metrics[k].append(v)
      
       return {k: np.mean(v) for k, v in epoch_metrics.items()}
  
   def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
       x = batch['x'].to(self.device)
       y = batch['y'].to(self.device)
      
       # Forward pass
       y_pred, metadata = self.model(x)
      
       # Compute losses
       recon_loss = F.mse_loss(y_pred, y)
       sparsity_loss = 0.1 * (1 - metadata['sparsity'])
      
       total_loss = recon_loss + sparsity_loss
      
       # Backward pass
       self.optimizer.zero_grad()
       total_loss.backward()
       self.optimizer.step()
      
       return {
           'loss': total_loss.item(),
           'recon_loss': recon_loss.item(),
           'sparsity': metadata['sparsity'].item()
       }


class LagrangianSourceIntegration(nn.Module):
   """Lagrangian formulation of multi-source integration"""
   def __init__(
       self,
       n_sources: int,
       input_dim: int,
       hidden_dim: int = 128,
       sparse_topk: int = 10,
       rho: float = 1.0
   ):
       super().__init__()
       self.n_sources = n_sources
       self.sparse_topk = sparse_topk
       self.rho = rho
      
       # Source-specific transforms
       self.source_transforms = nn.ModuleList([
           nn.Sequential(
               nn.Linear(input_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, input_dim)
           ) for _ in range(n_sources)
       ])
      
       # Initialize primal and dual variables
       self.lambda_weights = nn.Parameter(torch.ones(n_sources) / n_sources)
       self.mu = nn.Parameter(torch.zeros(1))
       self.nu = nn.Parameter(torch.zeros(n_sources))
      
       # Source confidence networks
       self.confidence_nets = nn.ModuleList([
           nn.Sequential(
               nn.Linear(input_dim, 32),
               nn.ReLU(),
               nn.Linear(32, 1),
               nn.Sigmoid()
           ) for _ in range(n_sources)
       ])
  
   def augmented_lagrangian_loss(
       self,
       x: torch.Tensor,
       y: torch.Tensor,
       outputs: torch.Tensor,
       confidences: torch.Tensor
   ) -> Tuple[torch.Tensor, Dict]:
       # Get normalized weights
       weights = self.lambda_weights #F.softmax(self.lambda_weights, dim=0)
      
       # Compute source-specific losses
       source_losses = []
       for i in range(self.n_sources):
           mse = F.mse_loss(outputs[:, i], y, reduction='none')
           weighted_mse = (mse * confidences[:, i]).mean()
           source_losses.append(weighted_mse)
      
       source_losses = torch.stack(source_losses)  # [n_sources]
      
       # Weighted reconstruction loss with temperature scaling
       temperature = 1.0
       weighted_loss = (weights * source_losses).sum()
      
       # Stronger constraint penalties
       g_lambda = (1 - weights.sum()).abs()
       h_lambda = torch.relu(-weights)
      
       # Adjusted penalty terms
       constraint_penalty = 100.0 * (g_lambda + h_lambda.sum())
      
       # L1 regularization for sparsity
       l1_reg = 0.1 * weights.abs().sum()
      
       # Entropy regularization
       entropy = -0.01 * (weights * torch.log(weights + 1e-6)).sum()
      
       # Combined loss
       loss = weighted_loss + constraint_penalty + l1_reg + entropy
      
       return loss, {
           'weighted_loss': weighted_loss,
           'g_lambda': g_lambda,
           'h_lambda': h_lambda,
           'source_losses': source_losses,
           'weights': weights,
           'reconstruction_loss': F.mse_loss(outputs.mean(dim=1), y)
       }




  
   def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
       batch_size = x.shape[0]
      
       # Process through source transforms
       outputs = []
       confidences = []
      
       for i in range(self.n_sources):
           output = self.source_transforms[i](x)
           confidence = self.confidence_nets[i](output)
           outputs.append(output)
           confidences.append(confidence)
      
       outputs = torch.stack(outputs, dim=1)  # [batch_size, n_sources, dim]
       confidences = torch.stack(confidences, dim=1)  # [batch_size, n_sources, 1]
      
       # Always normalize weights using softmax
       normalized_weights = self.lambda_weights #F.softmax(self.lambda_weights, dim=0)
       weights = normalized_weights.view(1, -1, 1)
      
       # Combine using normalized weights
       combined_weights = weights * confidences
       combined_weights = combined_weights / (combined_weights.sum(dim=1, keepdim=True) + 1e-6)
      
       output = (outputs * combined_weights).sum(dim=1)
      
       return output, {
           'weights': normalized_weights,  # Return normalized weights
           'outputs': outputs,
           'confidences': confidences
       }






class TwoTimeScaleOptimizer:
   """Two-time-scale optimizer with adaptive learning rates and gradient handling"""
   def __init__(
       self,
       model: LagrangianSourceIntegration,
       eta_theta: float = 1e-4,
       eta_lambda: float = 1e-3,
       clipgrad: float = 0.5,
       weight_decay: float = 1e-4,
       beta1: float = 0.9,
       beta2: float = 0.999
   ):
       self.model = model
       self.eta_theta = eta_theta
       self.eta_lambda = eta_lambda
       self.clipgrad = clipgrad
       self.epoch = 0
      
       # Network parameters optimizer with weight decay
       self.theta_optimizer = torch.optim.AdamW(
           [p for n, p in model.named_parameters()
            if not any(x in n for x in ['lambda_weights'])],
           lr=eta_theta,
           weight_decay=weight_decay,
           betas=(beta1, beta2)
       )
      
       # Weight parameters optimizer without weight decay
       self.lambda_optimizer = torch.optim.AdamW(
           [model.lambda_weights],
           lr=eta_lambda,
           weight_decay=0.0,
           betas=(beta1, beta2)
       )
      
       # Learning rate schedulers
       self.theta_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
           self.theta_optimizer,
           T_max=100,
           eta_min=1e-6
       )
      
       self.lambda_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
           self.lambda_optimizer,
           T_max=100,
           eta_min=1e-5
       )
      
       # Initialize momentum buffers for weight updates
       self.weight_momentum = torch.zeros_like(model.lambda_weights.data)
       self.weight_velocity = torch.zeros_like(model.lambda_weights.data)
      
   def zero_grad(self):
       """Zero all gradients"""
       self.theta_optimizer.zero_grad()
       self.lambda_optimizer.zero_grad()
  
   def step(self, loss_dict: Dict[str, torch.Tensor]):
       """Perform optimization step with two time scales and gradient handling"""
       # Compute gradients
       loss_dict['loss'].backward()
      
       # Clip gradients for stability
       torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
      
       # Update network parameters (faster timescale)
       self.theta_optimizer.step()
      
       # Update weight parameters (slower timescale) with momentum
       self.lambda_optimizer.step()
      
       with torch.no_grad():
           # Get current weights and apply temperature scaling
           weights = self.model.lambda_weights.data
           temperature = max(0.1, 1.0 * (0.95 ** self.epoch))
           weights = (weights / temperature).softmax(dim=0)
          
           # Add small noise to break symmetry
           noise = 0.01 * torch.randn_like(weights)
           weights = weights + noise
          
           # Project onto simplex and ensure minimum weight
           weights = self.project_simplex(weights)
           weights = torch.clamp(weights, min=0.01)
          
           # Apply momentum
           self.weight_momentum = 0.9 * self.weight_momentum + 0.1 * (weights - self.model.lambda_weights.data)
           weights = self.model.lambda_weights.data + self.weight_momentum
          
           # Final normalization
           weights = F.softmax(weights, dim=0)
           self.model.lambda_weights.data.copy_(weights)
      
       # Update learning rates
       self.theta_scheduler.step()
       self.lambda_scheduler.step()




   def get_lr(self) -> Dict[str, float]:
       """Get current learning rates"""
       return {
           'theta_lr': self.theta_optimizer.param_groups[0]['lr'],
           'lambda_lr': self.lambda_optimizer.param_groups[0]['lr']
       }
  
   def state_dict(self) -> Dict:
       """Get optimizer state"""
       return {
           'theta_optimizer': self.theta_optimizer.state_dict(),
           'lambda_optimizer': self.lambda_optimizer.state_dict(),
           'theta_scheduler': self.theta_scheduler.state_dict(),
           'lambda_scheduler': self.lambda_scheduler.state_dict(),
           'weight_momentum': self.weight_momentum,
           'weight_velocity': self.weight_velocity,
           'epoch': self.epoch
       }
  
   def load_state_dict(self, state_dict: Dict):
       """Load optimizer state"""
       self.theta_optimizer.load_state_dict(state_dict['theta_optimizer'])
       self.lambda_optimizer.load_state_dict(state_dict['lambda_optimizer'])
       self.theta_scheduler.load_state_dict(state_dict['theta_scheduler'])
       self.lambda_scheduler.load_state_dict(state_dict['lambda_scheduler'])
       self.weight_momentum = state_dict['weight_momentum']
       self.weight_velocity = state_dict['weight_velocity']
       self.epoch = state_dict['epoch']
  
   def update_epoch(self, epoch: int):
       """Update epoch counter"""
       self.epoch = epoch
  
   @staticmethod
   def project_simplex(v: torch.Tensor) -> torch.Tensor:
       """Project onto probability simplex using efficient algorithm"""
       v_sorted, _ = torch.sort(v, descending=True)
       cssv = torch.cumsum(v_sorted, dim=0)
       rho = (cssv - 1) / torch.arange(1, len(v) + 1, device=v.device)
       rho_star = rho[torch.where(v_sorted > rho)[0][-1]]
       return torch.maximum(v - rho_star, torch.zeros_like(v))
  
   def adjust_learning_rates(self, val_loss: float):
       """Adjust learning rates based on validation loss"""
       self.theta_scheduler.step(val_loss)
       self.lambda_scheduler.step(val_loss)
  
   def clip_and_scale_gradients(self):
       """Clip and scale gradients for better training stability"""
       # Clip network gradients
       torch.nn.utils.clip_grad_norm_(
           [p for p in self.model.parameters() if p.requires_grad],
           self.clipgrad
       )
      
       # Scale weight gradients based on magnitude
       if self.model.lambda_weights.grad is not None:
           grad_norm = self.model.lambda_weights.grad.norm()
           if grad_norm > self.clipgrad:
               self.model.lambda_weights.grad.mul_(self.clipgrad / grad_norm)






# 3. Comparative Trainer Implementation
class ComparativeTrainer:
   """Trainer for comparing Softmax and Lagrangian approaches"""
   def __init__(
       self,
       softmax_model: LargeScaleSourceIntegration,
       lagrangian_model: LagrangianSourceIntegration,
       lr_softmax: float = 1e-3,
       lr_theta: float = 1e-3,
       lr_lambda: float = 1e-6,
       device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
   ):
       self.softmax_model = softmax_model.to(device)
       self.lagrangian_model = lagrangian_model.to(device)
       self.device = device
      
       # Initialize optimizers
       self.softmax_optimizer = torch.optim.Adam(
           softmax_model.parameters(),
           lr=lr_softmax
       )
       self.lagrangian_optimizer = TwoTimeScaleOptimizer(
           lagrangian_model,
           eta_theta=lr_theta,
           eta_lambda=lr_lambda
       )
      
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
           # Train both models
           softmax_metrics = self.train_step_softmax(batch)
           lagrangian_metrics = self.train_step_lagrangian(batch)
          
           # Store metrics
           for k, v in softmax_metrics.items():
               epoch_metrics['softmax'][k].append(v)
           for k, v in lagrangian_metrics.items():
               epoch_metrics['lagrangian'][k].append(v)
      
       return {
           'softmax': {k: np.mean(v) for k, v in epoch_metrics['softmax'].items()},
           'lagrangian': {k: np.mean(v) for k, v in epoch_metrics['lagrangian'].items()}
       }
  
   def train_step_softmax(
       self,
       batch: Dict[str, torch.Tensor]
   ) -> Dict[str, float]:
       x = batch['x'].to(self.device)
       y = batch['y'].to(self.device)
      
       # Forward pass
       y_pred, metadata = self.softmax_model(x)
       mse_loss = F.mse_loss(y_pred, y)
      
       # Compute losses
       recon_loss = F.mse_loss(y_pred, y)
       sparsity_loss = 0.1 * (1 - metadata['sparsity'])
       total_loss = recon_loss + sparsity_loss
      
       # Backward pass
       self.softmax_optimizer.zero_grad()
       total_loss.backward()
       self.softmax_optimizer.step()
      
       return {
           'loss': total_loss.item(),
           'mse': mse_loss.item(),
           'recon_loss': recon_loss.item(),
           'sparsity': metadata['sparsity'].item(),
           'weight_entropy': -(metadata['weights'] *
                             torch.log(metadata['weights'] + 1e-6)).sum(dim=-1).mean().item()
       }
  
   def train_step_lagrangian(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
       x = batch['x'].to(self.device)
       y = batch['y'].to(self.device)
      
       # Zero gradients
       self.lagrangian_optimizer.zero_grad()
      
       # Forward pass
       y_pred, metadata = self.lagrangian_model(x)


  
       mse_loss = F.mse_loss(y_pred, y)
      
       # Compute Lagrangian loss
       lagrangian_loss, loss_dict = self.lagrangian_model.augmented_lagrangian_loss(
           x, y, metadata['outputs'], metadata['confidences']
       )
      
       # Optimizer step
       loss_dict['loss'] = lagrangian_loss
       self.lagrangian_optimizer.step(loss_dict)
      
       return {
           'loss': lagrangian_loss.item(),
           'mse': mse_loss.item(),
           'weighted_loss': loss_dict['weighted_loss'].item(),
           'constraint_violation': loss_dict['g_lambda'].abs().item(),
           'min_weight': metadata['weights'].min().item(),
           'max_weight': metadata['weights'].max().item()
       }


  
   def validate(
       self,
       dataloader: DataLoader
   ) -> Dict[str, Dict[str, float]]:
       self.softmax_model.eval()
       self.lagrangian_model.eval()
      
       val_metrics = {
           'softmax': defaultdict(list),
           'lagrangian': defaultdict(list),
           'mse': defaultdict(list)
       }
      
       with torch.no_grad():
           for batch in dataloader:
               x = batch['x'].to(self.device)
               y = batch['y'].to(self.device)
              
               # Evaluate Softmax model
               y_pred_soft, meta_soft = self.softmax_model(x)
               soft_loss = F.mse_loss(y_pred_soft, y)
               val_metrics['softmax']['loss'].append(soft_loss.item())


               val_metrics['softmax']['mse'].append(soft_loss.item())
              
               # Evaluate Lagrangian model
               y_pred_lag, meta_lag = self.lagrangian_model(x)
               lag_loss = F.mse_loss(y_pred_lag, y)
               val_metrics['lagrangian']['loss'].append(lag_loss.item())
               val_metrics['lagrangian']['mse'].append(lag_loss.item())
      
       return {
           'softmax': {k: np.mean(v) for k, v in val_metrics['softmax'].items()},
           'lagrangian': {k: np.mean(v) for k, v in val_metrics['lagrangian'].items()},
           'mse': {k: np.mean(v) for k, v in val_metrics['mse'].items()},
       }




def plot_predictions(
   models: Dict[str, nn.Module],
   test_loader: DataLoader,
   device: str,
   save_path: str,
   epoch: int,
   num_samples: int = 4
):
   """Plot predictions from both models on same plots for direct comparison"""
   fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
   if num_samples == 1:
       axes = [axes]
  
   # Get a batch of test data
   batch = next(iter(test_loader))
   x = batch['x'].to(device)
   y = batch['y'].to(device)
  
   # Use same indices for all plots
   indices = np.random.choice(len(x), num_samples, replace=False)
  
   # Color scheme
   colors = {
       'Initial': 'gray',
       'True': 'black',
       'Softmax': 'blue',
       'Lagrangian': 'red'
   }
  
   # Generate predictions
   predictions = {}
   for model_name, model in models.items():
       model.eval()
       with torch.no_grad():
           pred, metadata = model(x)
           predictions[model_name] = {
               'pred': pred.detach().cpu(),
               'metadata': {
                   k: v.detach().cpu() if torch.is_tensor(v) else v
                   for k, v in metadata.items()
               }
           }
  
   # Plot samples
   for i, idx in enumerate(indices):
       ax = axes[i]
      
       # Plot initial condition
       ax.plot(range(len(x[idx])), x[idx].cpu(), '-',
              color=colors['Initial'], label='Initial', alpha=0.5)
      
       # Plot true solution
       ax.plot(range(len(y[idx])), y[idx].cpu(), '-',
              color=colors['True'], label='True', alpha=0.7)
      
       # Plot predictions from both models
       for model_name, pred_dict in predictions.items():
           pred = pred_dict['pred'][idx]
           ax.plot(range(len(pred)), pred, '--',
                  color=colors[model_name],
                  label=f'{model_name}',
                  alpha=0.7)
          
           # Add confidence bands if available
           if 'confidences' in pred_dict['metadata']:
               conf = pred_dict['metadata']['confidences']
               if isinstance(conf, torch.Tensor):
                   # Handle different confidence shapes
                   if len(conf.shape) == 3:  # [batch, sources, 1]
                       conf = conf[idx].mean(dim=0)  # Average over sources
                   elif len(conf.shape) == 2:  # [batch, 1]
                       conf = conf[idx]
                  
                   # Expand confidence to match prediction size if needed
                   if conf.shape[-1] == 1:
                       conf = conf.expand(pred.shape[0])
                  
                   conf = conf.numpy()
                   pred_np = pred.numpy()
                  
                   # Ensure shapes match
                   if conf.shape == pred_np.shape:
                       ax.fill_between(range(len(pred)),
                                     pred_np - conf,
                                     pred_np + conf,
                                     color=colors[model_name],
                                     alpha=0.1,
                                     label=f'{model_name} Confidence')
      
       ax.set_title(f'Sample {i+1}')
       ax.set_xlabel('Spatial Position')
       ax.set_ylabel('Loss')
       ax.legend(loc='upper right')
       ax.grid(True)
  
   plt.tight_layout()
   os.makedirs(save_path, exist_ok=True)
   plt.savefig(f'{save_path}/predictions_epoch_{epoch}.png')
   plt.close()








def plot_solver_outputs(
   models: Dict[str, nn.Module],
   test_loader: DataLoader,
   device: str,
   save_path: str,
   epoch: int,
   num_samples: int = 4
):
   """Plot solver outputs and predictions for both models on same plots"""
   fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
   if num_samples == 1:
       axes = [axes]
  
   # Get a batch of test data
   batch = next(iter(test_loader))
   x = batch['x'].to(device)  # Initial conditions
   y = batch['y'].to(device)  # True solutions
  
   # Generate predictions from both models
   predictions = {}
   for model_name, model in models.items():
       model.eval()
       with torch.no_grad():
           pred, metadata = model(x)
           predictions[model_name] = {
               'output': pred,
               'metadata': metadata
           }
  
   # Use same indices for all plots
   sample_indices = np.random.choice(len(x), num_samples, replace=False)
  
   # Color scheme
   colors = {
       'initial': 'gray',
       'true': 'black',
       'Softmax': 'blue',
       'Lagrangian': 'red'
   }
  
   # Plot samples
   for i, idx in enumerate(sample_indices):
       ax = axes[i]
      
       # Plot initial condition
       ax.plot(range(len(x[idx])), x[idx].cpu(), '-',
              color=colors['initial'], label='Initial Condition', alpha=0.5)
      
       # Plot true solution
       ax.plot(range(len(y[idx])), y[idx].cpu(), '-',
              color=colors['true'], label='True Solution', alpha=0.7)
      
       # Plot predictions from both models
       for model_name, pred_dict in predictions.items():
           ax.plot(range(len(pred_dict['output'][idx])),
                  pred_dict['output'][idx].cpu(), '--',
                  color=colors[model_name],
                  label=f'{model_name} Prediction',
                  alpha=0.7)
          
           # Add confidence bands if available
           if 'confidences' in pred_dict['metadata']:
               conf = pred_dict['metadata']['confidences'][idx]
               if isinstance(conf, torch.Tensor):
                   conf = conf.cpu().numpy()
                   y_pred = pred_dict['output'][idx].cpu().numpy()
                   ax.fill_between(range(len(y_pred)),
                                 y_pred - conf,
                                 y_pred + conf,
                                 color=colors[model_name],
                                 alpha=0.1,
                                 label=f'{model_name} Confidence')
      
       # Add weights information
       weight_info = []
       for model_name, pred_dict in predictions.items():
           if 'weights' in pred_dict['metadata']:
               weights = pred_dict['metadata']['weights']
               if isinstance(weights, torch.Tensor):
                   weights = weights.detach().cpu().numpy()
                   top_weights = np.sort(weights)[-3:]
                   weight_info.append(f'{model_name} top weights: {top_weights:.3f}')
      
       if weight_info:
           ax.set_title(f'Sample {i+1}\n' + '\n'.join(weight_info))
       else:
           ax.set_title(f'Sample {i+1}')
      
       ax.set_xlabel('Spatial Position')
       ax.set_ylabel('Value')
       ax.legend(loc='upper right')
       ax.grid(True, alpha=0.3)
  
   plt.tight_layout()
   plt.savefig(f'{save_path}/solver_outputs_epoch_{epoch}.png')
   plt.close()


def plot_error_evolution(
   models: Dict[str, nn.Module],
   test_loader: DataLoader,
   device: str,
   save_path: str,
   epoch: int
):
   """Plot error evolution over the spatial domain for both models"""
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
  
   batch = next(iter(test_loader))
   x = batch['x'].to(device)
   y = batch['y'].to(device)
  
   colors = {'Softmax': 'blue', 'Lagrangian': 'red'}
  
   # Plot spatial error distribution
   for model_name, model in models.items():
       model.eval()
       with torch.no_grad():
           pred, metadata = model(x)
          
           # Compute pointwise error
           error = torch.abs(pred - y)
           mean_error = error.mean(dim=0)
           std_error = error.std(dim=0)
          
           # Plot mean error with confidence bands
           ax1.plot(range(len(mean_error)),
                   mean_error.cpu(),
                   '-',
                   color=colors[model_name],
                   label=f'{model_name} Mean Error')
           ax1.fill_between(range(len(mean_error)),
                          (mean_error - std_error).cpu(),
                          (mean_error + std_error).cpu(),
                          alpha=0.2,
                          color=colors[model_name])
          
           # Plot error histogram
           ax2.hist(error.cpu().numpy().flatten(),
                   bins=50,
                   alpha=0.5,
                   color=colors[model_name],
                   label=f'{model_name} Error Distribution')
  
   ax1.set_title('Spatial Error Distribution')
   ax1.set_xlabel('Spatial Position')
   ax1.set_ylabel('Mean Absolute Error')
   ax1.grid(True)
   ax1.legend()
  
   ax2.set_title('Error Histogram')
   ax2.set_xlabel('Absolute Error')
   ax2.set_ylabel('Count')
   ax2.grid(True)
   ax2.legend()
  
   plt.tight_layout()
   plt.savefig(f'{save_path}/error_evolution_epoch_{epoch}.png')
   plt.close()


# Add a new function to plot weight evolution
def plot_weight_evolution(
   trainer: ComparativeTrainer,
   save_path: str,
   epoch: int
):
   """Plot evolution of weights for both models"""
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
  
   # Plot Softmax weights
   if hasattr(trainer.softmax_model, 'weight_network'):
       weights = trainer.softmax_model.weight_network[-1].weight.data.cpu().numpy()
       ax1.hist(weights.flatten(), bins=50, alpha=0.7, color='blue')
       ax1.set_title('Softmax Model Weight Distribution')
       ax1.set_xlabel('Weight Value')
       ax1.set_ylabel('Count')
       ax1.grid(True)
  
   # Plot Lagrangian weights
   if hasattr(trainer.lagrangian_model, 'lambda_weights'):
       weights = trainer.softmax_model.weight_network[-1].weight.detach().cpu().numpy()
       ax2.hist(weights.flatten(), bins=50, alpha=0.7, color='red')
       ax2.set_title('Lagrangian Model Weight Distribution')
       ax2.set_xlabel('Weight Value')
       ax2.set_ylabel('Count')
       ax2.grid(True)
  
   plt.tight_layout()
   plt.savefig(f'{save_path}/weight_evolution_epoch_{epoch}.png')
   plt.close()






def plot_error_distribution(
   models: Dict[str, nn.Module],
   test_loader: DataLoader,
   device: str,
   save_path: str,
   epoch: int
):
   os.makedirs(save_path, exist_ok=True)   
   """Plot error distribution for both models"""
   errors = {name: [] for name in models.keys()}
  
   for model_name, model in models.items():
       model.eval()
       with torch.no_grad():
           for batch in test_loader:
               x = batch['x'].to(device)
               y = batch['y'].to(device)
               y_pred, _ = model(x)
               error = torch.abs(y_pred - y).mean(dim=1)
               errors[model_name].extend(error.cpu().numpy())
  
   plt.figure(figsize=(10, 6))
   for model_name, error_list in errors.items():
       plt.hist(error_list, bins=50, alpha=0.5, label=model_name)
  
   plt.title('Error Distribution')
   plt.xlabel('Mean Absolute Error')
   plt.ylabel('Count')
   plt.legend()
   plt.grid(True)
   plt.savefig(f'{save_path}/error_dist_epoch_{epoch}.png')
   plt.close()




# 4. Enhanced Visualization
def plot_comparative_metrics(trainer: ComparativeTrainer, epoch: int):
   """Plot comparison between Softmax and Lagrangian approaches"""
   fig, axes = plt.subplots(2, 3, figsize=(18, 12))
  
   # Training Loss
   axes[0,0].plot(
       trainer.metrics['softmax']['loss'],
       label='Softmax'
   )
   axes[0,0].plot(
       trainer.metrics['lagrangian']['loss'],
       label='Lagrangian'
   )
   axes[0,0].set_title('Training Loss')
   axes[0,0].legend()
  
   # Weight Distribution
   axes[0,1].hist(
       trainer.softmax_model.weight_network[-1].weight.data.cpu().numpy().flatten(),
       alpha=0.5,
       label='Softmax',
       bins=30
   )
   axes[0,1].hist(
       trainer.lagrangian_model.lambda_weights.data.cpu().numpy(),
       alpha=0.5,
       label='Lagrangian',
       bins=30
   )
   axes[0,1].set_title('Weight Distribution')
   axes[0,1].legend()
  
   # Sparsity
   if 'sparsity' in trainer.metrics['softmax']:
       axes[0,2].plot(
           trainer.metrics['softmax']['sparsity'],
           label='Softmax'
       )
       if 'constraint_violation' in trainer.metrics['lagrangian']:
           axes[0,2].plot(
               [1 - v for v in trainer.metrics['lagrangian']['constraint_violation']],
               label='Lagrangian'
           )
       axes[0,2].set_title('Sparsity Pattern')
       axes[0,2].legend()
  
   # Reconstruction Error
   axes[1,0].plot(
       trainer.metrics['softmax']['recon_loss'],
       label='Softmax'
   )
   axes[1,0].plot(
       trainer.metrics['lagrangian']['weighted_loss'],
       label='Lagrangian'
   )
   axes[1,0].set_title('Reconstruction Error')
   axes[1,0].legend()
  
   # Constraint Violation
   axes[1,1].plot(
       trainer.metrics['lagrangian']['constraint_violation'],
       label='Simplex Constraint'
   )
   axes[1,1].set_yscale('log')
   axes[1,1].set_title('Constraint Violation (Lagrangian)')
  
   # Weight Range
   axes[1,2].fill_between(
       range(len(trainer.metrics['lagrangian']['min_weight'])),
       trainer.metrics['lagrangian']['min_weight'],
       trainer.metrics['lagrangian']['max_weight'],
       alpha=0.3,
       label='Lagrangian Weight Range'
   )
   axes[1,2].set_title('Weight Range Evolution')
  
   plt.tight_layout()
   plt.savefig(f'comparison_epoch_{epoch}.png')
   plt.close()


def plot_predictions_and_weights(
   models: Dict[str, nn.Module],
   test_loader: DataLoader,
   device: str,
   save_path: str,
   epoch: int,
   num_samples: int = 4,
   top_k: int = 10  # Number of top weights to show
):
   """Plot predictions and source weights for both models"""
   # Create a figure with two rows of subplots
   fig = plt.figure(figsize=(15, 5*num_samples))
   gs = plt.GridSpec(num_samples, 3, figure=fig)
  
   # Get a batch of test data
   batch = next(iter(test_loader))
   x = batch['x'].to(device)
   y = batch['y'].to(device)
  
   # Use same indices for all plots
   indices = np.random.choice(len(x), num_samples, replace=False)
  
   # Color scheme
   colors = {
       'Initial': 'gray',
       'True': 'black',
       'Softmax': 'blue',
       'Lagrangian': 'red'
   }
  
   # Generate predictions and get weights
   predictions = {}
   weights = {}
   for model_name, model in models.items():
       model.eval()
       with torch.no_grad():
           pred, metadata = model(x)
           predictions[model_name] = pred.detach().cpu()
          
           # Extract weights based on model type
           if model_name == 'Softmax':
               raw_weights = metadata['weights'].detach().cpu()
               # Get top-k weights
               if len(raw_weights.shape) > 1:
                   # If weights are per-sample
                   top_weights, _ = torch.topk(raw_weights, min(top_k, raw_weights.size(-1)), dim=-1)
               else:
                   # If weights are global
                   top_weights, _ = torch.topk(raw_weights, min(top_k, len(raw_weights)))
               weights[model_name] = top_weights
           else:  # Lagrangian
               raw_weights = model.lambda_weights.detach().cpu() #F.softmax(model.lambda_weights, dim=0).detach().cpu()
               # Get top-k weights
               top_weights, _ = torch.topk(raw_weights, min(top_k, len(raw_weights)))
               weights[model_name] = top_weights
  
   # Plot samples and weights
   for i, idx in enumerate(indices):
       # Prediction plot (left)
       ax_pred = fig.add_subplot(gs[i, 0:2])
      
       # Plot initial condition and true solution
       ax_pred.plot(range(len(x[idx])), x[idx].cpu(), '-',
                   color=colors['Initial'], label='Initial', alpha=0.5)
       ax_pred.plot(range(len(y[idx])), y[idx].cpu(), '-',
                   color=colors['True'], label='True', alpha=0.7)
      
       # Plot predictions from both models
       for model_name in predictions:
           ax_pred.plot(range(len(predictions[model_name][idx])),
                       predictions[model_name][idx],
                       '--',
                       color=colors[model_name],
                       label=f'{model_name}',
                       alpha=0.7)
      
       ax_pred.set_title(f'Sample {i+1} Predictions')
       ax_pred.set_xlabel('Spatial Position')
       ax_pred.set_ylabel('Value')
       ax_pred.legend(loc='upper right')
       ax_pred.grid(True)
      
       # Weights plot (right)
       ax_weights = fig.add_subplot(gs[i, 2])
      
       # Plot weights for both models
       width = 0.35
       x_pos = np.arange(top_k)
      
       for j, (model_name, model_weights) in enumerate(weights.items()):
           if len(model_weights.shape) > 1:
               # If weights are per-sample
               weights_to_plot = model_weights[idx]
           else:
               # If weights are global
               weights_to_plot = model_weights
          
           ax_weights.bar(x_pos + j*width,
                        weights_to_plot,
                        width,
                        label=model_name,
                        color=colors[model_name],
                        alpha=0.7)
      
       ax_weights.set_title(f'Top {top_k} Source Weights')
       ax_weights.set_xlabel('Source Index')
       ax_weights.set_ylabel('Weight Value')
       ax_weights.legend()
       ax_weights.grid(True, alpha=0.3)
  
   plt.tight_layout()
   os.makedirs(save_path, exist_ok=True)
   plt.savefig(f'{save_path}/predictions_and_weights_epoch_{epoch}.png')
   plt.close()




# 5. Main Training Script
def main():
   # Parameters
   n_samples = 1000
   input_dim = 64
   n_sources = 128
   batch_size = 32
   n_epochs = 200


   save_dir = 'results'
  
   # Create dataset
   #dataset = MultiSourceDataset(n_samples, input_dim, n_sources)
   # Create dataset
   dataset = NavierStokes1DDataset(n_samples, input_dim, n_epochs)


   train_size = int(0.8 * len(dataset))
   val_size = len(dataset) - train_size
   train_dataset, val_dataset = torch.utils.data.random_split(
       dataset, [train_size, val_size]
   )
  
   train_loader = DataLoader(
       train_dataset,
       batch_size=batch_size,
       shuffle=True
   )
   val_loader = DataLoader(
       val_dataset,
       batch_size=batch_size
   )
  
   # Create models
   softmax_model = LargeScaleSourceIntegration(
       n_sources=n_sources,
       input_dim=input_dim,
       hidden_dim=128,
       sparse_topk=10
   )
  
   lagrangian_model = LagrangianSourceIntegration(
       n_sources=n_sources,
       input_dim=input_dim,
       hidden_dim=128,
       sparse_topk=10,
       rho=1.0
   )
  
   # Create trainer
   trainer = ComparativeTrainer(
       softmax_model=softmax_model,
       lagrangian_model=lagrangian_model,
       lr_softmax=1e-3,
       lr_theta=1e-3,
       lr_lambda=1e-3
   )
  
   # Training loop
   for epoch in range(n_epochs):
       print(f"\nEpoch {epoch+1}/{n_epochs}")
      
       # Train
       metrics = trainer.train_epoch(train_loader)
      
       # Validate
       val_metrics = trainer.validate(val_loader)


      
       trainer.lagrangian_optimizer.adjust_learning_rates(val_metrics['lagrangian']['loss'])
      


       # Print metrics
       print("\nSoftmax Model:")
       for k, v in metrics['softmax'].items():
           print(f"{k}: {v:.4f}")
       print(f"Train MSE: {metrics['softmax']['mse']:.4f}")
       print(f"Validation MSE: {val_metrics['softmax']['mse']:.4f}")
      


       print(f"Validation loss: {val_metrics['softmax']['loss']:.4f}")
      
       print("\nLagrangian Model:")
       for k, v in metrics['lagrangian'].items():
           print(f"{k}: {v:.4f}")
       print(f"Train MSE: {metrics['lagrangian']['mse']:.4f}")
       print(f"Validation MSE: {val_metrics['lagrangian']['mse']:.4f}")
      
       print(f"Validation loss: {val_metrics['lagrangian']['loss']:.4f}")
      
       # Store metrics
       for k, v in metrics['softmax'].items():
           trainer.metrics['softmax'][k].append(v)
       for k, v in metrics['lagrangian'].items():
           trainer.metrics['lagrangian'][k].append(v)
      
       # Plot comparison
       if (epoch + 1) % 10 == 0:
           plot_comparative_metrics(trainer, epoch + 1)


           models = {
               'Softmax': softmax_model,
               'Lagrangian': lagrangian_model
           }
           plot_predictions(models, val_loader, trainer.device, save_dir, epoch + 1)
           plot_error_distribution(models, val_loader, trainer.device, save_dir, epoch + 1)
           plot_weight_evolution(trainer, save_dir, epoch + 1)
           plot_predictions_and_weights(
               models={'Softmax': softmax_model, 'Lagrangian': lagrangian_model},
               test_loader=val_loader,
               device=trainer.device,
               save_path=save_dir,
               epoch=epoch + 1
           )


   models = {
       'Softmax': softmax_model,
       'Lagrangian': lagrangian_model
   }
   plot_predictions(models, val_loader, trainer.device, save_dir, epoch + 1)
   plot_error_distribution(models, val_loader, trainer.device, save_dir, epoch + 1)
   plot_weight_evolution(trainer, save_dir, epoch + 1)
   plot_predictions_and_weights(
       models={'Softmax': softmax_model, 'Lagrangian': lagrangian_model},
       test_loader=val_loader,
       device=trainer.device,
       save_path=save_dir,
       epoch=epoch + 1
   )
if __name__ == "__main__":
   main()












