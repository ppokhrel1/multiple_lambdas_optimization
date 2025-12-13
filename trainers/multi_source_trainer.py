import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from collections import defaultdict

class LargeScaleSourceIntegration(nn.Module):
    def __init__(self, n_sources: int, input_dim: int, hidden_dim: int = 128, sparse_topk: int = 10):
        super().__init__()
        self.n_sources = n_sources
        self.sparse_topk = sparse_topk
        
        # Better initialization for source transforms
        self.source_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(n_sources)
        ])
        
        # Add proper initialization
        for transform in self.source_transforms:
            for layer in transform:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        self.weight_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(hidden_dim, n_sources)
        )
        
        nn.init.xavier_uniform_(self.weight_network[0].weight)
        nn.init.xavier_uniform_(self.weight_network[3].weight)
        
        self.confidence_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for _ in range(n_sources)
        ])
        
    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        
        logits = self.weight_network(x)
        weights = F.softmax(logits, dim=-1)
        
        # Use more aggressive sparsity
        topk_values, topk_indices = torch.topk(weights, k=min(self.sparse_topk, self.n_sources), dim=-1)
        
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
            
            if batch_outputs:  # Check if we have any outputs for this batch
                selected_outputs.append(torch.cat(batch_outputs, dim=0))
                selected_confidences.append(torch.cat(batch_confidences, dim=0))
                selected_weights.append(torch.cat(batch_weights, dim=0))
        
        # FIX: Check if selected_outputs list is empty (not the tensor)
        if len(selected_outputs) > 0:
            selected_outputs = torch.stack(selected_outputs)
            selected_confidences = torch.stack(selected_confidences)
            selected_weights = torch.stack(selected_weights)
            
            combined_weights = selected_weights * selected_confidences
            combined_weights = combined_weights / (combined_weights.sum(dim=1, keepdim=True) + 1e-8)
            output = (selected_outputs * combined_weights).sum(dim=1)
            
            return output, {
                'weights': weights,
                'confidences': selected_confidences.squeeze(-1),
                'sparsity': (weights > 0.01).float().mean()
            }
        else:
            # Fallback: average of all sources
            all_outputs = []
            for i in range(self.n_sources):
                output = self.source_transforms[i](x)
                all_outputs.append(output)
            output = torch.stack(all_outputs).mean(dim=0)
            
            return output, {
                'weights': weights,
                'confidences': torch.ones(batch_size, 1, device=x.device),
                'sparsity': (weights > 0.01).float().mean()
            }

class LagrangianSourceIntegration(nn.Module):
    def __init__(self, n_sources: int, input_dim: int, hidden_dim: int = 128, 
                 sparse_topk: int = 10, rho: float = 1.0):
        super().__init__()
        self.n_sources = n_sources
        self.sparse_topk = sparse_topk
        self.rho = rho
      
        # Better initialization for source transforms
        self.source_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(n_sources)
        ])
        
        # Proper initialization
        for transform in self.source_transforms:
            for layer in transform:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        # FIX: Initialize lambda_weights with more diversity and proper scale
        self.lambda_weights = nn.Parameter(torch.randn(n_sources) * 0.5 + 1.0/n_sources)
        self.mu = nn.Parameter(torch.zeros(1))
        self.nu = nn.Parameter(torch.zeros(n_sources))
      
        self.confidence_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for _ in range(n_sources)
        ])
  
    def augmented_lagrangian_loss(self, x: torch.Tensor, y: torch.Tensor, 
                                outputs: torch.Tensor, confidences: torch.Tensor):
        # FIX: Use softmax for stable weights during loss computation
        weights = F.softmax(self.lambda_weights, dim=0)
        
        source_losses = []
        for i in range(self.n_sources):
            # FIX: Use Huber loss for better gradient behavior
            loss = F.smooth_l1_loss(outputs[:, i], y)
            weighted_loss = (loss * confidences[:, i]).mean()
            source_losses.append(weighted_loss)
        
        source_losses = torch.stack(source_losses)
        
        # FIX: Normalize weights in loss computation
        normalized_weights = F.softmax(self.lambda_weights, dim=0)
        weighted_loss = (normalized_weights * source_losses).sum()
        
        # FIX: Softer constraints with better scaling
        g_lambda = (1 - normalized_weights.sum()).abs()
        h_lambda = torch.relu(-normalized_weights)
        
        # FIX: Reduced constraint penalty
        constraint_penalty = 1.0 * (g_lambda + h_lambda.sum())
        
        # FIX: Reduced L1 regularization
        l1_reg = 0.01 * normalized_weights.abs().sum()
        
        # FIX: Positive entropy regularization to encourage exploration
        entropy = 0.1 * (normalized_weights * torch.log(normalized_weights + 1e-8)).sum()
        
        loss = weighted_loss + constraint_penalty + l1_reg - entropy
        
        return loss, {
            'weighted_loss': weighted_loss,
            'g_lambda': g_lambda,
            'h_lambda': h_lambda,
            'source_losses': source_losses,
            'weights': normalized_weights,
            'reconstruction_loss': F.mse_loss(outputs.mean(dim=1), y)
        }
  
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size = x.shape[0]
        outputs = []
        confidences = []
        
        for i in range(self.n_sources):
            output = self.source_transforms[i](x)
            confidence = self.confidence_nets[i](output)
            outputs.append(output)
            confidences.append(confidence)
        
        outputs = torch.stack(outputs, dim=1)
        confidences = torch.stack(confidences, dim=1)
        
        # FIX: Always use softmax for stable training
        normalized_weights = F.softmax(self.lambda_weights, dim=0)
        weights = normalized_weights.view(1, -1, 1)
        
        combined_weights = weights * confidences
        combined_weights = combined_weights / (combined_weights.sum(dim=1, keepdim=True) + 1e-8)
        output = (outputs * combined_weights).sum(dim=1)
        
        return output, {
            'weights': normalized_weights,
            'outputs': outputs,
            'confidences': confidences
        }

class TwoTimeScaleOptimizer:
    def __init__(self, model: nn.Module, eta_theta: float = 1e-3, eta_lambda: float = 1e-2, 
                 clipgrad: float = 1.0, weight_decay: float = 1e-4):
        self.model = model
        self.eta_theta = eta_theta
        self.eta_lambda = eta_lambda
        self.clipgrad = clipgrad
        self.epoch = 0
      
        # FIX: Better parameter separation
        self.theta_params = []
        self.lambda_params = []
        
        for name, param in model.named_parameters():
            if 'lambda_weights' in name:
                self.lambda_params.append(param)
            elif 'mu' in name or 'nu' in name:
                # FIX: Include dual parameters in lambda optimization
                self.lambda_params.append(param)
            else:
                self.theta_params.append(param)
      
        # FIX: Higher learning rates and weight decay
        self.theta_optimizer = torch.optim.AdamW(
            self.theta_params, 
            lr=eta_theta, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # FIX: Separate optimizer for lambda parameters with higher LR
        self.lambda_optimizer = torch.optim.AdamW(
            self.lambda_params,
            lr=eta_lambda,
            weight_decay=0.0,  # No weight decay for lambda weights
            betas=(0.9, 0.999)
        )
  
    def zero_grad(self):
        self.theta_optimizer.zero_grad()
        self.lambda_optimizer.zero_grad()
  
    def step(self, loss_dict: Dict[str, torch.Tensor]):
        # FIX: Better gradient handling
        loss = loss_dict['loss']
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.theta_params, self.clipgrad)
        torch.nn.utils.clip_grad_norm_(self.lambda_params, self.clipgrad)
        
        # Update parameters
        self.theta_optimizer.step()
        self.lambda_optimizer.step()
        
        # FIX: Project weights to simplex with more diversity
        with torch.no_grad():
            if hasattr(self.model, 'lambda_weights'):
                current_weights = self.model.lambda_weights.data
                
                # Add adaptive noise based on epoch
                noise_level = max(0.01, 0.1 * (0.95 ** self.epoch))
                noise = noise_level * torch.randn_like(current_weights)
                
                # Project to simplex
                projected = self.project_simplex(current_weights + noise)
                
                # Ensure minimum diversity
                min_weight = 0.01 / len(projected)
                projected = torch.clamp(projected, min=min_weight)
                projected = projected / projected.sum()
                
                self.model.lambda_weights.data.copy_(projected)
        
        self.epoch += 1

    @staticmethod
    def project_simplex(v: torch.Tensor) -> torch.Tensor:
        """Project onto probability simplex"""
        v_sorted, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(v_sorted, dim=0)
        rho = torch.arange(1, len(v) + 1, device=v.device, dtype=v.dtype)
        rho = (cssv - 1.0) / rho
        idx = (v_sorted > rho).nonzero()
        if len(idx) > 0:
            rho_star = rho[idx[-1]]
        else:
            rho_star = 0.0
        return torch.clamp(v - rho_star, min=0.0)
  
    def adjust_learning_rates(self, val_loss: float):
        """Simple learning rate adjustment"""
        # Reduce LR if validation loss plateaus
        if hasattr(self, 'last_val_loss') and val_loss > self.last_val_loss:
            for param_group in self.theta_optimizer.param_groups:
                param_group['lr'] *= 0.95
            for param_group in self.lambda_optimizer.param_groups:
                param_group['lr'] *= 0.95
        self.last_val_loss = val_loss

class MultiSourceTrainer:
    def __init__(self, model, lr: float = 1e-3, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.metrics = defaultdict(list)
        
    def train_epoch(self, dataloader):
        self.model.train()
        epoch_metrics = defaultdict(list)
        
        for batch in dataloader:
            metrics = self.train_step(batch)
            for k, v in metrics.items():
                epoch_metrics[k].append(v)
        
        return {k: np.mean(v) for k, v in epoch_metrics.items()}
    
    def train_step(self, batch):
        x = batch['x'].to(self.device)
        y = batch['y'].to(self.device)
        
        y_pred, metadata = self.model(x)
        
        recon_loss = F.mse_loss(y_pred, y)
        sparsity_loss = 0.1 * (1 - metadata['sparsity'])
        
        total_loss = recon_loss + sparsity_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'sparsity': metadata['sparsity'].item()
        }

