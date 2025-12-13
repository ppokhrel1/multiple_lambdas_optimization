import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict
from collections import defaultdict
import numpy as np
from .multi_source_trainer import TwoTimeScaleOptimizer

class ComparativeTrainer:
    def __init__(self, softmax_model: nn.Module, lagrangian_model: nn.Module, 
                 train_loader: DataLoader, val_loader: DataLoader,
                 lr_softmax: float = 1e-3, lr_theta: float = 1e-3, lr_lambda: float = 1e-6,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.softmax_model = softmax_model.to(device)
        self.lagrangian_model = lagrangian_model.to(device)
        self.device = device
        
        self.softmax_optimizer = torch.optim.Adam(softmax_model.parameters(), lr=lr_softmax)
        self.lagrangian_optimizer = TwoTimeScaleOptimizer(
            lagrangian_model,
            eta_theta=lr_theta,
            eta_lambda=lr_lambda
        )
        
        self.metrics = {
            'softmax': defaultdict(list),
            'lagrangian': defaultdict(list)
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, Dict[str, float]]:
        self.softmax_model.train()
        self.lagrangian_model.train()
        
        epoch_metrics = {
            'softmax': defaultdict(list),
            'lagrangian': defaultdict(list)
        }
        
        for batch in dataloader:
            softmax_metrics = self.train_step_softmax(batch)
            lagrangian_metrics = self.train_step_lagrangian(batch)
            
            for k, v in softmax_metrics.items():
                epoch_metrics['softmax'][k].append(v)
            for k, v in lagrangian_metrics.items():
                epoch_metrics['lagrangian'][k].append(v)
        
        return {
            'softmax': {k: np.mean(v) for k, v in epoch_metrics['softmax'].items()},
            'lagrangian': {k: np.mean(v) for k, v in epoch_metrics['lagrangian'].items()}
        }
    
    def train_step_softmax(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        #print(batch.keys())
        x = batch['x'].to(self.device)
        y = batch['u'].to(self.device)
        
        y_pred, metadata = self.softmax_model(x)
        mse_loss = F.mse_loss(y_pred, y)
        
        recon_loss = F.mse_loss(y_pred, y)
        sparsity_loss = 0.1 * (1 - metadata['sparsity'])
        total_loss = recon_loss + sparsity_loss
        
        self.softmax_optimizer.zero_grad()
        total_loss.backward()
        self.softmax_optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'mse': mse_loss.item(),
            'recon_loss': recon_loss.item(),
            'sparsity': metadata['sparsity'].item(),
            'weight_entropy': -(metadata['weights'] * torch.log(metadata['weights'] + 1e-6)).sum(dim=-1).mean().item()
        }
    
    def train_step_lagrangian(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        x = batch['x'].to(self.device)
        y = batch['u'].to(self.device)
        
        self.lagrangian_optimizer.zero_grad()
        
        y_pred, metadata = self.lagrangian_model(x)
        mse_loss = F.mse_loss(y_pred, y)
        
        lagrangian_loss, loss_dict = self.lagrangian_model.augmented_lagrangian_loss(
            x, y, metadata['outputs'], metadata['confidences']
        )
        
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
    
    def validate(self, dataloader: DataLoader) -> Dict[str, Dict[str, float]]:
        self.softmax_model.eval()
        self.lagrangian_model.eval()
        
        val_metrics = {
            'softmax': defaultdict(list),
            'lagrangian': defaultdict(list)
        }
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch['x'].to(self.device)
                y = batch['u'].to(self.device)
                
                y_pred_soft, meta_soft = self.softmax_model(x)
                soft_loss = F.mse_loss(y_pred_soft, y)
                val_metrics['softmax']['loss'].append(soft_loss.item())
                val_metrics['softmax']['mse'].append(soft_loss.item())
                
                y_pred_lag, meta_lag = self.lagrangian_model(x)
                lag_loss = F.mse_loss(y_pred_lag, y)
                val_metrics['lagrangian']['loss'].append(lag_loss.item())
                val_metrics['lagrangian']['mse'].append(lag_loss.item())
        
        return {
            'softmax': {k: np.mean(v) for k, v in val_metrics['softmax'].items()},
            'lagrangian': {k: np.mean(v) for k, v in val_metrics['lagrangian'].items()}
        }