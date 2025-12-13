import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict
from collections import defaultdict
import numpy as np
from common.loss_functions import HuberLoss
from common.optimizers import TwoTimeScaleLagrangianOptimizer

class BaseTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                 learning_rate: float = 1e-4, delta: float = 1.0, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.huber = HuberLoss(delta=delta)
      
        if hasattr(model, 'use_lagrangian') and model.use_lagrangian:
            self.optimizer = TwoTimeScaleLagrangianOptimizer(
                model,
                eta_theta=learning_rate,
                eta_lambda=learning_rate * 0.1
            )
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
      
        self.metrics = defaultdict(list)

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        epoch_metrics = defaultdict(list)
      
        for batch in self.train_loader:
            metrics = self.train_step(batch)
            if np.isnan(metrics['loss']):
                continue
            for k, v in metrics.items():
                epoch_metrics[k].append(v)
      
        return {k: np.mean(v) for k, v in epoch_metrics.items()}

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        x = batch['x'].to(self.device) if 'x' in batch else batch['grid'].to(self.device)
        u = batch['u'].to(self.device) if 'u' in batch else batch['solution'].to(self.device)
        regime_idx = batch.get('regime_idx', None)
        if regime_idx is not None:
            regime_idx = regime_idx.to(self.device)
      
        self.optimizer.zero_grad()
      
        if hasattr(self.model, 'compute_loss'):
            if hasattr(self.model, 'use_lagrangian') and self.model.use_lagrangian:
                loss, metadata = self.model.compute_loss(x, u)
                if torch.isnan(loss):
                    return {'loss': float('nan'), 'constraint_violation': float('nan'), 'min_weight': float('nan')}
                loss.backward()
                if hasattr(self.optimizer, 'step'):
                    self.optimizer.step({
                        'loss': loss,
                        'g_lambda': metadata.get('g_lambda', torch.tensor(0.0)),
                        'h_lambda': metadata.get('h_lambda', torch.tensor(0.0))
                    })
                metrics = {
                    'loss': loss.item(),
                    'constraint_violation': metadata.get('g_lambda', torch.tensor(0.0)).abs().item(),
                    'min_weight': metadata.get('weights', torch.tensor(0.0)).min().item()
                }
                if 'huber_loss' in metadata:
                    metrics['huber_loss'] = metadata['huber_loss']
                return metrics
            else:
                output, metadata = self.model(x)
                loss = self.huber(output, u)
                if torch.isnan(loss):
                    return {'loss': float('nan'), 'regime_accuracy': 0.0}
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                regime_accuracy = 0.0
                if regime_idx is not None and 'regime_weights' in metadata:
                    regime_accuracy = (metadata['regime_weights'].argmax(dim=1) == regime_idx).float().mean().item()
                return {
                    'loss': loss.item(),
                    'huber_loss': loss.item(),
                    'regime_accuracy': regime_accuracy
                }
        else:
            output, metadata = self.model(x)
            loss = self.huber(output, u)
            if torch.isnan(loss):
                return {'loss': float('nan')}
            loss.backward()
            self.optimizer.step()
            return {'loss': loss.item()}

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        val_metrics = defaultdict(list)
      
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch['x'].to(self.device) if 'x' in batch else batch['grid'].to(self.device)
                u = batch['u'].to(self.device) if 'u' in batch else batch['solution'].to(self.device)
                regime_idx = batch.get('regime_idx', None)
                if regime_idx is not None:
                    regime_idx = regime_idx.to(self.device)
              
                try:
                    if hasattr(self.model, 'compute_loss'):
                        output, metadata = self.model(x)
                        huber_loss = self.huber(output, u)
                        loss, loss_metadata = self.model.compute_loss(x, u)
                      
                        metrics = {
                            'loss': loss.item(),
                            'huber_loss': huber_loss.item(),
                            'constraint_violation': loss_metadata.get('g_lambda', torch.tensor(0.0)).abs().item()
                        }
                    else:
                        output, metadata = self.model(x)
                        huber_loss = self.huber(output, u)
                        regime_accuracy = 0.0
                        if regime_idx is not None and 'regime_weights' in metadata:
                            regime_accuracy = (metadata['regime_weights'].argmax(dim=1) == regime_idx).float().mean().item()
                        metrics = {
                            'loss': huber_loss.item(),
                            'huber_loss': huber_loss.item(),
                            'regime_accuracy': regime_accuracy
                        }
                  
                    for k, v in metrics.items():
                        val_metrics[k].append(v)
                except RuntimeError as e:
                    continue
      
        return {k: np.mean(v) for k, v in val_metrics.items()}