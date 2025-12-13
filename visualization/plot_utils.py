import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from typing import Dict, List
import torch.nn.functional as F
from matplotlib.lines import Line2D
from common.base_classes import PhysicsRegime

def plot_solver_outputs(lagrangian_model, softmax_model, dataset, epoch: int, 
                       save_dir: str = 'plots', num_samples: int = 4):
    os.makedirs(save_dir, exist_ok=True)
    if hasattr(lagrangian_model, 'eval'):
        lagrangian_model.eval()
    if hasattr(softmax_model, 'eval'):
        softmax_model.eval()
  
    fig, axes = plt.subplots(num_samples, 2, figsize=(20, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
  
    colors = {
        'Initial': 'gray',
        'True': 'black',
        'FNO': 'blue',
        'WENO': 'red',
        'Boundary': 'green',
        'Multiscale': 'purple',
        'Combined': 'orange'
    }
  
    with torch.no_grad():
        indices = np.random.randint(len(dataset), size=num_samples)
        x_coords = torch.linspace(0, 1, dataset.input_dim)
      
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            x = sample['x'].to(lagrangian_model.device).unsqueeze(0)
            u = sample['u']
          
            for j, (model, title) in enumerate([
                (lagrangian_model, 'Lagrangian'),
                (softmax_model, 'Softmax')
            ]):
                output, metadata = model(x, return_all=True)
                ax = axes[i, j]
              
                ax.plot(x_coords, x[0].cpu(), '-', color=colors['Initial'], label='Initial', alpha=0.5)
                ax.plot(x_coords, u.cpu(), '-', color=colors['True'], label='True', linewidth=2)
              
                if 'solver_outputs' in metadata:
                    solver_outputs = metadata['solver_outputs'][0]
                    weights = metadata['weights']
                  
                    if isinstance(weights, torch.Tensor):
                        weights = weights.detach().cpu()
                        if weights.dim() > 1:
                            weights = weights.squeeze()
                  
                    solver_names = ['FNO', 'WENO', 'Boundary', 'Multiscale']
                    for k, (solver_output, name) in enumerate(zip(solver_outputs, solver_names)):
                        weight_value = weights[k].item() if isinstance(weights, torch.Tensor) else weights[k]
                        ax.plot(x_coords, solver_output.cpu(), '--', color=colors[name], alpha=0.5,
                              label=f'{name} (w={weight_value:.2f})')
              
                if output.dim() > 1:
                    combined_output = output[0]
                else:
                    combined_output = output
                ax.plot(x_coords, combined_output.cpu(), '-', color=colors['Combined'], label='Combined', linewidth=2)
              
                regime_info = ""
                if hasattr(metadata, 'get') and metadata.get('regimes') is not None:
                    regime = metadata['regimes'][0]
                    regime_info = f" - Regime: {regime}"
              
                ax.set_title(f'{title} - Sample {i+1}{regime_info}')
                ax.set_xlabel('Spatial Position')
                ax.set_ylabel('Loss')
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                ax.grid(True)
  
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_outputs_epoch_{epoch}.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_training_metrics(metrics: Dict[str, Dict], epoch: int, save_dir: str = 'plots'):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
  
    colors = {
        'lagrangian': {'train': 'royalblue', 'val': 'lightblue'},
        'softmax': {'train': 'salmon', 'val': 'lightcoral'}
    }
  
    ax = axes[0, 0]
    for model_type in ['lagrangian', 'softmax']:
        if f'train_loss' in metrics[model_type]:
            ax.plot(metrics[model_type]['train_loss'], label=f'{model_type.capitalize()} Train',
                   color=colors[model_type]['train'], linestyle='-')
            ax.plot(metrics[model_type]['val_loss'], label=f'{model_type.capitalize()} Val',
                   color=colors[model_type]['val'], linestyle='--')
    ax.set_title('Loss Comparison')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
  
    ax = axes[0, 1]
    for model_type in ['lagrangian', 'softmax']:
        if f'train_huber_loss' in metrics[model_type]:
            ax.plot(metrics[model_type]['train_huber_loss'], label=f'{model_type.capitalize()} Train',
                   color=colors[model_type]['train'], linestyle='-')
            ax.plot(metrics[model_type]['val_huber_loss'], label=f'{model_type.capitalize()} Val',
                   color=colors[model_type]['val'], linestyle='--')
    ax.set_title('Huber Loss Comparison')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Huber Loss')
    ax.legend()
    ax.grid(True)
  
    ax = axes[1, 0]
    if 'constraint_violation' in metrics['lagrangian']:
        ax.semilogy(metrics['lagrangian']['constraint_violation'], label='Lagrangian Constraint',
                   color=colors['lagrangian']['train'])
    if 'balance_loss' in metrics['softmax']:
        ax.semilogy(metrics['softmax']['balance_loss'], label='Softmax Balance',
                   color=colors['softmax']['train'])
    ax.set_title('Constraint/Balance Metrics')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value (log scale)')
    ax.legend()
    ax.grid(True)
  
    ax = axes[1, 1]
    for model_type in ['lagrangian', 'softmax']:
        if 'train_regime_accuracy' in metrics[model_type]:
            ax.plot(metrics[model_type]['train_regime_accuracy'], label=f'{model_type.capitalize()} Train',
                   color=colors[model_type]['train'], linestyle='-')
            ax.plot(metrics[model_type]['val_regime_accuracy'], label=f'{model_type.capitalize()} Val',
                   color=colors[model_type]['val'], linestyle='--')
    ax.set_title('Regime Classification Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
  
    plt.figtext(0.99, 0.01, f'Epoch {epoch}', ha='right', va='bottom', fontsize=8)
    plt.figtext(0.01, 0.01, '2024', ha='left', va='bottom', fontsize=8)
  
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_metrics_epoch_{epoch}.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_comparison(metrics: Dict, epoch: int, save_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
  
    if 'train_loss' in metrics['lagrangian']:
        axes[0, 0].plot(metrics['lagrangian']['train_loss'], label='Lagrangian', color='blue')
        axes[0, 0].plot(metrics['softmax']['train_loss'], label='Softmax', color='red')
        axes[0, 0].set_title('Training Loss Comparison')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
  
    if 'val_loss' in metrics['lagrangian']:
        axes[0, 1].plot(metrics['lagrangian']['val_loss'], label='Lagrangian', color='blue')
        axes[0, 1].plot(metrics['softmax']['val_loss'], label='Softmax', color='red')
        axes[0, 1].set_title('Validation Loss Comparison')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
  
    if 'weights_mean' in metrics['lagrangian']:
        axes[1, 0].plot(metrics['lagrangian']['weights_mean'], label='Lagrangian Mean', color='blue')
        axes[1, 0].plot(metrics['softmax']['weights_mean'], label='Softmax Mean', color='red')
        axes[1, 0].set_title('Weight Evolution')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Weight Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
  
    if 'constraint_violation' in metrics['lagrangian']:
        axes[1, 1].semilogy(metrics['lagrangian']['constraint_violation'], label='Lagrangian Constraint', color='blue')
    if 'balance_loss' in metrics['softmax']:
        axes[1, 1].semilogy(metrics['softmax']['balance_loss'], label='Softmax Balance', color='red')
    if 'constraint_violation' in metrics['lagrangian'] or 'balance_loss' in metrics['softmax']:
        axes[1, 1].set_title('Constraint/Balance Metrics')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Value (log scale)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
  
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_epoch_{epoch}.png')
    plt.close()

def plot_predictions(models: Dict[str, torch.nn.Module], test_loader, device: str, 
                    save_path: str, epoch: int, num_samples: int = 4):
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = [axes]
  
    batch = next(iter(test_loader))
    x = batch['x'].to(device)
    y = batch['u'].to(device)
  
    indices = np.random.choice(len(x), num_samples, replace=False)
  
    colors = {
        'Initial': 'gray',
        'True': 'black',
        'Softmax': 'blue',
        'Lagrangian': 'red'
    }
  
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
  
    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.plot(range(len(x[idx])), x[idx].cpu(), '-', color=colors['Initial'], label='Initial', alpha=0.5)
        ax.plot(range(len(y[idx])), y[idx].cpu(), '-', color=colors['True'], label='True', alpha=0.7)
      
        for model_name, pred_dict in predictions.items():
            pred = pred_dict['pred'][idx]
            ax.plot(range(len(pred)), pred, '--', color=colors[model_name], label=f'{model_name}', alpha=0.7)
          
            if 'confidences' in pred_dict['metadata']:
                conf = pred_dict['metadata']['confidences']
                if isinstance(conf, torch.Tensor):
                    if len(conf.shape) == 3:
                        conf = conf[idx].mean(dim=0)
                    elif len(conf.shape) == 2:
                        conf = conf[idx]
                    if conf.shape[-1] == 1:
                        conf = conf.expand(pred.shape[0])
                    conf = conf.numpy()
                    pred_np = pred.numpy()
                    if conf.shape == pred_np.shape:
                        ax.fill_between(range(len(pred)), pred_np - conf, pred_np + conf,
                                      color=colors[model_name], alpha=0.1, label=f'{model_name} Confidence')
      
        ax.set_title(f'Sample {i+1}')
        ax.set_xlabel('Spatial Position')
        ax.set_ylabel('Loss')
        ax.legend(loc='upper right')
        ax.grid(True)
  
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/predictions_epoch_{epoch}.png')
    plt.close()

def plot_error_distribution(models: Dict[str, torch.nn.Module], test_loader, device: str, 
                           save_path: str, epoch: int):
    os.makedirs(save_path, exist_ok=True)
    errors = {name: [] for name in models.keys()}
  
    for model_name, model in models.items():
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                x = batch['x'].to(device)
                y = batch['u'].to(device)
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

def plot_weight_evolution(trainer, save_path: str, epoch: int):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
  
    solver_names = ['FNO', 'WENO', 'DeepONet', 'MultiRes']
    colors = ['blue', 'red', 'green', 'purple']
  
    if hasattr(trainer.softmax_model, 'weight_network'):
        weights = trainer.softmax_model.weight_network[-1].weight.data.cpu().numpy()
        ax1.hist(weights.flatten(), bins=50, alpha=0.7, color='blue')
        ax1.set_title('Softmax Model Weight Distribution')
        ax1.set_xlabel('Weight Value')
        ax1.set_ylabel('Count')
        ax1.grid(True)
  
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

def plot_predictions_and_weights(models: Dict[str, torch.nn.Module], test_loader, device: str, 
                               save_path: str, epoch: int, num_samples: int = 4, top_k: int = 10):
    fig = plt.figure(figsize=(15, 5*num_samples))
    gs = plt.GridSpec(num_samples, 3, figure=fig)
  
    batch = next(iter(test_loader))
    x = batch['x'].to(device)
    y = batch['u'].to(device)
  
    indices = np.random.choice(len(x), num_samples, replace=False)
  
    colors = {
        'Initial': 'gray',
        'True': 'black',
        'Softmax': 'blue',
        'Lagrangian': 'red'
    }
  
    predictions = {}
    weights = {}
    for model_name, model in models.items():
        model.eval()
        with torch.no_grad():
            pred, metadata = model(x)
            predictions[model_name] = pred.detach().cpu()
            raw_weights = metadata['weights'].detach().cpu()
            if model_name == 'Softmax':
                if len(raw_weights.shape) > 1:
                    top_weights, _ = torch.topk(raw_weights, min(top_k, raw_weights.size(-1)), dim=-1)
                else:
                    top_weights, _ = torch.topk(raw_weights, min(top_k, len(raw_weights)))
                weights[model_name] = top_weights
            else:
                raw_weights = model.lambda_weights.detach().cpu()
                top_weights, _ = torch.topk(raw_weights, min(top_k, len(raw_weights)))
                weights[model_name] = top_weights
  
    for i, idx in enumerate(indices):
        ax_pred = fig.add_subplot(gs[i, 0:2])
        ax_pred.plot(range(len(x[idx])), x[idx].cpu(), '-', color=colors['Initial'], label='Initial', alpha=0.5)
        ax_pred.plot(range(len(y[idx])), y[idx].cpu(), '-', color=colors['True'], label='True', alpha=0.7)
      
        for model_name in predictions:
            ax_pred.plot(range(len(predictions[model_name][idx])), predictions[model_name][idx],
                       '--', color=colors[model_name], label=f'{model_name}', alpha=0.7)
      
        ax_pred.set_title(f'Sample {i+1} Predictions')
        ax_pred.set_xlabel('Spatial Position')
        ax_pred.set_ylabel('Value')
        ax_pred.legend(loc='upper right')
        ax_pred.grid(True)
      
        ax_weights = fig.add_subplot(gs[i, 2])
        width = 0.35
        x_pos = np.arange(top_k)
      
        for j, (model_name, model_weights) in enumerate(weights.items()):
            if len(model_weights.shape) > 1:
                weights_to_plot = model_weights[idx]
            else:
                weights_to_plot = model_weights
            ax_weights.bar(x_pos + j*width, weights_to_plot, width, label=model_name,
                         color=colors[model_name], alpha=0.7)
      
        ax_weights.set_title(f'Top {top_k} Source Weights')
        ax_weights.set_xlabel('Source Index')
        ax_weights.set_ylabel('Weight Value')
        ax_weights.legend()
        ax_weights.grid(True, alpha=0.3)
  
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/predictions_and_weights_epoch_{epoch}.png')
    plt.close()