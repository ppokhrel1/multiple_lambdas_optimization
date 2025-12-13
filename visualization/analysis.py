import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from typing import Dict
import torch.nn.functional as F
from matplotlib.lines import Line2D

def plot_detailed_analysis(trainers, dataset, epoch, save_dir):
    plt.close('all')
    fig = plt.figure(figsize=(20, 8))
    gs = plt.GridSpec(1, 3)
  
    ax1 = fig.add_subplot(gs[0, 0])
    solver_names = ['FNO', 'WENO', 'DeepONet', 'MultiRes']
    colors = ['blue', 'red', 'green', 'purple']
  
    weights_history = trainers['lagrangian'].all_metrics['train_weight_mean']
    epochs = range(len(weights_history))
    ax1.plot(epochs, weights_history, label='Lagrangian', color='red', alpha=0.7)
  
    weights_history = trainers['softmax'].all_metrics['train_weight_mean']
    epochs = range(len(weights_history))
    ax1.plot(epochs, weights_history, label='Softmax', color='blue', alpha=0.7)
  
    if len(trainers['softmax'].all_metrics['train_weight_std']) > 0:
        ax1.fill_between(range(len(trainers['softmax'].all_metrics['train_weight_std'])),
                        np.array(trainers['softmax'].all_metrics['train_weight_mean']) -
                        np.array(trainers['softmax'].all_metrics['train_weight_std']),
                        np.array(trainers['softmax'].all_metrics['train_weight_mean']) +
                        np.array(trainers['softmax'].all_metrics['train_weight_std']),
                        color='blue', alpha=0.2)
        ax1.fill_between(range(len(trainers['lagrangian'].all_metrics['train_weight_std'])),
                        np.array(trainers['lagrangian'].all_metrics['train_weight_mean']) -
                        np.array(trainers['lagrangian'].all_metrics['train_weight_std']),
                        np.array(trainers['lagrangian'].all_metrics['train_weight_mean']) +
                        np.array(trainers['lagrangian'].all_metrics['train_weight_std']),
                        color='red', alpha=0.2)

    ax1.set_title('Weight Evolution')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Weight Value')
    ax1.grid(True)
    ax1.legend()
    ax1.set_ylim(0, 0.5)
  
    ax3 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(solver_names))
    width = 0.35
  
    for i, (name, trainer) in enumerate(trainers.items()):
        with torch.no_grad():
            sample = dataset[0]
            grid = sample['grid'].to(trainer.device).unsqueeze(0)
            _, meta = trainer.model(grid)
            weights = meta['weights'][0].cpu().numpy()
            ax3.bar(x + i*width, weights, width, label=name, alpha=0.7)
  
    ax3.set_title('Current Weight Distribution')
    ax3.set_xticks(x + width/2)
    ax3.set_xticklabels(solver_names, rotation=45)
    ax3.set_ylabel('Weight Value')
    ax3.legend()
    ax3.grid(True)
    ax3.set_ylim(0, 0.5)
  
    ax4 = fig.add_subplot(gs[0, 2])
    for name, trainer in trainers.items():
        ax4.plot(trainer.all_metrics['train_total_loss'], label=f'{name} Train', alpha=0.7)
        if 'val_total_loss' in trainer.all_metrics:
            ax4.plot(trainer.all_metrics['val_total_loss'], label=f'{name} Val', linestyle='--', alpha=0.7)
  
    ax4.set_title('Loss Evolution')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_yscale('log')
    ax4.grid(True)
    ax4.legend()
  
    plt.suptitle(f'Training Analysis - Epoch {epoch}', y=1.02, fontsize=18)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/detailed_analysis_epoch_{epoch}.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_true_solutions(sample, save_dir, epoch):
    titles = ['Velocity-U', 'Velocity-V', 'Vorticity']
    target = sample['solution']
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for i, title in enumerate(titles):
        im = axes[i].imshow(target[i].cpu(), cmap='RdBu')
        plt.colorbar(im, ax=axes[i])
        axes[i].set_title(f'True {title}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/true_solutions_epoch_{epoch}.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_solver_weights(all_outputs, save_dir, epoch, colors):
    fig, ax = plt.subplots(figsize=(10, 6))
    solver_names = ['FNO', 'WENO', 'DeepONet', 'MultiRes']
    x = np.arange(len(solver_names))
    width = 0.35
    
    lag_weights = [all_outputs['Lagrangian'][1]['weights'][0][i].item() for i in range(len(solver_names))]
    soft_weights = [all_outputs['Softmax'][1]['weights'][0][i].item() for i in range(len(solver_names))]
    
    ax.bar(x - width/2, lag_weights, width, label='Lagrangian', color=colors['Lagrangian'])
    ax.bar(x + width/2, soft_weights, width, label='Softmax', color=colors['Softmax'])
    
    ax.set_xlabel('Solvers')
    ax.set_ylabel('Weight Value')
    ax.set_title('Solver Weights Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(solver_names)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/solver_weights_epoch_{epoch}.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_comparison_separate(trainers, dataset, epoch, save_dir):
    colors = {
        'FNO': 'blue',
        'WENO': 'red',
        'DeepONet': 'green',
        'MultiRes': 'purple',
        'Combined': 'orange',
        'Lagrangian': 'crimson',
        'Softmax': 'navy'
    }
    
    sample_idx = np.random.randint(len(dataset))
    sample = dataset[sample_idx]
    
    plot_true_solutions(sample, save_dir, epoch)
    
    all_outputs = {}
    
    for method_name, trainer in trainers.items():
        device = next(trainer.model.parameters()).device
        grid = sample['grid'].to(device).unsqueeze(0)
        with torch.no_grad():
            output, metadata = trainer.model(grid)
        all_outputs[method_name.capitalize()] = (output, metadata)
    
    plot_solver_weights(all_outputs, save_dir, epoch, colors)

def plot_error_evolution(models: Dict[str, torch.nn.Module], test_loader, device: str, 
                        save_path: str, epoch: int):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
  
    batch = next(iter(test_loader))
    x = batch['x'].to(device)
    y = batch['y'].to(device)
  
    colors = {'Softmax': 'blue', 'Lagrangian': 'red'}
  
    for model_name, model in models.items():
        model.eval()
        with torch.no_grad():
            pred, metadata = model(x)
            error = torch.abs(pred - y)
            mean_error = error.mean(dim=0)
            std_error = error.std(dim=0)
          
            ax1.plot(range(len(mean_error)), mean_error.cpu(), '-', color=colors[model_name],
                   label=f'{model_name} Mean Error')
            ax1.fill_between(range(len(mean_error)), (mean_error - std_error).cpu(),
                           (mean_error + std_error).cpu(), alpha=0.2, color=colors[model_name])
          
            ax2.hist(error.cpu().numpy().flatten(), bins=50, alpha=0.5, color=colors[model_name],
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