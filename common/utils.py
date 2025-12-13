import os
import matplotlib.pyplot as plt
import torch

def create_directory(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
  
def save_plot(fig: plt.Figure, path: str, filename: str):
    create_directory(path)
    fig.savefig(os.path.join(path, filename))
    plt.close(fig)
  
def plot_2d_solution(true_sol: torch.Tensor, pred_sol: torch.Tensor, title: str = "Solution Comparison"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(true_sol.cpu().numpy(), cmap='viridis')
    ax1.set_title("True Solution")
    plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(pred_sol.cpu().numpy(), cmap='viridis')
    ax2.set_title("Predicted Solution")
    plt.colorbar(im2, ax=ax2)
    fig.suptitle(title)
    return fig