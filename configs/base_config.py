from dataclasses import dataclass
from typing import List
import torch

@dataclass
class BaseConfig:
    n_samples: int = 2000
    input_dim: int = 64
    hidden_dim: int = 128
    batch_size: int = 16
    n_epochs: int = 500
    learning_rate: float = 1e-4
    rho: float = 0.1
    temperature: float = 2.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir: str = 'results'


@dataclass
class MultiSourceConfig:
    n_samples: int = 1000
    input_dim: int = 64
    n_sources: int = 128
    batch_size: int = 32
    n_epochs: int = 200
    # FIX: Better learning rates
    lr_softmax: float = 1e-3
    lr_theta: float = 1e-3
    lr_lambda: float = 1e-2  # Higher for lambda weights
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir: str = 'results'

@dataclass
class PDESolverConfig:
    domain_size: int = 32
    n_samples: int = 1024
    batch_size: int = 16
    n_epochs: int = 400
    lr_theta: float = 1e-3
    lr_lambda: float = 1e-4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir: str = "results"