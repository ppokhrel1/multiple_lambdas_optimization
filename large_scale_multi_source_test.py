"""
Four-way comparison for large-scale source integration
----------------------------------------------------
Softmax
Single-time-scale Lagrangian
Two-time-scale Lagrangian
ADMM  (new)
"""

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


# ------------------------------------------------------------------------------
# 0. 1-D Navier–Stokes / Burgers data set  (your original code)
# ------------------------------------------------------------------------------
class NavierStokes1DDataset(Dataset):
    def __init__(self, n_samples: int, input_dim: int, n_sources: int):
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.n_sources = n_sources

        self.dt = 0.001
        self.dx = 2.0 / input_dim
        self.Re = 100
        self.nu = 1.0 / self.Re

        self.x = torch.zeros(n_samples, input_dim)
        for i in range(n_samples):
            self.x[i] = self.generate_initial_condition()

        self.y = torch.zeros(n_samples, input_dim)
        for i in range(n_samples):
            solution = self.solve_burgers(self.x[i])
            self.y[i] = solution[-1]

        self.y += 0.1 * torch.randn_like(self.y)

    def generate_initial_condition(self) -> torch.Tensor:
        x = torch.linspace(-1, 1, self.input_dim)
        cond = torch.randint(0, 3, (1,)).item()
        if cond == 0:
            u = torch.sin(np.pi * x)
        elif cond == 1:
            u = torch.exp(-10 * x**2)
        else:
            u = torch.zeros_like(x)
            u[x < 0] = 1.0
            u[x >= 0] = -1.0
            u = F.conv1d(u.view(1, 1, -1), torch.ones(1, 1, 5)/5, padding=2).squeeze()
        return u

    def solve_burgers(self, u: torch.Tensor, n_steps: int = 100) -> torch.Tensor:
        sols = [u.clone()]
        u_cur = u.clone()
        for _ in range(n_steps):
            du = torch.zeros_like(u_cur)
            du[1:-1] = (u_cur[2:] - u_cur[:-2]) / (2 * self.dx)
            du[0] = (u_cur[1] - u_cur[0]) / self.dx
            du[-1] = (u_cur[-1] - u_cur[-2]) / self.dx

            d2u = torch.zeros_like(u_cur)
            d2u[1:-1] = (u_cur[2:] - 2*u_cur[1:-1] + u_cur[:-2]) / (self.dx**2)
            d2u[0] = (u_cur[2] - 2*u_cur[1] + u_cur[0]) / (self.dx**2)
            d2u[-1] = (u_cur[-1] - 2*u_cur[-2] + u_cur[-3]) / (self.dx**2)

            u_cur = u_cur + self.dt * (-u_cur * du + self.nu * d2u)
            u_cur[0] = u_cur[-2]
            u_cur[-1] = u_cur[1]
            sols.append(u_cur.clone())
        return torch.stack(sols)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {'x': self.x[idx], 'y': self.y[idx]}


# ------------------------------------------------------------------------------
# 1. Softmax model  (your original code)
# ------------------------------------------------------------------------------
class LargeScaleSourceIntegration(nn.Module):
    def __init__(self, n_sources: int, input_dim: int, hidden_dim: int = 128, sparse_topk: int = 10):
        super().__init__()
        self.n_sources = n_sources
        self.sparse_topk = sparse_topk
        self.source_transforms = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))
            for _ in range(n_sources)
        ])
        self.weight_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_sources)
        )
        self.confidence_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
            for _ in range(n_sources)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        bsz = x.shape[0]
        logits = self.weight_network(x)
        weights = F.softmax(logits, dim=-1)
        topv, topi = torch.topk(weights, self.sparse_topk, dim=-1)

        sel_out, sel_conf, sel_w = [], [], []
        for b in range(bsz):
            bout, bconf, bw = [], [], []
            for idx in topi[b]:
                o = self.source_transforms[idx](x[b:b+1])
                c = self.confidence_nets[idx](o)
                w = weights[b:b+1, idx:idx+1]
                bout.append(o); bconf.append(c); bw.append(w)
            sel_out.append(torch.cat(bout, 0))
            sel_conf.append(torch.cat(bconf, 0))
            sel_w.append(torch.cat(bw, 0))
        sel_out = torch.stack(sel_out)
        sel_conf = torch.stack(sel_conf).squeeze(-1)
        sel_w = torch.stack(sel_w).squeeze(-1)

        comb_w = sel_w * sel_conf
        comb_w = comb_w / (comb_w.sum(1, keepdim=True) + 1e-6)
        out = (sel_out * comb_w.unsqueeze(-1)).sum(1)
        return out, {'weights': weights, 'confidences': sel_conf, 'sparsity': (weights > 0.01).float().mean()}


# ------------------------------------------------------------------------------
# 2. Lagrangian models  (your original code, untouched)
# ------------------------------------------------------------------------------

class LagrangianSourceIntegration(nn.Module):
    def __init__(self, n_sources: int, input_dim: int, hidden_dim: int = 128, sparse_topk: int = 10, rho: float = 1.0):
        super().__init__()
        self.n_sources = n_sources
        self.sparse_topk = sparse_topk
        self.rho = rho
        self.source_transforms = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))
            for _ in range(n_sources)
        ])
        self.lambda_weights = nn.Parameter(torch.ones(n_sources) / n_sources)
        self.confidence_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
            for _ in range(n_sources)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        outs, confs = [], []
        for i in range(self.n_sources):
            o = self.source_transforms[i](x)
            c = self.confidence_nets[i](o)
            outs.append(o)
            confs.append(c)
        outs = torch.stack(outs, 1)
        confs = torch.stack(confs, 1)
        w = self.lambda_weights.softmax(0).view(1, -1, 1)
        comb = w * confs
        comb = comb / (comb.sum(1, keepdim=True) + 1e-6)
        pred = (outs * comb).sum(1)
        return pred, {'weights': w.squeeze(), 'outputs': outs, 'confidences': confs}

    def augmented_lagrangian_loss(self, x, y, outs, confs):
        w = self.lambda_weights.softmax(0)
        src_loss = []
        for i in range(self.n_sources):
            mse = F.mse_loss(outs[:, i], y, reduction='none').mean(dim=1)
            src_loss.append((mse * confs[:, i].squeeze(-1)).mean())
        src_loss = torch.stack(src_loss)
        weighted = (w * src_loss).sum()
        g = (1 - w.sum()).abs()
        h = torch.relu(-w).sum()
        constraint = 100.0 * (g + h)
        l1 = 0.1 * w.abs().sum()
        entr = -0.01 * (w * torch.log(w + 1e-6)).sum()
        loss = weighted + constraint + l1 + entr
        return loss, {'weighted_loss': weighted, 'g_lambda': g, 'h_lambda': h,
                      'source_losses': src_loss, 'weights': w,
                      'reconstruction_loss': F.mse_loss(outs.mean(1), y)}


# ------------------------------------------------------------------
# 3. ADMM model  (NEW – drop-in)
# ------------------------------------------------------------------
class ADMMSourceIntegration(nn.Module):
    def __init__(self, n_sources: int, input_dim: int, hidden_dim: int = 128, rho: float = 1.0):
        super().__init__()
        self.n_sources = n_sources
        self.rho = rho
        self.source_transforms = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))
            for _ in range(n_sources)
        ])
        self.confidence_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
            for _ in range(n_sources)
        ])
        # consensus & dual
        self.z = nn.Parameter(torch.ones(n_sources) / n_sources)
        self.u = nn.Parameter(torch.zeros(n_sources))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        outs, confs = [], []
        for i in range(self.n_sources):
            o = self.source_transforms[i](x)
            c = self.confidence_nets[i](o)
            outs.append(o)
            confs.append(c)
        outs = torch.stack(outs, 1)
        confs = torch.stack(confs, 1)
        w = self.z.softmax(0).view(1, -1, 1)
        comb = w * confs
        comb = comb / (comb.sum(1, keepdim=True) + 1e-6)
        pred = (outs * comb).sum(1)
        return pred, {'weights': w.squeeze(), 'outputs': outs, 'confidences': confs,
                      'z': self.z, 'u': self.u}

    def admm_loss(self, x, y, outs, confs):
        w = self.z.softmax(0)
        src = []
        for i in range(self.n_sources):
            mse = F.mse_loss(outs[:, i], y, reduction='none').mean(dim=1)
            src.append((mse * confs[:, i].squeeze(-1)).mean())
        recon = (w * torch.stack(src)).sum()
        consensus = 0.5 * self.rho * (self.z - self.z.detach() + self.u).pow(2).sum()
        loss = recon + consensus
        return loss, {'loss': loss, 'recon_loss': recon, 'consensus': consensus, 'weights': w}


# ------------------------------------------------------------------
# 4. Optimisers  (your original code + ADMM optimiser)
# ------------------------------------------------------------------
class SingleTimeScaleOptimizer:
    def __init__(self, model: LagrangianSourceIntegration, eta: float = 1e-4, clipgrad: float = 0.5,
                 weight_decay: float = 1e-4, beta1: float = 0.9, beta2: float = 0.999):
        self.model = model
        self.epoch = 0
        theta_params = [p for n, p in model.named_parameters() if 'lambda_weights' not in n]
        self.opt = torch.optim.AdamW([{'params': theta_params, 'weight_decay': weight_decay},
                                      {'params': [model.lambda_weights], 'weight_decay': 0.0}],
                                     lr=eta, betas=(beta1, beta2))
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=100, eta_min=1e-6)
        self.mom = torch.zeros_like(model.lambda_weights.data)
        self.vel = torch.zeros_like(model.lambda_weights.data)

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self, loss_dict):
        loss_dict['loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.opt.step()
        with torch.no_grad():
            temp = max(0.1, 1.0 * (0.95 ** self.epoch))
            w = (self.model.lambda_weights / temp).softmax(0)
            noise = 0.01 * torch.randn_like(w)
            w = self._project_simplex(w + noise)
            w = torch.clamp(w, min=0.01)
            self.mom = 0.9 * self.mom + 0.1 * (w - self.model.lambda_weights.data)
            final = self.model.lambda_weights.data + self.mom
            self.model.lambda_weights.data.copy_(final.softmax(0))
        self.sched.step()

    @staticmethod
    def _project_simplex(v):
        s, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(s, 0)
        rho = (cssv - 1) / torch.arange(1, len(v)+1, device=v.device, dtype=v.dtype)
        idx = torch.where(s > rho)[0]
        rho_star = rho[idx[-1]] if idx.numel() else torch.tensor(0.0, device=v.device, dtype=v.dtype)
        return torch.maximum(v - rho_star, torch.zeros_like(v))

    def state_dict(self):
        return {'opt': self.opt.state_dict(), 'sched': self.sched.state_dict(),
                'mom': self.mom, 'vel': self.vel, 'epoch': self.epoch}

    def load_state_dict(self, d):
        self.opt.load_state_dict(d['opt'])
        self.sched.load_state_dict(d['sched'])
        self.mom, self.vel, self.epoch = d['mom'], d['vel'], d['epoch']

    def update_epoch(self, e):
        self.epoch = e


class TwoTimeScaleOptimizer:
    def __init__(self, model: LagrangianSourceIntegration, eta_theta: float = 1e-4, eta_lambda: float = 1e-3,
                 clipgrad: float = 0.5, weight_decay: float = 1e-4, beta1: float = 0.9, beta2: float = 0.999):
        self.model = model
        self.epoch = 0
        self.theta_opt = torch.optim.AdamW([p for n, p in model.named_parameters() if 'lambda_weights' not in n],
                                           lr=eta_theta, weight_decay=weight_decay, betas=(beta1, beta2))
        self.lambda_opt = torch.optim.AdamW([model.lambda_weights], lr=eta_lambda, weight_decay=0.0, betas=(beta1, beta2))
        self.theta_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.theta_opt, T_max=100, eta_min=1e-6)
        self.lambda_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.lambda_opt, T_max=100, eta_min=1e-5)
        self.mom = torch.zeros_like(model.lambda_weights.data)
        self.vel = torch.zeros_like(model.lambda_weights.data)

    def zero_grad(self):
        self.theta_opt.zero_grad()
        self.lambda_opt.zero_grad()

    def step(self, loss_dict):
        loss_dict['loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.theta_opt.step()
        self.lambda_opt.step()
        with torch.no_grad():
            temp = max(0.1, 1.0 * (0.95 ** self.epoch))
            w = (self.model.lambda_weights / temp).softmax(0)
            noise = 0.01 * torch.randn_like(w)
            w = self._project_simplex(w + noise)
            w = torch.clamp(w, min=0.01)
            self.mom = 0.9 * self.mom + 0.1 * (w - self.model.lambda_weights.data)
            final = self.model.lambda_weights.data + self.mom
            self.model.lambda_weights.data.copy_(final.softmax(0))
        self.theta_sched.step()
        self.lambda_sched.step()

    @staticmethod
    def _project_simplex(v):
        s, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(s, 0)
        rho = (cssv - 1) / torch.arange(1, len(v)+1, device=v.device, dtype=v.dtype)
        idx = torch.where(s > rho)[0]
        rho_star = rho[idx[-1]] if idx.numel() else torch.tensor(0.0, device=v.device, dtype=v.dtype)
        return torch.maximum(v - rho_star, torch.zeros_like(v))

    def state_dict(self):
        return {'theta_opt': self.theta_opt.state_dict(), 'lambda_opt': self.lambda_opt.state_dict(),
                'theta_sched': self.theta_sched.state_dict(), 'lambda_sched': self.lambda_sched.state_dict(),
                'mom': self.mom, 'vel': self.vel, 'epoch': self.epoch}

    def load_state_dict(self, d):
        self.theta_opt.load_state_dict(d['theta_opt'])
        self.lambda_opt.load_state_dict(d['lambda_opt'])
        self.theta_sched.load_state_dict(d['theta_sched'])
        self.lambda_sched.load_state_dict(d['lambda_sched'])
        self.mom, self.vel, self.epoch = d['mom'], d['vel'], d['epoch']

    def update_epoch(self, e):
        self.epoch = e


# ------------------------------------------------------------------
# 5. ADMM optimiser  (NEW)
# ------------------------------------------------------------------
class ADMMOptimizer:
    def __init__(self, model: ADMMSourceIntegration, lr: float = 1e-3, clipgrad: float = 0.5,
                 weight_decay: float = 1e-4):
        self.model = model
        self.rho = model.rho
        self.theta_opt = torch.optim.AdamW(
            [p for n, p in model.named_parameters() if n not in ('z', 'u')],
            lr=lr, weight_decay=weight_decay
        )
        self.clipgrad = clipgrad

    def zero_grad(self):
        self.theta_opt.zero_grad()

    def step(self, loss_dict):
        # θ-update
        loss_dict['loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
        self.theta_opt.step()

        # z-update  (exact projection)
        with torch.no_grad():
            z_hat = self.model.z - self.model.u
            z_proj = self._project_simplex(z_hat)
            self.model.z.copy_(z_proj)

        # u-update  (dual ascent)
        with torch.no_grad():
            self.model.u.add_(self.model.z - z_proj)

    @staticmethod
    def _project_simplex(v: torch.Tensor) -> torch.Tensor:
        s, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(s, 0)
        rho = (cssv - 1) / torch.arange(1, len(v)+1, device=v.device, dtype=v.dtype)
        idx = torch.where(s > rho)[0]
        rho_star = rho[idx[-1]] if idx.numel() else torch.tensor(0.0, device=v.device, dtype=v.dtype)
        return torch.maximum(v - rho_star, torch.zeros_like(v))

    def state_dict(self):
        return {'theta_opt': self.theta_opt.state_dict(), 'z': self.model.z.data, 'u': self.model.u.data}

    def load_state_dict(self, d):
        self.theta_opt.load_state_dict(d['theta_opt'])
        self.model.z.data.copy_(d['z'])
        self.model.u.data.copy_(d['u'])


# ------------------------------------------------------------------
# 6. Comparative trainer  (extended with ADMM)
# ------------------------------------------------------------------
class ComparativeTrainer:
    def __init__(self, softmax_model: LargeScaleSourceIntegration,
                 lagrangian_model: LagrangianSourceIntegration,
                 single_time_scale_lagrangian_model: LagrangianSourceIntegration,
                 admm_model: ADMMSourceIntegration,
                 lr_softmax: float = 1e-3,
                 lr_theta: float = 1e-3,
                 lr_lambda: float = 1e-3,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.softmax_model = softmax_model.to(device)
        self.lagrangian_model = lagrangian_model.to(device)
        self.admm_model = admm_model.to(device)
        self.single_time_scale_lagrangian_model = single_time_scale_lagrangian_model.to(device) 

        self.softmax_opt = torch.optim.Adam(softmax_model.parameters(), lr=lr_softmax)
        self.lagrangian_opt = TwoTimeScaleOptimizer(lagrangian_model, eta_theta=lr_theta, eta_lambda=lr_lambda)

        self.single_time_scale_lagrangian_opt = SingleTimeScaleOptimizer(
            single_time_scale_lagrangian_model, eta=lr_theta
        )
        self.admm_opt = ADMMOptimizer(admm_model, lr=lr_theta)

        self.metrics = {'softmax': defaultdict(list),
                        'lagrangian': defaultdict(list),
                        'single_time_scale_lagrangian': defaultdict(list),
                        'admm': defaultdict(list)}

    # ----------------------------------------------------------
    # one epoch
    # ----------------------------------------------------------
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, Dict[str, float]]:
        for m in (self.softmax_model, self.lagrangian_model, self.admm_model, self.single_time_scale_lagrangian_model):
            m.train()
        epoch = {'softmax': defaultdict(list), 'lagrangian': defaultdict(list), 'admm': defaultdict(list),
                 'single_time_scale_lagrangian': defaultdict(list),}
        for batch in dataloader:
            for k, v in self.train_step_softmax(batch).items():
                epoch['softmax'][k].append(v)
            for k, v in self.train_step_lagrangian(batch).items():
                epoch['lagrangian'][k].append(v)
            for k, v in self.train_step_admm(batch).items():
                epoch['admm'][k].append(v)
            for k, v in self.train_step_single_lagrangian(batch).items():
                epoch['single_time_scale_lagrangian'][k].append(v)
        return {k: {kk: np.mean(vv) for kk, vv in v.items()} for k, v in epoch.items()}

    # ----------------------------------------------------------
    # individual steps
    # ----------------------------------------------------------
    def train_step_softmax(self, batch):
        x = batch['x'].to(self.device)
        y = batch['y'].to(self.device)
        y_pred, meta = self.softmax_model(x)
        recon = F.mse_loss(y_pred, y)
        sparse = 0.1 * (1 - meta['sparsity'])
        loss = recon + sparse
        self.softmax_opt.zero_grad()
        loss.backward()
        self.softmax_opt.step()
        return {'loss': loss.item(), 'mse': recon.item(), 'recon_loss': recon.item(),
                'sparsity': meta['sparsity'].item(),
                'weight_entropy': -(meta['weights'] * torch.log(meta['weights'] + 1e-6)).sum(-1).mean().item()}

    def train_step_lagrangian(self, batch):
        x = batch['x'].to(self.device)
        y = batch['y'].to(self.device)
        self.lagrangian_opt.zero_grad()
        y_pred, meta = self.lagrangian_model(x)
        mse = F.mse_loss(y_pred, y)
        lag_loss, stats = self.lagrangian_model.augmented_lagrangian_loss(x, y, meta['outputs'], meta['confidences'])
        stats['loss'] = lag_loss
        self.lagrangian_opt.step(stats)
        return {'loss': lag_loss.item(), 'mse': mse.item(), 'weighted_loss': stats['weighted_loss'].item(),
                'constraint_violation': stats['g_lambda'].abs().item(),
                'min_weight': meta['weights'].min().item(), 'max_weight': meta['weights'].max().item()}

    def train_step_single_lagrangian(self, batch):
        x = batch['x'].to(self.device)
        y = batch['y'].to(self.device)
        self.single_time_scale_lagrangian_opt.zero_grad()
        y_pred, meta = self.single_time_scale_lagrangian_model(x)
        mse = F.mse_loss(y_pred, y)
        lag_loss, stats = self.single_time_scale_lagrangian_model.augmented_lagrangian_loss(x, y, meta['outputs'], meta['confidences'])
        stats['loss'] = lag_loss
        self.single_time_scale_lagrangian_opt.step(stats)
        return {'loss': lag_loss.item(), 'mse': mse.item(), 'weighted_loss': stats['weighted_loss'].item(),
                'constraint_violation': stats['g_lambda'].abs().item(),
                'min_weight': meta['weights'].min().item(), 'max_weight': meta['weights'].max().item()}
    
    def train_step_admm(self, batch):
        x = batch['x'].to(self.device)
        y = batch['y'].to(self.device)
        self.admm_opt.zero_grad()
        y_pred, meta = self.admm_model(x)
        mse = F.mse_loss(y_pred, y)
        lag_loss, stats = self.admm_model.admm_loss(x, y, meta['outputs'], meta['confidences'])
        stats['loss'] = lag_loss
        self.admm_opt.step(stats)
        return {'loss': lag_loss.item(), 'mse': mse.item(), 'recon_loss': stats['recon_loss'].item(),
                'consensus': stats['consensus'].item(),
                'min_weight': meta['weights'].min().item(), 'max_weight': meta['weights'].max().item()}

    # ----------------------------------------------------------
    # validation
    # ----------------------------------------------------------
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, Dict[str, float]]:
        for m in (self.softmax_model, self.lagrangian_model, self.admm_model):
            m.eval()
        val = {'softmax': defaultdict(list), 'lagrangian': defaultdict(list), 'admm': defaultdict(list),
               'single_time_scale_lagrangian': defaultdict(list),}
        for batch in dataloader:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            for name, model in [('softmax', self.softmax_model),
                                ('lagrangian', self.lagrangian_model),
                                ('admm', self.admm_model),
                                ('single_time_scale_lagrangian', self.single_time_scale_lagrangian_model)]:
                pred, _ = model(x)
                loss = F.mse_loss(pred, y)
                val[name]['loss'].append(loss.item())
                val[name]['mse'].append(loss.item())
        return {k: {kk: np.mean(vv) for kk, vv in v.items()} for k, v in val.items()}


# ------------------------------------------------------------------
# 7. Plotting utilities  (your original code – unchanged)
# ------------------------------------------------------------------
def plot_predictions(models: Dict[str, nn.Module], test_loader: DataLoader, device: str,
                     save_path: str, epoch: int, num_samples: int = 4):
    batch = next(iter(test_loader))
    x, y = batch['x'].to(device), batch['y'].to(device)
    indices = np.random.choice(len(x), num_samples, replace=False)
    colors = {'Initial': 'gray', 'True': 'black', 'Softmax': 'blue',
              'Lagrangian': 'red', 'ADMM': 'green', 'single_time_scale_lagrangian_model': 'orange'}
    preds = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            pred, meta = model(x)
            preds[name] = {'pred': pred.detach().cpu(), 'meta': {k: v.detach().cpu() if torch.is_tensor(v) else v
                                                                 for k, v in meta.items()}}
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = [axes]
    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.plot(range(len(x[idx])), x[idx].cpu(), '-', color=colors['Initial'], label='Initial', alpha=0.5)
        ax.plot(range(len(y[idx])), y[idx].cpu(), '-', color=colors['True'], label='True', alpha=0.7)
        for name, dic in preds.items():
            p = dic['pred'][idx]
            ax.plot(range(len(p)), p, '--', color=colors[name], label=name, alpha=0.7)
        ax.set_title(f'Sample {i+1}')
        ax.legend()
        ax.grid(True)
    os.makedirs(save_path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{save_path}/predictions_epoch_{epoch}.png')
    plt.close()


def plot_error_distribution(models: Dict[str, nn.Module], test_loader: DataLoader, device: str,
                            save_path: str, epoch: int):
    errors = {n: [] for n in models}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch['x'].to(device), batch['y'].to(device)
                pred, _ = model(x)
                errors[name].extend(torch.abs(pred - y).mean(dim=1).cpu().numpy())
    plt.figure(figsize=(10, 6))
    for name, err in errors.items():
        plt.hist(err, bins=50, alpha=0.5, label=name)
    plt.title('Error Distribution')
    plt.xlabel('Mean Absolute Error')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/error_dist_epoch_{epoch}.png')
    plt.close()


# ------------------------------------------------------------------
# 8. Main  (extended with ADMM)
# ------------------------------------------------------------------
def main():
    n_samples, input_dim, n_sources, batch_size, n_epochs = 1000, 64, 128, 32, 200
    save_dir = 'results'

    dataset = NavierStokes1DDataset(n_samples, input_dim, n_sources)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    softmax_model = LargeScaleSourceIntegration(n_sources=n_sources, input_dim=input_dim, hidden_dim=128, sparse_topk=10)
    lagrangian_model = LagrangianSourceIntegration(n_sources=n_sources, input_dim=input_dim, hidden_dim=128, sparse_topk=10, rho=1.0)
    single_time_scale_lagrangian_model = LagrangianSourceIntegration(n_sources=n_sources, input_dim=input_dim, hidden_dim=128, sparse_topk=10, rho=1.0)
    admm_model = ADMMSourceIntegration(n_sources=n_sources, input_dim=input_dim, hidden_dim=128, rho=1.0)

    trainer = ComparativeTrainer(softmax_model=softmax_model,
                                 lagrangian_model=lagrangian_model,
                                 single_time_scale_lagrangian_model=single_time_scale_lagrangian_model,
                                 admm_model=admm_model,
                                 lr_softmax=1e-4,
                                 lr_theta=1e-4,
                                 lr_lambda=1e-4)

    for epoch in range(n_epochs):
        print(f'\nEpoch {epoch+1}/{n_epochs}')
        metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)

        for k, v in metrics['softmax'].items():
            trainer.metrics['softmax'][k].append(v)
        for k, v in metrics['lagrangian'].items():
            trainer.metrics['lagrangian'][k].append(v)
        for k, v in metrics['admm'].items():
            trainer.metrics['admm'][k].append(v)
        for k, v in metrics['single_time_scale_lagrangian'].items():
            trainer.metrics['single_time_scale_lagrangian'][k].append(v)

        print('Softmax  – train MSE:', metrics['softmax']['mse'], 'val MSE:', val_metrics['softmax']['mse'])
        print('Lagrangian – train MSE:', metrics['lagrangian']['mse'], 'val MSE:', val_metrics['lagrangian']['mse'])
        print('Single-time-scale Lagrangian – train MSE:', metrics['single_time_scale_lagrangian']['mse'],
              'val MSE:', val_metrics['single_time_scale_lagrangian']['mse'])
        print('ADMM     – train MSE:', metrics['admm']['mse'], 'val MSE:', val_metrics['admm']['mse'])

        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            models = {'Softmax': softmax_model, 'Lagrangian': lagrangian_model, 'single_time_scale_lagrangian_model': single_time_scale_lagrangian_model, 
                    'ADMM': admm_model}
            plot_predictions(models, val_loader, trainer.device, save_dir, epoch + 1)
            plot_error_distribution(models, val_loader, trainer.device, save_dir, epoch + 1)


if __name__ == '__main__':
    main()