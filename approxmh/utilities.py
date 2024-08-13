import torch
import jax
from jax import numpy as jnp
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt
from .y_utils import *


PROJECT_PATH = Path(__file__).parent.parent
MODEL_DIR = Path(PROJECT_PATH, "models")
CHECKPOINT_DIR = Path(PROJECT_PATH, "gan_checkpoints")


# torch.tensor to jax.numpy.ndarray
### TODO: make this work for jax on cuda
def torch_to_jax(tensor):
    if isinstance(tensor, jnp.ndarray):
        return tensor
    return jnp.array(tensor.detach().cpu().numpy())


def estimate_mean_and_std(function, n_runs=10, *args, **kwargs):
    values = torch.tensor([function(*args, **kwargs) for _ in range(n_runs)])
    mean = values.mean(dim=0)
    std = values.std(dim=0)
    return mean, std


def estimate_quartiles(function, n_runs=10, *args, **kwargs):
    values = torch.tensor([function(*args, **kwargs) for i in range(n_runs)])
    lower = values.quantile(q=0.25, interpolation='higher')
    upper = values.quantile(q=0.75, interpolation='lower')
    return lower, upper


# Blame the developers of torch.distributions.mixture_same_family.sample()
def sample_by_batches(distribution, n_samples, batch_size):
    n_batches = (n_samples - 1) // batch_size + 1
    return torch.cat([distribution.sample((batch_size,)) for _ in range(n_batches)])[:n_samples]


def dataloader_from_tensor(X, batch_size):
    dataset = TensorDataset(X)
    dataloader = DataLoader(X, batch_size=batch_size, shuffle=True)
    return dataloader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_on_grid(func, xlim, ylim, d, device='cpu'):
    x = np.linspace(*xlim, d)
    y = np.linspace(*ylim, d)
    X, Y = np.meshgrid(x, y)
    X, Y = torch.tensor(X), torch.tensor(Y)
    points = torch.zeros(d, d, 2, device=device)
    points[..., 0] = X
    points[..., 1] = Y
    return func(points.flatten(end_dim=1)).unflatten(dim=0, sizes=(d, d))

def plot_2d_torch_function(ax, func, xlim, ylim, dpi, device='cpu', **kwargs):
    values = evaluate_on_grid(func, xlim, ylim, dpi, device=device)
    img = ax.imshow(values.detach().cpu(), origin='lower', extent=[*xlim, *ylim], aspect='auto', **kwargs)
    return img


def visualize_distribution(distribution, xlim=None, ylim=None, levels=30, dpi=30, proj_dims=(0, 1), plot_samples=True, sample_size=4000):
    fixed_coordinates = torch.zeros(distribution.dim)
    fig, ax = plt.subplots(figsize=(5, 5))

    if plot_samples:
        # Implementation of torch.distributions.mixture_same_family.sample() causes MemoryOverflow without batched sampling
        sample = sample_by_batches(distribution, sample_size, batch_size=1024)[:, proj_dims]
        ax.scatter(*pl(sample), zorder=4, alpha=0.5, s=150, edgecolors='none', marker='.')
        if xlim is None:
            xlim = (sample[:, 0].min().item(), sample[:, 0].max().item())
        if ylim is None:
            ylim = (sample[:, 1].min().item(), sample[:, 1].max().item())

    x = torch.linspace(*xlim, dpi)
    y = torch.linspace(*ylim, dpi)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = fixed_coordinates.repeat(dpi * dpi, 1)
    points[:, proj_dims[0]] = X.ravel()
    points[:, proj_dims[1]] = Y.ravel()
    Z = distribution.log_prob(points.to(distribution.device)).reshape(dpi, dpi)
    
    ax.contour(to_numpy(X), to_numpy(Y), to_numpy(Z), levels=levels)
    # ax.imshow(to_numpy(Z), origin='lower', extent=[*xlim, *ylim], aspect='auto')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(f'x_{proj_dims[0]}')
    ax.set_ylabel(f'x_{proj_dims[1]}')
    ax.set_title(f'{distribution.friendly_name} {" (Projection)" if distribution.dim > 2 else ""}')
