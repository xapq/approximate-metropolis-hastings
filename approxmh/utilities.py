import torch
import jax
from jax import numpy as jnp
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

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

def plot_2d_torch_function(ax, func, xlim, ylim, d, device='cpu', **kwargs):
    x = np.linspace(*xlim, d)
    y = np.linspace(*ylim, d)
    X, Y = np.meshgrid(x, y)
    X, Y = torch.tensor(X), torch.tensor(Y)
    points = torch.zeros(d, d, 2, device=device)
    points[..., 0] = X
    points[..., 1] = Y
    values = func(points.flatten(end_dim=1)).unflatten(dim=0, sizes=(d, d))
    img = ax.imshow(values.detach().cpu(), origin='lower', extent=[*xlim, *ylim], aspect='auto', **kwargs)
    return img

