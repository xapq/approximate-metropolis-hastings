from abc import ABC, abstractmethod
import numpy as np
import numpy.random as rng
from numpy.random import default_rng
import torch
import torch.distributions as td
import torch.nn.functional as F
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from torch import nn
import math

# from .linear_regression import RegressionDataset
# from .logistic_regression import ClassificationDataset


torchType = torch.float32


class Distribution(ABC):
    """
    Base class for a custom target distribution
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.device = kwargs.get("device", "cpu")
        self.torchType = torchType
        self.xlim, self.ylim = [-1, 1], [-1, 1]
        self.scale_2d_log_prob = 1
        # self.device_zero = torch.tensor(0., dtype=self.torchType, device=self.device)
        # self.device_one = torch.tensor(1., dtype=self.torchType, device=self.device)

    def prob(self, x):
        """
        The method returns target density, estimated at point x
        Input:
        x - datapoint
        Output:
        density - p(x)
        """
        # You should define the class for your custom distribution
        return self.log_prob(x).exp()

    @abstractmethod
    def log_prob(self, x):
        """
        The method returns target logdensity, estimated at point x
        Input:
        x - datapoint
        Output:
        log_density - log p(x)
        """
        # You should define the class for your custom distribution
        raise NotImplementedError

    def energy(self, x):
        """
        The method returns target logdensity, estimated at point x
        Input:
        x - datapoint
        Output:
        energy = -log p(x)
        """
        # You should define the class for your custom distribution
        return -self.log_prob(x)

    def sample(self, n):
        """
        The method returns samples from the distribution
        Input:
        n - amount of samples
        Output:
        samples - samples from the distribution
        """
        # You should define the class for your custom distribution
        raise NotImplementedError

    def __call__(self, x):
        return self.log_prob(x)

    def log_prob_2d_slice(self, z):
        raise NotImplementedError

    def plot_2d(self, fig=None, ax=None):
        if fig is None and ax is None:
            fig, ax = plt.subplots()

        x = np.linspace(*self.xlim, 100)
        y = np.linspace(*self.ylim, 100)
        xx, yy = np.meshgrid(x, y)
        z = torch.FloatTensor(np.stack([xx, yy], -1))
        vals = (self.log_prob_2d_slice(z) / self.scale_2d_log_prob).exp()

        if ax is not None:
            ax.imshow(
                vals.flip(0),
                extent=[*self.xlim, *self.ylim],
                cmap="Greens",
                alpha=0.5,
                aspect="auto",
            )
        else:
            plt.imshow(
                vals.flip(0),
                extent=[*self.xlim, *self.ylim],
                cmap="Greens",
                alpha=0.5,
                aspect="auto",
            )

        return fig, self.xlim, self.ylim

def create_gaussian_mixture(means, cov_matricies):
    return td.MixtureSameFamily(
        td.Categorical(torch.ones(means.shape[0], device=means.device)),
        td.MultivariateNormal(means, cov_matricies)
    )


def create_gaussian_lattice(shape, step, variance, device='cpu'):
    dim = len(shape)
    cov_matrix = variance * torch.eye(dim, device=device)
    means = torch.stack(torch.meshgrid(
        *(float(step) * torch.arange(d_i) for d_i in shape),
        indexing='xy'
    )).flatten(start_dim=1).T.to(device)
    return create_gaussian_mixture(means, cov_matrix)

def create_random_gaussian_mixture(dim, n_components, mean_lim=(0, 1), variance_lim=(0.1, 1), device='cpu', seed=None):
    gen = torch.Generator()
    if seed is None:
        gen.seed()
    else:
        gen.manual_seed(seed)
    scale_to = lambda x, lim: x * (lim[1] - lim[0]) + lim[0]
    means = scale_to(torch.rand(n_components, dim, generator=gen), mean_lim)
    principal_components = torch.linalg.qr(torch.rand(n_components, dim, dim, generator=gen)).Q
    component_variances = scale_to(torch.rand(n_components, dim, 1, generator=gen), variance_lim) * torch.eye(dim)
    cov_matricies = principal_components @ component_variances @ principal_components.transpose(-2, -1)
    return create_gaussian_mixture(means.to(device), cov_matricies.to(device))

# returns (n_mixture_components, n_samples)-shaped tensor
def scaled_mahalanobis_distance(gaussian_mixture, x):
    batched_normal = gaussian_mixture.component_distribution
    means = batched_normal.mean  # shape (n_comp, d)
    precision_matricies = batched_normal.precision_matrix  # shape (n_comp, d, d)
    deltas = x - means.unsqueeze(-2)  # shape (n_comp, n_samples, d)
    squared_norm = torch.matmul(
        deltas.unsqueeze(-2),  # shape (n_comp, n_samples, 1, d)
        torch.matmul(
            precision_matricies.unsqueeze(-3),  # shape (n_comp, 1, d, d)
            deltas.unsqueeze(-1)  # shape (n_comp, n_sampled, d, 1)
        )
    )[..., 0, 0]
    return (squared_norm / means.shape[-1]).sqrt()

# return (n_components,) shaped tensor, containing number of samples within `k` stds for each component
def get_mode_coverage(gaussian_mixture, x, k=2):
    dists = scaled_mahalanobis_distance(gaussian_mixture, x)
    counts = (dists < k).sum(axis=-1)
    return counts


class IndependentMultivariateNormal(Distribution):
    def __init__(self, mean, std, **kwargs):
        super().__init__()
        self.mean = mean  # (*batch_dims, data_dim)
        self.std = std
        self.batch_dims = self.mean.shape[:-1]
        self.data_dim = self.mean.shape[-1]
        self.batch_idxs = torch.arange(len(self.batch_dims))

    # takes (*sample_shape, *batch_dims, data_dim)-tensor and returns (*sample_shape, *batch_dims)-tensor
    def log_prob(self, x):
        # x = x.movedim(self.batch_idxs, len(sample_shape) + self.batch_idxs)
        log_density = (
            -((x - self.mean) ** 2) / (2 * self.std ** 2)
            - self.std.log()
            - math.log(math.sqrt(2 * math.pi))
        ).sum(axis=-1)
        # log_density = log_density.movedim(len(sample_shape) + self.batch_idxs, self.batch_idxs)
        return log_density
    
    # returns (*sample_shape, *batch_dims, data_dim)-shaped tensor
    def rsample(self, sample_shape=torch.Size([])):
        x = torch.randn(*sample_shape, *self.batch_dims, self.data_dim, device=self.std.device)
        x *= self.std
        x += self.mean
        # x = x.movedim(len(sample_shape) + self.batch_idxs, self.batch_idxs)
        return x

    def sample(self, sample_shape=torch.Size([])):
        with torch.no_grad():
            return self.rsample(sample_shape)


class DoubleWell(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = kwargs.get("device", "cpu")
        self.a = kwargs.get("a", 1.0)
        self.b = kwargs.get("b", 1.0)
        self.c = kwargs.get("b", 1.0)
        self.d = kwargs.get("b", 1.0)

    def log_prob(self, xy):
        x = xy[..., 0]
        y = xy[..., 1]
        return -self.a * x**4 / 4 + self.b * x**2 / 2 - self.c * x - self.d * y**2 / 2


class Serpentine(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = kwargs.get("device", "cpu")
        self.n_sections = kwargs.get("n_sections", 1)
        self.section_width = kwargs.get("section_width", 1.)
        self.fence_length = kwargs.get("fence_length", 0.8)
        self.height = 1.
        self.width = self.n_sections * self.section_width

    def log_prob(self, x):
        res = torch.zeros(x.shape[:-1], device=self.device)
        res[(x[..., 0] < 0.) ^ (x[..., 0] > self.width) ^ (x[..., 1] < 0.) ^ (x[..., 1] > self.height)] = -100.
        return res

class Funnel(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = kwargs.get("device", "cpu")
        self.a = (kwargs.get("a", 1.0) * torch.ones(1)).to(self.device)
        self.dim = kwargs.get("dim", 16)
        self.distr1 = torch.distributions.Normal(torch.zeros(1).to(self.device), self.a)
        # self.distr2 = lambda z1: torch.distributions.MultivariateNormal(torch.zeros(self.dim-1), (2*self.b*z1).exp()*torch.eye(self.dim-1))
        # self.distr2 = lambda z1: -(z[...,1:]**2).sum(-1) * (-2*self.b*z1).exp() - np.log(self.dim) + 2*self.b*z1
        self.xlim = [-2, 10]
        self.ylim = [-30, 30]
        self.scale_2d_log_prob = 20  # 30.0

    def log_prob(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Returns:
            log p(x)
        """

        normal_first = torch.distributions.Normal(torch.zeros(x.shape[:-1], device=x.device), torch.exp(x[..., 0] / 2.))
        return normal_first.log_prob(x[..., 1:].permute(-1, *range(x.ndim-1))).sum(0) + \
            self.distr1.log_prob(x[..., 0])

    def sample(self, sample_shape):
        samples  = torch.randn(*sample_shape, self.dim, device=self.device)
        samples[..., 0] *= self.a
        samples[..., 1:] *= (samples[..., 0].unsqueeze(-1) / 2).exp()
        return samples

    def log_prob_2d_slice(self, z, dim1=0, dim2=1):
        if dim1 == 0 or dim2 == 0:
            logprob1 = self.distr1.log_prob(z[..., 0])
            dim2 = dim2 if dim2 != 0 else dim1
            z1 = z[..., 0]
            # logprob2 = self.distr2(z[...,0])
            logprob2 = (
                -0.5 * (z[..., dim2] ** 2) * torch.exp(-2 * self.b * z1) - self.b * z1
            )
        # else:
        #     logprob2 = -(z[...,dim2]**2) * (-2*self.b*z1).exp() - np.log(self.dim) + 2*self.b*z1
        return logprob1 + logprob2

    def plot_2d_countour(self, ax):
        x = np.linspace(-15, 15, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        inp = torch.from_numpy(np.stack([X, Y], -1))
        Z = self.log_prob(inp.reshape(-1, 2)).reshape(inp.shape[:-1])

        # levels = np.quantile(Z, np.linspace(0.9, 0.99, 5))
        ax.contour(
            X,
            Y,
            Z.exp(),
            # levels = levels,
            levels=3,
            alpha=1.0,
            cmap="inferno",
        )

    def plot_2d(self, fig=None, ax=None):
        if fig is None and ax is None:
            fig, ax = plt.subplots()

            xlim = [-2, 13]
            ylim = [-60, 60]
        else:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

        x = np.linspace(*xlim, 100)
        y = np.linspace(*ylim, 100)
        xx, yy = np.meshgrid(x, y)
        z = torch.FloatTensor(np.stack([xx, yy], -1))
        vals = (self.log_prob_2d_slice(z) / self.scale_2d_log_prob).exp()
        if ax is not None:
            ax.imshow(
                vals.flip(0),
                extent=[*xlim, *ylim],
                cmap="Greens",
                alpha=0.5,
                aspect="auto",
            )
        else:
            plt.imshow(
                vals.flip(0),
                extent=[*xlim, *ylim],
                cmap="Greens",
                alpha=0.5,
                aspect="auto",
            )

        return fig, xlim, ylim


class Banana(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = kwargs.get("device", "cpu")
        self.b = kwargs.get("b", 0.02)  # * torch.ones(1).to(self.device)
        self.sigma = kwargs.get("sigma", 100.0)  # * torch.ones(1).to(self.device)
        self.dim = kwargs.get("dim", 32)
        self.xlim = [-1, 5]
        self.ylim = [-2, 2]
        self.scale_2d_log_prob = 2.0
        # assert self.dim % 2 == 0, 'Dimension should be divisible by 2'

    def __repr__(self):
        return f"banana_dim{self.dim}_b{self.b}_sigma{self.sigma}"

    def sample(self, size=(1,)):
        even = torch.arange(0, self.dim, 2, device=self.device)
        odd = torch.arange(1, self.dim, 2, device=self.device)
        samples = torch.randn((*size, self.dim), device=self.device)
        samples[..., even] *= self.sigma
        samples[..., odd] += self.b * samples[..., even] ** 2 - (self.sigma ** 2) * self.b
        return samples

    def log_prob(self, z, x=None):
        # n = self.dim/2
        even = np.arange(0, self.dim, 2)
        odd = np.arange(1, self.dim, 2)

        ll = -0.5 * (
            z[..., odd] - self.b * z[..., even] ** 2 + (self.sigma**2) * self.b
        ) ** 2 - ((z[..., even]) ** 2) / (2 * self.sigma**2)
        return ll.sum(-1)

    def log_prob_2d_slice(self, z, dim1=0, dim2=1):
        if dim1 % 2 == 0 and dim2 % 2 == 1:
            ll = (
                -((z[..., dim1] - z[..., dim2] ** 2) ** 2) / self.Q
                - (z[..., dim1] - 1) ** 2
            )
        return ll  # .sum(-1)
        