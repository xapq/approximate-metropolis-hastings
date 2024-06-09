from dataclasses import dataclass
from abc import ABC, abstractmethod
import jax
from jax import numpy as jnp
from jax.scipy.stats import gaussian_kde
import torch
import ot

from .utilities import torch_to_jax


class DistributionMetric(ABC):
    @abstractmethod
    def __call__(self, sample1, sample2):
        pass
    
    @abstractmethod
    def name():
        pass


# W_2
class WassersteinMetric(DistributionMetric):
    def __call__(self, sample1, sample2):
        a = torch.ones(sample1.shape[0]) / sample1.shape[0]
        b = torch.ones(sample2.shape[0]) / sample2.shape[0]
        M = ot.dist(sample1, sample2)
        W = ot.emd2(a, b, M, numThreads='max', numItermax=500_000) ** 0.5
        return W
    
    def name(self):
        return 'Wasserstein Metric'


# W_2
class WassersteinMetric1d(DistributionMetric):
    def __call__(self, sample1, sample2):
        cost = ot.wasserstein_1d(sample1, sample2, p=2) ** 0.5
        return cost
    
    def name(self):
        return 'Wasserstein Metric'


class TotalVariation1d(DistributionMetric):
    def __init__(self, n_density_samples=1000):
        self.n_density_samples = n_density_samples

    "All dimensions except the last one are considered batch dimensions"
    def __call__(self, sample1, sample2):
        sample1 = torch_to_jax(sample1)
        sample2 = torch_to_jax(sample2)
        density1 = gaussian_kde(sample1)
        density2 = gaussian_kde(sample2)
        x_min = jnp.minimum(sample1.min(axis=-1), sample2.min(axis=-1))
        x_max = jnp.maximum(sample1.max(axis=-1), sample2.max(axis=-1))
        eval_points = jnp.linspace(x_min, x_max, self.n_density_samples)
        return (
            0.5
            * jnp.abs(density1(eval_points) - density2(eval_points)).mean()
            * (x_max - x_min)
        ).item()

    def name(self):
        return 'Total Variation'


### TODO: add full support for batched data
class SlicedDistributionMetric(DistributionMetric):
    def __init__(self, metric_1d, n_projections, mode='mean'):
        self.metric_1d = metric_1d
        self.n_projections = n_projections
        self.mode = mode

    def __call__(self, sample1, sample2):
        data_dim = sample1.shape[-1]
        assert(sample2.shape[-1] == data_dim)
        projection_vectors = torch.randn(data_dim, self.n_projections, device=sample1.device)
        sample1_projections = torch.matmul(sample1, projection_vectors)
        sample2_projections = torch.matmul(sample2, projection_vectors)
        metric_1d_values = []
        for i in range(self.n_projections):
            metric_1d_values.append(
                self.metric_1d(sample1_projections[..., i], sample2_projections[..., i])
            )
        if self.mode == 'mean':
            return sum(metric_1d_values) / self.n_projections
        if self.mode == 'max':
            return max(metric_1d_values)
        raise ValueError('Invalid mode')

    def name(self):
        return f'{self.n_projections}-{self.mode.capitalize()} Sliced {self.metric_1d.name()}'


### TODO: remove duplicate code with SlicedDistributionMetric
class CoordinateDistributionMetric(DistributionMetric):
    def __init__(self, metric_1d, mode='mean'):
        self.metric_1d = metric_1d
        self.mode = mode

    def __call__(self, sample1, sample2):
        assert(sample1.shape[-1] == sample2.shape[-1])
        metric_1d_values = []
        for i in range(self.n_projections):
            metric_1d_values.append(
                self.metric_1d(sample1[..., i], sample2[..., i])
            )
        if self.mode == 'mean':
            return sum(metric_1d_values) / self.n_projections
        if self.mode == 'max':
            return max(metric_1d_values)
        raise ValueError('Invalid mode')

    def name(self):
        return f'{self.mode.capitalize()} Coordinate {self.metric_1d.name()}'


# def average_emd(
#     key: jnp.ndarray, true: jnp.ndarray, other: jnp.ndarray, n_kde_samples: int, n_projections: int
# ) -> ValueTracker:
#     tracker = ValueTracker()
#     keys = jax.random.split(key, n_projections)
#     for i in trange(n_projections, leave=False):
#         tracker.update(emd_2d(keys[i], true, other, n_kde_samples))
#     return tracker


# def emd_2d(key: jnp.ndarray, xs_true: jnp.ndarray, xs_pred: jnp.ndarray, n_kde_samples: int):
#     proj = create_random_2d_projection(key, xs_true)
#     return earth_movers_distance_2d(proj.project(xs_true), proj.project(xs_pred), n_kde_samples)


# def earth_movers_distance_2d(xs_true, xs_pred, n_kde_samples):
#     print(xs_true.shape, xs_pred.shape)
#     M = np.linalg.norm(xs_true[None, :, :] - xs_pred[:, None, :], axis=-1, ord=2)**2
#     emd = ot.lp.emd2([], [], M)
#     return emd