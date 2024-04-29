from dataclasses import dataclass

import jax
import torch
import numpy as np
from jax import numpy as jnp
from scipy.stats import gaussian_kde
from tqdm.auto import trange
import matplotlib.pyplot as plt
import ot


def coordinate_total_variation(
    true: torch.tensor,
    other: torch.tensor,
    n_kde_samples: int = 1000,
    mode='mean',  # can be either 'mean' or 'max'
):
    D = true.shape[-1]
    total_variations = []
    for i in range(D):
        total_variations.append(total_variation_1d(true[..., i], other[..., i], n_kde_samples))
    if mode == 'mean':
        return sum(total_variations) / D
    if mode == 'max':
        return max(total_variations)
    raise ValueError('Invalid mode')


class ValueTracker:
    def __init__(self):
        self.values = []

    def update(self, value: float) -> None:
        self.values.append(value)

    def __len__(self):
        return len(self.values)

    def mean(self) -> float:
        return jnp.mean(jnp.array(self.values))

    def std(self) -> float:
        return jnp.std(jnp.array(self.values), ddof=1)

    def std_of_mean(self) -> float:
        return jnp.std(jnp.array(self.values)) / jnp.sqrt(len(self))

    def max(self):
        return jnp.max(jnp.array(self.values))

    def last(self) -> float:
        return self.values[-1]


@dataclass
class Projector:
    x0: jnp.ndarray
    v: jnp.ndarray

    def project(self, xs: jnp.ndarray) -> jnp.ndarray:
        return (xs - self.x0[None]) @ self.v


def create_random_projection(key: jnp.ndarray, xs: jnp.ndarray) -> Projector:
    x0 = jnp.mean(xs, 0)
    v = jax.random.normal(key, [len(x0)])
    v = v / jnp.linalg.norm(v)
    return Projector(x0, v)


def sliced_total_variation(
    true: torch.tensor,
    other: torch.tensor,
    n_projections: int,
    n_kde_samples: int,
    mode='mean',  # can be either 'mean' or 'max'
):
    
    true = to_jax(true)
    other = to_jax(other)
    tracker = ValueTracker()
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, n_projections)
    for i in range(n_projections):  # can use trange
        tracker.update(random_projection_total_variation(keys[i], true, other, n_kde_samples))
    
    if mode == 'mean':
        return tracker.mean()
    if mode == 'max':
        return tracker.max()
    raise ValueError('Invalid mode')


def random_projection_total_variation(
    key: jnp.ndarray,
    xs_true: jnp.ndarray,
    xs_pred: jnp.ndarray,
    n_kde_samples: int,
):
    proj = create_random_projection(key, xs_true)
    return total_variation_1d(
        proj.project(xs_true),
        proj.project(xs_pred),
        n_kde_samples,
        is_input_jax=True
    )


def total_variation_1d(xs_true, xs_pred, n_samples=1000, is_input_jax=False):
    if not is_input_jax:
        xs_true = to_jax(xs_true)
        xs_pred = to_jax(xs_pred)
    true_density = gaussian_kde(xs_true)
    pred_density = gaussian_kde(xs_pred)

    x_min = min(xs_true.min(), xs_pred.min())
    x_max = max(xs_true.max(), xs_pred.max())

    points = np.linspace(x_min, x_max, n_samples)

    return (
        0.5
        * np.abs(true_density(points) - pred_density(points)).mean()
        * (x_max - x_min)
    )


# torch.tensor to jax.numpy.ndarray
def to_jax(tensor: torch.tensor):
    return jnp.array(tensor.detach().cpu().numpy())


def wasserstein_metric(sample1, sample2):
    a = torch.ones(sample1.shape[0]) / sample1.shape[0]
    b = torch.ones(sample2.shape[0]) / sample2.shape[0]
    M = ot.dist(sample1, sample2)
    gamma, log = ot.emd(a, b, M, log=True, numThreads='max', numItermax=500_000)
    return log['cost']


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