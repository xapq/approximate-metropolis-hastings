from abc import ABC, abstractmethod
import torch
import torch.distributions as td
import numpy as np
import math


class Distribution(ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = kwargs.get("device", "cpu")
        self.dim = kwargs.get("dim")
        self.name = kwargs.get("name")
        self.torchType = torch.float32  # ?

    def __call__(self, x):
        return self.log_prob(x)

    def __repr__(self):
        return self.name

    def prob(self, x):
        return self.log_prob(x).exp()

    @abstractmethod
    def log_prob(self, x):
        raise NotImplementedError

    def energy(self, x):
        return -self.log_prob(x)

    def rsample(self, sample_shape):
        raise NotImplementedError
    
    def sample(self, sample_shape=torch.Size([])):
        with torch.no_grad():
            return self.rsample(sample_shape)


class GaussianMixture(Distribution):
    def __init__(self, means, cov_matricies, **kwargs):
        super().__init__(device=means.device, dim=means.shape[-1], **kwargs)
        self.n_components = means.shape[-2]
        self.torch_base = td.MixtureSameFamily(
            td.Categorical(torch.ones(self.n_components, device=self.device)),
            td.MultivariateNormal(means, cov_matricies)
        )
        self.friendly_name = kwargs.get("friendly_name", f'{self.n_components}-Component MoG')

    def sample(self, sample_shape=torch.Size([])):
        return self.torch_base.sample(sample_shape)

    def log_prob(self, x):
        return self.torch_base.log_prob(x)


class PoorlyConditionedGaussian(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min_variance = kwargs.get("min_variance", 1e-1)
        self.max_variance = kwargs.get("max_variance", 1e+1)
        self.variances = torch.tensor(np.geomspace(self.min_variance, self.max_variance, num=self.dim), device=self.device)
        gen = torch.Generator(device=self.device)
        if "seed" in kwargs:
            self.seed = kwargs["seed"]
            gen.manual_seed(self.seed)
        else:
            self.seed = gen.seed()
        self.Q = torch.linalg.qr(torch.rand(self.dim, self.dim, generator=gen, device=self.device)).Q
        self.name = f'pcg_{self.min_variance}_{self.max_variance}_seed{self.seed}'
        self.friendly_name = f'{self.dim}D Poorly Conditioned Gaussian'

    def rsample(self, sample_shape=torch.Size([])):
        result = torch.randn(*sample_shape, self.dim, device=self.device)
        result *= self.variances ** 0.5
        result = result @ self.Q.T
        return result

    def log_prob(self, x):
        x = x @ self.Q
        x /= self.variances ** 0.5
        return (-x ** 2 / 2).sum(axis=-1)


def create_gaussian_lattice(shape, variance, device='cpu'):
    dim = len(shape)
    cov_matrix = variance * torch.eye(dim, device=device)
    means = torch.stack(torch.meshgrid(
        *(1. * torch.arange(d_i) for d_i in shape),
        indexing='xy'
    )).flatten(start_dim=1).T.to(device)
    return GaussianMixture(
        means, cov_matrix, 
        name=f'gaussian_lattice{shape}_var{variance}',
        friendly_name=f'{"x".join(map(str, shape))} Gaussian Lattice'
    )


def create_random_gaussian_mixture(dim, n_components, mean_lim=(0, 1), variance_lim=(0.1, 1), device='cpu', seed=None):
    gen = torch.Generator()
    if seed is None:
        seed = gen.seed()
    else:
        gen.manual_seed(seed)
    scale_to = lambda x, lim: x * (lim[1] - lim[0]) + lim[0]
    means = scale_to(torch.rand(n_components, dim, generator=gen), mean_lim)
    principal_components = torch.linalg.qr(torch.rand(n_components, dim, dim, generator=gen)).Q
    component_variances = scale_to(torch.rand(n_components, dim, 1, generator=gen), variance_lim) * torch.eye(dim)
    cov_matricies = principal_components @ component_variances @ principal_components.transpose(-2, -1)
    return GaussianMixture(
        means.to(device), cov_matricies.to(device),
        name=f'mog{n_components}_{dim}d_vlim{variance_lim}_seed{seed}'
    )


def create_serpentine(n_sections=1, section_width=1., section_height=5., device='cpu'):
    thickness = 0.05
    length = 0.25
    n_components = 2 * n_sections - 1
    means = torch.zeros((n_components, 2), device=device)
    stds = torch.ones((n_components, 2), device=device) * thickness
    for i in range(n_sections):
        means[2 * i][0] = i * section_width
        stds[2 * i][1] = length * section_height
        if i + 1 < n_sections:
            means[2 * i + 1][0] = (i + 0.5) * section_width
            means[2 * i + 1][1] = 0.5 * (-1)**(i%2) * section_height
            stds[2 * i + 1][0] = length * section_width
    cov_matricies = torch.diag_embed(stds ** 2)
    return GaussianMixture(
        means, cov_matricies,
        name=f'{n_sections}sec_{section_width}x{section_height}_len{length}_thick{thickness}_serpentine',
        friendly_name=f'{n_sections}-Section Serpentine'
    )


class IndependentMultivariateNormal(Distribution):
    def __init__(self, mean, std, **kwargs):
        super().__init__(dim=mean.shape[-1], device=mean.device, **kwargs)
        self.mean = mean  # (*batch_dims, data_dim)
        self.std = std    # (*batch_dims, data_dim)
        self.batch_dims = self.mean.shape[:-1]

    def __getitem__(self, key):
        if len(self.batch_dims) == 0:
            raise ValueError('Cannot index distribution that is not batched')
        return IndependentMultivariateNormal(self.mean[key], self.std[key])

    @property
    def friendly_name(self):
        return 'Independent Multivariate Normal'

    # takes (*sample_shape, *batch_dims, data_dim)-tensor and returns (*sample_shape, *batch_dims)-tensor
    def log_prob(self, x):
        log_density = (
            -((x - self.mean) ** 2) / (2 * self.std ** 2)
            - self.std.log()
            - math.log(math.sqrt(2 * math.pi))
        ).sum(axis=-1)
        return log_density
    
    # returns (*sample_shape, *batch_dims, data_dim)-shaped tensor
    def rsample(self, sample_shape=torch.Size([])):
        x = torch.randn(*sample_shape, *self.batch_dims, self.dim, device=self.device)
        x *= self.std
        x += self.mean
        return x

    def move_to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.device = device


class DoubleWell(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = kwargs.get("device", "cpu")
        self.a = kwargs.get("a", 1.0)
        self.b = kwargs.get("b", 1.0)
        self.c = kwargs.get("b", 1.0)
        self.d = kwargs.get("b", 1.0)
        self.friendly_name = "Double Well"

    def log_prob(self, xy):
        x = xy[..., 0]
        y = xy[..., 1]
        return -self.a * x**4 / 4 + self.b * x**2 / 2 - self.c * x - self.d * y**2 / 2


class Funnel(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = kwargs.get("a", 1.0)
        self.distr1 = td.Normal(torch.zeros(1).to(self.device), self.a)  # Distribution along first coordinate
        self.name = f'funnel_{self.dim}d_a{self.a}'
        self.friendly_name = f'{self.dim}-Dimensional a={self.a} Funnel'

    def log_prob(self, x: torch.FloatTensor) -> torch.FloatTensor:
        normal_first = td.Normal(torch.zeros(x.shape[:-1], device=x.device), torch.exp(x[..., 0] / 2.))
        return normal_first.log_prob(x[..., 1:].permute(-1, *range(x.ndim-1))).sum(0) + \
            self.distr1.log_prob(x[..., 0])

    def sample(self, sample_shape):
        samples  = torch.randn(*sample_shape, self.dim, device=self.device)
        samples[..., 0] *= self.a
        samples[..., 1:] *= (samples[..., 0].unsqueeze(-1) / 2).exp()
        return samples


class Banana(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.b = kwargs.get("b", 0.02)
        self.sigma = kwargs.get("sigma", 100.0)
        self.name = f"banana_dim{self.dim}_b{self.b}_sigma{self.sigma}"

    def sample(self, size=(1,)):
        even = torch.arange(0, self.dim, 2, device=self.device)
        odd = torch.arange(1, self.dim, 2, device=self.device)
        samples = torch.randn((*size, self.dim), device=self.device)
        samples[..., even] *= self.sigma
        samples[..., odd] += self.b * samples[..., even] ** 2 - (self.sigma ** 2) * self.b
        return samples

    def log_prob(self, z, x=None):
        even = torch.arange(0, self.dim, 2)
        odd = torch.arange(1, self.dim, 2)
        ll = -0.5 * (
            z[..., odd] - self.b * z[..., even] ** 2 + (self.sigma**2) * self.b
        ) ** 2 - ((z[..., even]) ** 2) / (2 * self.sigma**2)
        return ll.sum(-1)


# returns (n_mixture_components, n_samples)-shaped tensor
def scaled_mahalanobis_distance(gaussian_mixture, x):
    if isinstance(gaussian_mixture, GaussianMixture):
        gaussian_mixture = gaussian_mixture.torch_base  # polymorphism :)
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
