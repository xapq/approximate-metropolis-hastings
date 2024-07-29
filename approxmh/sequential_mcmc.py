import torch
import math
from abc import ABC, abstractmethod


class DensityMixture:
    def __init__(self, distribution1, power1, distribution2, power2):
        self.distribution1 = distribution1
        self.power1 = power1
        self.distribution2 = distribution2
        self.power2 = power2

    def log_prob(self, x):
        return self.power1 * self.distribution1.log_prob(x) + self.power2 * self.distribution2.log_prob(x)


class MarkovKernel(ABC):
    @abstractmethod
    def step(self, x, n_steps=1):
        '''
        Apply the kernel to x `n_steps` times
        '''
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, x, y):
        '''
        Return the transition density from x to y if it exists
        '''
        raise NotImplementedError


class ULAKernel(MarkovKernel):
    def __init__(self, stationary_distribution, time_step):
        super().__init__()
        self.negative_energy = stationary_distribution.log_prob
        self.time_step = time_step

    def step(self, x):
        minus_grad_U = self._grad_negative_energy(x)
        noise = torch.randn_like(x)
        y = x + self.time_step * minus_grad_U + math.sqrt(2 * self.time_step) * noise
        return y

    def log_prob(self, x, y):
        minus_grad_U = self._grad_negative_energy(x)
        standard_normal = torch.distributions.Normal(0., 1.)
        return standard_normal.log_prob((y - x - self.time_step * minus_grad_U) / math.sqrt(2 * self.time_step)).sum(axis=-1)

    def _grad_negative_energy(self, x):
        x = x.clone().detach().requires_grad_(True)
        sum_negative_energy = self.negative_energy(x).sum()
        return torch.autograd.grad(sum_negative_energy, x)[0]


def run_annealed_importance_sampling(
    p_0,
    p_n,
    n_steps : int,
    n_particles : int,
    transition_kernel,
    n_kernel_steps=1,
    resample=True,
    ess_threshold=0.5
):
    '''
    Use annealed importance sampling to generate a weighted sample from p_N using samples from p_0
    
    This function works with tractable (allowing transition density evaluation) forward kernels and
    uses $L_{k-1}(x_k, x_{k-1})=M_n(x_{k-1}, x_{k})$ as reverse kernels, where $M$ denotes forward
    kernels.
    
    Parameters
    ----------
    p_0
        Unnormalized starting distribution from which we can sample
    p_n
        Unnormalized target distribution from which we wish to sample
    n_steps
        Number of interpolating steps
    n_kernel_steps
        Number of times the transition kernel is applied at each interpolating step
    n_particles
        Number of particles
    transition_kernel
        Function that returns a Markov kernel with a given stationary distribution

    Returns
    -------
    (torch.tensor, torch.tensor)
        Weighted sample from p_N (log_weights, values)
        The expected value of each weight is the ratio of the normalizing constants of p_n and p_0
    '''
    # Annealing schedule
    beta = torch.linspace(0, 1, n_steps + 1)
    # Intermediate distributions
    gamma = [DensityMixture(p_0, 1 - beta[t], p_n, beta[t]) for t in range(n_steps + 1)]
    # Particles
    X = p_0.sample((n_particles,))
    X.requires_grad_(False)
    # Logarithmic weights
    # logW = -p_0.log_prob(X)
    logW = torch.zeros(*X.shape[:-1]).to(X.device)
    
    for t in range(1, n_steps+ 1):
        # Markov kernel with stationary distribution gamma_t
        M_t = transition_kernel(gamma[t])
        X_next = M_t.step(X)
        X_next.requires_grad_(False)
        # Incremental importance weights (adding and then subtracting gamma_t(X_t) is redundant if particles were not resampled during that step)
        logW += M_t.log_prob(X_next, X) - M_t.log_prob(X, X_next) + gamma[t].log_prob(X_next) - gamma[t-1].log_prob(X)
        # Update particles
        X = X_next
        # Resampling
        if resample:
            normalized_weights = logW.exp()
            weight_sum = normalized_weights.sum(dim=-1, keepdims=True)
            normalized_weights /= weight_sum
            effective_sample_size = 1. / (normalized_weights ** 2).sum(dim=0)
            need_resampling = effective_sample_size < ess_threshold * n_particles  # batch indicies that require resampling
            if torch.any(need_resampling):
                # print('Resampling', need_resampling.sum().item())
                sample_indicies = torch.multinomial(normalized_weights[:, need_resampling].T, n_particles)
                # print('X[:, need_resampling]', X[:, need_resampling].shape)
                # print('sample_indicies', sample_indicies.shape)
                X[:, need_resampling] = X[sample_indicies.T, need_resampling]
            # print(f'ESS: {effective_sample_size.item():0.0f}')
            '''
            if effective_sample_size < ess_threshold * n_particles:
                print(f'Resampling')
                sample_indicies = torch.multinomial(normalized_weights, n_particles, replacement=True)
                X = X[sample_indicies]
                logW = torch.ones_like(n_particles) * (weight_sum / n_particles).log()
            '''
    
    # logW += p_n.log_prob(X)
    return logW, X


def ais_ula_log_mean_weight(
    *args,
    ula_time_step=None,
    return_variance=False,
    **kwargs
):
    transition_kernel = lambda distr: ULAKernel(distr, ula_time_step)
    log_weights, samples = run_annealed_importance_sampling(*args, transition_kernel=transition_kernel, **kwargs)
    log_mean_weight = torch.logsumexp(log_weights, axis=0) - math.log(kwargs["n_particles"])
    if return_variance:
        weight_variance = torch.var(log_weights.exp(), axis=0)
        return log_mean_weight, weight_variance
    return log_mean_weight
