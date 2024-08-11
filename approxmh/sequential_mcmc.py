import torch
import math, statistics
from abc import ABC, abstractmethod
from .distributions import IndependentMultivariateNormal


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


class LangevinKernel(MarkovKernel):
    # mh_corrected=False -> ULA, mh_corrected=True -> MALA
    def __init__(self, stationary_distribution, time_step, mh_corrected: bool):
        super().__init__()
        self.negative_energy = stationary_distribution.log_prob
        self.time_step = time_step
        self.mh_corrected = mh_corrected

    def step(self, x, return_acc_prob=True):
        y = self.step_distribution(x).sample()
        acc_prob = self.acceptance_probability(x, y)
        if self.mh_corrected:
            rejected = torch.rand_like(acc_prob) > acc_prob
            y += (x - y) * rejected.unsqueeze(-1)
        if return_acc_prob:
            return y, acc_prob.detach()
            # print(acc_prob.mean().item())
        return y

    def log_prob(self, x, y):
        if self.mh_corrected:
            raise NotImplementedError('MALA transition probabilites are intractable')
        return self.step_distribution(x).log_prob(y)
        
    def step_distribution(self, x):
        return IndependentMultivariateNormal(
            mean=x + self.time_step * self._grad_negative_energy(x),
            std=torch.as_tensor(2 * self.time_step, device=x.device).sqrt(),
        )
    
    # Metropolis-Hastings acceptance probability
    def acceptance_probability(self, x, y):
        return torch.minimum(torch.exp(
            self.negative_energy(y) + self.step_distribution(y).log_prob(x) -
            self.negative_energy(x) - self.step_distribution(x).log_prob(y)
        ), torch.tensor(1.))
    
    def _grad_negative_energy(self, x):
        x = x.clone().detach().requires_grad_(True)
        sum_negative_energy = self.negative_energy(x).sum()
        return torch.autograd.grad(sum_negative_energy, x)[0]


def create_annealing_schedule(n_steps, scheme, scale=1.):
    if scheme == 'linear':
        return torch.linspace(0, 1, n_steps + 1)
    if scheme == 'sigmoidal':
        beta_tilda = torch.sigmoid(scale * torch.linspace(-1, 1, n_steps + 1))
    elif scheme == 'logit':
        beta_tilda = torch.logit(scale * torch.linspace(-1, 1, n_steps + 1))
    elif scheme == 'sine':
        beta_tilda = torch.sin(scale * 0.5 * math.pi * torch.linspace(0, 1, n_steps + 1))
    else:
        raise ValueError('Annealing scheme must be one of [linear, sigmoidal, logit, sine].')
        
    beta = (beta_tilda - beta_tilda[0]) / (beta_tilda[n_steps] - beta_tilda[0])
    return beta


def run_annealed_importance_sampling(
    p_0,
    p_n,
    n_steps : int,
    n_particles : int,
    transition_kernel,
    kernel_type=None,
    n_kernel_steps=1,
    resample=False,
    ess_threshold=0.5,
    annealing_scheme='linear',
    annealing_scale=1.,
    return_acc_rate=False,
    return_all_steps=False
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
    beta = create_annealing_schedule(n_steps, annealing_scheme, annealing_scale)
    if n_kernel_steps > 1:
        beta = torch.cat([torch.tensor([0.]), torch.repeat_interleave(beta[1:], repeats=n_kernel_steps)])
        n_steps *= n_kernel_steps
    # Intermediate distributions
    gamma = [DensityMixture(p_0, 1 - beta[t], p_n, beta[t]) for t in range(n_steps + 1)]
    # Particles
    X = p_0.sample((n_particles,)).requires_grad_(False)
    # Logarithmic weights
    logW = torch.zeros(*X.shape[:-1]).to(X.device)
    # Acceptance Rates
    acc_rates = []
    # Full particle history
    if return_all_steps:
        all_X = [X.clone().detach()]
        all_logW = [logW.clone().detach()]
    
    for t in range(1, n_steps + 1):
        # Markov kernel with stationary distribution gamma_t
        M_t = transition_kernel(gamma[t])
        X_next, acc_rate = M_t.step(X, return_acc_prob=True)
        X_next.requires_grad_(False)
        acc_rates.append(acc_rate.mean(dim=0).detach())
        
        # Incremental importance weights (adding and then subtracting gamma_t(X_t) is redundant when resample=False)
        if kernel_type == 'almost_invertible': 
            logW += M_t.log_prob(X_next, X) - M_t.log_prob(X, X_next) + gamma[t].log_prob(X_next) - gamma[t-1].log_prob(X)
        elif kernel_type == 'invariant':
            logW += gamma[t].log_prob(X) - gamma[t-1].log_prob(X)
        else:
            raise ValueError('kernel_type must be one of [almost_invertible, invariant]')

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
                sample_indicies = torch.multinomial(normalized_weights[:, need_resampling].T, n_particles)
                X[:, need_resampling] = X[sample_indicies.T, need_resampling]
                logW[:, need_resampling] = weight_sum / n_particles
        # Logging
        if return_all_steps:
            all_X.append(X.clone().detach())
            all_logW.append(logW.clone().detach())
    
    acc_rate = torch.stack(acc_rates, dim=-1)
    if return_all_steps:
        X = torch.stack(all_X, dim=-1)
        logW = torch.stack(all_logW, dim=-1)
    else:
        acc_rate = acc_rates.mean(dim=-1)

    if return_acc_rate:
        return logW, X, acc_rate
    return logW, X


def ais_langevin_log_norm_constant_ratio(
    *args,
    n_particles=None,
    batch_particles=None,
    mh_corrected=None,
    time_step=None,
    return_acc_rate=True,
    **kwargs
):
    if batch_particles is None:
        batch_particles = n_particles
    
    transition_kernel = lambda distr: LangevinKernel(distr, time_step, mh_corrected)
    kernel_type = 'invariant' if mh_corrected else 'almost_invertible'

    log_weights = []
    acc_rate = []
    for i in range(0, n_particles, batch_particles):
        batch_log_weights, _, batch_acc_rate = run_annealed_importance_sampling(
            *args, transition_kernel=transition_kernel, kernel_type=kernel_type, 
            n_particles=min(batch_particles, n_particles - i), return_acc_rate=True, **kwargs
        )
        log_weights.append(batch_log_weights.detach())
        acc_rate.append(batch_acc_rate)

    log_weights = torch.cat(log_weights)
    acc_rate = torch.stack(acc_rate).mean(dim=0)
    
    log_mean_weight = torch.logsumexp(log_weights, axis=0) - math.log(n_particles)
    weight_variance = torch.var(log_weights.exp(), axis=0)
    if return_acc_rate:
        return log_mean_weight, weight_variance, acc_rate
    return log_mean_weight, weight_variance
