from abc import ABC, abstractmethod
import torch
import math
from .sequential_mcmc import run_annealed_importance_sampling, LangevinKernel, ULAKernel, MALAKernel


class MonteCarloLikelihoodEstimator(ABC):
    def __init__(self, **kwargs):
        self.model = kwargs.get("model")
        self.L = kwargs.get("L", 64)
        self.batchL = kwargs.get("batchL", self.L)

    def __call__(self, x, return_variance=False, **kwargs):
        estimates = []
        for i in range(0, self.L, self.batchL):
            estimates.append(self._batch_estimate(x, min(self.batchL, self.L - i), **kwargs).detach())
        estimates = torch.cat(estimates)
        estimate = torch.logsumexp(estimates, dim=0) - math.log(self.L)
        info_dict = kwargs.get('info_dict', dict())
        for key, value in info_dict.items():
            info_dict[key] = torch.stack(value).mean(dim=0)
        if return_variance:
            variance = torch.var(estimates.exp(), dim=0)
            return estimate, variance
        return estimate

    @abstractmethod
    def _batch_estimate(self, x, L, **kwargs):
        raise NotImplementedError

    @property
    @abstractmethod
    def friendly_name(self):
        raise NotImplementedError


class IWLikelihoodEstimator(MonteCarloLikelihoodEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _batch_estimate(self, x, L, **kwargs):
        encoder_dist = self.model.encoder_distribution(x)
        zs = encoder_dist.sample((L,))
        log_p_zs = self.model.latent_sampling_distribution.log_prob(zs)
        decoder_dists = self.model.decoder_distribution(zs)
        log_p_x_cond_zs = decoder_dists.log_prob(x)
        log_q_zs = encoder_dist.log_prob(zs)
        point_estimates = log_p_x_cond_zs + log_p_zs - log_q_zs
        return point_estimates

    @property
    def friendly_name(self):
        return f'IW'


class SISLikelihoodEstimator(MonteCarloLikelihoodEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_steps = kwargs.get("n_steps", 10)
        self.kernel_type = kwargs.get("kernel_type", "mala")
        self.precondition = kwargs.get("precondition", False)
        self.time_step = kwargs.get("time_step")
        self.n_kernel_steps = kwargs.get("n_kernel_steps", 1)
        self.resample = kwargs.get("resample", False)
        self.annealing_schedule = kwargs.get("annealing_schedule", "linear")
        self.annealing_scale = kwargs.get("annealing_scale", 1.)

        if self.kernel_type == "mala":
            self.kernel = MALAKernel
            self.kernel_property = "invariant"
        elif self.kernel_type == "ula":
            self.kernel = ULAKernel
            self.kernel_property = "almost_reversible"
        else:
            raise ValueError("kernel_type must be one of ['ula', 'mala']")

    def _batch_estimate(self, x, L, **kwargs):
        time_step = self.time_step
        if self.precondition:
            _, log_encoder_variance = self.model.encoding_parameters(x)
            time_step *= torch.exp(log_encoder_variance)
        kernel_factory = lambda distribution: self.kernel(distribution, time_step)
        log_weights, particles, acc_rate = run_annealed_importance_sampling(
            self.model.encoder_distribution(x), 
            self.model.posterior(x),
            self.n_steps,
            L,
            kernel_factory,
            self.kernel_property,
            n_kernel_steps=self.n_kernel_steps,
            resample=self.resample,
            return_acc_rate=True,
             **kwargs
        )
        if 'info_dict' in kwargs:
            kwargs['info_dict']['acceptance_rate'].append(acc_rate)
        return log_weights

    @property
    def friendly_name(self):
        step_str = self.n_steps if self.n_kernel_steps == 1 else "x" + self.n_kernel_steps
        return f'{step_str}-Step {"Precond. " if self.precondition else ""}{self.kernel_type.upper()} SIS'
