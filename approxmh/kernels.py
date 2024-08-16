import torch
from abc import ABC, abstractmethod
from .distributions import IndependentMultivariateNormal


class MarkovKernel(ABC):
    @abstractmethod
    def step(self, x, n_steps=1):
        '''
        Apply the kernel to `x` 1 time
        '''
        raise NotImplementedError

    def multistep(self, x, n_steps):
        '''
        Apply the kernel to `x` `n_steps` times and return a
        (n_steps, *x.shape)-shaped tensor containing all intermediate steps
        '''
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, x, y):
        '''
        Return the transition density from x to y if it exists
        '''
        raise NotImplementedError


class iSIRKernel(MarkovKernel):
    def __init__(self, proposal, target, n_particles, **kwargs):
        super().__init__()
        self.proposal = proposal
        self.target = target
        self.n_particles = n_particles  # number of proposals generated at each stap
        if "proposal_log_prob" in kwargs:
            self.proposal_log_prob = kwargs["proposal_log_prob"]
        else:
            self.proposal_log_prob = self.proposal.log_prob

    def step(self, x):
        particles = torch.zeros((self.n_particles + 1, self.target.dim), device='cpu')
        particles[:-1] = self.proposal.sample((self.n_particles,))
        particles[-1] = x
        weights = self._calculate_weight(particles)
        return particles[torch.multinomial(weights, num_samples=1)[0]]

    def multistep(self, x, n_steps):
        if x is None:
            x = self.proposal.sample((1,)).squeeze(dim=0)
        particles = torch.zeros((self.n_particles + 1, n_steps, self.target.dim))
        weights = torch.zeros((self.n_particles + 1, n_steps))
        particles[:-1] = self.proposal.sample((self.n_particles, n_steps))
        weights[:-1] = self._calculate_weight(particles[:-1])
        particles[-1][0] = x
        weights[-1][0] = self._calculate_weight(x.unsqueeze(0)).squeeze(0)
        current_state_idx = -1
        for i in range(n_steps):
            new_state_idx = torch.multinomial(weights[:, i], num_samples=1)[0]
            particles[-1][i] = particles[new_state_idx][i]
            if i + 1 < n_steps:
                particles[-1][i + 1] = particles[-1][i]
                weights[-1][i + 1] = weights[new_state_idx][i]
        return particles[-1]

    def log_prob(self, x, y):
        raise NotImplementedError('i-SIR kernels do not admit a transition density')

    def _calculate_weight(self, x):
        original_shape = x.shape
        x = x.flatten(end_dim=-2)
        result = torch.exp(self.target.log_prob(x) - self.proposal_log_prob(x))
        return result.unflatten(dim=0, sizes=original_shape[:-1])


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
            raise NotImplementedError('MALA kernels do not admit a transition density')
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
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            sum_negative_energy = self.negative_energy(x).sum()
        return torch.autograd.grad(sum_negative_energy, x)[0]


class ULAKernel(LangevinKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, mh_corrected=False, **kwargs)


class MALAKernel(LangevinKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, mh_corrected=True, **kwargs)
