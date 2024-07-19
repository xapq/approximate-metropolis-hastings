import torch


class DensityMixture:
    def __init__(self, distribution1, power1, distribution2, power2):
        self.distribution1 = distribution1
        self.power1 = power1
        self.distribution2 = distribution2
        self.power2 = power2

    def log_prob(self, x):
        return self.power1 * self.distribution1.log_prob(x) + self.power2 * self.distribution2.log_prob(x)


def run_annealed_importance_sampling(
    p_0,
    p_n,
    n : int,
    K : int,
    transition_kernel,
    n_kernel_steps : int = 1
):
    '''
    Use annealed importance sampling to generate a weighted sample from p_N using samples from p_0
    
    Parameters
    ----------
    p_0
        Unnormalized starting distribution from which we can sample
    p_n
        Unnormalized target distribution from which we wish to sample
    n
        Number of interpolating steps
    K
        Number of particles
    transition_kernel
        Function that returns a Markov kernel with a given stationary distribution
    n_kernel_steps
        Number of times a kernel is applied at each interpolating step
    Returns
    -------
    (torch.tensor, torch.tensor)
        Weighted sample from p_N (log weights, values)
    '''
    # Annealing schedule
    beta = torch.linspace(0, 1, n + 1)
    # Intermediate distributions
    gamma = [DensityMixture(p_0, 1 - beta[t], p_n, beta[t]) for t in range(n + 1)]
    # Particles
    X = p_0.sample((K,))
    # Logarithms of weights
    logW = -p_0.log_prob(X)

    for t in range(1, n + 1):
        kernel = transition_kernel(gamma[t])
        X_next = X
        for i in range(n_kernel_steps):
            X_next = kernel(X_next)
        # Incremental importance weights
        logW += gamma[t - 1].log_prob(X) - gamma[t - 1].log_prob(X_next)
        
    logW += p_n.log_prob(X)























