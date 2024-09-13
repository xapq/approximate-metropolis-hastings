'''
Taken from
https://github.com/svsamsonov/ex2mcmc_new/blob/31d422ad704a145422db8cd19f9d3907bfb2e608/ex2mcmc/pyro_samplers.py
'''
import torch
from pyro.infer import MCMC
from pyro.infer import NUTS as pyro_nuts


def NUTS(
    start,
    target,
    n_samples: int,
    burn_in: int,
    *,
    step_size: float,
    verbose: bool = False,
) -> torch.FloatTensor:
    """
    No-U-Turn Sampler

    Args:
        start - strating points of shape [n_chains x dim]
        target - target distribution instance with method "log_prob"
        n_samples - number of last samples from each chain to return
        burn_in - number of first samples from each chain to throw away
        step_size - step size for drift term
        verbose - whether to show iterations' bar

    Returns:
        tensor of chains with shape [n_samples, n_chains, dim], acceptance rates
        for each iteration
    """
    x = start.clone()
    x.requires_grad_(False)

    def energy(z):
        z = z["points"]
        return -target.log_prob(z).sum()

    kernel = pyro_nuts(potential_fn=energy, step_size=step_size, full_mass=False)

    init_params = {"points": x}
    mcmc_true = MCMC(
        kernel=kernel,
        num_samples=n_samples,
        initial_params=init_params,
        warmup_steps=burn_in,
    )
    mcmc_true.run()

    q_true = mcmc_true.get_samples(group_by_chain=True)["points"]
    samples_true = q_true.view(-1, *start.shape)

    return samples_true