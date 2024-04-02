import numpy as np
from scipy.stats import bernoulli
import torch

# Returns acceptance rate and samples
def metropolis_hastings_with_noise(target, proposal, n_samples, burn_in=100, noise_std=0):
    n_steps = burn_in + n_samples
    samples = proposal.sample((n_steps,))
    target_log_prob = target.log_prob(samples)
    proposal_prob = proposal.prob(samples)
    noise = noise = np.random.normal(0, noise_std, size=(n_steps, 2))
    acc_noise = np.random.uniform(0, 1, n_steps)
    n_accepted = 0
    for t in range(1, n_steps):
        accept_prob = (
            np.exp(target_log_prob[t] - target_log_prob[t - 1])
            * (proposal_prob[t - 1] + noise[t][0]) / (proposal_prob[t] + noise[t][1])
        )
        if acc_noise[t] < accept_prob:  # accept
            n_accepted += (t >= burn_in)
        else:  # reject
            for arr in (samples, target_log_prob, proposal_prob):
                arr[t] = arr[t - 1]  # copy previous sample
    acc_rate = n_accepted / n_samples
    return acc_rate, samples[burn_in:].numpy(force=True)


def approximate_metropolis_hastings(target, proposer, proposal_log_prob_estimator, n_samples, burn_in=None):
    if burn_in is None:
        burn_in = n_samples // 10 
    n_samples += burn_in

    samples = proposer((n_samples,))
    target_log_prob = target.log_prob(samples)
    proposal_log_prob = proposal_log_prob_estimator(samples)
    acc_noise = torch.rand(n_samples)
    n_accepted = 0
    for t in range(1, n_samples):
        accept_prob = torch.exp(
            target_log_prob[t] - target_log_prob[t - 1] + proposal_log_prob[t - 1] - proposal_log_prob[t]
        )
        if acc_noise[t] < accept_prob:  # accept
            n_accepted += (t >= burn_in)
        else:  # reject
            for arr in (samples, target_log_prob, proposal_log_prob):
                arr[t] = arr[t - 1]  # copy previous sample
    acc_rate = n_accepted / (n_samples - burn_in)
    return acc_rate, samples[burn_in:]


def approximate_metropolis_hastings_reevaluation(target, proposer, proposal_log_prob_estimator, n_samples, burn_in=None):
    if burn_in is None:
        burn_in = n_samples // 10 
    n_samples += burn_in

    samples = proposer((n_samples,))
    target_log_prob = target.log_prob(samples)
    proposal_log_prob = proposal_log_prob_estimator(samples)
    acc_noise = torch.rand(n_samples)
    n_accepted = 0
    for t in range(1, n_samples):
        accept_prob = torch.exp(
            target_log_prob[t] - target_log_prob[t - 1] + proposal_log_prob[t - 1] - proposal_log_prob[t]
        )
        if acc_noise[t] < accept_prob:  # accept
            n_accepted += (t >= burn_in)
        else:  # reject
            samples[t] = samples[t - 1]
            target_log_prob[t] = target_log_prob[t - 1]
            proposal_log_prob[t] = proposal_log_prob_estimator(samples[t])
    acc_rate = n_accepted / (n_samples - burn_in)
    return acc_rate, samples[burn_in:]


def approximate_metropolis_hastings_multiple_estimates(target, proposer, proposal_log_prob_estimator, n_estimates, n_samples, burn_in=None):
    if burn_in is None:
        burn_in = n_samples // 10 
    n_samples += burn_in

    samples = proposer((n_samples,))
    sample_indicies = torch.arange(n_samples)
    target_log_prob = target.log_prob(samples)
    proposal_log_prob_estimates = torch.stack([proposal_log_prob_estimator(samples) for _ in range(n_estimates)], axis=4)
    print(proposal_log_prob_estimates.shape)
    est_index = 0
    acc_noise = torch.rand(n_samples)
    n_accepted = 0
    for t in range(1, n_samples):
        last = sample_indicies[t - 1]
        proposal_log_prob_cur = proposal_log_prob_estimates[t][est_index]
        proposal_log_prob_last = proposal_log_prob_estimates[last][est_index]
        accept_prob = torch.exp(
            target_log_prob[t] - target_log_prob[last] + proposal_log_prob_last - proposal_log_prob_cur
        )
        if acc_noise[t] < accept_prob:  # accept
            n_accepted += (t >= burn_in)
        else:  # reject
            sample_indicies[t] = sample_indicies[t - 1]
        est_index = (estimate_index + 1) % n_estimates
    acc_rate = n_accepted / (n_samples - burn_in)
    return acc_rate, samples[sample_indicies[burn_in:]]
