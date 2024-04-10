import numpy as np
from scipy.stats import bernoulli
import torch
import numbers
import matplotlib.pyplot as plt
from y_utils import *


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


def metropolis_hastings_filter(target, proposal_samples, proposal_log_prob_estimator, burn_in=None, n_estimates=1, visualize=False):
    '''
    Parameters
    ----------
    target : torch.distribution
        Target distribution
    proposal_samples : torch.tensor
        Samples from the proposal distribution
    proposal_log_prob_estimator : function
        Function estimating log-probability of objects under the proposal distribution
    burn_in : positive integer
        Number of initial MH samples to discard. 1/20th of the samples by default
    n_estimates : positive integer
        Number of times to evaluate the proposal log-probability estimate.
    Returns
    -------
    float
        Acceptance rate
    1d integer torch.tensor
        Indicies of proposal samples chosen by the Metropolis-Hastings algorithm
    '''
    n_samples = proposal_samples.shape[0]
    if burn_in is None:
        burn_in = n_samples // 20
    sample_indicies = torch.arange(n_samples)
    target_log_prob = target.log_prob(proposal_samples)
    proposal_log_probs = torch.stack([proposal_log_prob_estimator(proposal_samples) for _ in range(n_estimates)], dim=-1)    
    acc_noise = torch.rand(n_samples)
    
    n_accepted = 0
    est_index = 0
    for t in range(1, n_samples):
        index_last = sample_indicies[t - 1]
        cur_proposal_log_prob = proposal_log_probs[t][est_index]
        last_proposal_log_prob = proposal_log_probs[index_last][est_index]
        accept_prob = torch.exp(
            target_log_prob[t] - target_log_prob[index_last] + last_proposal_log_prob - cur_proposal_log_prob
        )
        if acc_noise[t] < accept_prob:  # accept
            n_accepted += (t >= burn_in)
        else:  # reject
            sample_indicies[t] = index_last
        est_index = (est_index + 1) % n_estimates
    acc_rate = n_accepted / (n_samples - burn_in)
    
    if visualize:
        fig, ax = plt.subplots(figsize=(16, 8))
        log_ratios_accepted = (target_log_prob - proposal_log_probs[..., 0])[sample_indicies]
        log_ratios_all = target_log_prob - proposal_log_probs[..., 0]
        ax.plot(np.arange(1, n_samples + 1), to_numpy(log_ratios_accepted.exp()), label='Accepted sample target/proposal', zorder=3)
        ax.plot(np.arange(1, n_samples + 1), to_numpy(log_ratios_all.exp()), label='Proposed sample target/proposal')
        # ax.plot(np.arange(1, n_samples + 1), to_numpy(target_log_prob.exp()), label='Sample target density', zorder=0)
        ax.plot(np.arange(1, n_samples + 1), to_numpy(target_log_prob.exp()), label='Sample target density')
        ax.plot(np.arange(1, n_samples + 1), to_numpy(proposal_log_probs[..., 0].exp()), label='Sample proposal density', zorder=4)
        ax.set_yscale('log')
        # ax.set_ylabel('Target / Proposal log-ratio')
        ax.set_xlabel('Sample #')
        ax.axvline(x=burn_in, label='Burn-in', color='r', linestyle='--')
        ax.set_title(f'Metropolis-Hastings Diagnostics\nA/R={acc_rate * 100:0.1f}%')
        ax.legend()
    
    return acc_rate, sample_indicies[burn_in:]


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
    return acc_rate, samples_indicies[burn_in:]


def log_prob_cutoff_filter(target, cutoff, samples):
    cut_indicies = target.log_prob(samples) > cutoff
    acc_rate = cut_indicies.sum() / len(cut_indicies)
    return acc_rate, cut_indicies