import numpy as np
from scipy.stats import bernoulli
import torch
import numbers
import statistics
import matplotlib.pyplot as plt

from .vae import VAE, VAETrainer
from .kernels import LangevinKernel, iSIRKernel
from .distributions import IndependentMultivariateNormal
from .y_utils import *


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


def metropolis_hastings_filter(target, proposal_samples, proposal_log_prob_estimator, burn_in=None,
                               n_estimates=1, max_density_ratio=None, visualize=False, return_acc_probs=False):
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
    torch.tensor
        Samples from the Metropolis-Hastings algorithm
    '''
    def open_tuple(x):
        if isinstance(x, tuple):
            return x[0]
        return x
    
    n_samples = proposal_samples.shape[0]
    if burn_in is None:
        burn_in = n_samples // 20
    
    target_log_prob = target.log_prob(proposal_samples)
    proposal_log_probs = torch.stack([open_tuple(proposal_log_prob_estimator(proposal_samples)) for _ in range(n_estimates)], dim=-1)
    density_ratios = target_log_prob.unsqueeze(-1) - proposal_log_probs
    
    if max_density_ratio is not None:
        non_outlier = density_ratios.median(dim=1).values < max_density_ratio
        density_ratios = density_ratios[non_outlier]
        target_log_prob = target_log_prob[non_outlier]
        proposal_log_probs = proposal_log_probs[non_outlier]
        proposal_samples = proposal_samples[non_outlier]
        new_n_samples = density_ratios.shape[0]
        print(f'MH discarded {n_samples - new_n_samples} outlier(s)')
        n_samples = new_n_samples

    sample_indicies = torch.arange(n_samples)
    acc_noise = torch.rand(n_samples)
    density_ratios = density_ratios.to('cpu')
    acc_probs = []

    for t in range(1, n_samples):
        index_last = sample_indicies[t - 1]
        cur_density_ratio = density_ratios[t][0]
        last_density_ratio = density_ratios[index_last][(t - index_last) % n_estimates]
        accept_prob = torch.exp(
            cur_density_ratio - last_density_ratio
        ).item()
        if acc_noise[t] > accept_prob:  # reject
            sample_indicies[t] = index_last
        acc_probs.append(min(accept_prob, 1.))
    
    n_accepted = torch.ne(sample_indicies[burn_in:], sample_indicies[burn_in-1 : -1]).sum().item()
    acc_rate = n_accepted / (n_samples - burn_in)
    
    if visualize:
        fig, ax = plt.subplots(figsize=(16, 8))
        log_ratios_accepted = (density_ratios[..., 0])[sample_indicies]
        log_ratios_all = density_ratios[..., 0]
        ax.plot(np.arange(1, n_samples + 1), to_numpy(log_ratios_accepted.exp()), label='Accepted sample target/proposal', zorder=3)
        ax.plot(np.arange(1, n_samples + 1), to_numpy(log_ratios_all.exp()), label='Proposed sample target/proposal')
        ax.plot(np.arange(1, n_samples + 1), to_numpy(target_log_prob.exp()), label='Sample target density')
        #for i in range(n_estimates):
        #    ax.scatter(np.arange(1, n_samples + 1), to_numpy(proposal_log_probs[..., i].exp()), label='Sample proposal density', color='tab:red', zorder=4)
        ax.set_yscale('log')
        ax.set_ylabel('Target / Proposal log-ratio')
        ax.set_xlabel('Sample #')
        ax.axvline(x=burn_in, label='Burn-in', color='r', linestyle='--')
        ax.set_title(f'Metropolis-Hastings Diagnostics\nA/R={acc_rate * 100:0.1f}%')
        ax.legend()

    mh_indicies = sample_indicies[burn_in:]
    return1 = acc_probs[burn_in:] if return_acc_probs else acc_rate
    return return1, proposal_samples[mh_indicies]


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


class MetropolisHastingsFilter:
    def __init__(self, proposal, target, **kwargs):
        self.proposal = proposal
        self.target = target
        if "proposal_log_prob" in kwargs:
            self.proposal_log_prob = kwargs["proposal_log_prob"]
        else:
            self.proposal_log_prob = self.proposal.log_prob

    def apply(self, proposal_samples):
        n_samples = proposal_samples.shape[0]
        log_weights = self._calculate_log_weights(proposal_samples)
        log_weights = log_weights.to("cpu")
        mh_indicies = torch.arange(n_samples)
        acc_noise = torch.rand(n_samples)
        for t in range(1, n_samples):
            last_index = mh_indicies[t - 1]
            acc_prob = torch.exp(log_weights[t] - log_weights[last_index]).item()
            if acc_noise[t] > acc_prob:  # reject
                mh_indicies[t] = last_index
        return proposal_samples[mh_indicies]

    def _calculate_log_weights(self, x):
        return self.target.log_prob(x) - self.proposal_log_prob(x)


class VAEGlobalMHFilter:
    def __init__(self, vae, target):
        self.vae = vae
        self.target = target

    def apply(self, x, z):
        assert(x.shape[0] == z.shape[0])
        n_samples = x.shape[0]
        with torch.no_grad():
            log_weights = self.target.log_prob(x) + \
                          self.vae.encoder_distribution(x).log_prob(z) - \
                          self.vae.joint_log_prob(x, z)
        log_weights = log_weights.to("cpu")
        mh_indicies = torch.arange(n_samples)
        acc_noise = torch.rand(n_samples)
        for t in range(n_samples):
            last_index = mh_indicies[t - 1]
            acc_prob = torch.exp(log_weights[t] - log_weights[last_index]).item()
            if acc_noise[t] > acc_prob:  # reject
                mh_indicies[t] = last_index
        return x[mh_indicies]


# The naming of this class may no longer be appropriate
class VAEMetropolisWithinGibbsSampler:
    def __init__(self, vae, target, latent_noise_variance=0.):
        self.vae = vae
        self.target = target
        self.noise_sigma = latent_noise_variance ** 0.5

    @torch.no_grad()
    def sample(self, n_samples, x0=None):
        latent_noise = self.noise_sigma * torch.randn(n_samples, self.vae.latent_dim)
        x = x0 if x0 is not None else self.vae.sample() # current x
        x_posterior = self.vae.encoder_distribution(x)
        samples = []
        acc_noise = torch.rand(n_samples)
        for t in range(n_samples):
            # update z
            z1 = self.vae.encode(x)
            z1_conditional = self.vae.decoder_distribution(z1)
            if self.noise_sigma == 0.:
                z2 = z1
                z_log_weight = 0
                z2_conditional = z1_conditional
            else:
                p_z2_cond_z1 = self.latent_step_distribution(z1)
                z2 = p_z2_cond_z1.sample()
                p_z1_cond_z2 = self.latent_step_distribution(z2)
                z_log_weight = p_z1_cond_z2.log_prob(z1) - p_z2_cond_z1.log_prob(z2)
                z2_conditional = self.vae.decoder_distribution(z2)
            # update x
            new_x = z2_conditional.sample()
            new_x_posterior = self.vae.encoder_distribution(new_x)
            acc_prob = torch.exp(
                z1_conditional.log_prob(x) + new_x_posterior.log_prob(z2) + self.target.log_prob(new_x)
                - z2_conditional.log_prob(new_x) - x_posterior.log_prob(z1) - self.target.log_prob(x)
                + z_log_weight
            )
            print('log p(x|z1) - log q(z1|x)\t\t', (z1_conditional.log_prob(x) - x_posterior.log_prob(z1)).item())
            print('log q(z2|new_x) - log p(new_x|z2)\t', (new_x_posterior.log_prob(z2) - z2_conditional.log_prob(new_x)).item())
            print('log \\pi(new_x) - log \\pi(x)\t\t', (self.target.log_prob(new_x) - self.target.log_prob(x)).item())
            # print('log p(new_x|z2)', (z2_conditional.log_prob(new_x)).item())
            print('Acc. Prob.', acc_prob.item())
            if acc_noise[t] < acc_prob:  # accept
                x = new_x
                x_posterior = new_x_posterior
            samples.append(x.clone())
        samples = torch.stack(samples, dim=0).squeeze(1)
        return samples

    def latent_step_distribution(self, z):
        # return IndependentMultivariateNormal(z, self.noise_sigma)
        return IndependentMultivariateNormal((1 - self.noise_sigma**2) ** 0.5 * z, self.noise_sigma)

    # This is trash
    @torch.no_grad()
    def sample_effectively(self, n_samples):
        raise ValueError("Don't use this pls")
        zs = self.vae.prior.sample((n_samples,))
        decoder_distributions = self.vae.decoder_distribution(zs)
        xs = decoder_distributions.sample()
        target_log_probs = self.target.log_prob(xs)
        encoder_distributions = self.vae.encoder_distribution(xs)
        decoder_distributions.move_to('cpu')
        encoder_distributions.move_to('cpu')
        xs = xs.to('cpu')
        zs = zs.to('cpu')
        mh_indicies = torch.arange(n_samples)
        acc_noise = torch.rand(n_samples)
        for t in range(1, n_samples):
            z = zs[t]
            l = mh_indicies[t - 1]  # index of last accepted proposal
            acc_prob = torch.exp(
                target_log_probs[t] + encoder_distributions[t].log_prob(z) - decoder_distributions[t].log_prob(xs[t]) -
                target_log_probs[l] - encoder_distributions[l].log_prob(z) + decoder_distributions[t].log_prob(xs[l])
            )
            if acc_noise[t] > acc_prob:  # reject
                mh_indicies[t] = l
        return xs[mh_indicies]

    # when updating x while keeping z fixed
    @torch.no_grad()
    def _log_weight(self, x, x_posterior, z, z_conditional):
        return self.target.log_prob(x) + x_posterior.log_prob(z) - z_conditional.log_prob(x)


class LocalGlobalSampler:
    def __init__(self, **kwargs):
        self.target = kwargs.get("target")
        self.global_model = kwargs.get("global_model")
        self.n_local_steps = kwargs.get("n_local_steps")
        self.global_kernel = iSIRKernel(
            proposal=self.global_model,
            target=self.target,
            n_particles=kwargs.get("n_isir_particles", 10),
            proposal_log_prob=kwargs.get("model_likelihood_estimate")
        )
        self.local_step_size = kwargs.get("local_step_size")
        self.device = kwargs.get("device", "cpu")
        
        self.local_kernel = LangevinKernel(self.target, self.local_step_size, mh_corrected=True)
        self.local_steps_left = 0 # number of local steps to do before the next global step
        self.current_state = kwargs.get("starting_state", torch.zeros(self.target.dim, device=self.device))

    def sample(self, n_samples):
        samples = []
        local_acc_probs = []
        for t in range(n_samples):
            if self.local_steps_left > 0:
                self.current_state, acc_prob = self.local_kernel.step(self.current_state, return_acc_prob=True)
                local_acc_probs.append(acc_prob)
                self.local_steps_left -= 1
            else:
                self.current_state = self.global_kernel.step(self.current_state)
                self.local_steps_left = self.n_local_steps
            samples.append(self.current_state)
        local_acc_rate = torch.mean(torch.stack(local_acc_probs), dim=0)
        return torch.stack(samples), local_acc_rate


class AdaptiveVAESampler:
    def __init__(self, **kwargs):
        self.target = kwargs["target"]
        self.model = kwargs["model"]
        self.use_probability_cutoff = kwargs.get("use_probability_cutoff", True)
        '''
        self.global_kernel = iSIRKernel(
            proposal=self.global_model,
            target=self.target,
            n_particles=kwargs.get("n_isir_particles", 10),
            proposal_log_prob=kwargs.get("model_likelihood_estimate")
        )
        '''
        self.global_filter = MetropolisHastingsFilter(
            proposal=self.model,
            target = self.target,
            proposal_log_prob=kwargs.get("model_log_prob")
        )
        self.device = kwargs.get("device", "cpu")
        # self.retrain_frequency = kwargs.get("retrain_frequency")
        
        self.sample_history = kwargs.get("initial_sample", torch.tensor([]))
        self.current_state = None

    # forgetting_alpha -- how much of the previous history we remember
    def retrain(self, forgetting_alpha=0.5, warm_start=True, **kwargs):
        if self.sample_history.size(dim=0) == 0:
            print('Not enough samples to train on')
            return
        if not warm_start:
            self.model.init_weights()
        model_trainer = VAETrainer(model=self.model, target=self.target, device=self.device, **kwargs)
        model_trainer.fit(x_train=self.sample_history.to(self.device), **kwargs)
        self.probability_cutoff = ProbabilityCutoff(self.target, self.sample_history.shape[0])
        #if clear_sample_history:
        #    self.sample_history = torch.tensor([])
        self.sample_history = self.sample_history[torch.randperm(int(len(self.sample_history) * forgetting_alpha))]

    def sample(self, n_samples=1, add_to_history=True):
        self.model.eval()
        with torch.no_grad():
            model_sample = self.model.sample((n_samples,))
            if self.use_probability_cutoff:
                model_sample = self.probability_cutoff.apply(model_sample)
            corrected_sample = self.global_filter.apply(model_sample)
            print(model_sample.shape[0] - corrected_sample.shape[0], 'samples cut')
            # self.current_state = corrected_sample[-1]
            if add_to_history:
                self.sample_history = torch.cat((self.sample_history, corrected_sample))
        return corrected_sample


def get_log_prob_quantile(target, q=0, N=2000):
    samples = target.sample((N, ))
    log_probs = target.log_prob(samples)
    return log_probs.quantile(q).item()


def log_prob_cutoff_filter(target, samples, cutoff_min, cutoff_max=torch.inf, return_indicies=True):
    sample_log_prob = target.log_prob(samples)
    cut_indicies = (cutoff_min < sample_log_prob) & (sample_log_prob < cutoff_max)
    acc_rate = cut_indicies.sum() / len(cut_indicies)
    if return_indicies:
        return acc_rate.item(), cut_indicies
    return acc_rate.item(), samples[cut_indicies]


class ProbabilityCutoff:
    def __init__(self, distribution, sample_size):
        self.distribution = distribution
        self.min_log_prob = torch.min(self.distribution.log_prob(self.distribution.sample((sample_size,))))

    def apply(self, sample):
        return sample[self.distribution.log_prob(sample) >= self.min_log_prob]
