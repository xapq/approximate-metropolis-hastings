import torch
from pytorch_warmup import LinearWarmup
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
import torch.nn.init as init
import numpy as np
import math
from samplers import metropolis_hastings_filter, log_prob_cutoff_filter
import matplotlib.pyplot as plt
from IPython.display import clear_output
from y_utils import *
from distribution_metrics import SlicedDistributionMetric, WassersteinMetric1d


def dataloader_from_tensor(X, batch_size):
    dataset = TensorDataset(X)
    dataloader = DataLoader(X, batch_size=batch_size, shuffle=True)
    return dataloader

def SequentialFC(dims, activation):
    network = nn.Sequential(nn.Linear(dims[0], dims[1]))
    for i in range(1, len(dims) - 1):
        network.append(activation())
        # network.append(nn.Dropout(p=0.01))
        network.append(nn.BatchNorm1d(num_features=dims[i]))
        network.append(nn.Linear(dims[i], dims[i + 1]))
    return network        


'''
Variational Autoencoder
Encoder and Decoder are made up of FC layers
'''
class VAE(torch.nn.Module):
    def __init__(self, data_dim, hidden_dims, latent_dim, device='cpu'):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        encoder_dims = [data_dim, *hidden_dims, 2 * latent_dim]
        decoder_dims = [latent_dim, *hidden_dims[::-1], 2 * data_dim]
        activation = lambda : nn.ReLU(inplace=True)
        self.encoder = SequentialFC(encoder_dims, activation)
        self.decoder = SequentialFC(decoder_dims, activation)
        self.device = device

        self.prior = torch.distributions.MultivariateNormal(torch.zeros(latent_dim, device=device), torch.diag(torch.ones(latent_dim, device=device)))
        self.latent_sampling_distribution = self.prior
        self.std_factor = 1
        
        self.to(device)

    def __repr__(self):
        return f'y02vae_D{self.data_dim}_layers{self.hidden_dims}'

    def forward(self, x):
        return reconstruct(x)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def reconstruct(self, x):
        return self.decode(self.encode(x))

    # parameters of the distribution q(z|x)
    def encoding_parameters(self, x):
        mean_z, log_var_z = self.encoder(x).chunk(2, dim=-1)
        return mean_z, log_var_z

    # parameters of the distribution p(x|z)
    def decoding_parameters(self, z):
        mean_x, log_var_x = self.decoder(z).chunk(2, dim=-1)
        return mean_x, log_var_x

    # sample distribution q(z|x)
    def encode(self, x):
        mean_z, log_var_z = self.encoding_parameters(x)
        z = self.reparameterize(mean_z, log_var_z)
        return z

    # sample distribution p(x|z)
    def decode(self, z):
        mean_x, log_var_x = self.decoding_parameters(z)
        x = self.reparameterize(mean_x, log_var_x)
        return x

    def sample(self, sample_shape=(1,)):
        z = self.sample_latent(sample_shape)
        return self.decode(z)

    def sample_prior(self, sample_shape=(1,)):
        return self.prior.sample(sample_shape)

    def sample_latent(self, sample_shape=(1,)):
        return self.latent_sampling_distribution.sample(sample_shape)

    ### standard deviation multipler for ancestral sampling
    def set_std_factor(self, std_factor):
        self.std_factor = std_factor
        self.latent_sampling_distribution = torch.distributions.MultivariateNormal(
            torch.zeros(self.latent_dim, device=self.device),
            torch.eye(self.latent_dim, device=self.device) * std_factor ** 2
        )

    # beta -- smoothing constant for log-sum-exp
    # assumes self.latent_sampling_distribution hasn't changed since sampling x
    def iw_log_marginal_estimate(self, x, L, beta=1, batch_L=64):
        point_estimates = []
        for i in range(0, L, batch_L):
            point_estimates.append(self._iw_log_marginal_estimate_batch(x, min(batch_L, L - i)))
        point_estimates = torch.cat(point_estimates)
        estimate = (torch.logsumexp(point_estimates.double() / beta, dim=0) - np.log(point_estimates.shape[0])) * beta
        return estimate
    
    def _iw_log_marginal_estimate_batch(self, x, L):
        self.eval()
        with torch.no_grad():
            mean_z, log_var_z = self.encoding_parameters(x)
            std_z = torch.exp(0.5 * log_var_z)
            m = torch.distributions.normal.Normal(mean_z, std_z)
            zs = m.sample((L,))
            log_p_zs = self.latent_sampling_distribution.log_prob(zs)
            mean_x_cond_zs, log_var_x_cond_zs = map(lambda t: t.unflatten(dim=0, sizes=(L, -1)),
                                                      self.decoding_parameters(zs.flatten(end_dim=1)))
            log_p_x_cond_zs = mean_field_log_prob(x - mean_x_cond_zs, log_var_x_cond_zs.exp())
            log_q_zs = mean_field_log_prob(zs - mean_z, log_var_z.exp())
            point_estimates = log_p_x_cond_zs + log_p_zs - log_q_zs
            return point_estimates

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)

    def save_knowledge(self, filename):
        save_dict = {
            'model_state_dict': self.state_dict()
        }
        torch.save(save_dict, filename)

    def load_knowledge(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval()


## TODO: Merge with FlowTrainer
class VAETrainer:
    def __init__(self, model, target, **kwargs):
        self.model = model
        self.target = target
        self.distribution_metric = kwargs.get("distribution_metric", SlicedDistributionMetric(WassersteinMetric1d(), self.model.data_dim))
        self.device = kwargs.get("device", "cpu")
        self.batch_size = kwargs.get("batch_size", 64)
        self.grad_clip = kwargs.get("grad_clip", 1.0)
        self.scheduler = kwargs.get("scheduler", None)
        self.no_kl_penalty_epochs = kwargs.get("no_kl_penalty_epochs", 40)
        self.kl_annealing_epochs = kwargs.get("kl_annealing_epochs", 500)
        # optimizer choice
        optimizer = kwargs.get("optimizer", "adam")
        lr = kwargs.get("lr", 1e-3)
        wd = kwargs.get("wd", 1e-4)
        momentum = kwargs.get("momentum",0.9)
        if isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, str):
            if optimizer.lower() == "adam":
                self.optimizer = torch.optim.Adam(
                    model.parameters(), lr=lr, weight_decay=wd
                )
            elif optimizer.lower() == "sgd":
                self.optimizer = torch.optim.SGD(
                    model.parameters(), lr=lr, weight_decay=wd, momentum=momentum
                )
            else:
                raise ValueError
        # optimizer choice finished
        self.warmup_scheduler = LinearWarmup(self.optimizer, warmup_period=20)

        self.plot_interval = 50
        self.evaluate_samples_interval = 50
        self.n_eval_samples = 2000
        self.epoch = 0
        self.train_loss_hist = []
        self.val_loss_hist = []
        self.best_loss = float('inf')
        self.sample_scores = []
        self.cut_sample_scores = []
        self.mh_sample_scores = []
        self.best_model_weights = None
        
        self.model.to(self.device)
        self.model.init_weights()

    def fit(self, x_train, **kwargs):
        x_val = kwargs.get("x_val", self.target.sample((x_train.shape[0] // 10,)))
        n_epochs = kwargs.get("n_epochs", 800)
        plot_interval = kwargs.get("plot_interval", n_epochs)
        train_loader = dataloader_from_tensor(x_train, self.batch_size)
        val_loader = dataloader_from_tensor(x_val, self.batch_size)

        # filters & sample score
        L = 64
        beta = 1
        cutoff_quantile = 0
        model_log_prob_estimator = lambda x : self.model.iw_log_marginal_estimate(x, L=L, beta=beta, batch_L=32)
        cutoff_log_prob = self.target.log_prob(x_train).quantile(cutoff_quantile)
        sample_score = lambda sample: self.distribution_metric(x_train[:self.n_eval_samples], sample).item()

        for epoch_id in range(n_epochs):
            self.epoch += 1
            assert(not torch.stack([p.isnan().any() for p in self.model.parameters()]).any())

            train_loss = self.run_epoch(train_loader, is_train=True)
            val_loss = self.run_epoch(val_loader, is_train=False)
            self.train_loss_hist.append(train_loss)
            self.val_loss_hist.append(val_loss)
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_weights = self.model.state_dict()

            if self.epoch % self.evaluate_samples_interval == 0:
                self.model.eval()
                model_samples = self.model.sample((self.n_eval_samples,))
                self.model.train()
                self.sample_scores.append(sample_score(model_samples))
                cut_acc_rate, cut_samples = log_prob_cutoff_filter(self.target, model_samples, cutoff_log_prob, return_indicies=False)
                if cut_samples.shape[0] == 0:
                    cut_samples = torch.zeros((1, self.model.data_dim), device=self.device)
                self.cut_sample_scores.append(sample_score(cut_samples))
                mh_acc_rate, mh_samples = metropolis_hastings_filter(self.target, cut_samples, model_log_prob_estimator, return_indicies=False)
                if mh_samples.shape[0] == 0:
                    mh_samples = torch.zeros((1, self.model.data_dim), device=self.device)
                self.mh_sample_scores.append(sample_score(mh_samples))
            
            if plot_interval is not None and (self.epoch % self.plot_interval == 0 or epoch_id == n_epochs - 1):
                self.show_training_plot()

            with self.warmup_scheduler.dampening():
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
            
        self.model.load_state_dict(self.best_model_weights)
        self.model.eval()

    def run_epoch(self, data_loader, is_train):
        self.model.train(is_train)
        avg_loss = 0.0
        for x in data_loader:
            if is_train:
                recon_loss, kl_div = self.loss_components(x)
                intermediate_loss = recon_loss + self.kl_loss_factor() * kl_div
                loss = recon_loss + kl_div
                self.optimizer.zero_grad()
                intermediate_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            else:
                with torch.no_grad():
                    loss = self.loss(x)
            avg_loss += loss.item() * x.size(0)
        avg_loss /= len(data_loader.dataset)
        return avg_loss

    def loss(self, x):
        recon_loss, kl_div = self.loss_components(x)
        return recon_loss + kl_div

    def loss_components(self, x):
        mean_z, log_var_z = self.model.encoding_parameters(x)
        z = self.model.reparameterize(mean_z, log_var_z)
        mean_recon_x, log_var_recon_x = self.model.decoding_parameters(z)
        # BULLSHIT => recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        recon_loss = -mean_field_log_prob(x - mean_recon_x, log_var_recon_x.exp()).mean()
        kl_div = -0.5 * (1 + log_var_z - mean_z.pow(2) - log_var_z.exp()).sum(dim=1).mean()
        return recon_loss, kl_div

    def kl_loss_factor(self):
        if self.epoch <= self.no_kl_penalty_epochs:
            return 0.5 / self.kl_annealing_epochs
        return min(1., (self.epoch - self.no_kl_penalty_epochs) / self.kl_annealing_epochs)

    def show_training_plot(self):
        plot_from = 30
        clear_output(wait=True)
        fig, axs = plt.subplots(figsize=(10, 10), nrows=2, ncols=2)

        ax = axs[0][0]
        epoch_list = np.arange(self.epoch) + 1
        ax.plot(epoch_list[plot_from:], self.train_loss_hist[plot_from:], label='Train Loss')
        ax.plot(epoch_list[plot_from:], self.val_loss_hist[plot_from:], label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('VAE Training')

        ax = axs[1][0]
        epoch_list = epoch_list[epoch_list % self.evaluate_samples_interval == 0]
        ax.plot(epoch_list, self.sample_scores, label='Raw VAE samples')
        ax.plot(epoch_list, self.cut_sample_scores, label='Log-prob cutoff Samples')
        ax.plot(epoch_list, self.mh_sample_scores, label='MH samples')
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(self.distribution_metric.name())

        def sample_scatter(ax, samples, **kwargs):
            ax.scatter(*pl(samples[:, (0, 1)]), alpha=0.5, s=1.5, **kwargs)
        
        ax = axs[0][1]
        target_samples = self.target.sample((self.n_eval_samples,))
        sample_scatter(ax, target_samples, label='Target Samples')
        good_xlim = ax.get_xlim()
        good_ylim = ax.get_ylim()
        sample_scatter(ax, self.model.sample((self.n_eval_samples,)), label='VAE Samples')
        
        ax = axs[1][1]
        reconstructed_target = self.model.reconstruct(target_samples)
        sample_scatter(ax, target_samples, label='Target Samples')
        sample_scatter(ax, reconstructed_target, label='Reconstructed Target Samples', color='green')
        
        for ax in (axs[0][1], axs[1][1]):
            ax.set_xlim(good_xlim)
            ax.set_ylim(good_ylim)
        for ax in axs.flatten():
            ax.legend()
        plt.show()
        print(f'Epoch {self.epoch}')
        print(f'\tTrain loss: {self.train_loss_hist[-1]:.4f}')
        print(f'\tValidation loss: {self.val_loss_hist[-1]:.4f}')
        print(f'\tLearning rate: {self.optimizer.param_groups[0]["lr"]}')


def get_filename(model, target):
    filename = f'{model}__{target}.pt'
    return filename
