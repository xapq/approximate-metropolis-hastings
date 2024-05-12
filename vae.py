import torch
from pytorch_warmup import LinearWarmup
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
# from torchvision import datasets
# from torchvision import transforms
from torch.nn import functional as F
import torch.nn.init as init
import numpy as np
import math
from samplers import metropolis_hastings_filter, log_prob_cutoff_filter
import matplotlib.pyplot as plt
from IPython.display import clear_output
from y_utils import *
from warmup_schedules import CubicWarmup


def dataloader_from_tensor(X, batch_size):
    dataset = TensorDataset(X)
    dataloader = DataLoader(X, batch_size=batch_size, shuffle=True)
    return dataloader


class BatchReshape(nn.Module):
    def __init__(self, shape):
        super(BatchReshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)


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

        '''train_info = {
            'epoch': 0,
            'kl_penalty': 0,
            'train_losses': [],
            'val_losses': [],
        }'''

        self.prior = torch.distributions.MultivariateNormal(torch.zeros(latent_dim, device=device), torch.diag(torch.ones(latent_dim, device=device)))
        self.latent_sampling_distribution = None
        self.std_factor = None
        self.latent_mean = None
        self.latent_std = None
        
        self.to(device)
        self.__init_weights()

    def __repr__(self):
        return f'y02vae_D{self.data_dim}_layers{self.hidden_dims}'

    LOSS_THRESHOLD = 1e-4
    PLOT_INTERVAL = 30
    EVALUATE_SAMPLES_INTERVAL = 80
    
    def fit_distribution(self, target, N_train, optimizer, scheduler=None, max_epochs=5000, no_kl_penalty_epochs=10, kl_annealing_epochs=100, early_stopping_epochs=1000, batch_size=64, distribution_metric=None):
        N_val = N_train // 10
        n_eval_samples = min(500, N_train)
        self.warmup_scheduler = LinearWarmup(optimizer, warmup_period=50)

        X_train = target.sample((N_train,))
        X_val = target.sample((N_val,))
        train_loader = dataloader_from_tensor(X_train, batch_size)
        val_loader = dataloader_from_tensor(X_val, batch_size)

        # filters & sample score
        L = 64
        beta = 1
        cutoff_quantile = 1e-4
        model_log_prob_estimator = lambda x : self.iw_log_marginal_estimate(x, L=L, beta=beta, batch_L=32)
        cutoff_log_prob = target.log_prob(X_train).quantile(cutoff_quantile)
        sample_score = lambda sample: distribution_metric(X_train[:n_eval_samples], sample).item()
        target_samples = target.sample((2000,))
        best_sample_score = sample_score(target.sample((n_eval_samples,)))

        def epoch_kl_penalty(epoch):
            if epoch <= no_kl_penalty_epochs:
                return 0.01
            return min(1., (epoch - no_kl_penalty_epochs) / kl_annealing_epochs)
        
        train_losses = []
        val_losses = []
        sample_scores = []
        cut_sample_scores = []
        mh_sample_scores = []
        best_val_loss = float('inf')
        best_model_weights = None
        lr = optimizer.param_groups[0]['lr']
        no_improvement_epochs = 0
        
        for epoch in range(1, max_epochs + 1):
            lr = optimizer.param_groups[0]['lr']
            #print('Epoch', epoch, 'lr', lr)
            assert(not torch.stack([p.isnan().any() for p in self.parameters()]).any())
            
            #if cur_lr < lr:
            #    lr = cur_lr
            #    self.load_state_dict(best_model_weights)

            no_penalty_epochs = 1
            cur_kl_penalty = epoch_kl_penalty(epoch)
            train_recon_loss, train_kl_div = self.__run_epoch(train_loader, cur_kl_penalty, optimizer)
            train_loss = train_recon_loss + train_kl_div
            train_losses.append(train_loss)
            val_recon_loss, val_kl_div = self.__run_epoch(val_loader, cur_kl_penalty)
            val_loss = val_recon_loss +  val_kl_div
            val_losses.append(val_loss)

            # calculate sample quality
            if epoch % self.EVALUATE_SAMPLES_INTERVAL == 0:
                self.eval()
                latent_train = self.encode(X_train)
                self.latent_mean = torch.mean(latent_train, dim=0)
                self.latent_std = torch.std(latent_train, dim=0)
                self.set_std_factor(1)
                model_samples = self.sample((n_eval_samples,))
                sample_scores.append(sample_score(model_samples))
                cut_acc_rate, cut_indicies = log_prob_cutoff_filter(target, model_samples, cutoff_log_prob)
                cut_samples = model_samples[cut_indicies]
                if cut_samples.shape[0] == 0:
                    cut_samples = torch.zeros((1, self.data_dim), device=self.device)
                cut_sample_scores.append(sample_score(cut_samples))
                mh_acc_rate, mh_indicies = metropolis_hastings_filter(target, cut_samples, model_log_prob_estimator)
                mh_samples = cut_samples[mh_indicies]
                mh_sample_scores.append(sample_score(mh_samples))
    
            if epoch % self.PLOT_INTERVAL == 0:
                self.eval()
                latent_train = self.encode(X_train)
                self.latent_mean = torch.mean(latent_train, dim=0)
                self.latent_std = torch.std(latent_train, dim=0)
                self.set_std_factor(1)
                self.plot_losses(train_losses, val_losses, sample_scores, cut_sample_scores, mh_sample_scores, best_sample_score, lr, target_samples, distribution_metric)
    
            # Autostopping
            if val_loss < (1 - self.LOSS_THRESHOLD) * best_val_loss:
                best_val_loss = val_loss
                best_model_weights = self.state_dict()
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
                if min(no_improvement_epochs, epoch - kl_annealing_epochs) > early_stopping_epochs:
                    break

            with self.warmup_scheduler.dampening():
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
    
        self.load_state_dict(best_model_weights)
        self.eval()

    def plot_losses(self, train_losses, val_losses, sample_scores, cut_sample_scores, mh_sample_scores, best_sample_score, lr, target_samples, distribution_metric, plot_from=5):
        clear_output(wait=True)
        epoch = len(val_losses)
        fig, axs = plt.subplots(figsize=(10, 10), nrows=2, ncols=2)

        ax = axs[0][0]
        epoch_list = np.arange(plot_from, epoch + 1)
        ax.plot(epoch_list, train_losses[plot_from-1:], label='Train Loss')
        ax.plot(epoch_list, val_losses[plot_from-1:], label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('VAE Training on Synthetic Data')
        # ax.set_yscale('log')
        ax = axs[1][0]
        epoch_list = np.arange(1, epoch + 1)
        epoch_list = epoch_list[epoch_list % self.EVALUATE_SAMPLES_INTERVAL == 0]
        ax.plot(epoch_list, sample_scores, label='Raw VAE samples')
        ax.plot(epoch_list, cut_sample_scores, label='Log-prob cutoff Samples')
        ax.plot(epoch_list, mh_sample_scores, label='MH samples')
        ax.axhline(best_sample_score, label='Best possible score', linestyle='--', color='black')
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(distribution_metric.name())

        proj_dims = (0, 1)
        ax = axs[0][1]
        ax.scatter(*pl(target_samples[:, proj_dims]), alpha=0.5, label='Target Samples', s=1.5)
        good_xlim = ax.get_xlim()
        good_ylim = ax.get_ylim()
        ax.scatter(*pl(self.sample((2000,))[:, proj_dims]), alpha=0.5, label='VAE Samples', s=1.5)
        ax = axs[1][1]
        ax.scatter(*pl(target_samples[:, proj_dims]), alpha=0.5, label='Target Samples', s=1.5)
        reconstructed_target = self.reconstruct(target_samples)
        ax.scatter(*pl(reconstructed_target[:, proj_dims]), alpha=0.5, label='Reconstructed Target Samples', color='green', s=1.5)
        for ax in (axs[0][1], axs[1][1]):
            ax.set_xlim(good_xlim)
            ax.set_ylim(good_ylim)
        
        for ax in axs.flatten():
            ax.legend()
        plt.show()
        print(f'Epoch {epoch}')
        print(f'\tTrain loss: {train_losses[-1]:.4f}')
        print(f'\tValidation loss: {val_losses[-1]:.4f}')
        print(f'\tLearning rate: {lr}')

    def forward(self, x):
        assert(False)
        pass
        # mean_z, log_var_z = self.encode(x)
        # z = self.reparameterize(mean_z, log_var_z)
        # mean_x, log_var_x = self.decode(z)
        # return mean_x, log_var_x, mean_z, log_var_z

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

    def sample_prior(self, n=1, std_factor=1):
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=self.device) * std_factor
        return z

    def sample_latent(self, sample_shape=(1,)):
        return self.latent_sampling_distribution.sample(sample_shape)

    def sample(self, sample_shape=(1,)):
        z = self.sample_latent(sample_shape)
        return self.decode(z)

    ### standard deviation multipler for ancestral sampling
    def set_std_factor(self, std_factor):
        self.std_factor = std_factor
        self.latent_sampling_distribution = torch.distributions.MultivariateNormal(self.latent_mean, self.latent_std.diag() * std_factor ** 2)

    # beta -- smoothing constant for log-sum-exp
    def iw_log_marginal_estimate(self, x, L, beta=1, batch_L=64):
        point_estimates = []
        for i in range(0, L, batch_L):
            point_estimates.append(self.__iw_log_marginal_estimate_batch(x, min(batch_L, L - i)))
        point_estimates = torch.cat(point_estimates)
        
        # log-sum-exp trick
        #point_estimates, indicies = torch.sort(point_estimates, dim=0)
        #point_estimates = point_estimates[n_outliers : L - n_outliers]
        estimate = (torch.logsumexp(point_estimates.double() / beta, dim=0) - np.log(point_estimates.shape[0])) * beta
        #max_est = torch.max(point_estimates)
        #estimate = (point_estimates - max_est).to(torch.float64).exp().mean(dim=0).log() + max_est
        return estimate

    # Returns reconstruction loss and KL divergence from prior
    def loss_components(self, x):
        mean_z, log_var_z = self.encoding_parameters(x)
        z = self.reparameterize(mean_z, log_var_z)
        mean_recon_x, log_var_recon_x = self.decoding_parameters(z)
        # recon_x = self.reparameterize(mean_recon_x, log_var_recon_x)
        # recon_x, mean_z, log_var_z = self(x)
        
        #recon_loss = F.mse_loss(recon_x, x, reduction='mean') # <= BULLSHIT
        recon_loss = -mean_field_log_prob(x - mean_recon_x, log_var_recon_x.exp()).mean()
        kl_div = -0.5 * (1 + log_var_z - mean_z.pow(2) - log_var_z.exp()).sum(dim=1).mean()
        return recon_loss, kl_div

    def loss(self, x):
        recon_loss, kl_div = self.loss_components(x)
        return recon_loss + kl_div

    def save_knowledge(self, filename):
        save_dict = {
            'model_state_dict': self.state_dict(),
            'latent_mean': self.latent_mean,
            'latent_std': self.latent_std
        }
        torch.save(save_dict, filename)

    def load_knowledge(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.latent_mean = checkpoint['latent_mean']
        self.latent_std = checkpoint['latent_std']
        self.eval()
    
    def __run_epoch(self, data_loader, cur_kl_penalty, optimizer=None):
        is_train = optimizer is not None
        if is_train:
            self.train()
        else:
            self.eval()
        avg_recon_loss = 0.0
        avg_kl_div = 0.0
        for x in data_loader:
            if is_train:
                optimizer.zero_grad()
            with torch.set_grad_enabled(is_train):
                recon_loss, kl_div = self.loss_components(x.to(self.device))
                loss = recon_loss + cur_kl_penalty * kl_div
                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)
                    optimizer.step()
            avg_recon_loss += recon_loss.item() * x.size(0)
            avg_kl_div += kl_div.item() * x.size(0)
        avg_recon_loss /= len(data_loader.dataset)
        avg_kl_div /= len(data_loader.dataset)
        return avg_recon_loss, avg_kl_div
    
    def __iw_log_marginal_estimate_batch(self, x, L):
        self.eval()
        with torch.no_grad():
            mean_z, log_var_z = self.encoding_parameters(x)
            std_z = torch.exp(0.5 * log_var_z)
            m = torch.distributions.normal.Normal(mean_z, std_z)
            zs = m.sample((L,))
            log_p_zs = self.latent_sampling_distribution.log_prob(zs)  # HEHE idea
            mean_x_cond_zs, log_var_x_cond_zs = map(lambda t: t.unflatten(dim=0, sizes=(L, -1)),
                                                      self.decoding_parameters(zs.flatten(end_dim=1)))
            # log_p_x_cond_zs = normal_density_log(dec_zs, torch.ones(self.data_dim, device=self.device), x)
            log_p_x_cond_zs = mean_field_log_prob(x - mean_x_cond_zs, log_var_x_cond_zs.exp())
            # log_q_zs = normal_density_log(mean_z, std_z, zs)
            log_q_zs = mean_field_log_prob(zs - mean_z, log_var_z.exp())
            point_estimates = log_p_x_cond_zs + log_p_zs - log_q_zs
            return point_estimates

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)


def get_filename(model, target):
    filename = f'{model}__{target}.pt'
    return filename
