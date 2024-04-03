import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
# from torchvision import datasets
# from torchvision import transforms
from torch.nn import functional as F
import torch.nn.init as init
import numpy as np
import math


import matplotlib.pyplot as plt
from IPython.display import clear_output

from y_utils import *

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
        encoder_dims = [data_dim, *hidden_dims, 2 * latent_dim]
        decoder_dims = [latent_dim, *hidden_dims[::-1], data_dim]
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
        return f'y01vae_dim_{self.data_dim}'
        
    def fit_distribution(self, target, kl_penalty, N_train, optimizer, scheduler=None, max_epochs=5000, kl_annealing_epochs=1, early_stopping_epochs=1000, batch_size=64):
        LOSS_THRESHOLD = 1e-4
        PLOT_INTERVAL = 20
        N_val = N_train // 10
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_weights = None
        lr = optimizer.param_groups[0]['lr']
        no_improvement_epochs = 0

        X_train = target.sample((N_train,))
        train_loader = dataloader_from_tensor(X_train, batch_size)
        val_loader = dataloader_from_tensor(target.sample((N_val,)), batch_size)
        
        for epoch in range(1, max_epochs + 1):
            cur_lr = optimizer.param_groups[0]['lr']
            if cur_lr < lr:
                lr = cur_lr
                self.load_state_dict(best_model_weights)
            
            cur_kl_penalty = min(1, epoch / kl_annealing_epochs) * kl_penalty
            train_recon_loss, train_kl_div = self.__run_epoch(train_loader, cur_kl_penalty, optimizer)
            train_loss = train_recon_loss + kl_penalty * train_kl_div
            train_losses.append(train_loss)
            val_recon_loss, val_kl_div = self.__run_epoch(val_loader, cur_kl_penalty)
            val_loss = val_recon_loss + kl_penalty * val_kl_div
            val_losses.append(val_loss)
    
            if epoch % PLOT_INTERVAL == 0:
                self.plot_losses(train_losses, val_losses, lr)
    
            # Autostopping
            if val_loss < (1 - LOSS_THRESHOLD) * best_val_loss:
                best_val_loss = val_loss
                best_model_weights = self.state_dict()
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
                if min(no_improvement_epochs, epoch - kl_annealing_epochs) > early_stopping_epochs:
                    break
            if scheduler is not None:
                scheduler.step(val_loss)
    
        self.load_state_dict(best_model_weights)

        latent_train = self.sample_q(X_train)
        self.latent_mean = torch.mean(latent_train, dim=0)
        self.latent_std = torch.std(latent_train, dim=0)
        self.set_std_factor(1)
        self.eval()

    def plot_losses(self, train_losses, val_losses, lr, plot_from=50):
        epoch = len(val_losses)
        clear_output(wait=True)
        plt.figure(figsize=(10, 5))
        epoch_list = np.arange(plot_from, epoch + 1)
        plt.plot(epoch_list, train_losses[plot_from-1:], label='Train Loss')
        plt.plot(epoch_list, val_losses[plot_from-1:], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('VAE Training on Synthetic Data')
        plt.yscale('log')
        plt.legend()
        plt.show()
        print(f'Epoch {epoch}')
        print(f'\tTrain loss: {train_losses[-1]:.4f}')
        print(f'\tValidation loss: {val_losses[-1]:.4f}')
        print(f'\tLearning rate: {lr:.6f}')

    def forward(self, x):
        mean_z, log_var_z = self.encode(x)
        z = self.reparameterize(mean_z, log_var_z)
        return self.decoder(z), mean_z, log_var_z

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def reconstruct(self, x):
        return self.forward(x)[0]

    def encode(self, x):
        mean_z, log_var_z = self.encoder(x).chunk(2, dim=-1)
        return mean_z, log_var_z

    def sample_q(self, x):
        mean_z, log_var_z = self.encode(x)
        z = torch.normal(mean_z, torch.exp(0.5 * log_var_z))
        return z.squeeze()

    def sample_prior(self, n=1, std_factor=1):
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=self.device) * std_factor
        return z

    def sample_latent(self, sample_shape=(1,)):
        return self.latent_sampling_distribution.sample(sample_shape)

    def sample(self, sample_shape=(1,)):
        return self.decoder(self.latent_sampling_distribution.sample(sample_shape))

    ### standard deviation multipler for ancestral sampling
    def set_std_factor(self, std_factor):
        self.std_factor = std_factor
        self.latent_sampling_distribution = torch.distributions.MultivariateNormal(self.latent_mean, self.latent_std.diag() * std_factor ** 2)

    # n_outliers -- amount of highest and lowest point estimates not taken into account
    def iw_log_marginal_estimate(self, x, L, batch_L=64):
        point_estimates = []
        for i in range(0, L, batch_L):
            point_estimates.append(self.__iw_log_marginal_estimate_batch(x, min(batch_L, L - i)))
        point_estimates = torch.cat(point_estimates)
        # log-sum-exp trick
        #point_estimates, indicies = torch.sort(point_estimates, dim=0)
        #point_estimates = point_estimates[n_outliers : L - n_outliers]
        estimate = torch.logsumexp(point_estimates.type(torch.DoubleTensor), dim=0) - np.log(point_estimates.shape[0])
        #max_est = torch.max(point_estimates)
        #estimate = (point_estimates - max_est).to(torch.float64).exp().mean(dim=0).log() + max_est
        return estimate

    # Returns reconstruction loss and KL divergence from prior
    def loss_components(self, x):
        recon_x, mean_z, log_var_z = self(x)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_div = -0.5 * (1 + log_var_z - mean_z.pow(2) - log_var_z.exp()).sum(dim=1).mean()
        return recon_loss, kl_div

    def loss(self, x, kl_penalty):
        recon_loss, kl_div = self.loss_components(x)
        return recon_loss + kl_penalty * kl_div

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
                    optimizer.step()
            avg_recon_loss += recon_loss.item() * x.size(0)
            avg_kl_div += kl_div.item() * x.size(0)
        avg_recon_loss /= len(data_loader.dataset)
        avg_kl_div /= len(data_loader.dataset)
        return avg_recon_loss, avg_kl_div
    
    def __iw_log_marginal_estimate_batch(self, x, L):
        self.eval()
        with torch.no_grad():
            mean_z, log_var_z = self.encode(x)
            std_z = torch.exp(0.5 * log_var_z)
            m = torch.distributions.normal.Normal(mean_z, std_z)
            zs = m.sample((L,))
            dec_zs = self.decoder(zs)
            log_p_zs = self.latent_sampling_distribution.log_prob(zs)  # HEHE idea
            log_p_x_cond_zs = normal_density_log(dec_zs, torch.ones(self.data_dim, device=self.device), x)
            log_q_zs = normal_density_log(mean_z, std_z, zs)
            point_estimates = log_p_x_cond_zs + log_p_zs - log_q_zs
            return point_estimates

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)