# inspired by FlowMCMC from https://github.com/svsamsonov/ex2mcmc_new/blob/31d422ad704a145422db8cd19f9d3907bfb2e608/ex2mcmc/sampling_utils/adaptive_mc.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

from .y_utils import pl
from .utilities import dataloader_from_tensor


class FlowTrainer:
    def __init__(self, flow, target, **kwargs):
        self.model = flow
        self.target = target
        self.device = kwargs.get("device", "cpu")
        self.batch_size = kwargs.get("batch_size", 64)
        self.grad_clip = kwargs.get("grad_clip", 1.0)
        self.scheduler = kwargs.get("scheduler", None)
        self.epoch = 0
        self.train_loss_hist = []
        self.val_loss_hist = []
        self.best_val_loss = float("inf")
        self.best_model_weights = None
        
        optimizer = kwargs.get("optimizer", "adam")
        lr = kwargs.get("lr", 1e-3)
        wd = kwargs.get("wd", 1e-4)
        momentum = kwargs.get("momentum",0.9)
        if isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, str):
            if optimizer.lower() == "adam":
                self.optimizer = torch.optim.Adam(
                    flow.parameters(), lr=lr, weight_decay=wd
                )
            elif optimizer.lower() == "sgd":
                self.optimizer = torch.optim.SGD(
                    flow.parameters(), lr=lr, weight_decay=wd, momentum=momentum
                )
            else:
                raise ValueError

        self.model.to(self.device)

    def fit(self, x_train, **kwargs):
        x_val = kwargs.get("x_val", self.target.sample((x_train.shape[0] // 10,)))
        n_epochs = kwargs.get("n_epochs", 800)
        plot_interval = kwargs.get("plot_interval", n_epochs)
        train_loader = dataloader_from_tensor(x_train, self.batch_size)
        val_loader = dataloader_from_tensor(x_val, self.batch_size)
        
        for epoch_id in range(n_epochs):
            self.epoch += 1
            train_loss = self.run_epoch(train_loader, is_train=True)
            val_loss = self.run_epoch(val_loader, is_train=False)
            self.train_loss_hist.append(train_loss)
            self.val_loss_hist.append(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_weights = self.model.state_dict()
            if self.epoch % plot_interval == 0 or epoch_id == n_epochs - 1:
                self.show_training_plot()
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
                loss = self.loss(x)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            else:
                with torch.no_grad():
                    loss = self.loss(x)
            avg_loss += loss.item() * x.size(0)
        avg_loss /= len(data_loader.dataset)
        return avg_loss

    # TODO: this can be a weighted sum of forward and backward KL divergences
    def loss(self, target_samples):
        return -self.model.log_prob(target_samples).mean()

    def show_training_plot(self):
        clear_output(wait=True)
        fig, axs = plt.subplots(figsize=(10, 5), ncols=2)

        ax = axs[0]
        plot_from = 5
        ax.plot(np.arange(plot_from, self.epoch) + 1, self.train_loss_hist[plot_from:], label='Train')
        ax.plot(np.arange(plot_from, self.epoch) + 1, self.val_loss_hist[plot_from:], label='Validation')
        ax.set_ylim(min(min(self.val_loss_hist), min(self.train_loss_hist)), self.train_loss_hist[plot_from])
        ax.set_xlabel('Step')
        ax.set_ylabel('Negative Log Likelihood')
        ax.legend()

        n_plot_samples = 2000
        proj_dims = (0, 1)
        def sample_scatter(sample, **kwargs):
            axs[1].scatter(*pl(sample[:, proj_dims]), alpha=0.5, s=1.5, **kwargs)
        sample_scatter(self.target.sample((n_plot_samples,)), label='Target Samples')
        good_xlim = axs[1].get_xlim()
        good_ylim = axs[1].get_ylim()
        sample_scatter(self.model.sample((n_plot_samples,)), label='Flow Samples')
        axs[1].set_xlim(good_xlim)
        axs[1].set_ylim(good_ylim)
        axs[1].legend()

        fig.suptitle('Flow Training')
        plt.show()
        print(f'Epoch {self.epoch}')
        print(f'\tTrain Loss: {self.train_loss_hist[-1]:.4f}')
        print(f'\tValidation Loss: {self.val_loss_hist[-1]:.4f}')
        print(f'\tLearning rate: {self.optimizer.param_groups[0]["lr"]}')


class AdaptiveFlowTrainer:
    def __init__(self, flow, target, **kwargs):
        self.model = flow
        self.target = target
        self.device = kwargs.get("device", "cpu")
        self.forward_kl_factor = kwargs.get("forward_kl_factor", 1.)
        self.backward_kl_factor = kwargs.get("backward_kl_factor", 1.)
        self.grad_clip = kwargs.get("grad_clip", 1.0)
        self.scheduler = kwargs.get("scheduler", None)
        self.loss_history = []
        
        optimizer = kwargs.get("optimizer", "adam")
        lr = kwargs.get("lr", 1e-3)
        wd = kwargs.get("wd", 1e-4)
        momentum = kwargs.get("momentum",0.9)
        if isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, str):
            if optimizer.lower() == "adam":
                self.optimizer = torch.optim.Adam(
                    flow.parameters(), lr=lr, weight_decay=wd
                )
            elif optimizer.lower() == "sgd":
                self.optimizer = torch.optim.SGD(
                    flow.parameters(), lr=lr, weight_decay=wd, momentum=momentum
                )
            else:
                raise ValueError

        self.model.to(self.device)

    def sample_and_train(self, batch_size=64):
        model_samples = self.model.rsample((batch_size,))
        # P - target distribution, Q - model distribution
        logp = self.target.log_prob(model_samples)
        logq = self.model.log_prob(model_samples)
        logw = logp - logq  # unnormalized log importance weights
        importance_weights = torch.softmax(logw.detach(), dim=0)
        forward_kl = -(importance_weights * logq).sum()
        backward_kl = -logw.mean()
        loss = self.forward_kl_factor * forward_kl + self.backward_kl_factor * backward_kl
    
        self.loss_history.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return model_samples

    def train_on_sample(self, training_sample):
        # Forward KL
        logq = self.model.log_prob(training_sample)
        forward_kl = -logq.mean()
        # Backward KL
        batch_size = training_sample.shape[0]
        model_sample = self.model.rsample((batch_size,))
        logp = self.target.log_prob(model_sample)
        logq = self.model.log_prob(model_sample)
        backward_kl = (logq - logp).mean()
        # Loss
        loss = self.forward_kl_factor * forward_kl + self.backward_kl_factor * backward_kl
        # Update
        self.loss_history.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()



























    