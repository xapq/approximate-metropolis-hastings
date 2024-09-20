# inspired by FlowMCMC from https://github.com/svsamsonov/ex2mcmc_new/blob/31d422ad704a145422db8cd19f9d3907bfb2e608/ex2mcmc/sampling_utils/adaptive_mc.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

from .y_utils import pl
from .utilities import dataloader_from_tensor
from .model_trainers import ModelTrainer


class FlowTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = kwargs.get("batch_size", 64)
        self.best_val_loss = float("inf")
        self.best_model_weights = None

    def fit(self, x_train, **kwargs):
        x_val = kwargs.get("x_val", self.target.sample((x_train.shape[0] // 10,)))
        n_epochs = kwargs.get("n_epochs", 800)
        plot_interval = kwargs.get("plot_interval", n_epochs)
        train_loader = dataloader_from_tensor(x_train, self.batch_size)
        val_loader = dataloader_from_tensor(x_val, self.batch_size)
        
        for epoch_id in range(n_epochs):
            self.epoch += 1
            train_loss = self.process_data_loader(train_loader, is_train=True)
            val_loss = self.process_data_loader(val_loader, is_train=False)
            self.record_loss(train_loss, train=True)
            self.record_loss(val_loss, train=False)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_weights = self.model.state_dict()
            if self.epoch % plot_interval == 0 or epoch_id == n_epochs - 1:
                self.show_training_plot()
            self._finish_epoch()
        self.model.load_state_dict(self.best_model_weights)
        self.model.eval()

    # TODO: this can be a weighted sum of forward and backward KL divergences
    def loss(self, target_samples):
        return -self.model.log_prob(target_samples).mean()

    def show_training_plot(self):
        clear_output(wait=True)
        fig, axs = plt.subplots(figsize=(10, 5), ncols=2)

        ax = axs[0]
        plot_from = 5
        for train in (True, False):
            epoch_list, loss_list = self.get_loss_history(train=train)
            ax.plot(epoch_list[plot_from:], loss_list[plot_from:], label='Train' if train else 'Validation')
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
        print(f'\tTrain Loss: {self.train_loss_history[-1][1]:.4f}')
        print(f'\tValidation Loss: {self.val_loss_history[-1][1]:.4f}')
        print(f'\tLearning rate: {self.optimizer.param_groups[0]["lr"]}')
