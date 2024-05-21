# inspired by FlowMCMC from https://github.com/svsamsonov/ex2mcmc_new/blob/31d422ad704a145422db8cd19f9d3907bfb2e608/ex2mcmc/sampling_utils/adaptive_mc.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from y_utils import pl


class FlowTrainer:
    def __init__(self, flow, target, **kwargs):
        self.flow = flow
        self.target = target
        self.device = kwargs.get("device", "cpu")
        self.batch_size = kwargs.get("batch_size", 64)
        self.grad_clip = kwargs.get("grad_clip", 1.0)
        self.step = 0
        self.loss_hist = []
        self.plot_interval = 50
        
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

        self.flow.to(self.device)

    def fit(self, n_steps=800, plot_interval=None):
        self.flow.train()
        for step_id in range(n_steps):
            self.step += 1
            target_samples = self.target.sample((self.batch_size,))
            loss = self.loss(target_samples)
            self.loss_hist.append(loss.item())
            if plot_interval is not None and (self.step % self.plot_interval == 0 or step_id == n_steps - 1):
                self.show_training_plot()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.flow.parameters(),
                self.grad_clip,
            )
            self.optimizer.step()
        self.flow.eval()

    # TODO: this can be a weighted sum of forward and backward KL divergences
    def loss(self, target_samples):
        return -self.flow.log_prob(target_samples).mean()

    def show_training_plot(self):
        clear_output(wait=True)
        fig, axs = plt.subplots(figsize=(10, 5), ncols=2)

        ax = axs[0]
        ax.plot(np.arange(self.step) + 1, self.loss_hist)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Flow Training')

        n_plot_samples = 2000
        proj_dims = (0, 1)
        def sample_scatter(sample, **kwargs):
            axs[1].scatter(*pl(sample[:, proj_dims]), alpha=0.5, s=1.5, **kwargs)
        sample_scatter(self.target.sample((n_plot_samples,)), label='Target Samples')
        sample_scatter(self.flow.sample((n_plot_samples,)), label='Flow Samples')
        axs[1].legend()
        
        plt.show()
        print(f'Step {self.step}')
        print(f'\tLoss: {self.loss_hist[-1]:.4f}')
        print(f'\tLearning rate: {self.optimizer.param_groups[0]["lr"]}')
            