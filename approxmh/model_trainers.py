from abc import ABC, abstractmethod
from tqdm import tqdm
import torch
from pytorch_warmup import LinearWarmup
from .y_utils import mean_field_log_prob


class ModelTrainer:
    def __init__(self, model, target, **kwargs):
        # Properties
        self.model = model
        self.target = target
        self.grad_clip = kwargs.get("grad_clip", None)
        # Optimizer
        optimizer = kwargs.get("optimizer", "adam")
        lr = kwargs.get("lr", 1e-3)
        wd = kwargs.get("wd", 1e-4)
        if isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, str):
            if optimizer.lower() == "adam":
                self.optimizer = torch.optim.Adam(
                    model.parameters(), lr=lr, weight_decay=wd
                )
            elif optimizer.lower() == "sgd":
                momentum = kwargs.get("momentum", 0.9)
                self.optimizer = torch.optim.SGD(
                    model.parameters(), lr=lr, weight_decay=wd, momentum=momentum
                )
            else:
                raise ValueError("Invalid optimizer specification")
        else:
            self.optimizer = None
        # Learning rate scheduler
        scheduler = kwargs.get("scheduler")
        if isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            self.scheduler = scheduler
        elif isinstance(scheduler, str):
            if scheduler.lower() == "steplr":
                step_size = kwargs.get("scheduler_step_size")
                gamma = kwargs.get("scheduler_gamma", 0.1)
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)
            else:
                raise ValueError("Invalid scheduler specification")
        else:
            self.scheduler = None
        # Warmup scheduler
        warmup_epochs = kwargs.get("warmup_epochs", 1)
        self.warmup_scheduler = LinearWarmup(self.optimizer, warmup_period=warmup_epochs)
        # Model trainer state
        self.epoch = 0
        self.train_loss_history = []
        self.val_loss_history = []

    @abstractmethod
    def loss(self, x):
        raise NotImplementedError

    def run_classic_epoch(self, train_loader, val_loader=None):
        train_loss = self.process_data_loader(train_loader, is_train=True)
        self.record_loss(train_loss, train=True)
        if val_loader is not None:
            val_loss = self.process_data_loader(val_loader, is_train=False)
            self.record_loss(val_loss, train=False)
        self._finish_epoch()

    # Calculate the loss on a training or validation set and possibly update model parameters
    def process_data_loader(self, data_loader, is_train=False):
        self.model.train(is_train)
        sum_losses = 0
        with torch.set_grad_enabled(is_train):
            for x in tqdm(data_loader):
                x = x.to(self.model.device)
                loss = self.loss(x)
                if is_train:
                    self._step(loss)
                sum_losses += loss.detach() * x.size(0)
        return sum_losses.item() / len(data_loader.dataset)

    # Update epoch number, learning rate, loss function if needed
    def _finish_epoch(self):
        with self.warmup_scheduler.dampening():
            if self.scheduler is not None:
                self.scheduler.step()
        self.epoch += 1

    # Update model weights based on a loss
    def _step(self, loss_value):
        self.optimizer.zero_grad()
        loss_value.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

    def record_loss(self, loss_scalar, train=False):
        loss_history = self.train_loss_history if train else self.val_loss_history
        loss_history.append((self.epoch, loss_scalar))

    def get_loss_history(self, train=False):
        loss_history = self.train_loss_history if train else self.val_loss_history
        return zip(*loss_history)

    def clear_loss_history(self):
        self.train_loss_history = []
        self.val_loss_history = []


class AdaptiveVAETrainer(ModelTrainer):
    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.no_kl_penalty_epochs = kwargs.get("no_kl_penalty_epochs", 0)
        self.kl_annealing_epochs = kwargs.get("kl_annealing_epochs", 1)
        self.model_log_likelihood = kwargs.get("model_log_likelihood")  # only needed for adaptive training

    def loss(self, x):
        return torch.mean(self.elementwise_losses(x))

    def elementwise_losses(self, x):
        reconstruction_loss, kl_divergence = self.loss_components(x)
        return reconstruction_loss + self.get_kl_loss_factor() * kl_divergence
        
    def loss_components(self, x):
        mean_z, log_var_z = self.model.encoding_parameters(x)
        z = self.model.reparameterize(mean_z, log_var_z)
        reconstruction_distribution = self.model.decoder_distribution(z)
        reconstruction_loss = -reconstruction_distribution.log_prob(x)
        kl_divergence = -0.5 * (1 + log_var_z - mean_z.pow(2) - log_var_z.exp()).sum(dim=1)
        return reconstruction_loss, kl_divergence

    def get_kl_loss_factor(self):
        if self.epoch < self.no_kl_penalty_epochs:
            return 0.01
        return min(1., (self.epoch - self.no_kl_penalty_epochs + 1) / self.kl_annealing_epochs)

    def sample_and_train(self, sample_size=128):
        model_samples = self.model.rsample((sample_size,))
        # P - target distribution, Q - model distribution
        logp = self.target.log_prob(model_samples)
        with torch.no_grad():
            logq = self.model_log_likelihood(model_samples)
        logw = logp.detach() - logq
        assert(not logw.requires_grad)
        
        importance_weights = torch.softmax(logw.detach(), dim=0)
        loss = torch.sum(importance_weights * self.elementwise_losses(model_samples))

        self.train_loss_history[self.epoch] = loss.item()
        self.step(loss)
        return model_samples


class AdaptiveFlowTrainer(ModelTrainer):
    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.forward_kl_factor = kwargs.get("forward_kl_factor", 1.)
        self.backward_kl_factor = kwargs.get("backward_kl_factor", 1.)

    def loss(self, x):
        # Forward KL
        logq = self.model.log_prob(x)
        forward_kl = -logq.mean()
        # Backward KL
        batch_size = x.shape[0]
        model_sample = self.model.rsample((batch_size,))
        logp = self.target.log_prob(model_sample)
        logq = self.model.log_prob(model_sample)
        backward_kl = (logq - logp).mean()
        # Loss
        return self.forward_kl_factor * forward_kl + self.backward_kl_factor * backward_kl

    def sample_and_train(self, sample_size=128):
        model_samples = self.model.rsample((sample_size,))
        # P - target distribution, Q - model distribution
        logp = self.target.log_prob(model_samples)
        logq = self.model.log_prob(model_samples)
        logw = logp - logq  # unnormalized log importance weights
        importance_weights = torch.softmax(logw.detach(), dim=0)
        forward_kl = -(importance_weights * logq).sum()
        backward_kl = -logw.mean()
        loss = self.forward_kl_factor * forward_kl + self.backward_kl_factor * backward_kl
        self.loss_history.append(loss.item())
        self.step(loss)
        return model_samples