from abc import ABC, abstractmethod
from collections import defaultdict
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
        self.logs = defaultdict(list)

    @abstractmethod
    def loss(self, x):
        raise NotImplementedError

    def run_classic_epoch(self, train_loader, val_loader=None):
        train_loss = self.process_data_loader(train_loader, is_train=True)
        self.log_scalar('train_loss', train_loss)
        if val_loader is not None:
            val_loss = self.process_data_loader(val_loader, is_train=False)
            self.log_scalar('val_loss', val_loss)
        self._finish_epoch()

    # Calculate the loss on a training or validation set and possibly update model parameters
    def process_data_loader(self, data_loader, is_train=False):
        log_prefix = 'train_' if is_train else 'val_'
        log_dict = defaultdict(float)  # for accumulating information to log
        self.model.train(is_train)
        sum_losses = 0
        with torch.set_grad_enabled(is_train):
            for x in tqdm(data_loader):
                x = x.to(self.model.device)
                loss = self.loss(x, log_dict)
                if is_train:
                    self._step(loss)
                sum_losses += loss.detach() * x.size(0)
        epoch_loss = sum_losses.item() / len(data_loader.dataset)
        self.log_scalar(log_prefix + 'loss', epoch_loss)
        for value_name in log_dict:
            self.log_scalar(log_prefix + value_name, log_dict[value_name].item() / len(data_loader.dataset))
        return epoch_loss

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

    def log_scalar(self, log_name, data):
        self.logs[log_name].append((self.epoch, data))
    
    def get_log(self, log_name, from_epoch=0):
        epochs_data = torch.tensor(list(zip(*self.logs[log_name])))  # epochs and data stacked
        if from_epoch > 0:
            epochs_data = epochs_data[:, epochs_data[0] >= from_epoch]
        return epochs_data

    def clear_logs(self):
        self.logs = defaultdict(list)


class AdaptiveVAETrainer(ModelTrainer):
    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.no_latent_loss_epochs = kwargs.get("no_latent_loss_epochs", 0)
        self.loss_annealing_epochs = kwargs.get("loss_annealing_epochs", 1)
        self.model_log_likelihood = kwargs.get("model_log_likelihood")  # only needed for adaptive training
        self.beta = kwargs.get("beta", 1.)  # extra weighting factor for latent space loss

    def loss(self, x, log_dict=None):
        mean_z, log_var_z = self.model.encoding_parameters(x)
        z = self.model.reparameterize(mean_z, log_var_z)
        reconstruction_distribution = self.model.decoder_distribution(z)
        reconstruction_loss = -reconstruction_distribution.log_prob(x).mean()
        latent_loss = -0.5 * (1 + log_var_z - mean_z.pow(2) - log_var_z.exp()).sum(dim=1).mean()
        if log_dict is not None:
            log_dict['recon_loss'] += reconstruction_loss.detach() * x.shape[0]
            log_dict['latent_loss'] += latent_loss.detach() * x.shape[0]
        return reconstruction_loss + self.beta * self.get_latent_loss_factor() * latent_loss

    def get_latent_loss_factor(self):
        if self.epoch < self.no_latent_loss_epochs:
            return 0.01
        return min(1., (self.epoch - self.no_latent_loss_epochs + 1) / self.loss_annealing_epochs)

    '''
    def elementwise_losses(self, x):
        reconstruction_loss, kl_loss = self.loss_components(x)
        return reconstruction_loss + self.get_latent_loss_factor() * kl_loss
        
    def loss_components(self, x):
        mean_z, log_var_z = self.model.encoding_parameters(x)
        z = self.model.reparameterize(mean_z, log_var_z)
        reconstruction_distribution = self.model.decoder_distribution(z)
        reconstruction_loss = -reconstruction_distribution.log_prob(x)
        kl_loss = -0.5 * (1 + log_var_z - mean_z.pow(2) - log_var_z.exp()).sum(dim=1)
        return reconstruction_loss, kl_loss
    
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
    '''


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