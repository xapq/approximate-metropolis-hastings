import torch
from pathlib import Path
from torch_mimicry.nets import sngan
from .utilities import CHECKPOINT_DIR
from .distributions import Distribution, IndependentMultivariateNormal


"""
See 'Your GAN is Secretly an Energy-based Model and You Should Use Discriminator Driven Latent Sampling'
https://arxiv.org/pdf/2003.06060
"""
class SNGANLatentSpace(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.temperature = kwargs.get('temperature', 1.)

        self.dim = 128  # dimensionality of the GAN latent space
        self.generator = sngan.SNGANGenerator32().to(self.device)
        self.discriminator = sngan.SNGANDiscriminator32().to(self.device)
        self.prior = IndependentMultivariateNormal(
            torch.zeros(self.dim, device=self.device),
            torch.ones(self.dim, device=self.device)
        )
        self.name = f'sngan_latent_space_temp_{self.temperature}'
        self.friendly_name = f'SNGAN Temperature {self.temperature} Latent Space'
        
        self.generator.load_state_dict(torch.load(Path(CHECKPOINT_DIR, 'CIFAR10_SNGAN_Hinge_netG.pth'), map_location=self.device)['model_state_dict'])
        self.generator.eval()
        # print('Loaded generator')
        self.discriminator.load_state_dict(torch.load(Path(CHECKPOINT_DIR, 'CIFAR10_SNGAN_Hinge_netD.pth'), map_location=self.device)['model_state_dict'])
        self.discriminator.eval()
        # print('Loaded discriminator')

    def log_prob(self, z):
        return -self.energy(z) / self.temperature
    
    def energy(self, z):
        with torch.no_grad():
            return -self.prior.log_prob(z) - self.discriminator(self.generator(z)).squeeze(-1)
