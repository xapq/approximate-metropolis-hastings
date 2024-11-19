import torch
from pathlib import Path
from .utilities import CHECKPOINT_DIR
from .distributions import Distribution
from .nonlocal_net import NonlocalNet

class MnistEBMDistribution(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # maybe add temperature?
        CHECKPOINT_FILE = 'MNIST_NonlocalNet_100000.pth'
        self.model = NonlocalNet(n_c=1)
        self.model.load_state_dict(torch.load(Path(CHECKPOINT_DIR, CHECKPOINT_FILE), map_location=self.device, weights_only=True))
        self.model.eval()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.name = f'mnist_ebm'
        self.friendly_name = f'MNIST EBM'
        self.upsampler = torch.nn.Upsample(size=32)

    def log_prob(self, x, need_upsample=True):
        if need_upsample:
            x = self.upsampler(x)
        return -self.model(x)
