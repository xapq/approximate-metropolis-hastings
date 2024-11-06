import torch
import math


def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def print_tensors_on_gpu():
    import gc

    def human_readable_size(size, decimal_places=2):
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024.0:
                break
            size /= 1024.0
        return f"{size:.{decimal_places}f} {unit}"
    
    def tensor_size_in_bytes(tensor):
        return tensor.element_size() * tensor.nelement()
    
    # List to hold tuples of (tensor size in bytes, tensor object)
    tensors_on_gpu = []
    
    # Iterate over all objects, filter for tensors on the GPU
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                # Add tensor and its size to the list
                tensors_on_gpu.append((tensor_size_in_bytes(obj), obj))
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data) and obj.data.is_cuda:
                # Handle nn.Parameter and similar types
                tensors_on_gpu.append((tensor_size_in_bytes(obj.data), obj.data))
        except:
            pass
    
    # Sort the list of tensors by size in descending order
    tensors_on_gpu.sort(key=lambda x: x[0], reverse=True)
    
    # Print the sorted list of tensors
    print('Tensors on GPU (shape, size):')
    for size, tensor in tensors_on_gpu:
        print(f"({list(tensor.shape)},\t{human_readable_size(size)})",  end='\n')


def to_numpy(x):
    return x.detach().cpu().numpy()


# For passing as an argument to matplotlib.pyplot.scatter
def pl(x):
    return x.T.detach().cpu().numpy()

def unify_clim(*args):
    vmin = float('inf')
    vmax = float('-inf')
    for arg in args:
        vmin_i, vmax_i = arg.get_clim()
        vmin = min(vmin, vmin_i)
        vmax = max(vmax, vmax_i)
    for arg in args:
        arg.set_clim(vmin, vmax)


def log_sum_exp(x):
    x_max = torch.max(x)
    return torch.log(torch.sum(torch.exp((x - x_max).to(torch.float64)))) + x_max


class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)

    def transform(self, data):
        if self.mean is None or self.std is None:
            raise ValueError("fit method must be called before transform")
        return (data - self.mean) / self.std

    def inverse_transform(self, normalized_data):
        if self.mean is None or self.std is None:
            raise ValueError("fit method must be called before inverse_transform")
        return normalized_data * self.std + self.mean

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


def normal_density_log(mean, std_dev, x):
    normal_dist = torch.distributions.MultivariateNormal(mean, torch.diag_embed(std_dev ** 2))
    log_density = normal_dist.log_prob(x)
    return log_density


def approximate_with_gaussian(x):
    mean = x.mean(dim=0)
    variances = x.var(dim=0, unbiased=True)
    cov = torch.diag(variances)
    return torch.distributions.MultivariateNormal(mean, cov)


# adapted from pytorch gaussian_nll_loss
def mean_field_log_prob(
    displacement: torch.Tensor,
    var: torch.Tensor,
    full: bool = False,
    eps: float = 1e-6
) -> torch.Tensor:
    r"""Calculate negative log likelihood of multivariate Gaussian distributions with diagonal covariances and expectation 0 at given points

    Args:
        input: (*, D) tensor. Expectation of the Gaussian distribution.
        target: (*, D) tensor. Sample from the Gaussian distribution.
        var: (*, D) tensor of positive variances in each dimension, one for each of the expectations
        full (bool, optional): include the constant term in the loss calculation. Default: ``False``.
        eps (float, optional): value added to var, for stability. Default: 1e-6.
    """
    
    # Check var size
    # If var.size == input.size, the case is heteroscedastic and no further checks are needed.
    # Otherwise:
    #if var.size() != displacement.size():
    #    print(var.size(), displacement.size())
    #    raise ValueError("var is of incorrect size")

    # Entries of var must be non-negative
    ### CHECKING THIS TAKES 97% OF TOTAL mean_field_log_prob TIME
    # if torch.any(var < 0):
    #    raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    # Calculate the loss
    D = displacement.shape[-1]
    loss = 0.5 * (torch.log(var) + (displacement)**2 / var).sum(dim=-1)
    if full:
        loss += 0.5 * D * math.log(2 * math.pi)

    return -loss
