import torch
import jax

# torch.tensor to jax.numpy.ndarray
### TODO: make this work for jax on cuda
def torch_to_jax(tensor):
    if isinstance(tensor, jnp.ndarray):
        return tensor
    return jnp.array(tensor.detach().cpu().numpy())
