"""A structure sampler bound to one trunk output."""

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

from jfold.sampling import core
from jfold.sampling.core import Denoise, SamplerSettings


class StructureSampler(eqx.Module):
    """Samples structures for one trunk output. It holds the denoiser's weights, so pass it to
    jitted functions as an argument."""

    denoise: Denoise
    atom_mask: Float[Array, " N"]
    settings: SamplerSettings

    def sample(self, key: PRNGKeyArray) -> Float[Array, "N 3"]:
        """Draw one structure. Use `jax.vmap` over keys for several."""
        return core.sample(self.denoise, self.atom_mask, self.settings, key=key)
