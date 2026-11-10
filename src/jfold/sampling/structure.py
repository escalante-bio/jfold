"""A structure sampler bound to one trunk output."""

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

from jfold.sampling import core
from jfold.sampling.core import Denoise, SamplerSettings


class StructureSampler(eqx.Module):
    """Samples structures from one trunk output: the model's denoiser with its
    step-invariant context bound (computed once, when the sampler is built), the atoms the
    denoiser moves, and the sampler's settings.

    Build one with a model's `structure_sampler(...)`. It holds the denoiser's weights, so
    pass it to jitted functions as an argument.
    """

    denoise: Denoise
    atom_mask: Float[Array, " N"]  # the atoms it centres and aligns on (not padding)
    settings: SamplerSettings

    def sample(self, key: PRNGKeyArray) -> Float[Array, "N 3"]:
        """One structure, from fresh noise; `jax.vmap` over keys for several."""
        return core.sample(self.denoise, self.atom_mask, self.settings, key=key)
