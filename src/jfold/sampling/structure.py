"""A structure sampler bound to one trunk output."""

import dataclasses

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from jfold.sampling import core, schedules
from jfold.sampling.core import Denoise, Noise, SamplerSettings


class StructureSampler(eqx.Module):
    """Samples structures from one trunk output: the model's denoiser with its
    step-invariant context bound (computed once, when the sampler is built, and reused by
    every sample), the atoms the denoiser moves, and the sampler's settings.

    Build one with a model's `structure_sampler(...)`. It holds the model's weights, so
    pass it to jitted functions as an argument.
    """

    denoise: Denoise
    atom_mask: Float[Array, " N"]  # the atoms it centres and aligns on (not padding)
    settings: SamplerSettings

    @property
    def num_steps(self) -> int:
        return self.settings.num_steps

    def sample(self, key: PRNGKeyArray) -> Float[Array, "N 3"]:
        """One structure, from fresh noise; `jax.vmap` over keys for several."""
        return core.sample(self.denoise, self.atom_mask, self.settings, key=key)

    def integrate(self, noise: Noise) -> Float[Array, "N 3"]:
        """The structure from the given draws (deterministic)."""
        return core.integrate(self.denoise, self.atom_mask, self.settings, noise)

    def with_settings(
        self, settings: SamplerSettings | None = None, **changes
    ) -> "StructureSampler":
        """The same trunk output sampled with other settings: `settings`, and/or
        `changes` to their fields (e.g. `update=exponential_update`)."""
        settings = self.settings if settings is None else settings
        return dataclasses.replace(self, settings=dataclasses.replace(settings, **changes))


def _denoise(model, context, x, sigma):
    # No key: the sampler draws all its noise itself (`Noise`), and a diffusion module
    # with dropout switched on raises rather than reusing one key at every step.
    return model.denoise(context, x, sigma)


def structure_sampler(
    model: eqx.Module,
    context: eqx.Module,
    *,
    num_steps: int | None = None,
    schedule: schedules.Schedule | None = None,
    **changes,
) -> StructureSampler:
    """`model`'s sampler for `context` (from `model.denoiser_context`): upstream's
    settings (`model.sampling`) for `num_steps` steps on `schedule`, with `changes` to any
    of their fields. `StructureModel.structure_sampler` calls this."""
    settings = model.sampling.settings(num_steps, schedule=schedule)
    return StructureSampler(
        denoise=eqx.Partial(_denoise, model, context),
        atom_mask=context.atom_mask.astype(jnp.float32),
        settings=dataclasses.replace(settings, **changes),
    )
