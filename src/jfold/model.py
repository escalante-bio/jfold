"""The `StructureModel` that every jfold model is built as, and the AF3 recycling trunk."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from jfold.blocks.confidence import PairHead
from jfold.blocks.msa import AbstractMSAModule
from jfold.blocks.pairformer import AbstractPairformer
from jfold.blocks.templates import TemplateEmbedder
from jfold.nn import LayerNorm, Linear, split_key
from jfold.sampling import StructureSampler, UpstreamSampler
from jfold.sampling.schedules import Schedule
from jfold.slots import (
    AbstractConfidenceModule,
    AbstractDiffusionModule,
    AbstractInputEmbedder,
    AbstractTrunk,
    AbstractTrunkInit,
    ConfidenceLogits,
    Features,
    TrunkInputs,
    TrunkState,
)


class Recycle(eqx.Module):
    """Return `init + linear(norm(previous))`."""

    norm: LayerNorm
    linear: Linear

    def __call__(self, init: Float[Array, "... C"], previous: Float[Array, "... C"]):
        return init + self.linear(self.norm(previous))


class Trunk(AbstractTrunk[Features]):
    """Run one pass of AF3's recycling trunk (Algorithm 1, lines 8-13)."""

    recycle_s: Recycle
    recycle_z: Recycle
    pairformer: AbstractPairformer
    templates: TemplateEmbedder | None = None
    msa: AbstractMSAModule | None = None

    def init_state(self, inputs: TrunkInputs, *, key: PRNGKeyArray | None = None) -> TrunkState:
        return TrunkState(s=jnp.zeros_like(inputs.s_init), z=jnp.zeros_like(inputs.z_init))

    def __call__(
        self,
        features: Features,
        inputs: TrunkInputs,
        state: TrunkState,
        *,
        key: PRNGKeyArray | None = None,
    ) -> TrunkState:
        k_templates, k_msa, k_pairformer = split_key(key, 3)
        mask = features.token_mask
        z = self.recycle_z(inputs.z_init, state.z)
        if self.templates is not None and features.template_features is not None:
            z = z + self.templates(
                z, features.template_features, features.template_mask, mask, key=k_templates
            )
        if self.msa is not None and features.msa_features is not None:
            _, z = self.msa(
                z, features.msa_features, inputs.s_inputs, features.msa_mask, mask, key=k_msa
            )
        s = self.recycle_s(inputs.s_init, state.s)
        s, z = self.pairformer(s, z, mask, key=k_pairformer)
        return TrunkState(s=s, z=z)


class StructureModel[F](eqx.Module):
    """A structure model whose parts are slots that all read the same features `F`:

        inputs, state = model.run_trunk(features, num_passes=4, key=k)
        coords = model.structure_sampler(features, inputs, state).sample(k)
        logits = model.confidence_logits(features, inputs, state, coords, key=k)

    To run the trunk passes yourself, call the trunk slot the way `run_trunk` does:

        inputs = model.embed(features, key=k)
        state = model.trunk.init_state(inputs, key=k)
        state = model.trunk(features, inputs, state, key=k)  # one pass, repeat as needed
        state = model.trunk.finish(features, state, key=k)
    """

    input_embedder: AbstractInputEmbedder[F]
    trunk_init: AbstractTrunkInit[F]
    trunk: AbstractTrunk[F]
    diffusion: AbstractDiffusionModule[F]
    distogram: PairHead
    confidence: AbstractConfidenceModule[F] | None = None
    sampling: UpstreamSampler | None = None

    def embed(self, features: F, *, key: PRNGKeyArray | None = None) -> TrunkInputs:
        """Compute the trunk inputs that every pass reads."""
        return self.trunk_init(features, self.input_embedder(features), key=key)

    def run_trunk(self, features: F, *, num_passes: int, key: PRNGKeyArray):
        """Run `embed`, `init_state`, `num_passes` trunk passes under `lax.scan`, and
        `finish`, returning the trunk inputs and final state."""
        if num_passes < 1:
            raise ValueError(f"num_passes must be at least 1, got {num_passes}")
        k_embed, k_init, k_passes, k_finish = jax.random.split(key, 4)
        inputs = self.embed(features, key=k_embed)
        state = self.trunk.init_state(inputs, key=k_init)

        def one_pass(state, k):
            return self.trunk(features, inputs, state, key=k), None

        state, _ = jax.lax.scan(one_pass, state, jax.random.split(k_passes, num_passes))
        return inputs, self.trunk.finish(features, state, key=k_finish)

    def structure_sampler(
        self,
        features: F,
        inputs: TrunkInputs,
        state: TrunkState,
        *,
        num_steps: int | None = None,
        schedule: Schedule | None = None,
        **changes,
    ) -> StructureSampler:
        """Build a structure sampler from the finished trunk `state`, with keyword `changes`
        overriding any upstream sampler setting."""
        context = self.diffusion.prepare(features, inputs, state)
        settings = self.sampling.settings(num_steps, schedule=schedule)
        return StructureSampler(
            denoise=eqx.Partial(self.diffusion, context),
            atom_mask=context.atom_mask.astype(jnp.float32),
            settings=dataclasses.replace(settings, **changes),
        )

    def confidence_logits(
        self,
        features: F,
        inputs: TrunkInputs,
        state: TrunkState,
        coords: Float[Array, "N 3"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> ConfidenceLogits:
        """Return the confidence logits for the structure `coords`."""
        if self.confidence is None:
            raise NotImplementedError("this model has no confidence head")
        return self.confidence(features, inputs, state, coords, key=key)
