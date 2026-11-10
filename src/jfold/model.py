"""`StructureModel`, the composition every jfold model is (its parts are slots typed by
`jfold.slots`), and the AF3 recycling trunk (`Trunk`)."""

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
    """`init + linear(norm(previous))`."""

    norm: LayerNorm
    linear: Linear

    def __call__(self, init: Float[Array, "... C"], previous: Float[Array, "... C"]):
        return init + self.linear(self.norm(previous))


class Trunk(AbstractTrunk[Features]):
    """AF3 Algorithm 1's recycling loop, starting from zeros; one pass (lines 8-13)
    recycles the previous single and pair, adds templates, runs the MSA module, then the
    Pairformer. Boltz-2 and ProMERA add the MSA module's output to its input
    (`ResidualMSA`). `key` may be None without dropout and MSA subsampling."""

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
    """A structure model: each part is a slot, filled by a shared AF3 module (e.g. `Trunk`)
    or the model's own (e.g. ESMFold2's Parcae trunk), all reading the same features `F`:

        inputs, state = model.run_trunk(features, num_passes=4, key=k)
        coords = model.structure_sampler(features, inputs, state).sample(k)
        logits = model.confidence_logits(features, inputs, state, coords, key=k)

    To drive the trunk passes yourself (where gradients stop, a state carried between
    calls), call the trunk slot, as `run_trunk` does:

        inputs = model.embed(features, key=k)
        state = model.trunk.init_state(inputs, key=k)
        state = model.trunk(features, inputs, state, key=k)  # one pass; repeat at will
        state = model.trunk.finish(features, state, key=k)  # the state the heads read

    The other parts are called the same way: `model.diffusion.prepare(...)` and
    `model.diffusion(context, x_noisy, t)` (what a sampler runs), `model.distogram(state.z)`.

    Keys feed dropout (off by default) and the trunk's random draws (MSA subsampling,
    ESMFold2's starting state and MSA column mask); where only dropout reads them they may
    be None."""

    input_embedder: AbstractInputEmbedder[F]
    trunk_init: AbstractTrunkInit[F]
    trunk: AbstractTrunk[F]
    diffusion: AbstractDiffusionModule[F]
    distogram: PairHead
    confidence: AbstractConfidenceModule[F] | None = None
    # The upstream sampler's settings (`structure_sampler` uses them).
    sampling: UpstreamSampler | None = None

    def embed(self, features: F, *, key: PRNGKeyArray | None = None) -> TrunkInputs:
        """What every trunk pass reads, computed once."""
        return self.trunk_init(features, self.input_embedder(features), key=key)

    def run_trunk(self, features: F, *, num_passes: int, key: PRNGKeyArray):
        """The whole trunk: `embed`, the trunk's `init_state`, `num_passes` passes (3
        recycles is 4 passes) and `finish`. Returns the trunk inputs and the state the heads
        read. The passes run under `lax.scan`, so they compile once."""
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
        """A sampler for structures from this trunk output (`state` after `finish`): the
        denoiser with its context computed once, and upstream's settings (`sampling`) for
        `num_steps` steps on `schedule`, with `changes` to any of their fields (e.g.
        `update=exponential_update`)."""
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
        """Confidence logits for the structure `coords`."""
        if self.confidence is None:
            raise NotImplementedError("this model has no confidence head")
        return self.confidence(features, inputs, state, coords, key=key)
