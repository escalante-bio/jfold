"""Whole models: `StructureModel`, the AF3-family composition every jfold model is (the
input embedding, the trunk, the diffusion module and the heads, each a slot typed by
`jfold.slots`), with a system's inputs in one `Features` bundle; and the AF3 recycling
trunk (`Trunk`, AF3 Algorithm 1)."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from jfold.blocks.atoms import Atoms
from jfold.blocks.confidence import ConfidenceLogits, PairHead
from jfold.blocks.msa import AbstractMSAModule
from jfold.blocks.pairformer import AbstractPairformer
from jfold.blocks.templates import TemplateEmbedder
from jfold.nn import LayerNorm, Linear, split_key
from jfold.sampling import StructureSampler, UpstreamSampler, structure_sampler
from jfold.sampling.schedules import Schedule
from jfold.slots import (
    AbstractConfidenceModule,
    AbstractDiffusionModule,
    AbstractInputEmbedder,
    AbstractTrunk,
    AbstractTrunkInit,
)


class Features(eqx.Module):
    """A system's inputs, in the model's vocabulary (`jfold.models.*.features` build them).
    Masks are False for padding."""

    atoms: Atoms
    token_features: Float[Array, "T F"]
    relative_positions: Float[Array, "T T R"]  # jfold.blocks.inputs.relative_position_features
    token_mask: Bool[Array, " T"]
    asym_id: Int[Array, " T"]
    rep_atom: Int[Array, " T"]  # each token's representative atom (confidence)
    atom_position: Int[Array, " N"]  # each atom's index within its token
    bonds: Float[Array, "T T B"] | None = None
    # Model-specific extra pair inputs, which the model's trunk init embeds (Boltz-2's
    # contact features).
    pair_features: Float[Array, "T T P"] | None = None
    keep_aggregate: Bool[Array, " T"] | None = None  # see InputEmbedder
    msa_features: Float[Array, "S T Fm"] | None = None
    msa_mask: Bool[Array, "S T"] | None = None
    template_features: Float[Array, "Nt T T Ft"] | None = None
    template_mask: Bool[Array, " Nt"] | None = None
    # A model's own inputs, read by its own modules: ESMFold2's language-model input,
    # OpenDDE's structural tokens, BioEmu2's conditioning.
    extras: eqx.Module | None = None


class TrunkInputs(eqx.Module):
    """What every trunk pass reads: `s_inputs`, the initial single and pair, and anything
    else a model computes once per forward pass (`extras`: ESMFold2's language-model pair,
    embedded relative positions and bonds, and column-masked MSA)."""

    s_inputs: Float[Array, "T Ci"]
    s_init: Float[Array, "T Cs"]
    z_init: Float[Array, "T T Cz"]
    extras: eqx.Module | None = None


class TrunkState(eqx.Module):
    """The trunk's single and pair. A pair-only trunk (ESMFold2's) carries the single it
    starts from, unchanged."""

    s: Float[Array, "T Cs"]
    z: Float[Array, "T T Cz"]


class Recycle(eqx.Module):
    """`init + linear(norm(previous))`."""

    norm: LayerNorm
    linear: Linear

    def __call__(self, init: Float[Array, "... C"], previous: Float[Array, "... C"]):
        return init + self.linear(self.norm(previous))


class Trunk(AbstractTrunk):
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
        del key
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


class StructureModel(eqx.Module):
    """A structure model: each part a slot, filled by the module of that type a model uses
    (the shared AF3 ones, e.g. `Trunk`, or its own, e.g. ESMFold2's Parcae trunk):

        inputs, state = model.run_trunk(features, num_passes=4, key=k)
        coords = model.structure_sampler(features, inputs, state).sample(k)
        logits = model.confidence_logits(features, inputs, state, coords, key=k)

    `run_trunk` is `embed`, `init_state`, one `step` per pass and `finish`. They are
    separate methods for callers that drive the passes themselves (where gradients stop,
    inputs reused across evaluations). `denoiser_context` and `denoise` split the diffusion
    module the same way (the context once per trunk, a step per noise level).

    Keys feed dropout (off by default) and the trunk's random draws (MSA subsampling,
    ESMFold2's starting state and MSA column mask); where only dropout reads them they may
    be None."""

    input_embedder: AbstractInputEmbedder
    trunk_init: AbstractTrunkInit
    trunk: AbstractTrunk
    diffusion: AbstractDiffusionModule
    distogram: PairHead
    confidence: AbstractConfidenceModule | None = None
    # The upstream sampler's settings (`structure_sampler` uses them).
    sampling: UpstreamSampler | None = None

    def embed(self, features: Features, *, key: PRNGKeyArray | None = None) -> TrunkInputs:
        """What every trunk pass reads, computed once."""
        k_inputs, k_init = split_key(key)
        s_inputs = self.input_embedder(features, key=k_inputs)
        return self.trunk_init(features, s_inputs, key=k_init)

    def init_state(self, inputs: TrunkInputs, *, key: PRNGKeyArray | None = None) -> TrunkState:
        """The state the first `step` starts from."""
        return self.trunk.init_state(inputs, key=key)

    def step(
        self,
        features: Features,
        inputs: TrunkInputs,
        state: TrunkState,
        *,
        key: PRNGKeyArray | None = None,
    ) -> TrunkState:
        """One trunk pass."""
        return self.trunk(features, inputs, state, key=key)

    def finish(
        self, features: Features, state: TrunkState, *, key: PRNGKeyArray | None = None
    ) -> TrunkState:
        """The state the heads read, from the last pass's."""
        return self.trunk.finish(features, state, key=key)

    def run_trunk(self, features: Features, *, num_passes: int, key: PRNGKeyArray):
        """The whole trunk: `embed`, `init_state`, `num_passes` passes (the first, then
        recycles: 3 recycles is 4 passes) and `finish`. Returns the trunk inputs and the
        state the heads read. The passes run under `lax.scan`, so they compile once whatever
        their number."""
        if num_passes < 1:
            raise ValueError(f"num_passes must be at least 1, got {num_passes}")
        k_embed, k_init, k_passes, k_finish = jax.random.split(key, 4)
        inputs = self.embed(features, key=k_embed)
        state = self.init_state(inputs, key=k_init)

        def one_pass(state, k):
            return self.step(features, inputs, state, key=k), None

        state, _ = jax.lax.scan(one_pass, state, jax.random.split(k_passes, num_passes))
        return inputs, self.finish(features, state, key=k_finish)

    def denoiser_context(
        self, features: Features, inputs: TrunkInputs, state: TrunkState
    ) -> eqx.Module:
        """Everything a denoising step needs that depends only on the trunk (`state` as the
        heads read it, after `finish`)."""
        return self.diffusion.prepare(features, inputs, state)

    def denoise(
        self,
        context: eqx.Module,
        x_noisy: Float[Array, "N 3"],
        t: Float[Array, ""],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "N 3"]:
        """The denoised positions at noise level `t`."""
        return self.diffusion(context, x_noisy, t, key=key)

    def structure_sampler(
        self,
        features: Features,
        inputs: TrunkInputs,
        state: TrunkState,
        *,
        num_steps: int | None = None,
        schedule: Schedule | None = None,
        **changes,
    ) -> StructureSampler:
        """A sampler for structures from this trunk output (`state` as the heads read it,
        after `finish`): the denoiser's context computed once, upstream's settings
        (`sampling`) for `num_steps` steps on `schedule`, `changes` to any of their fields
        (e.g. `update=exponential_update`)."""
        context = self.denoiser_context(features, inputs, state)
        return structure_sampler(self, context, num_steps=num_steps, schedule=schedule, **changes)

    def distogram_logits(self, state: TrunkState) -> Float[Array, "T T B"]:
        """Distogram logits from the state the heads read."""
        return self.distogram(state.z)

    @property
    def has_confidence(self) -> bool:
        """Whether `confidence_logits` is available."""
        return self.confidence is not None

    def confidence_logits(
        self,
        features: Features,
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
