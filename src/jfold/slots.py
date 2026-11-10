"""The slots of `jfold.model.StructureModel`, and the values that pass between them.

Each part of a model is a slot: an abstract type here, which the shared AF3 module of that
kind (`jfold.model.Trunk`, `jfold.blocks.denoiser.DiffusionModule`, ...) or a model's own
subclasses. Slots are generic in the features they read (`F`): most models read `Features`,
and a model's own inputs travel in `Features.extras`; a model whose inputs are of another
kind has its own type (AtlasFold's `AtlasFoldFeatures`).

Keys feed dropout and a model's random draws; a slot whose module draws nothing takes none."""

import abc

import equinox as eqx
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from jfold.blocks.atoms import Atoms


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
    # Extra pair inputs added to the initial pair; a model's own trunk init embeds them
    # first (Boltz-2's contact features).
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


class ConfidenceLogits(eqx.Module):
    pae: Float[Array, "T T Bpae"]
    pde: Float[Array, "T T Bpde"] | None
    plddt: Float[Array, "N_or_T Bplddt"]
    resolved: Float[Array, "N_or_T 2"] | None


class AbstractInputEmbedder[F](eqx.Module):
    """`s_inputs`, from the features."""

    @abc.abstractmethod
    def __call__(self, features: F) -> Float[Array, "T Ci"]: ...


class AbstractTrunkInit[F](eqx.Module):
    """`TrunkInputs`, from the features and `s_inputs`."""

    @abc.abstractmethod
    def __call__(
        self, features: F, s_inputs: Float[Array, "T Ci"], *, key: PRNGKeyArray | None = None
    ) -> TrunkInputs: ...


class AbstractTrunk[F](eqx.Module):
    """The trunk: its starting state, one pass (a call), and the state the heads read after
    the last pass (`finish`). `StructureModel.run_trunk` is the whole loop."""

    @abc.abstractmethod
    def init_state(self, inputs: TrunkInputs, *, key: PRNGKeyArray | None = None) -> TrunkState:
        """The state the first pass starts from."""

    @abc.abstractmethod
    def __call__(
        self,
        features: F,
        inputs: TrunkInputs,
        state: TrunkState,
        *,
        key: PRNGKeyArray | None = None,
    ) -> TrunkState:
        """One pass."""

    def finish(
        self, features: F, state: TrunkState, *, key: PRNGKeyArray | None = None
    ) -> TrunkState:
        """The state the heads read: the last pass's, as it is."""
        return state


class AbstractDiffusionModule[F](eqx.Module):
    """The denoiser: `prepare` computes what depends only on the trunk (a context with an
    `atom_mask`, the atoms it moves), once; a call is one denoising step."""

    @abc.abstractmethod
    def prepare(self, features: F, inputs: TrunkInputs, state: TrunkState) -> eqx.Module: ...

    @abc.abstractmethod
    def __call__(
        self, context: eqx.Module, x_noisy: Float[Array, "N 3"], t: Float[Array, ""]
    ) -> Float[Array, "N 3"]:
        """The denoised positions at noise level `t`."""


class AbstractConfidenceModule[F](eqx.Module):
    """Confidence logits for a predicted structure."""

    @abc.abstractmethod
    def __call__(
        self,
        features: F,
        inputs: TrunkInputs,
        state: TrunkState,
        coords: Float[Array, "N 3"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> ConfidenceLogits: ...
