"""Abstract types for the parts of `jfold.model.StructureModel` and the values passed
between them. The parts are generic in the features type `F` they read, usually
`Features`."""

import abc

import equinox as eqx
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from jfold.blocks.atoms import Atoms


class Features(eqx.Module):
    """The featurized inputs for one system, with masks False for padding."""

    atoms: Atoms
    token_features: Float[Array, "T F"]
    relative_positions: Float[Array, "T T R"]  # jfold.blocks.inputs.relative_position_features
    token_mask: Bool[Array, " T"]
    asym_id: Int[Array, " T"]
    rep_atom: Int[Array, " T"]  # each token's representative atom, used for confidence
    atom_position: Int[Array, " N"]  # each atom's index within its token
    bonds: Float[Array, "T T B"] | None = None
    # Extra pair inputs, such as Boltz-2's contacts, embedded by a model's own trunk init.
    pair_features: Float[Array, "T T P"] | None = None
    keep_aggregate: Bool[Array, " T"] | None = None  # see InputEmbedder
    msa_features: Float[Array, "S T Fm"] | None = None
    msa_mask: Bool[Array, "S T"] | None = None
    template_features: Float[Array, "Nt T T Ft"] | None = None
    template_mask: Bool[Array, " Nt"] | None = None
    # Model-specific inputs, such as ESMFold2's language-model input.
    extras: eqx.Module | None = None


class TrunkInputs(eqx.Module):
    """The values every trunk pass reads, computed once per forward pass."""

    s_inputs: Float[Array, "T Ci"]
    s_init: Float[Array, "T Cs"]
    z_init: Float[Array, "T T Cz"]
    extras: eqx.Module | None = None


class TrunkState(eqx.Module):
    """The trunk's single and pair representations."""

    s: Float[Array, "T Cs"]
    z: Float[Array, "T T Cz"]


class ConfidenceLogits(eqx.Module):
    pae: Float[Array, "T T Bpae"]
    pde: Float[Array, "T T Bpde"] | None
    plddt: Float[Array, "N_or_T Bplddt"]
    resolved: Float[Array, "N_or_T 2"] | None


class AbstractInputEmbedder[F](eqx.Module):
    """Compute `s_inputs` from the features."""

    @abc.abstractmethod
    def __call__(self, features: F) -> Float[Array, "T Ci"]: ...


class AbstractTrunkInit[F](eqx.Module):
    """Compute `TrunkInputs` from the features and `s_inputs`."""

    @abc.abstractmethod
    def __call__(
        self, features: F, s_inputs: Float[Array, "T Ci"], *, key: PRNGKeyArray | None = None
    ) -> TrunkInputs: ...


class AbstractTrunk[F](eqx.Module):
    """A trunk with a starting state, one pass per call, and a `finish` step run after
    the last pass."""

    @abc.abstractmethod
    def init_state(self, inputs: TrunkInputs, *, key: PRNGKeyArray | None = None) -> TrunkState:
        """Return the state the first pass starts from."""

    @abc.abstractmethod
    def __call__(
        self,
        features: F,
        inputs: TrunkInputs,
        state: TrunkState,
        *,
        key: PRNGKeyArray | None = None,
    ) -> TrunkState:
        """Run one trunk pass."""

    def finish(
        self, features: F, state: TrunkState, *, key: PRNGKeyArray | None = None
    ) -> TrunkState:
        """Return the state the heads read, by default the last pass's state."""
        return state


class AbstractDiffusionModule[F](eqx.Module):
    """A denoiser whose `prepare` computes a context once per trunk output and whose call
    is one denoising step."""

    @abc.abstractmethod
    def prepare(self, features: F, inputs: TrunkInputs, state: TrunkState) -> eqx.Module: ...

    @abc.abstractmethod
    def __call__(
        self, context: eqx.Module, x_noisy: Float[Array, "N 3"], t: Float[Array, ""]
    ) -> Float[Array, "N 3"]:
        """Return the denoised positions at noise level `t`."""


class AbstractConfidenceModule[F](eqx.Module):
    """Compute confidence logits for a predicted structure."""

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
