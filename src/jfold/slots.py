"""The slots of `jfold.model.StructureModel`: the abstract type of each part, which a
model-specific module subclasses. Every slot reads the model's `Features` (and what the
earlier parts computed), so a model's own inputs reach its own modules through
`Features.extras` and `TrunkInputs.extras`."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

if TYPE_CHECKING:
    from jfold.blocks.confidence import ConfidenceLogits
    from jfold.model import Features, TrunkInputs, TrunkState


class AbstractInputEmbedder(eqx.Module):
    """`s_inputs`, from the features."""

    @abc.abstractmethod
    def __call__(
        self, features: Features, *, key: PRNGKeyArray | None = None
    ) -> Float[Array, "T Ci"]: ...


class AbstractTrunkInit(eqx.Module):
    """What every trunk pass reads (`TrunkInputs`): `s_inputs`, the initial single and
    pair, and anything else the model computes once per forward pass."""

    @abc.abstractmethod
    def __call__(
        self,
        features: Features,
        s_inputs: Float[Array, "T Ci"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> TrunkInputs: ...


class AbstractTrunk(eqx.Module):
    """The trunk: its starting state, one pass, and the state the heads read after the
    last pass (`finish`)."""

    @abc.abstractmethod
    def init_state(self, inputs: TrunkInputs, *, key: PRNGKeyArray | None = None) -> TrunkState:
        """The state the first pass starts from."""

    @abc.abstractmethod
    def __call__(
        self,
        features: Features,
        inputs: TrunkInputs,
        state: TrunkState,
        *,
        key: PRNGKeyArray | None = None,
    ) -> TrunkState:
        """One pass."""

    def finish(
        self, features: Features, state: TrunkState, *, key: PRNGKeyArray | None = None
    ) -> TrunkState:
        """The state the heads read: the last pass's, as it is."""
        del features, key
        return state


class AbstractDiffusionModule(eqx.Module):
    """The denoiser: `prepare` computes what depends only on the trunk (a context with an
    `atom_mask`, the atoms it moves), once; a call is one denoising step."""

    @abc.abstractmethod
    def prepare(self, features: Features, inputs: TrunkInputs, state: TrunkState) -> eqx.Module:
        ...

    @abc.abstractmethod
    def __call__(
        self,
        context: eqx.Module,
        x_noisy: Float[Array, "N 3"],
        t: Float[Array, ""],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "N 3"]:
        """The denoised positions at noise level `t`."""


class AbstractConfidenceModule(eqx.Module):
    """Confidence logits for a predicted structure."""

    @abc.abstractmethod
    def __call__(
        self,
        features: Features,
        inputs: TrunkInputs,
        state: TrunkState,
        coords: Float[Array, "N 3"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> ConfidenceLogits: ...
