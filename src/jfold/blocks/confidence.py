"""The confidence module of AF3 Algorithm 31 and the distogram head, on unbatched inputs
with masks False for padding."""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.nn import Identity
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from jfold.blocks.pairformer import PairformerStack
from jfold.nn import LayerNorm, Linear, SinglesToPair
from jfold.slots import (
    AbstractConfidenceModule,
    ConfidenceLogits,
    Features,
    TrunkInputs,
    TrunkState,
)


class DistanceEmbedding(eqx.Module):
    """Embed pair distances as one-hot bins, where bin `k` holds `(edges[k-1], edges[k]]`
    and bin 0 everything up to `edges[0]`."""

    edges: Float[Array, " E"]
    linear: Linear  # E + 1 bins -> Cz
    linear_raw: Linear | None = None  # Protenix also embeds the distance itself

    def __call__(self, d: Float[Array, "T T"]) -> Float[Array, "T T Cz"]:
        bins = jax.nn.one_hot((d[..., None] > self.edges).sum(-1), self.edges.shape[0] + 1)
        out = self.linear(bins)
        return out if self.linear_raw is None else out + self.linear_raw(d[..., None])


class PairHead(eqx.Module):
    """Return `linear(norm(z))`, or `linear(norm(z + zᵀ))` if `symmetric`, using
    `linear_inter` for pairs on different chains if set."""

    linear: Linear
    linear_inter: Linear | None = None
    norm: LayerNorm | Identity = eqx.field(default_factory=Identity)
    norm_terms: LayerNorm | Identity = eqx.field(default_factory=Identity)
    symmetric: bool = eqx.field(static=True, default=False)

    def __call__(
        self, z: Float[Array, "T T C"], same_chain: Bool[Array, "T T"] | None = None
    ) -> Float[Array, "T T B"]:
        z = self.norm_terms(z)
        if self.symmetric:
            z = z + jnp.swapaxes(z, 0, 1)
        z = self.norm(z)
        if self.linear_inter is None:
            return self.linear(z)
        return jnp.where(same_chain[..., None], self.linear(z), self.linear_inter(z))


def pairwise_distances(x: Float[Array, "T 3"]) -> Float[Array, "T T"]:
    """Return pairwise distances, with a finite gradient where a distance is zero."""
    sq = jnp.sum((x[:, None] - x[None, :]) ** 2, axis=-1)
    return jnp.where(sq > 0, jnp.sqrt(jnp.where(sq > 0, sq, 1.0)), 0.0)


class TokenHead(eqx.Module):
    """Compute per-token logits from the single."""

    linear: Linear
    norm: LayerNorm | Identity = eqx.field(default_factory=Identity)

    def __call__(self, s, atom_to_token, atom_position):
        return self.linear(self.norm(s))


class AtomHead(eqx.Module):
    """Compute per-atom logits from each atom's token single, with weights per atom
    position within the token."""

    weight: Float[Array, "P C B"]
    bias: Float[Array, "P B"] | None = None
    norm: LayerNorm | Identity = eqx.field(default_factory=Identity)

    def __call__(
        self,
        s: Float[Array, "T C"],
        atom_to_token: Int[Array, " N"],
        atom_position: Int[Array, " N"],
    ) -> Float[Array, "N B"]:
        # Gathering the weights per atom instead would materialize an [N, C, B] array.
        per_position = jnp.einsum("tc,pcb->tpb", self.norm(s), self.weight)
        logits = per_position[atom_to_token, atom_position]
        return logits if self.bias is None else logits + self.bias[atom_position]


class AbstractPairInputs(eqx.Module):
    """Embed extra pair inputs for the confidence module, such as Boltz-2's."""

    @abc.abstractmethod
    def __call__(self, features: Features) -> Float[Array, "T T Cz"]: ...


class ConfidenceModule(AbstractConfidenceModule[Features]):
    """The confidence module of AF3 Algorithm 31, with its heads."""

    singles_to_pair: SinglesToPair  # s_inputs -> Cz
    distances: DistanceEmbedding
    stack: PairformerStack
    pae: PairHead
    pde: PairHead
    plddt: AtomHead | TokenHead
    resolved: AtomHead | TokenHead | None  # ProMERA has none
    norm_s: LayerNorm | Identity = eqx.field(default_factory=Identity)
    norm_z: LayerNorm | Identity = eqx.field(default_factory=Identity)
    norm_s_inputs: LayerNorm | Identity = eqx.field(default_factory=Identity)
    linear_s_inputs_to_s: Linear | None = None
    clamp_s: float | None = eqx.field(static=True, default=None)  # applied before norm_s
    pair_inputs: AbstractPairInputs | None = None

    def __call__(
        self,
        features: Features,
        inputs: TrunkInputs,
        state: TrunkState,
        coords: Float[Array, "N 3"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> ConfidenceLogits:
        return self.logits(
            inputs.s_inputs,
            state.s,
            state.z,
            coords[features.rep_atom],
            features.asym_id,
            features.atoms.atom_to_token,
            features.atom_position,
            features.token_mask,
            None if self.pair_inputs is None else self.pair_inputs(features),
            key=key,
        )

    def logits(
        self,
        s_inputs: Float[Array, "T Ci"],
        s_trunk: Float[Array, "T Cs"],
        z_trunk: Float[Array, "T T Cz"],
        rep_coords: Float[Array, "T 3"],
        asym_id: Int[Array, " T"],
        atom_to_token: Int[Array, " N"],
        atom_position: Int[Array, " N"],
        mask: Bool[Array, " T"],
        pair_features: Float[Array, "T T Cz"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> ConfidenceLogits:
        """Compute the logits from arrays, with `rep_coords` the predicted positions of
        each token's representative atom."""
        s, z = s_trunk, z_trunk
        s_inputs = self.norm_s_inputs(s_inputs)
        if self.clamp_s is not None:
            s = jnp.clip(s, -self.clamp_s, self.clamp_s)
        s = self.norm_s(s)
        if self.linear_s_inputs_to_s is not None:
            s = s + self.linear_s_inputs_to_s(s_inputs)
        z = self.norm_z(z)
        if pair_features is not None:
            z = z + pair_features
        z = z + self.singles_to_pair(s_inputs)
        z = z + self.distances(pairwise_distances(rep_coords))
        s, z = self.stack(s, z, mask, key=key)
        same_chain = asym_id[:, None] == asym_id[None, :]
        return ConfidenceLogits(
            pae=self.pae(z, same_chain),
            pde=self.pde(z, same_chain),
            plddt=self.plddt(s, atom_to_token, atom_position),
            resolved=None
            if self.resolved is None
            else self.resolved(s, atom_to_token, atom_position),
        )
