"""Layers that build the trunk inputs: `s_inputs` (AF3 Algorithm 2), the relative position
features (Algorithm 3), and the initial single and pair (Algorithm 1, lines 1-5)."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from jfold.blocks.atoms import AtomAttentionEncoder
from jfold.nn import Linear, SinglesToPair
from jfold.slots import AbstractInputEmbedder, AbstractTrunkInit, Features, TrunkInputs


def probabilities(x: Array, num_classes: int) -> Float[Array, "... C"]:
    """Return class inputs as probability vectors, one-hot encoding integer indices and
    passing floats, such as a soft sequence, through unchanged."""
    if jnp.issubdtype(x.dtype, jnp.integer):
        return jax.nn.one_hot(x, num_classes)
    return x.astype(jnp.float32)


def _offset_one_hot(offset, condition, clip):
    """One-hot encode `offset` clipped to `[-clip, clip]`, with a last bin where
    `condition` is False."""
    index = jnp.where(condition, jnp.clip(offset + clip, 0, 2 * clip), 2 * clip + 1)
    return jax.nn.one_hot(index, 2 * clip + 2)


def relative_position_features(
    asym_id: Int[Array, " T"],
    residue_index: Int[Array, " T"],
    entity_id: Int[Array, " T"],
    token_index: Int[Array, " T"],
    sym_id: Int[Array, " T"],
    *,
    r_max: int = 32,
    s_max: int = 2,
    cyclic_period: Int[Array, " T"] | None = None,
    chain_offset: Bool[Array, "T T"] | None = None,
) -> Float[Array, "T T F"]:
    """Compute AF3 Algorithm 3's relative position features, `4 r_max + 2 s_max + 7`
    wide. Residue offsets wrap on chains with a positive `cyclic_period`, and
    `chain_offset` selects the pairs given a chain offset, by default those on the same
    entity."""
    same_chain = asym_id[:, None] == asym_id[None, :]
    same_residue = residue_index[:, None] == residue_index[None, :]
    same_entity = entity_id[:, None] == entity_id[None, :]
    d_residue = residue_index[:, None] - residue_index[None, :]
    if cyclic_period is not None:
        period = jnp.where(cyclic_period > 0, cyclic_period, 10000)[None, :]
        d_residue = (d_residue - period * jnp.round(d_residue / period)).astype(d_residue.dtype)
    d_token = token_index[:, None] - token_index[None, :]
    d_chain = sym_id[:, None] - sym_id[None, :]
    if chain_offset is None:
        chain_offset = same_entity
    return jnp.concatenate(
        [
            _offset_one_hot(d_residue, same_chain, r_max),
            _offset_one_hot(d_token, same_chain & same_residue, r_max),
            same_entity[..., None].astype(jnp.float32),
            _offset_one_hot(d_chain, chain_offset, s_max),
        ],
        axis=-1,
    )


class TrunkInit(AbstractTrunkInit[Features]):
    """Compute the initial single `linear_s(s_inputs)` and pair
    `linear_i(s_i) + linear_j(s_j)` plus relative positions, bonds and `pair_features`
    (AF3 Algorithm 1, lines 2-5)."""

    linear_s: Linear
    singles_to_pair: SinglesToPair
    relative_positions: Linear  # relative position features -> Cz
    bonds: Linear | None = None  # token bond features -> Cz
    # Protenix-lineage models build the pair from `s` after `linear_s`, not from s_inputs.
    pair_from_s_init: bool = eqx.field(static=True, default=False)

    def __call__(
        self, features: Features, s_inputs: Float[Array, "T Ci"], *, key: PRNGKeyArray | None = None
    ) -> TrunkInputs:
        s_init, z_init = self.initial(
            s_inputs, features.relative_positions, features.bonds, features.pair_features
        )
        return TrunkInputs(s_inputs=s_inputs, s_init=s_init, z_init=z_init)

    def initial(
        self,
        s_inputs: Float[Array, "T Ci"],
        relative_position_features: Float[Array, "T T F"],
        bond_features: Float[Array, "T T B"] | None = None,
        pair_features: Float[Array, "T T Cz"] | None = None,
    ) -> tuple[Float[Array, "T Cs"], Float[Array, "T T Cz"]]:
        """Return the initial single and pair, adding `pair_features` to the pair as is."""
        s = self.linear_s(s_inputs)
        z = self.singles_to_pair(s if self.pair_from_s_init else s_inputs)
        z = z + self.relative_positions(relative_position_features)
        if self.bonds is not None:
            z = z + self.bonds(bond_features)
        if pair_features is not None:
            z = z + pair_features
        return s, z


class InputEmbedder(AbstractInputEmbedder[Features]):
    """Compute `s_inputs` (AF3 Algorithm 2) from per-token atom aggregates, zeroed where
    `keep_aggregate` is False, and the token features, either concatenated or projected
    by `linear_tokens` and added."""

    encoder: AtomAttentionEncoder
    linear_tokens: Linear | None = None

    def __call__(self, features: Features) -> Float[Array, "T Ci"]:
        atoms, token_features = features.atoms, features.token_features
        q, c, p = self.encoder.condition(atoms)
        biases = self.encoder.transformer.pair_biases(p)
        a, _ = self.encoder(q, c, biases, atoms.atom_to_token, atoms.mask, token_features.shape[0])
        if features.keep_aggregate is not None:
            a = jnp.where(features.keep_aggregate[:, None], a, 0)
        if self.linear_tokens is None:
            return jnp.concatenate([a, token_features], axis=-1)
        return a + self.linear_tokens(token_features)
