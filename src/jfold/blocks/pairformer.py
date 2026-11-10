"""The Pairformer (AF3 Algorithm 17) and its blocks.

Inputs are unbatched: single `[T, Cs]`, pair `[T, T, Cz]`, and a token mask `[T]` that is
False for padding. Padded tokens never affect real ones; their own outputs are unspecified
but finite."""

import abc
import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.nn import Identity
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from jfold.nn import (
    Attention,
    Dropout,
    LayerNorm,
    Linear,
    PairBias,
    Transition,
    scan_layers,
    split_key,
)


def _pair_mask(mask: Bool[Array, " T"]) -> Bool[Array, "T T"]:
    return mask[:, None] & mask[None, :]


class TriangleMultiplication(eqx.Module):
    """Triangle update using outgoing (AF3 Algorithm 12) or incoming (Algorithm 13) edges."""

    norm_in: LayerNorm
    linear_p: Linear  # Cz -> 2C: the a and b projections, concatenated
    linear_g: Linear  # Cz -> 2C: their gates
    norm_out: LayerNorm
    linear_out: Linear
    linear_out_g: Linear
    outgoing: bool = eqx.field(static=True)

    def __call__(
        self, z: Float[Array, "T T Cz"], pair_mask: Bool[Array, "T T"]
    ) -> Float[Array, "T T Cz"]:
        z = self.norm_in(z)
        ab = self.linear_p(z) * jax.nn.sigmoid(self.linear_g(z)) * pair_mask[..., None]
        a, b = jnp.split(ab, 2, axis=-1)
        x = jnp.einsum("ikc,jkc->ijc" if self.outgoing else "kic,kjc->ijc", a, b)
        return self.linear_out(self.norm_out(x)) * jax.nn.sigmoid(self.linear_out_g(z))


class TriangleAttention(eqx.Module):
    """Triangle attention around the starting node (AF3 Algorithm 14): each row attends
    along itself, biased by the pair. Applied to the transposed pair it is the
    ending-node version (Algorithm 15).

    With `chunk_size` the rows run in blocks of that many, one block at a time, so the
    attention maps are `[chunk_size, heads, T, T]` instead of `[T, heads, T, T]`. Each
    block is rematerialized on the backward pass."""

    norm: LayerNorm
    linear_bias: Linear  # Cz -> heads
    attention: Attention
    chunk_size: int | None = eqx.field(default=None, static=True)

    def __call__(
        self, z: Float[Array, "T T Cz"], pair_mask: Bool[Array, "T T"]
    ) -> Float[Array, "T T Cz"]:
        z = self.norm(z)
        bias = jnp.moveaxis(self.linear_bias(z), -1, 0)  # [heads, T, T], shared by rows
        if self.chunk_size is None:
            return jax.vmap(lambda row, m: self.attention(row, row, bias, m))(z, pair_mask)
        num_rows = z.shape[0]
        size = min(self.chunk_size, num_rows)
        padding = -num_rows % size  # rows of zeros that attend to nothing, then dropped
        blocks = lambda x: jnp.pad(x, [(0, padding)] + [(0, 0)] * (x.ndim - 1)).reshape(
            -1, size, *x.shape[1:]
        )
        out = jax.lax.map(
            lambda block: _attend_rows(self.attention, bias, *block),
            (blocks(z), blocks(pair_mask)),
        )
        return out.reshape(-1, *out.shape[2:])[:num_rows]


@jax.checkpoint
def _attend_rows(
    attention: Attention,
    bias: Float[Array, "heads T T"],
    rows: Float[Array, "n T Cz"],
    mask: Bool[Array, "n T"],
) -> Float[Array, "n T Cz"]:
    return jax.vmap(lambda row, m: attention(row, row, bias, m))(rows, mask)


def with_triangle_chunks[M](model: M, chunk_size: int | None) -> M:
    """`model` with every `TriangleAttention` in it running `chunk_size` rows at a time
    (None: all rows at once). Slower, with less memory, also under `grad`."""
    is_attention = lambda x: isinstance(x, TriangleAttention)
    return jax.tree.map(
        lambda x: dataclasses.replace(x, chunk_size=chunk_size) if is_attention(x) else x,
        model,
        is_leaf=is_attention,
    )


class AttentionPairBias(eqx.Module):
    """Single-track self-attention biased by the pair (AF3 Algorithm 24, unconditioned)."""

    norm_s: LayerNorm
    pair_bias: PairBias
    attention: Attention

    def __call__(
        self,
        s: Float[Array, "T Cs"],
        z: Float[Array, "T T Cz"],
        mask: Bool[Array, " T"],
        bias: Float[Array, "heads T T"] | None = None,
    ) -> Float[Array, "T Cs"]:
        a = self.norm_s(s)
        pair_bias = self.pair_bias(z)
        if bias is not None:
            pair_bias = pair_bias + bias
        return self.attention(a, a, pair_bias, mask)


class SingleUpdate(eqx.Module):
    """The single-track half of a block: attention biased by the pair, then a transition."""

    attention: AttentionPairBias
    transition: Transition
    post_norm: LayerNorm | Identity = eqx.field(default_factory=Identity)  # Boltz-2's

    def __call__(
        self,
        s: Float[Array, "T Cs"],
        z: Float[Array, "T T Cz"],
        mask: Bool[Array, " T"],
        bias: Float[Array, "heads T T"] | None = None,
    ) -> Float[Array, "T Cs"]:
        s = s + self.attention(s, z, mask, bias)
        s = s + self.transition(s)
        return self.post_norm(s)


class PairformerLayer(eqx.Module):
    """One Pairformer block (AF3 Algorithm 17). Without `single` (MSA and template
    stacks) the single track passes through unchanged. Dropout acts on the triangle
    updates: `dropout_row` on the multiplications and the starting-node attention,
    `dropout_column` on the ending-node attention."""

    tri_mul_out: TriangleMultiplication
    tri_mul_in: TriangleMultiplication
    tri_att_start: TriangleAttention
    tri_att_end: TriangleAttention
    transition_z: Transition
    single: SingleUpdate | None
    dropout_row: Dropout | Identity = eqx.field(default_factory=Identity)
    dropout_column: Dropout | Identity = eqx.field(default_factory=Identity)

    def __call__(
        self,
        s: Float[Array, "T Cs"] | None,
        z: Float[Array, "T T Cz"],
        mask: Bool[Array, " T"],
        bias: Float[Array, "heads T T"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "T Cs"] | None, Float[Array, "T T Cz"]]:
        k1, k2, k3, k4 = split_key(key, 4)
        pair_mask = _pair_mask(mask)
        z = z + self.dropout_row(self.tri_mul_out(z, pair_mask), key=k1)
        z = z + self.dropout_row(self.tri_mul_in(z, pair_mask), key=k2)
        z = z + self.dropout_row(self.tri_att_start(z, pair_mask), key=k3)
        end = jnp.swapaxes(self.tri_att_end(jnp.swapaxes(z, 0, 1), pair_mask.T), 0, 1)
        z = z + self.dropout_column(end, key=k4)
        z = z + self.transition_z(z)
        return (s if self.single is None else self.single(s, z, mask, bias)), z


class PairMixerLayer(eqx.Module):
    """A PairMixer block: a Pairformer block without triangle attention."""

    tri_mul_out: TriangleMultiplication
    tri_mul_in: TriangleMultiplication
    transition_z: Transition
    single: SingleUpdate | None
    dropout_row: Dropout | Identity = eqx.field(default_factory=Identity)

    def __call__(
        self,
        s: Float[Array, "T Cs"] | None,
        z: Float[Array, "T T Cz"],
        mask: Bool[Array, " T"],
        bias: Float[Array, "heads T T"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "T Cs"] | None, Float[Array, "T T Cz"]]:
        k1, k2 = split_key(key)
        pair_mask = _pair_mask(mask)
        z = z + self.dropout_row(self.tri_mul_out(z, pair_mask), key=k1)
        z = z + self.dropout_row(self.tri_mul_in(z, pair_mask), key=k2)
        z = z + self.transition_z(z)
        return (s if self.single is None else self.single(s, z, mask, bias)), z


class AbstractPairformer(eqx.Module):
    """A trunk's pair stack: single and pair in, single and pair out (`PairformerStack`,
    or a model's own, e.g. Mol-1's)."""

    @abc.abstractmethod
    def __call__(
        self,
        s: Float[Array, "T Cs"] | None,
        z: Float[Array, "T T Cz"],
        mask: Bool[Array, " T"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "T Cs"] | None, Float[Array, "T T Cz"]]: ...


class PairformerStack(AbstractPairformer):
    """Identical layers (`PairformerLayer`s or `PairMixerLayer`s), stacked along a leading
    axis and scanned."""

    layers: PairformerLayer | PairMixerLayer

    def __call__(
        self,
        s: Float[Array, "T Cs"] | None,
        z: Float[Array, "T T Cz"],
        mask: Bool[Array, " T"],
        bias: Float[Array, "heads T T"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "T Cs"] | None, Float[Array, "T T Cz"]]:
        def step(layer, carry, k):
            return layer(*carry, mask, bias, key=k)

        return scan_layers(step, (s, z), self.layers, key=key)
