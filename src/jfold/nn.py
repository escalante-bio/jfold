"""Primitives shared by every block: linear and norm layers, gated attention, pair bias,
the SwiGLU transition, the pair from a single representation, dropout, and scanning over a
stack of identical layers.

Keys: a module whose only randomness is dropout takes `key=None` (dropout off, as it is by
default); `Dropout` raises if it is on and has no key."""

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray


class Linear(eqx.Module):
    """`x @ weight.T + bias`; the weight keeps torch's `[out, in]` layout."""

    weight: Float[Array, "out in"]
    bias: Float[Array, " out"] | None = None

    def __call__(self, x: Float[Array, "... in"]) -> Float[Array, "... out"]:
        y = jnp.einsum("...i,oi->...o", x, self.weight)
        return y if self.bias is None else y + self.bias


class LayerNorm(eqx.Module):
    """Last-axis layer norm; the scale and offset are optional."""

    weight: Float[Array, " d"] | None
    bias: Float[Array, " d"] | None
    eps: float = eqx.field(static=True, default=1e-5)

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "... d"]:
        x = jax.nn.standardize(x, epsilon=self.eps, algorithm="stable")
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class Attention(eqx.Module):
    """Gated multi-head attention: `softmax(q kᵀ / √c + bias) v` over the keys in
    `key_mask`, gated per channel by `sigmoid(linear_g(q_x))`, then projected by
    `linear_o`."""

    linear_q: Linear
    linear_k: Linear
    linear_v: Linear
    linear_g: Linear
    linear_o: Linear
    num_heads: int = eqx.field(static=True)

    def __call__(
        self,
        q_x: Float[Array, "q d"],
        kv_x: Float[Array, "k d"],
        bias: Float[Array, "heads q k"] | None,
        key_mask: Bool[Array, " k"],
    ) -> Float[Array, "q d"]:
        q = self._heads(self.linear_q(q_x))
        k = self._heads(self.linear_k(kv_x))
        v = self._heads(self.linear_v(kv_x))
        o = jax.nn.dot_product_attention(q, k, v, bias=bias, mask=key_mask[None, None, :])
        o = o * self._heads(jax.nn.sigmoid(self.linear_g(q_x)))
        return self.linear_o(o.reshape(*o.shape[:-2], -1))

    def _heads(self, x: Float[Array, "n d"]) -> Float[Array, "n heads c"]:
        return x.reshape(*x.shape[:-1], self.num_heads, -1)


class PairBias(eqx.Module):
    """Per-head attention bias from the pair: `linear(norm(z))`, heads first."""

    norm: LayerNorm
    linear: Linear  # Cz -> heads

    def __call__(self, z: Float[Array, "q k Cz"]) -> Float[Array, "heads q k"]:
        return jnp.moveaxis(self.linear(self.norm(z)), -1, 0)


class Transition(eqx.Module):
    """SwiGLU MLP behind a layer norm (AF3 Algorithm 11)."""

    norm: LayerNorm
    linear_a: Linear
    linear_b: Linear
    linear_out: Linear

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "... d"]:
        x = self.norm(x)
        return self.linear_out(jax.nn.silu(self.linear_a(x)) * self.linear_b(x))


class SinglesToPair(eqx.Module):
    """`z_ij` from a single representation: `linear_i(s_i) + linear_j(s_j)`, plus
    Boltz-2's `linear_out(in_i(s_i) * in_j(s_j))`."""

    linear_i: Linear
    linear_j: Linear
    product_i: Linear | None = None
    product_j: Linear | None = None
    product_out: Linear | None = None

    def __call__(self, s: Float[Array, "T C"]) -> Float[Array, "T T Cz"]:
        z = self.linear_i(s)[:, None] + self.linear_j(s)[None, :]
        if self.product_out is not None:
            z = z + self.product_out(self.product_i(s)[:, None] * self.product_j(s)[None, :])
        return z


class ReluMLP(eqx.Module):
    """`x -> linear(relu(x))` for each linear in turn."""

    layers: tuple[Linear, ...]

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "... d"]:
        for layer in self.layers:
            x = layer(jax.nn.relu(x))
        return x


class Dropout(eqx.Module):
    """Dropout with its mask shared along `shared_axes`, scaled by `1 / (1 - rate)`.
    OpenFold's rowwise dropout on a pair `[T, T, C]` shares the mask down the rows
    (`(0,)`); Boltz's draws one value per row (`(1, 2)`). Off while `inference` is set, as
    it is by default; `eqx.nn.inference_mode(model, False)` turns it on everywhere in a model.
    `key` may be None only while it is off."""

    rate: float = eqx.field(static=True)
    shared_axes: tuple[int, ...] = eqx.field(static=True)
    inference: bool = True

    def __call__(
        self, x: Float[Array, "..."], *, key: PRNGKeyArray | None = None
    ) -> Float[Array, "..."]:
        if self.inference or self.rate == 0:
            return x
        if key is None:
            raise ValueError("dropout is on (`inference=False`) and needs a key")
        shape = tuple(1 if i in self.shared_axes else d for i, d in enumerate(x.shape))
        keep = jax.random.bernoulli(key, 1 - self.rate, shape)
        return jnp.where(keep, x / (1 - self.rate), 0)


def split_key(key: PRNGKeyArray | None, num: int = 2) -> tuple[PRNGKeyArray | None, ...]:
    """`num` keys from `key`, or `num` Nones from None (no dropout, nothing to draw)."""
    if key is None:
        return (None,) * num
    return tuple(jax.random.split(key, num))


def stack[M: eqx.Module](layers: Sequence[M]) -> M:
    """Stack identical layers into one module whose arrays have a leading layer axis
    (non-array fields, identical across layers, are kept as they are)."""
    return jax.tree.map(lambda *xs: jnp.stack(xs) if eqx.is_array(xs[0]) else xs[0], *layers)


def scan_layers[C, M: eqx.Module](
    f: Callable[[M, C, PRNGKeyArray | None], C],
    carry: C,
    layers: M,
    *,
    key: PRNGKeyArray | None = None,
) -> C:
    """Apply `f(layer, carry, key)` for each layer of a stacked module, with a fresh key
    per layer (None for every layer without `key`), rematerializing each layer's
    activations on the backward pass."""
    arrays, static = eqx.partition(layers, eqx.is_array)
    num_layers = jax.tree.leaves(arrays)[0].shape[0]
    keys = None if key is None else jax.random.split(key, num_layers)

    @jax.checkpoint
    def body(carry, layer_and_key):
        layer, k = layer_and_key
        return f(eqx.combine(layer, static), carry, k), None

    return jax.lax.scan(body, carry, (arrays, keys))[0]
