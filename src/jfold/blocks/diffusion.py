"""The diffusion transformer (AF3 Algorithms 23-26): attention and transitions on a
representation `a`, conditioned on `s` through adaptive layer norms and output gates, with
attention biased by the pair.

Inputs are unbatched: `a` `[N, Ca]`, `s` `[N, Cs]`, and a mask `[N]` that is False for
padding. The same transformer runs over tokens, with pair biases `[heads, N, N]`, and over
atoms, with `LocalAttention` and windowed biases `[heads, NW, queries, keys]`."""

import equinox as eqx
import jax
from equinox.nn import Identity
from jaxtyping import Array, Bool, Float

from jfold.blocks.windows import LocalAttention, Windows
from jfold.nn import Attention, LayerNorm, Linear, PairBias, scan_layers


class AdaLN(eqx.Module):
    """Adaptive layer norm (AF3 Algorithm 26): normalize `a`, then scale and shift it by
    projections of `s`."""

    norm_a: LayerNorm  # no scale or offset
    norm_s: LayerNorm
    linear_scale: Linear
    linear_shift: Linear

    def __call__(self, a: Float[Array, "N Ca"], s: Float[Array, "N Cs"]) -> Float[Array, "N Ca"]:
        s = self.norm_s(s)
        return jax.nn.sigmoid(self.linear_scale(s)) * self.norm_a(a) + self.linear_shift(s)


class RenormKeys(eqx.Module):
    """Keys and values from a second adaptive norm of the queries' normalized input
    (the Protenix lineage's atom attention)."""

    adaln: AdaLN

    def __call__(self, a, x, s):
        return self.adaln(x, s)


class SeparateKeys(eqx.Module):
    """Keys and values from their own adaptive norm of the input (OpenFold3's atom
    attention)."""

    adaln: AdaLN

    def __call__(self, a, x, s):
        return self.adaln(a, s)


class ConditionedAttention(eqx.Module):
    """Self-attention with pair bias, conditioned on `s` (AF3 Algorithm 24): adaptive layer
    norm in, a gate from `s` out. Keys and values share the queries' normalized input
    unless `keys` normalizes them separately."""

    adaln: AdaLN
    attention: Attention | LocalAttention
    linear_gate: Linear  # Cs -> Ca
    keys: RenormKeys | SeparateKeys | None = None

    def __call__(
        self,
        a: Float[Array, "N Ca"],
        s: Float[Array, "N Cs"],
        bias: Float[Array, "heads ..."],
        mask: Bool[Array, " N"],
    ) -> Float[Array, "N Ca"]:
        x = self.adaln(a, s)
        kv = x if self.keys is None else self.keys(a, x, s)
        return jax.nn.sigmoid(self.linear_gate(s)) * self.attention(x, kv, bias, mask)


class ConditionedTransition(eqx.Module):
    """SwiGLU transition conditioned on `s` (AF3 Algorithm 25)."""

    adaln: AdaLN
    linear_a: Linear  # through SiLU
    linear_b: Linear
    linear_c: Linear | None  # Boltz multiplies in a third projection
    linear_out: Linear
    linear_gate: Linear  # Cs -> Ca

    def __call__(self, a: Float[Array, "N Ca"], s: Float[Array, "N Cs"]) -> Float[Array, "N Ca"]:
        x = self.adaln(a, s)
        b = jax.nn.silu(self.linear_a(x)) * self.linear_b(x)
        if self.linear_c is not None:
            b = b * self.linear_c(x)
        return jax.nn.sigmoid(self.linear_gate(s)) * self.linear_out(b)


class DiffusionTransformerLayer(eqx.Module):
    """One block of AF3 Algorithm 23."""

    attention: ConditionedAttention
    transition: ConditionedTransition
    post_norm: LayerNorm | Identity = eqx.field(default_factory=Identity)  # Boltz-2's

    def __call__(
        self,
        a: Float[Array, "N Ca"],
        s: Float[Array, "N Cs"],
        bias: Float[Array, "heads ..."],
        mask: Bool[Array, " N"],
    ) -> Float[Array, "N Ca"]:
        a = a + self.attention(a, s, bias, mask)
        a = a + self.transition(a, s)
        return self.post_norm(a)


class DiffusionTransformer(eqx.Module):
    """AF3 Algorithm 23: identical layers, stacked and scanned. Each layer's attention bias
    comes from the pair through its own `PairBias`; `pair_biases` depends only on the pair,
    so a sampler computes it once."""

    pair_bias: PairBias  # one per layer, stacked
    layers: DiffusionTransformerLayer  # stacked

    @property
    def windows(self) -> Windows:
        """The atom windows its attention runs in (an atom transformer's)."""
        return self.layers.attention.attention.windows

    def pair_biases(self, z: Float[Array, "... Cz"]) -> Float[Array, "layers heads ..."]:
        return jax.vmap(lambda pair_bias: pair_bias(z))(self.pair_bias)

    def __call__(
        self,
        a: Float[Array, "N Ca"],
        s: Float[Array, "N Cs"],
        biases: Float[Array, "layers heads ..."],
        mask: Bool[Array, " N"],
    ) -> Float[Array, "N Ca"]:
        def step(layer_and_bias, a, key):
            layer, bias = layer_and_bias
            return layer(a, s, bias, mask)

        return scan_layers(step, a, (self.layers, biases))
