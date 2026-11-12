"""Helpers for reading torch modules into jfold modules."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from equinox.nn import Identity

from jfold.blocks.diffusion import DiffusionTransformer
from jfold.blocks.windows import LocalAttention, Windows
from jfold.nn import Attention, Dropout, LayerNorm, Linear


def array(t) -> jnp.ndarray:
    return jnp.array(np.asarray(t.detach().cpu(), dtype=np.float32))


def _maybe(t):
    return None if t is None else array(t)


def linear(m) -> Linear:
    return Linear(weight=array(m.weight), bias=_maybe(m.bias))


def concat_outputs(*ms) -> Linear:
    """Merge bias-free linears into one `Linear` that returns their outputs concatenated."""
    if any(m.bias is not None for m in ms):
        raise ValueError("concat_outputs expects bias-free linears")
    return Linear(weight=jnp.concatenate([array(m.weight) for m in ms]))


def concat_inputs(*ms) -> Linear:
    """Merge linears into one `Linear` on their concatenated inputs, summing outputs."""
    biases = [array(m.bias) for m in ms if m.bias is not None]
    return Linear(
        weight=jnp.concatenate([array(m.weight) for m in ms], axis=1),
        bias=sum(biases) if biases else None,
    )


def negated_input(m) -> Linear:
    """Convert `m` to act on the negated input."""
    return Linear(weight=-array(m.weight), bias=_maybe(m.bias))


def layer_norm(m) -> LayerNorm:
    return LayerNorm(
        weight=_maybe(getattr(m, "weight", None)),
        bias=_maybe(getattr(m, "bias", None)),
        eps=float(m.eps),
    )


def num_heads(m) -> int:
    for name in ("no_heads", "num_heads", "n_heads"):
        if hasattr(m, name):
            return int(getattr(m, name))
    raise AttributeError(f"{type(m).__name__} has no head count")


def attention(
    m, *, q="linear_q", k="linear_k", v="linear_v", g="linear_g", o="linear_o", heads=None
):
    """Convert a torch module holding the five attention projections to `Attention`."""
    return Attention(
        linear_q=linear(getattr(m, q)),
        linear_k=linear(getattr(m, k)),
        linear_v=linear(getattr(m, v)),
        linear_g=linear(getattr(m, g)),
        linear_o=linear(getattr(m, o)),
        num_heads=num_heads(m) if heads is None else heads,
    )


def localize(transformer: DiffusionTransformer, windows: Windows) -> DiffusionTransformer:
    """Return the same transformer with its attention restricted to atom windows."""
    return eqx.tree_at(
        lambda t: t.layers.attention.attention,
        transformer,
        replace_fn=lambda attention: LocalAttention(attention=attention, windows=windows),
    )


def doubled_bias(m) -> Linear:
    """Convert `m` with its bias doubled, so that `m(a) + m(b) = doubled_bias(m)(a + b)`."""
    return Linear(weight=array(m.weight), bias=None if m.bias is None else 2 * array(m.bias))


def embedding(m) -> Linear:
    """Convert an `nn.Embedding` to a `Linear` on one-hot indices."""
    return Linear(weight=array(m.weight).T)


def below_first_bin(m) -> Linear:
    """Convert a linear on distance bins, adding a zero column for the bin below the
    first edge that upstream lacks."""
    w = array(m.weight)
    return Linear(
        weight=jnp.concatenate([jnp.zeros((w.shape[0], 1), w.dtype), w], axis=1),
        bias=_maybe(m.bias),
    )


def dropout(rate: float, shared_axes: tuple[int, ...]) -> Dropout | Identity:
    """Return `Dropout` at `rate`, or `Identity` if the rate is zero."""
    return Dropout(rate=float(rate), shared_axes=shared_axes) if rate else Identity()


# OpenFold's rowwise dropout shares its mask down the rows of a `[rows, columns, channels]`
# tensor, while Boltz's draws one value per row.
OPENFOLD_ROW, OPENFOLD_COLUMN = (0,), (1,)
BOLTZ_ROW, BOLTZ_COLUMN = (1, 2), (0, 2)
