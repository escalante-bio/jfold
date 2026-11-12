"""Sequence-local atom attention of AF3 Algorithm 7, where blocks of `queries` atoms
attend to `keys` atoms around them."""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from jfold.nn import Attention


class Windows(eqx.Module):
    queries: eqx.AbstractVar[int]
    keys: eqx.AbstractVar[int]

    @abc.abstractmethod
    def key_index(
        self, mask: Bool[Array, " N"]
    ) -> tuple[Int[Array, "NW keys"], Bool[Array, "NW keys"]]:
        """Return each block's key atom indices and a mask of the real ones."""

    def num_blocks(self, num_atoms: int) -> int:
        return -(-num_atoms // self.queries)

    def query_blocks(self, x: Float[Array, "N ..."]) -> Float[Array, "NW queries ..."]:
        """Split per-atom values into query blocks, zero-padded past the last atom."""
        num_blocks = self.num_blocks(x.shape[0])
        x = jnp.pad(x, [(0, num_blocks * self.queries - x.shape[0])] + [(0, 0)] * (x.ndim - 1))
        return x.reshape(num_blocks, self.queries, *x.shape[1:])

    def _centered(self, num_atoms: int) -> Int[Array, "NW keys"]:
        start = jnp.arange(self.num_blocks(num_atoms)) * self.queries
        return start[:, None] - (self.keys - self.queries) // 2 + jnp.arange(self.keys)


class CenteredWindows(Windows):
    """AF3's windows, centered on each query block, with out-of-range keys masked."""

    queries: int = eqx.field(static=True, default=32)
    keys: int = eqx.field(static=True, default=128)

    def key_index(self, mask):
        index = self._centered(mask.shape[0])
        in_range = (index >= 0) & (index < mask.shape[0])
        index = jnp.clip(index, 0, mask.shape[0] - 1)
        return index, in_range & mask[index]


class ShiftedWindows(Windows):
    """OpenFold3's windows, centered but shifted to stay within the real atoms."""

    queries: int = eqx.field(static=True, default=32)
    keys: int = eqx.field(static=True, default=128)

    def key_index(self, mask):
        index = self._centered(mask.shape[0])
        num_real = mask.sum()
        underflow = jax.nn.relu(-index[:, 0])
        overflow = jax.nn.relu(index[:, -1] - (num_real - 1))
        index = index + jnp.where(underflow > 0, underflow, -overflow)[:, None]
        in_range = (index >= 0) & (index < num_real)
        index = jnp.clip(index, 0, mask.shape[0] - 1)
        return index, in_range & mask[index]


class LocalAttention(eqx.Module):
    """`Attention` within windows, biased by the windowed atom pair."""

    attention: Attention
    windows: Windows

    def __call__(
        self,
        q_x: Float[Array, "N d"],
        kv_x: Float[Array, "N d"],
        bias: Float[Array, "heads NW queries keys"],
        key_mask: Bool[Array, " N"],
    ) -> Float[Array, "N d"]:
        index, valid = self.windows.key_index(key_mask)
        q = self.windows.query_blocks(q_x)
        out = jax.vmap(self.attention, in_axes=(0, 0, 1, 0))(q, kv_x[index], bias, valid)
        return out.reshape(-1, q_x.shape[-1])[: q_x.shape[0]]
