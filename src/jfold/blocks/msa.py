"""The MSA module of AF3 Algorithms 8-10, on an unbatched MSA `[S, T, Cm]` and pair
`[T, T, Cz]` with masks False for padding."""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.nn import Identity
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from jfold.blocks.pairformer import PairformerLayer, PairMixerLayer
from jfold.nn import Dropout, LayerNorm, Linear, Transition, scan_layers, split_key


class PairWeightedAveraging(eqx.Module):
    """AF3 Algorithm 10, where each MSA row averages over tokens with per-head weights
    from the pair, gated by the row."""

    norm_m: LayerNorm
    linear_v: Linear  # Cm -> heads * C
    norm_z: LayerNorm
    linear_b: Linear  # Cz -> heads
    linear_g: Linear  # Cm -> heads * C
    linear_o: Linear  # heads * C -> Cm
    num_heads: int = eqx.field(static=True)

    def __call__(
        self, m: Float[Array, "S T Cm"], z: Float[Array, "T T Cz"], mask: Bool[Array, " T"]
    ) -> Float[Array, "S T Cm"]:
        m = self.norm_m(m)
        heads = lambda x: x.reshape(*x.shape[:-1], self.num_heads, -1)
        b = jnp.moveaxis(self.linear_b(self.norm_z(z)), -1, 0)  # [heads, T, T]
        w = jax.nn.softmax(jnp.where(mask[None, None, :], b, -1e9), axis=-1)
        o = jnp.einsum("hij,sjhc->sihc", w, heads(self.linear_v(m)))
        o = o * heads(jax.nn.sigmoid(self.linear_g(m)))
        return self.linear_o(o.reshape(*o.shape[:-2], -1))


def outer_product_sum(norm, linear_a, linear_b, m, msa_mask):
    """Return the flattened sum of `a_si ⊗ b_sj` over real MSA rows, and the row counts."""
    m = norm(m)
    mask = msa_mask[..., None].astype(m.dtype)
    a, b = linear_a(m) * mask, linear_b(m) * mask
    outer = jnp.einsum("sic,sjd->ijcd", a, b)
    count = jnp.einsum("si,sj->ij", msa_mask.astype(m.dtype), msa_mask.astype(m.dtype))
    return outer.reshape(*outer.shape[:2], -1), count


class OuterProductMean(eqx.Module):
    """AF3 Algorithm 9 as in Boltz, projecting the mean of outer products over MSA rows
    to the pair."""

    norm: LayerNorm
    linear_a: Linear
    linear_b: Linear
    linear_out: Linear

    def __call__(
        self, m: Float[Array, "S T Cm"], msa_mask: Bool[Array, "S T"]
    ) -> Float[Array, "T T Cz"]:
        outer, count = outer_product_sum(self.norm, self.linear_a, self.linear_b, m, msa_mask)
        return self.linear_out(outer / jnp.maximum(count, 1)[..., None])


class ProjectedOuterProductMean(eqx.Module):
    """OpenFold's outer product mean, which projects the summed outer products and then
    divides by the row count plus `eps`."""

    norm: LayerNorm
    linear_a: Linear
    linear_b: Linear
    linear_out: Linear
    eps: float = eqx.field(static=True, default=1e-3)

    def __call__(
        self, m: Float[Array, "S T Cm"], msa_mask: Bool[Array, "S T"]
    ) -> Float[Array, "T T Cz"]:
        outer, count = outer_product_sum(self.norm, self.linear_a, self.linear_b, m, msa_mask)
        return self.linear_out(outer) / (count + self.eps)[..., None]


class MSAUpdate(eqx.Module):
    """Update the MSA with pair-weighted averaging and dropout, followed by a transition."""

    pair_weighted_averaging: PairWeightedAveraging
    transition: Transition
    dropout: Dropout | Identity = eqx.field(default_factory=Identity)

    def __call__(
        self,
        m: Float[Array, "S T Cm"],
        z: Float[Array, "T T Cz"],
        mask: Bool[Array, " T"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "S T Cm"]:
        m = m + self.dropout(self.pair_weighted_averaging(m, z, mask), key=key)
        return m + self.transition(m)


class MSALayer(eqx.Module):
    """One block of AF3 Algorithm 8, which returns the MSA unchanged if `msa` is None."""

    outer_product_mean: OuterProductMean | ProjectedOuterProductMean
    msa: MSAUpdate | None
    pair: PairformerLayer | PairMixerLayer  # with no single track
    outer_product_first: bool = eqx.field(static=True)

    def __call__(
        self,
        m: Float[Array, "S T Cm"],
        z: Float[Array, "T T Cz"],
        msa_mask: Bool[Array, "S T"],
        mask: Bool[Array, " T"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "S T Cm"], Float[Array, "T T Cz"]]:
        k_msa, k_pair = split_key(key)
        if self.outer_product_first:
            z = z + self.outer_product_mean(m, msa_mask)
        if self.msa is not None:
            m = self.msa(m, z, mask, key=k_msa)
        if not self.outer_product_first:
            z = z + self.outer_product_mean(m, msa_mask)
        _, z = self.pair(None, z, mask, key=k_pair)
        return m, z


DEFAULT_MAX_ROWS = 1024  # rows drawn per pass by ESMFold2, Boltz-2, OpenFold3 and ProMERA


def subsample_rows(
    msa_mask: Bool[Array, "S T"], num: int, key: PRNGKeyArray, *, keep_query: bool = True
) -> Int[Array, " num"]:
    """Draw `num` row indices uniformly without replacement, real rows first, always
    including row 0 if `keep_query`."""
    real = msa_mask.any(axis=-1)
    if not keep_query:
        scores = jax.random.gumbel(key, real.shape) - jnp.where(real, 0.0, jnp.inf)
        return jax.lax.top_k(scores, num)[1]
    rest = subsample_rows(msa_mask[1:], num - 1, key, keep_query=False)
    return jnp.concatenate([jnp.zeros(1, rest.dtype), rest + 1])


class AbstractMSAModule(eqx.Module):
    """A trunk's MSA module, which updates the pair from the MSA."""

    @abc.abstractmethod
    def __call__(
        self,
        z: Float[Array, "T T Cz"],
        msa_features: Float[Array, "S T F"],
        s_inputs: Float[Array, "T Ci"],
        msa_mask: Bool[Array, "S T"],
        mask: Bool[Array, " T"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "S T Cm"], Float[Array, "T T Cz"]]:
        """Return the final MSA and the updated pair."""


class MSAModule(AbstractMSAModule):
    """The MSA module of AF3 Algorithm 8, which draws at most `max_rows` real rows each
    pass, or all if None, keeping the query if `keep_query`. With `min_rows`, it masks
    out all but a uniformly random number of them, at least `min_rows`."""

    linear_msa: Linear  # MSA feature vector -> Cm
    linear_s: Linear  # s_inputs -> Cm
    layers: MSALayer | None  # stacked
    last: MSALayer | None = None
    max_rows: int | None = eqx.field(static=True, default=DEFAULT_MAX_ROWS)
    keep_query: bool = eqx.field(static=True, default=True)
    min_rows: int | None = eqx.field(static=True, default=None)

    def draw_rows(
        self, msa_mask: Bool[Array, "S T"], key: PRNGKeyArray | None
    ) -> tuple[Int[Array, " R"], Bool[Array, "R T"]] | None:
        """Return the drawn row indices and their mask, or None if every row is used."""
        num_rows = msa_mask.shape[0]
        num = num_rows if self.max_rows is None else min(num_rows, self.max_rows)
        random_depth = self.min_rows is not None and num_rows > self.min_rows
        if num == num_rows and not random_depth:
            return None
        if key is None:
            raise ValueError(f"an MSA of {num_rows} rows is drawn from at random: pass a key")
        k_rows, k_depth = jax.random.split(key)
        rows = subsample_rows(msa_mask, num, k_rows, keep_query=self.keep_query)
        drawn = msa_mask[rows]
        if random_depth:
            real = msa_mask.any(axis=-1).sum()
            depth = jax.random.randint(k_depth, (), jnp.minimum(self.min_rows, real), real + 1)
            drawn = drawn & (jnp.arange(num) < depth)[:, None]
        return rows, drawn

    def __call__(
        self,
        z: Float[Array, "T T Cz"],
        msa_features: Float[Array, "S T F"],
        s_inputs: Float[Array, "T Ci"],
        msa_mask: Bool[Array, "S T"],
        mask: Bool[Array, " T"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "S T Cm"], Float[Array, "T T Cz"]]:
        k_rows, k_layers, k_last = split_key(key, 3)
        draw = self.draw_rows(msa_mask, k_rows)
        if draw is not None:
            rows, msa_mask = draw
            msa_features = msa_features[rows]
        m = self.linear_msa(msa_features) + self.linear_s(s_inputs)

        def step(layer, carry, k):
            return layer(*carry, msa_mask, mask, key=k)

        if self.layers is not None:
            m, z = scan_layers(step, (m, z), self.layers, key=k_layers)
        if self.last is not None:
            m, z = self.last(m, z, msa_mask, mask, key=k_last)
        return m, z


class ResidualMSA(AbstractMSAModule):
    """Return `z + msa(z)` as Boltz-2 and ProMERA do, counting the incoming pair twice."""

    module: MSAModule

    def __call__(
        self,
        z: Float[Array, "T T Cz"],
        msa_features: Float[Array, "S T F"],
        s_inputs: Float[Array, "T Ci"],
        msa_mask: Bool[Array, "S T"],
        mask: Bool[Array, " T"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "S T Cm"], Float[Array, "T T Cz"]]:
        m, out = self.module(z, msa_features, s_inputs, msa_mask, mask, key=key)
        return m, z + out
