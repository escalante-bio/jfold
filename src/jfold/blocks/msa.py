"""The MSA module (AF3 Algorithms 8-10): MSA rows exchange information with the pair through
outer product means and pair-weighted averaging, between pair-only Pairformer layers.

Inputs are unbatched: MSA `[S, T, Cm]` with a mask `[S, T]`, pair `[T, T, Cz]` and a token
mask `[T]`; masks are False for padding."""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.nn import Identity
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from jfold.blocks.pairformer import PairformerLayer, PairMixerLayer
from jfold.nn import Dropout, LayerNorm, Linear, Transition, scan_layers, split_key


class PairWeightedAveraging(eqx.Module):
    """AF3 Algorithm 10: each MSA row averages over tokens with per-head weights from the
    pair, gated by the row."""

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
    """The sum over real MSA rows of `a_si ⊗ b_sj`, flattened, and the number of rows in it."""
    m = norm(m)
    mask = msa_mask[..., None].astype(m.dtype)
    a, b = linear_a(m) * mask, linear_b(m) * mask
    outer = jnp.einsum("sic,sjd->ijcd", a, b)
    count = jnp.einsum("si,sj->ij", msa_mask.astype(m.dtype), msa_mask.astype(m.dtype))
    return outer.reshape(*outer.shape[:2], -1), count


class OuterProductMean(eqx.Module):
    """AF3 Algorithm 9: the mean over MSA rows of outer products, projected to the pair
    (the Boltz lineage)."""

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
    """OpenFold's outer product mean (Protenix, OpenDDE, Mol-1, OpenFold3): the summed outer
    products are projected first, and the result, bias included, divided by the row count
    plus `eps`."""

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
    """The MSA's own update in a block: pair-weighted averaging (with dropout), then a
    transition."""

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
    """One block of AF3 Algorithm 8. AF3 updates the pair from the MSA first
    (`outer_product_first`); the Boltz lineage updates the MSA first. Without `msa` (a last
    block, whose MSA nothing reads) the MSA passes through."""

    outer_product_mean: OuterProductMean | ProjectedOuterProductMean
    msa: MSAUpdate | None
    pair: PairformerLayer | PairMixerLayer  # without a single track
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


DEFAULT_MAX_ROWS = 1024  # ESMFold2's, Boltz-2's, OpenFold3's and ProMERA's per-pass draw


def subsample_rows(
    msa_mask: Bool[Array, "S T"], num: int, key: PRNGKeyArray, *, keep_query: bool = True
) -> Int[Array, " num"]:
    """`num` rows drawn uniformly without replacement, real rows before padding. With
    `keep_query`, row 0 (the query) and `num - 1` others."""
    real = msa_mask.any(axis=-1)
    if not keep_query:
        scores = jax.random.gumbel(key, real.shape) - jnp.where(real, 0.0, jnp.inf)
        return jax.lax.top_k(scores, num)[1]
    rest = subsample_rows(msa_mask[1:], num - 1, key, keep_query=False)
    return jnp.concatenate([jnp.zeros(1, rest.dtype), rest + 1])


class AbstractMSAModule(eqx.Module):
    """A trunk's MSA module: the pair updated from the MSA (`MSAModule`, or `ResidualMSA`
    around one)."""

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
        """The final MSA and the pair."""


class MSAModule(AbstractMSAModule):
    """AF3 Algorithm 8: embed the MSA (the model's MSA feature vector, from
    `jfold.models.*.msa_features`, plus `s_inputs`), then update the pair through stacked
    `MSALayer`s and, when the last block differs (no MSA update), a final one.

    Each call (each trunk pass) draws afresh the MSA rows its upstream reads (`draw_rows`);
    a draw needs a `key`:

    - At most `max_rows` rows (`None`: every row), uniformly without replacement from the
      real rows. With `keep_query` (ESMFold2) the query is always among them; Boltz-2,
      ProMERA, OpenFold3 and OpenDDE draw it like any other row.
    - With `min_rows` (Protenix, Mol-1), a uniformly random number of those: from `min_rows`
      (every real row, if fewer) to every real row. The rows past that number are masked, so
      a pass computes on the whole draw.

    Row order does not change the pair."""

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
        """The rows a pass reads and their mask (rows past a random depth masked out), or
        None when it reads every row."""
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
        """The final MSA (before the last block's skipped update, if any) and the pair."""
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
    """The MSA module as Boltz-2's and ProMERA's trunks apply it: `z + msa(z)`, where the
    module's output pair already includes its input, so the incoming pair counts twice."""

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
