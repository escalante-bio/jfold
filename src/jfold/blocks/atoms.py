"""The atom attention encoder and decoder of AF3 Algorithms 5 and 6, on unbatched
per-atom arrays `[N, ...]`."""

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.nn import Identity
from jaxtyping import Array, Bool, Float, Int

from jfold.blocks.diffusion import DiffusionTransformer
from jfold.blocks.windows import Windows
from jfold.nn import LayerNorm, Linear, ReluMLP


class Atoms(eqx.Module):
    """A system's atoms, with their features, reference conformers and token indices."""

    features: Float[Array, "N F"]
    ref_pos: Float[Array, "N 3"]
    ref_space_uid: Int[Array, " N"]
    atom_to_token: Int[Array, " N"]
    mask: Bool[Array, " N"]


def mean_atoms_to_tokens(
    x: Float[Array, "N d"],
    atom_to_token: Int[Array, " N"],
    mask: Bool[Array, " N"],
    num_tokens: int,
) -> Float[Array, "T d"]:
    """Average `x` over each token's real atoms, giving zero for tokens with none."""
    total = jax.ops.segment_sum(x * mask[:, None], atom_to_token, num_tokens)
    count = jax.ops.segment_sum(mask.astype(x.dtype), atom_to_token, num_tokens)
    return total / jnp.maximum(count, 1)[:, None]


class AtomPairEmbedding(eqx.Module):
    """Embed windowed atom-pair features from the reference conformers (AF3 Algorithm 5,
    lines 2-6), zero unless both atoms are real and in the same conformer."""

    linear_offset: Linear  # 3 -> Cp
    linear_inv_sq_dist: Linear  # 1 -> Cp
    linear_valid: Linear  # 1 -> Cp

    def __call__(
        self,
        ref_pos: Float[Array, "N 3"],
        ref_space_uid: Int[Array, " N"],
        mask: Bool[Array, " N"],
        windows: Windows,
    ) -> Float[Array, "NW queries keys Cp"]:
        index, key_valid = windows.key_index(mask)
        offset = windows.query_blocks(ref_pos)[:, :, None] - ref_pos[index][:, None]
        same = windows.query_blocks(ref_space_uid)[:, :, None] == ref_space_uid[index][:, None]
        valid = same & windows.query_blocks(mask)[:, :, None] & key_valid[:, None]
        valid = valid[..., None].astype(offset.dtype)
        inv_sq_dist = 1 / (1 + jnp.sum(offset**2, axis=-1, keepdims=True))
        p = self.linear_offset(offset) + self.linear_inv_sq_dist(inv_sq_dist)
        return (p + self.linear_valid(valid)) * valid


class TrunkToAtoms(eqx.Module):
    """Add the trunk's single and pair to the atom single and windowed atom pair (AF3
    Algorithm 5, lines 9-11)."""

    norm_s: LayerNorm
    linear_s: Linear
    norm_z: LayerNorm
    linear_z: Linear

    def __call__(
        self,
        c: Float[Array, "N Ca"],
        p: Float[Array, "NW queries keys Cp"],
        s_trunk: Float[Array, "T Cs"],
        z_trunk: Float[Array, "T T Cz"],
        atom_to_token: Int[Array, " N"],
        mask: Bool[Array, " N"],
        windows: Windows,
    ) -> tuple[Float[Array, "N Ca"], Float[Array, "NW queries keys Cp"]]:
        c = c + self.linear_s(self.norm_s(s_trunk))[atom_to_token]
        index, _ = windows.key_index(mask)
        token_q = windows.query_blocks(atom_to_token)[:, :, None]
        token_k = atom_to_token[index][:, None]
        return c, p + self.linear_z(self.norm_z(z_trunk))[token_q, token_k]


class AtomAttentionEncoder(eqx.Module):
    """AF3 Algorithm 5, which turns atom features into per-token aggregates, without
    trunk conditioning or noisy positions if `trunk` is None."""

    embed_features: Linear  # the model's atom feature vector -> Ca
    pair_embedding: AtomPairEmbedding
    trunk: TrunkToAtoms | None
    linear_r: Linear | None  # noisy positions -> Ca
    linear_c_queries: Linear  # relu(c) of the query atom -> Cp
    linear_c_keys: Linear  # relu(c) of the key atom -> Cp
    pair_mlp: ReluMLP
    transformer: DiffusionTransformer  # within atom windows
    linear_token: Linear  # Ca -> token width
    # Protenix and OpenFold3 start the queries from the trunk-conditioned single.
    queries_from_conditioned: bool = eqx.field(static=True, default=False)

    def condition(
        self,
        atoms: Atoms,
        s_trunk: Float[Array, "T Cs"] | None = None,
        z_trunk: Float[Array, "T T Cz"] | None = None,
    ) -> tuple[Float[Array, "N Ca"], Float[Array, "N Ca"], Float[Array, "NW queries keys Cp"]]:
        """Compute the starting queries `q`, atom single `c` and windowed atom pair `p`,
        which do not depend on the noisy positions."""
        windows, mask = self.transformer.windows, atoms.mask
        c = self.embed_features(atoms.features)
        p = self.pair_embedding(atoms.ref_pos, atoms.ref_space_uid, mask, windows)
        q = c
        if self.trunk is not None:
            c, p = self.trunk(c, p, s_trunk, z_trunk, atoms.atom_to_token, mask, windows)
            if self.queries_from_conditioned:
                q = c
        index, _ = windows.key_index(mask)
        relu_c = jax.nn.relu(c)
        p = p + self.linear_c_queries(windows.query_blocks(relu_c))[:, :, None]
        p = p + self.linear_c_keys(relu_c[index])[:, None]
        return q, c, p + self.pair_mlp(p)

    def __call__(
        self,
        q: Float[Array, "N Ca"],
        c: Float[Array, "N Ca"],
        biases: Float[Array, "layers heads NW queries keys"],
        atom_to_token: Int[Array, " N"],
        mask: Bool[Array, " N"],
        num_tokens: int,
        r: Float[Array, "N 3"] | None = None,
    ) -> tuple[Float[Array, "T Ct"], Float[Array, "N Ca"]]:
        """Return the per-token aggregates and the atom queries for the decoder."""
        if self.linear_r is not None:
            q = q + self.linear_r(r)
        q = self.transformer(q, c, biases, mask)
        a = jax.nn.relu(self.linear_token(q))
        return mean_atoms_to_tokens(a, atom_to_token, mask, num_tokens), q


class AtomAttentionDecoder(eqx.Module):
    """AF3 Algorithm 6, which broadcasts token updates to atoms and returns a position
    update per atom."""

    linear_a: Linear  # token width -> Ca
    transformer: DiffusionTransformer  # within atom windows
    norm_out: LayerNorm | Identity  # Boltz-2 skips it when its transformer post-normalizes
    linear_out: Linear  # Ca -> 3

    def __call__(
        self,
        a: Float[Array, "T Ct"],
        q: Float[Array, "N Ca"],
        c: Float[Array, "N Ca"],
        biases: Float[Array, "layers heads NW queries keys"],
        atom_to_token: Int[Array, " N"],
        mask: Bool[Array, " N"],
    ) -> Float[Array, "N 3"]:
        q = q + self.linear_a(a)[atom_to_token]
        q = self.transformer(q, c, biases, mask)
        return self.linear_out(self.norm_out(q))
