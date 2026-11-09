"""Atom attention encoder and decoder (AF3 Algorithms 5 and 6).

The encoder's work splits by when it runs: `AtomAttentionEncoder.condition` depends only on
the reference conformer and the trunk, so a sampler runs it once; `__call__` adds the noisy
positions and runs the atom transformer at every step.

Inputs are unbatched: per-atom arrays `[N, ...]`, the atom-to-token index `[N]`, and an atom
mask `[N]` that is False for padding."""

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.nn import Identity
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from jfold.blocks.diffusion import DiffusionTransformer
from jfold.blocks.windows import CenteredWindows, ShiftedWindows
from jfold.nn import LayerNorm, Linear, ReluMLP


class Atoms(eqx.Module):
    """A system's atoms: the model's feature vector (`jfold.models.*.atom_features`), the
    reference conformers, and which token each atom belongs to."""

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
    """Mean over each token's real atoms (zero for a token without any)."""
    total = jax.ops.segment_sum(x * mask[:, None], atom_to_token, num_tokens)
    count = jax.ops.segment_sum(mask.astype(x.dtype), atom_to_token, num_tokens)
    return total / jnp.maximum(count, 1)[:, None]


class AtomPairEmbedding(eqx.Module):
    """Atom-pair features from the reference conformers, in windows (AF3 Algorithm 5,
    lines 2-6): the offset between two atoms' reference positions, its inverse squared
    length, and a flag; all zero unless both atoms are real and share a conformer."""

    linear_offset: Linear  # 3 -> Cp
    linear_inv_sq_dist: Linear  # 1 -> Cp
    linear_valid: Linear  # 1 -> Cp

    def __call__(
        self,
        ref_pos: Float[Array, "N 3"],
        ref_space_uid: Int[Array, " N"],
        mask: Bool[Array, " N"],
        windows: CenteredWindows | ShiftedWindows,
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
    """The trunk's single and pair, broadcast to atoms and windowed atom pairs (AF3
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
        windows: CenteredWindows | ShiftedWindows,
    ) -> tuple[Float[Array, "N Ca"], Float[Array, "NW queries keys Cp"]]:
        c = c + self.linear_s(self.norm_s(s_trunk))[atom_to_token]
        index, _ = windows.key_index(mask)
        token_q = windows.query_blocks(atom_to_token)[:, :, None]
        token_k = atom_to_token[index][:, None]
        return c, p + self.linear_z(self.norm_z(z_trunk))[token_q, token_k]


class AtomAttentionEncoder(eqx.Module):
    """AF3 Algorithm 5: atom features in, per-token aggregates out. Without `trunk` (the
    trunk's input embedder) there is no trunk conditioning and no noisy positions."""

    embed_features: Linear  # the model's atom feature vector -> Ca
    pair_embedding: AtomPairEmbedding
    trunk: TrunkToAtoms | None
    linear_r: Linear | None  # noisy positions -> Ca
    linear_c_queries: Linear  # relu(c) of the query atom -> Cp
    linear_c_keys: Linear  # relu(c) of the key atom -> Cp
    pair_mlp: ReluMLP
    transformer: DiffusionTransformer  # within atom windows
    linear_token: Linear  # Ca -> token width
    # Protenix and OpenFold3 start the queries from the trunk-conditioned single;
    # AF3 and Boltz from the atom features alone.
    queries_from_conditioned: bool = eqx.field(static=True, default=False)

    def condition(
        self,
        atoms: Atoms,
        s_trunk: Float[Array, "T Cs"] | None = None,
        z_trunk: Float[Array, "T T Cz"] | None = None,
    ) -> tuple[Float[Array, "N Ca"], Float[Array, "N Ca"], Float[Array, "NW queries keys Cp"]]:
        """Everything that does not depend on the noisy positions: the queries' starting
        point `q`, the atom single `c` and the windowed atom pair `p`."""
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
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "T Ct"], Float[Array, "N Ca"]]:
        """Per-token aggregates `a`, and the atom queries for the decoder's skip."""
        if self.linear_r is not None:
            q = q + self.linear_r(r)
        q = self.transformer(q, c, biases, mask, key=key)
        a = jax.nn.relu(self.linear_token(q))
        return mean_atoms_to_tokens(a, atom_to_token, mask, num_tokens), q


class AtomAttentionDecoder(eqx.Module):
    """AF3 Algorithm 6: token updates broadcast to atoms, the atom transformer, and a
    position update per atom."""

    linear_a: Linear  # token width -> Ca
    transformer: DiffusionTransformer  # within atom windows
    norm_out: LayerNorm | Identity  # Boltz-2 drops it when its transformer post-normalizes
    linear_out: Linear  # Ca -> 3

    def __call__(
        self,
        a: Float[Array, "T Ct"],
        q: Float[Array, "N Ca"],
        c: Float[Array, "N Ca"],
        biases: Float[Array, "layers heads NW queries keys"],
        atom_to_token: Int[Array, " N"],
        mask: Bool[Array, " N"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "N 3"]:
        q = q + self.linear_a(a)[atom_to_token]
        q = self.transformer(q, c, biases, mask, key=key)
        return self.linear_out(self.norm_out(q))
