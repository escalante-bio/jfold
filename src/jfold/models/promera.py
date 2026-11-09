"""ProMERA model pieces."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from jfold.blocks.atoms import Atoms, mean_atoms_to_tokens
from jfold.blocks.inputs import probabilities, relative_position_features
from jfold.model import Features
from jfold.models.common import arrays, msa_features
from jfold.nn import Linear
from jfold.slots import AbstractInputEmbedder


def atom_features(
    ref_pos: Float[Array, "N 3"],
    ref_charge: Float[Array, " N"],
    atom_mask: Bool[Array, " N"],
    ref_element: Int[Array, " N"],
    ref_atom_name_chars: Int[Array, "N 4"],
    atom_is_std: Bool[Array, " N"],
    ref_hydrogens: Int[Array, " N"],
) -> Float[Array, "N 397"]:
    """The atom feature vector ProMERA's encoder embeds: position, charge, mask, element,
    name characters (standard residues only) and hydrogen count."""
    name_chars = jax.nn.one_hot(ref_atom_name_chars, 64).reshape(ref_pos.shape[0], -1)
    return jnp.concatenate(
        [
            ref_pos,
            ref_charge[:, None],
            atom_mask[:, None],
            jax.nn.one_hot(ref_element, 128),
            name_chars * atom_is_std[:, None],
            jax.nn.one_hot(ref_hydrogens, 8),
        ],
        axis=-1,
    )


def token_features(res_type, profile, deletion_mean, is_epitope):
    """The token feature vector `[T, 70]`: residue-type probabilities `[T, 34]` (one-hot for
    a fixed sequence), MSA profile, deletion mean, epitope flag."""
    return jnp.concatenate([res_type, profile, deletion_mean[:, None], is_epitope[:, None]], -1)


class AtomAggregate(eqx.Module):
    """An atom encoder without transformer layers (ProMERA's input embedder): each token's
    mean over its atoms of `relu(linear_token(embed_features(atom features)))`."""

    embed_features: Linear
    linear_token: Linear

    def __call__(self, atoms: Atoms, num_tokens: int) -> Float[Array, "T Ct"]:
        a = jax.nn.relu(self.linear_token(self.embed_features(atoms.features)))
        return mean_atoms_to_tokens(a, atoms.atom_to_token, atoms.mask, num_tokens)


class PromeraInputEmbedder(AbstractInputEmbedder):
    """`s_inputs`: the atom aggregates, kept where `keep_aggregate` holds, concatenated with
    the token features."""

    encoder: AtomAggregate

    def __call__(self, features: Features, *, key: PRNGKeyArray | None = None) -> Float[Array, "T Ci"]:
        return self.embed(features.atoms, features.token_features, features.keep_aggregate)

    def embed(
        self,
        atoms: Atoms,
        token_features: Float[Array, "T F"],
        keep_aggregate: Bool[Array, " T"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "T Ci"]:
        a = self.encoder(atoms, token_features.shape[0])
        if keep_aggregate is not None:
            a = jnp.where(keep_aggregate[:, None], a, 0)
        return jnp.concatenate([a, token_features], axis=-1)


def features(f: dict) -> Features:
    """`jfold.model.Features` from a ProMERA feature dict (tinyprot's `AF3Featurizer`
    output after `promera.inference.utils.finalize_feats`; unbatched, atoms possibly
    padded as `promera.data.utils.collate` pads them). The atom aggregate is dropped for
    standard residues, as in the released model (`mask_std_feats`).

    Arrays may be numpy or JAX (traceable: runs inside `jit`). `restype` and `msa` are
    class indices or probability vectors over the 34 token types (a soft sequence)."""
    a = arrays(f)
    num_tokens = a["restype"].shape[0]
    num_atoms = a["ref_pos"].shape[0]
    atom_mask = a["atom_pad_mask"].astype(bool)
    atom_to_token = a["atom_to_token"]
    atom_index = jnp.arange(num_atoms)
    first_atom = jax.ops.segment_min(
        jnp.where(atom_mask, atom_index, num_atoms), atom_to_token, num_tokens
    )
    asym_id = a["asym_id"]
    is_epitope = a.get("is_epitope", jnp.zeros(num_tokens))
    return Features(
        atoms=Atoms(
            features=atom_features(
                a["ref_pos"].astype(jnp.float32),
                a["ref_charge"].astype(jnp.float32),
                atom_mask,
                a["ref_element"],
                a["ref_atom_name_chars"],
                a["atom_is_std"].astype(bool),
                a["ref_hydrogens"],
            ),
            ref_pos=a["ref_pos"].astype(jnp.float32),
            ref_space_uid=a["ref_space_uid"],
            atom_to_token=atom_to_token,
            mask=atom_mask,
        ),
        token_features=token_features(
            probabilities(a["restype"], 34),
            a["profile"].astype(jnp.float32),
            a["deletion_mean"].astype(jnp.float32),
            is_epitope.astype(jnp.float32),
        ),
        relative_positions=relative_position_features(
            asym_id,
            a["residue_index"],
            a["entity_id"],
            a["token_index"],
            a["sym_id"],
            chain_offset=asym_id[:, None] != asym_id[None, :],
        ),
        token_mask=a["token_pad_mask"].astype(bool),
        asym_id=asym_id,
        rep_atom=a["token_to_rep_atom"],
        atom_position=jnp.where(atom_mask, atom_index - first_atom[atom_to_token], 0),
        bonds=a["token_bonds"].astype(jnp.float32)[..., None],
        keep_aggregate=~a["is_std"].astype(bool),
        msa_features=msa_features(
            probabilities(a["msa"], 34),
            a["has_deletion"].astype(jnp.float32),
            a["deletion_value"].astype(jnp.float32),
        ),
        msa_mask=a["msa_mask"].astype(bool),
    )
