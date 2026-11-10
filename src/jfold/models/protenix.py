"""Protenix-lineage (Protenix, OpenDDE, Mol-1) model pieces."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jfold.blocks.atoms import Atoms
from jfold.blocks.inputs import probabilities, relative_position_features
from jfold.models.common import arrays, msa_features, token_features
from jfold.slots import Features


def atom_features(
    ref_pos: Float[Array, "N 3"],
    ref_charge: Float[Array, " N"],
    ref_mask: Float[Array, " N"],
    ref_element: Float[Array, "N 128"],
    ref_atom_name_chars: Float[Array, "N 4 64"],
) -> Float[Array, "N 389"]:
    """The atom feature vector, in the column order `convert.protenix.atom_encoder` lays out:
    position, arcsinh of the charge, mask, element and name characters; zero for atoms
    without a reference conformer."""
    features = jnp.concatenate(
        [
            ref_pos,
            jnp.arcsinh(ref_charge)[:, None],
            ref_mask[:, None],
            ref_element,
            ref_atom_name_chars.reshape(ref_pos.shape[0], -1),
        ],
        axis=-1,
    )
    return features * ref_mask[:, None]


def template_features(
    distogram, pseudo_beta_mask, aatype, unit_vector, backbone_frame_mask, asym_id
):
    """Each template's pair feature vector `[N, T, T, 108]`: distogram, pseudo-beta pair
    mask, residue type of the column then of the row token, unit vector and backbone-frame
    pair mask; pair features are zero between chains."""
    same_chain = (asym_id[:, None] == asym_id[None, :])[..., None]
    restype = jax.nn.one_hot(aatype, 32)
    n = aatype.shape[-1]
    return jnp.concatenate(
        [
            distogram * same_chain,
            pseudo_beta_mask[..., None] * same_chain,
            jnp.broadcast_to(restype[:, None, :, :], (*aatype.shape, n, 32)),
            jnp.broadcast_to(restype[:, :, None, :], (*aatype.shape, n, 32)),
            unit_vector * same_chain,
            backbone_frame_mask[..., None] * same_chain,
        ],
        axis=-1,
    )


def features(f: dict) -> Features:
    """`jfold.model.Features` from a Protenix feature dict (`protenix.data` featurization;
    numpy or JAX arrays, unbatched). Traceable: `restype` and `profile` may be soft, `msa`
    may be residue probabilities `[S, T, 32]` instead of indices, and the optional
    `atom_pad_mask` and `token_pad_mask` mark padding (e.g. atom slots left empty when a
    residue is swapped for one with fewer atoms). Without `msa` (or `template_aatype`) keys
    the trunk skips the MSA module (or the template embedder)."""
    a = arrays(f)
    num_atoms, num_tokens = a["ref_pos"].shape[0], a["asym_id"].shape[0]
    ref_mask = a["ref_mask"].astype(jnp.float32)
    atom_mask = a.get("atom_pad_mask", jnp.ones(num_atoms, bool)).astype(bool)
    token_mask = a.get("token_pad_mask", jnp.ones(num_tokens, bool)).astype(bool)
    msa = msa_mask = templates = template_mask = None
    if "msa" in a:
        msa = a["msa"]
        msa = msa_features(
            probabilities(msa, 32),
            a["has_deletion"].astype(jnp.float32),
            a["deletion_value"].astype(jnp.float32),
        )
        msa_mask = jnp.ones(msa.shape[:2], bool)
    if "template_aatype" in a:
        templates = template_features(
            a["template_distogram"],
            a["template_pseudo_beta_mask"],
            a["template_aatype"],
            a["template_unit_vector"],
            a["template_backbone_frame_mask"],
            a["asym_id"],
        )
        template_mask = jnp.ones(a["template_aatype"].shape[0], bool)
    return Features(
        atoms=Atoms(
            features=atom_features(
                a["ref_pos"],
                a["ref_charge"].astype(jnp.float32),
                ref_mask,
                a["ref_element"].astype(jnp.float32),
                a["ref_atom_name_chars"].astype(jnp.float32),
            ),
            ref_pos=a["ref_pos"],
            ref_space_uid=a["ref_space_uid"],
            atom_to_token=a["atom_to_token_idx"],
            mask=atom_mask,
        ),
        token_features=token_features(
            a["restype"].astype(jnp.float32),
            a["profile"].astype(jnp.float32),
            a["deletion_mean"].astype(jnp.float32),
        ),
        relative_positions=relative_position_features(
            a["asym_id"], a["residue_index"], a["entity_id"], a["token_index"], a["sym_id"]
        ),
        token_mask=token_mask,
        asym_id=a["asym_id"],
        rep_atom=jnp.nonzero(a["distogram_rep_atom_mask"], size=num_tokens)[0],
        atom_position=a["atom_to_tokatom_idx"],
        bonds=a["token_bonds"].astype(jnp.float32)[..., None],
        msa_features=msa,
        msa_mask=msa_mask,
        template_features=templates,
        template_mask=template_mask,
    )
