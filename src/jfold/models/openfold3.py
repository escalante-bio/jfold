"""OpenFold3 model pieces."""

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from jfold.blocks.atoms import Atoms
from jfold.blocks.inputs import relative_position_features
from jfold.model import Features
from jfold.models.common import arrays, as_array, msa_features, token_features


def atom_features(
    ref_pos: Float[Array, "N 3"],
    ref_charge: Float[Array, " N"],
    ref_mask: Float[Array, " N"],
    ref_element: Float[Array, "N E"],
    ref_atom_name_chars: Float[Array, "N 4 64"],
) -> Float[Array, "N F"]:
    """The atom feature vector, in the column order `convert.openfold3.atom_encoder` lays
    out: position, arcsinh of the charge, mask, element and name characters."""
    return jnp.concatenate(
        [
            ref_pos,
            jnp.arcsinh(ref_charge)[:, None],
            ref_mask[:, None],
            ref_element,
            ref_atom_name_chars.reshape(ref_pos.shape[0], -1),
        ],
        axis=-1,
    )


def template_features(
    distogram, pseudo_beta_mask, restype, unit_vector, backbone_frame_mask, asym_id
):
    """Each template's pair feature vector `[N, T, T, F]`: distogram, pseudo-beta pair mask,
    residue type of the row then of the column token, unit vector and backbone-frame pair
    mask; the masks are zero between chains."""
    same_chain = (asym_id[:, None] == asym_id[None, :])[..., None]
    pair = lambda m: (m[..., :, None] * m[..., None, :])[..., None] * same_chain
    n = restype.shape[-2]
    shape = (*restype.shape[:-1], n, restype.shape[-1])
    return jnp.concatenate(
        [
            distogram,
            pair(pseudo_beta_mask),
            jnp.broadcast_to(restype[..., :, None, :], shape),
            jnp.broadcast_to(restype[..., None, :, :], shape),
            unit_vector,
            pair(backbone_frame_mask),
        ],
        axis=-1,
    )


# OpenFold3's residue classes (`STANDARD_RESIDUES_WITH_GAP_3`) and, per class, the index
# within the residue of the atoms that can represent a token (-1: the residue has none), from
# `token_atom_constants.atom_name_to_index_by_restype`.
RESIDUES = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
    "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK",
    "A", "G", "C", "U", "N", "DA", "DG", "DC", "DT", "DN", "GAP",
)  # fmt: skip
_ATOM_INDEX = {
    "CB": [4] * 7 + [-1] + [4] * 13 + [-1] * 11,
    "CA": [1] * 21 + [-1] * 11,
    "C4": [-1] * 21 + [21, 22, 16, 16, -1, 20, 21, 15, 15, -1, -1],
    "C2": [-1] * 21 + [19, 19, 13, 13, -1, 18, 18, 12, 12, -1, -1],
}


def representative_atoms(restype, is_protein, is_dna, is_rna, is_atomized, start_atom_index):
    """Each token's representative atom (`get_token_representative_atoms`): CB (CA for
    glycine) for standard residues, C4 for purines, C2 for pyrimidines and the first atom
    of atomized tokens. `restype` is OpenFold3's one-hot `[T, 32]` of the featurized
    residues: the atoms are those of its residues, so for a soft sequence pass the fixed
    layout's own `rep_atom` to `features` instead."""
    restype = restype.astype(jnp.float32)

    def offset(atom):
        return restype @ np.array(_ATOM_INDEX[atom], np.float32)

    def any_of(*residues):
        return restype[:, np.array([RESIDUES.index(r) for r in residues])].sum(-1)

    standard = 1 - is_atomized
    protein, dna, rna = is_protein * standard, is_dna * standard, is_rna * standard
    glycine = protein * any_of("GLY")
    purine = dna * any_of("DA", "DG") + rna * any_of("A", "G")
    pyrimidine = dna * any_of("DC", "DT") + rna * any_of("C", "U")
    index = (
        start_atom_index
        + offset("CB") * protein * (1 - glycine)
        + offset("CA") * glycine
        + offset("C4") * purine
        + offset("C2") * pyrimidine
    ) * (protein + purine + pyrimidine + is_atomized)
    return jnp.round(index).astype(jnp.int32)


def features(batch: dict, rep_atom=None) -> Features:
    """`jfold.model.Features` from an OpenFold3 feature dict (its inference featurization;
    unbatched torch tensors, numpy or JAX arrays). All templates count, empty ones included,
    as in OpenFold3's template average.

    `restype`, `profile` and `msa` may be probabilities over OpenFold3's 32 residue classes
    (soft sequences), and every step is traceable, so the dict may be built inside `jit`.
    Each token's representative atom depends on the featurized atoms, so it is read off
    `restype` only when `rep_atom` is not given: pass it when `restype` is a soft sequence
    over a fixed atom layout."""
    f = arrays(batch)
    flt = lambda k: f[k].astype(jnp.float32)
    num_atoms = f["atom_to_token_index"].shape[0]
    if rep_atom is None:
        rep_atom = representative_atoms(
            f["restype"],
            f["is_protein"],
            f["is_dna"],
            f["is_rna"],
            f["is_atomized"],
            f["start_atom_index"],
        )
    return Features(
        atoms=Atoms(
            features=atom_features(
                flt("ref_pos"),
                flt("ref_charge"),
                flt("ref_mask"),
                flt("ref_element"),
                flt("ref_atom_name_chars"),
            ),
            ref_pos=flt("ref_pos"),
            ref_space_uid=f["ref_space_uid"],
            atom_to_token=f["atom_to_token_index"],
            mask=f["atom_mask"] > 0,
        ),
        token_features=token_features(flt("restype"), flt("profile"), flt("deletion_mean")),
        relative_positions=relative_position_features(
            f["asym_id"], f["residue_index"], f["entity_id"], f["token_index"], f["sym_id"]
        ),
        token_mask=f["token_mask"] > 0,
        asym_id=f["asym_id"],
        rep_atom=jnp.clip(as_array(rep_atom), 0, num_atoms - 1),
        atom_position=jnp.arange(num_atoms) - f["start_atom_index"][f["atom_to_token_index"]],
        bonds=flt("token_bonds")[..., None],
        msa_features=msa_features(flt("msa"), flt("has_deletion"), flt("deletion_value")),
        msa_mask=f["msa_mask"] > 0,
        template_features=template_features(
            flt("template_distogram"),
            flt("template_pseudo_beta_mask"),
            flt("template_restype"),
            flt("template_unit_vector"),
            flt("template_backbone_frame_mask"),
            f["asym_id"],
        ),
        template_mask=jnp.ones(f["template_restype"].shape[0], bool),
    )
