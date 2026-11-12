"""Featurize proteins for OpenDDE, which follows Mol-1 and adds the structural tokens its
diffusion runs on. Templates are rejected because OpenDDE's inference only takes them as
search hits to realign. See `docs/upstream-featurization/protenix.md`, section 3."""

import jax
import numpy as np

from jfold.featurize.sites import AtomSites
from jfold.featurize.sites import atom_sites as _atom_sites
from jfold.featurize.system import System
from jfold.featurize.views.protenix import (
    BACKBONE,
    JFOLD_KEYS,
    MOL1_TEMPLATE_AATYPE,
    Layout,
    empty_templates,
    layout,
    mol1_msa,
    residue_features,
    residues,
)

# Role ids from upstream's `STRUCTURAL_TOKEN_ROLES`.
PROTEIN_BB, PROTEIN_SC = 1, 2
NO_TWIN = -1

STRUCTURAL_KEYS = (
    "structural_token_index", "residue_token_group_id", "subtoken_role", "subtoken_role_id",
    "twin_token_idx", "parent_residue_idx", "atom_to_structural_token_idx",
    "atom_to_structural_tokatom_idx", "structural_distogram_rep_atom_mask",
    "structural_pae_rep_atom_mask", "structural_has_frame", "structural_frame_atom_index",
    "prev_parent_residue_idx", "next_parent_residue_idx", "structural_is_polymer",
    "structural_polymer_type", "structural_seq_pos",
)  # fmt: skip
KEYS = (
    *JFOLD_KEYS, "has_frame", "frame_atom_index", "is_ligand", "pae_rep_atom_mask",
    *STRUCTURAL_KEYS,
)  # fmt: skip


def structural_tokens(lay: Layout, frame_atom_index: np.ndarray) -> dict[str, np.ndarray]:
    """Split each protein residue into a backbone and a side-chain token as
    `tokenize_structural` does, keeping glycine whole."""
    num_atoms = len(lay.names)
    parent, role, twin = [], [], []
    atom_token = np.zeros(num_atoms, np.int64)
    atom_position = np.zeros(num_atoms, np.int64)
    centre = np.zeros(num_atoms, np.int64)
    for t, letter in enumerate(lay.sequence):
        atoms = range(lay.atom_start[t], lay.atom_start[t + 1])
        if letter == "G":
            groups = [(PROTEIN_BB, list(atoms), ("CA", "N", "C"))]
        else:
            groups = [
                (PROTEIN_BB, [i for i in atoms if lay.names[i] in BACKBONE], ("CA", "N", "C")),
                (PROTEIN_SC, [i for i in atoms if lay.names[i] not in BACKBONE], ("CB",)),
            ]
        first = len(parent)
        for r, members, centres in groups:
            index = len(parent)
            parent.append(t)
            role.append(r)
            twin.append(NO_TWIN if len(groups) == 1 else 2 * first + 1 - index)
            atom_token[members] = index
            atom_position[members] = np.arange(len(members))
            names = [lay.names[i] for i in members]
            centre[members[next((names.index(c) for c in centres if c in names), 0)]] = 1
    parent = np.array(parent, np.int64)
    role = np.array(role, np.int64)
    num = len(parent)
    residue = np.arange(lay.num_tokens)
    same_as_prev = np.concatenate([[False], lay.chain[1:] == lay.chain[:-1]])
    same_as_next = np.concatenate([lay.chain[:-1] == lay.chain[1:], [False]])
    return {
        "structural_token_index": np.arange(num, dtype=np.int64),
        "residue_token_group_id": parent,
        "subtoken_role": role,
        "subtoken_role_id": role.copy(),
        "twin_token_idx": np.array(twin, np.int64),
        "parent_residue_idx": parent.copy(),
        "atom_to_structural_token_idx": atom_token,
        "atom_to_structural_tokatom_idx": atom_position,
        "structural_distogram_rep_atom_mask": centre,
        "structural_pae_rep_atom_mask": centre.copy(),
        "structural_has_frame": np.ones(num, np.int64),
        "structural_frame_atom_index": frame_atom_index[parent],
        "prev_parent_residue_idx": np.where(same_as_prev, residue - 1, -1)[parent],
        "next_parent_residue_idx": np.where(same_as_next, residue + 1, -1)[parent],
        "structural_is_polymer": np.ones(num, np.int64),
        "structural_polymer_type": np.ones(num, np.int64),  # protein
        "structural_seq_pos": lay.res_id[parent],
    }


def features(
    system: System, *, key: jax.Array | None = None, use_msa: bool | None = None
) -> dict[str, np.ndarray]:
    """Return OpenDDE's feature dict, randomly rotating then translating each reference
    conformer if `key` is given."""
    system = system.without_templates("OpenDDE")
    if use_msa is None:
        use_msa = any(c.msa is not None for c in system.chains)
    lay = layout(system, residues("opendde"))
    f = residue_features(system, lay, key, translate_first=False)
    f |= mol1_msa(system, use_msa)
    f |= empty_templates(lay.num_tokens, MOL1_TEMPLATE_AATYPE)
    f |= structural_tokens(lay, f["frame_atom_index"])
    return {k: f[k] for k in KEYS}


def atom_sites(system: System) -> AtomSites:
    """Describe each real atom of `features(system)`, with `token` indexing residues rather
    than structural tokens."""
    return _atom_sites(system.chains, residues("opendde"), oxt=True)
