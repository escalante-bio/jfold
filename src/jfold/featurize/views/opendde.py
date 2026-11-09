"""OpenDDE's features, as its inference featurizer (`opendde.data.inference.infer_dataloader
.InferenceDataset.process_one`, `opendde_v1` config, templates off) produces them for
proteins (see `docs/upstream-featurization/protenix.md`, section 3): the dict `jfold.models.opendde
.features` takes, plus upstream's frame, PAE-mask and residue-level keys.

As Mol-1, except: OpenDDE's own reference conformers, posed by rotating then translating;
the structural tokens its diffusion runs on (each residue split into a backbone token and a
side-chain token, glycine kept whole); and no templates: OpenDDE's inference takes them only
as search hits on PDB entries (`templatesPath`, an `.a3m` or `.hhr`), realigned with kalign,
not as a given structure, so the view refuses them (section 3.1)."""

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

# `STRUCTURAL_TOKEN_ROLES`: atom 0, protein backbone 1, protein side chain 2, ...
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
# Further keys of upstream's dict: token frames, the PAE mask and the residue-level features.
KEYS = (
    *JFOLD_KEYS, "has_frame", "frame_atom_index", "is_ligand", "pae_rep_atom_mask",
    *STRUCTURAL_KEYS,
)  # fmt: skip


def structural_tokens(lay: Layout, frame_atom_index: np.ndarray) -> dict[str, np.ndarray]:
    """`tokenize_structural` for protein residues: a backbone token (`N CA C O OXT`, centre
    CA) then a side-chain token (the other atoms, centre CB), or one backbone token holding
    all of a glycine's atoms. Side-chain tokens inherit their residue's frame; a residue's
    neighbours are the previous and next residues of its chain."""
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
    """OpenDDE's feature dict for `system`. `key`: draw each residue's reference-conformer
    pose (rotate, then translate); None leaves the conformers centred and unposed.
    `use_msa`: the config's `use_msa`; by default, whether any chain has an MSA."""
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
    """What each real atom of `features(system)` is (chains in input order, `OXT` on each
    chain's last residue). `token` indexes residues: the tokens of the trunk, distogram and
    confidence heads (`atom_to_token_idx`); the structural tokens are internal to diffusion."""
    return _atom_sites(system.chains, residues("opendde"), oxt=True)
