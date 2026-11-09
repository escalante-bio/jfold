"""Protenix's features, as `protenix.data.template.featurize` produces them for proteins (see
`docs/upstream-featurization/protenix.md`): the dict `jfold.models.protenix.features` takes, plus
upstream's frame, confidence-mask, molecule-type and bond keys. A chain's template is placed
residue by residue (`protenix_templates`).

Also the Protenix-lineage code the Mol-1 and OpenDDE views share: the atom layout (CCD
heavy atoms, `OXT` on each chain's last residue), the reference-conformer encodings and pose,
the token ids, both MSA conventions (Protenix's, and Mol-1's, which OpenDDE shares) and the
template pair features.

The only random step upstream is each residue's reference-conformer pose, drawn from numpy's
global RNG. Here it is drawn from a JAX `key` with the same distribution; without a key the
conformers are left centred and unposed."""

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import numpy as np

from jfold.featurize.a3m import parse_a3m
from jfold.featurize.residues import ResidueTable
from jfold.featurize.sites import AtomSites
from jfold.featurize.sites import atom_sites as _atom_sites
from jfold.featurize.system import AMINO_ACIDS, THREE_LETTER, Chain, System, Template

# `STD_RESIDUES_WITH_GAP`: the 20 amino acids in `AMINO_ACIDS` order, UNK, RNA, DNA, gap.
NUM_CLASSES = 32
UNK, GAP = 20, 31
RESTYPE = {aa: i for i, aa in enumerate(AMINO_ACIDS)} | {"X": UNK}
# MSA characters (`HHBLITS_AA_TO_ID` then `correct_msa_restypes`; Mol-1's
# `MSA_PROTEIN_SEQ_TO_ID`): every uppercase letter and the gap.
MSA_CLASS = RESTYPE | {"B": 3, "Z": 6, "U": 4, "J": UNK, "O": UNK, "-": GAP}
MAX_MSA_ROWS = 16384
MAX_PAIRED_ROWS = MAX_MSA_ROWS // 2
SHORT_CHAIN = 4  # Mol-1 and OpenDDE ignore the MSA of chains this short or shorter
BACKBONE = ("N", "CA", "C", "O", "OXT")
NUM_ELEMENTS = 128
NUM_NAME_CHARS = 64
NUM_TEMPLATE_BINS = 39


def residues(name: str) -> ResidueTable:
    """The residue table `name` (`protenix`, `mol1` or `opendde`)."""
    return ResidueTable.load(name)


@dataclass(frozen=True)
class Layout:
    """The tokens (one per residue, chains in input order) and their atoms: each residue's
    table atoms in CCD order, without `OXT` except on a chain's last residue."""

    sequence: str  # [T] one-letter codes, chains concatenated
    chain: np.ndarray  # [T] chain index
    res_id: np.ndarray  # [T] 1-based position in the chain
    atom_start: np.ndarray  # [T + 1] first atom of each token
    names: tuple[str, ...]  # [N]
    element: np.ndarray  # [N] atomic number
    charge: np.ndarray  # [N] formal charge
    ref_mask: np.ndarray  # [N]
    conformer: np.ndarray  # [N, 3] float32, the residue's reference conformer, unposed
    atom_to_token: np.ndarray  # [N]
    tokatom: np.ndarray  # [N] position in the residue's CCD atom order

    @property
    def num_tokens(self) -> int:
        return len(self.sequence)

    def atom(self, token: int, name: str) -> int:
        """The global index of `token`'s atom `name`."""
        start, stop = self.atom_start[token], self.atom_start[token + 1]
        return int(start + self.names[start:stop].index(name))


def layout(system: System, table: ResidueTable) -> Layout:
    chain, res_id, names, element, charge, mask, coords, tokatom = ([] for _ in range(8))
    atom_start = [0]
    for c, ch in enumerate(system.chains):
        for i, letter in enumerate(ch.sequence):
            r = table[THREE_LETTER[letter]]
            keep = [j for j, a in enumerate(r.atoms) if a != "OXT" or i == len(ch.sequence) - 1]
            chain.append(c)
            res_id.append(i + 1)
            names += [r.atoms[j] for j in keep]
            element.append(r.elements[keep])
            charge.append(r.charges[keep])
            mask.append(r.extra["ref_mask"][keep])
            coords.append(r.conformers[0][keep])
            tokatom += keep
            atom_start.append(atom_start[-1] + len(keep))
    atom_start = np.array(atom_start, np.int64)
    return Layout(
        sequence="".join(ch.sequence for ch in system.chains),
        chain=np.array(chain, np.int64),
        res_id=np.array(res_id, np.int64),
        atom_start=atom_start,
        names=tuple(names),
        element=np.concatenate(element).astype(np.int64),
        charge=np.concatenate(charge).astype(np.int64),
        ref_mask=np.concatenate(mask).astype(np.int64),
        conformer=np.concatenate(coords).astype(np.float32),
        atom_to_token=np.repeat(np.arange(len(chain), dtype=np.int64), np.diff(atom_start)),
        tokatom=np.array(tokatom, np.int64),
    )


# Reference-conformer pose.


def centred(lay: Layout) -> np.ndarray:
    """Each residue's conformer minus its mean over the residue's atoms (`OXT` included),
    in float32, as `random_transform(..., centralize=True)`."""
    return np.concatenate(
        [
            lay.conformer[start:stop] - lay.conformer[start:stop].mean(axis=0)
            for start, stop in zip(lay.atom_start[:-1], lay.atom_start[1:], strict=True)
        ]
    )


def rotation_matrices(q: np.ndarray) -> np.ndarray:
    """Rotation matrices `[..., 3, 3]` of quaternions `[..., 4]` in scalar-last `(x, y, z,
    w)` order, normalized first: scipy's `Rotation.from_quat(q).as_matrix()`."""
    q = np.asarray(q, np.float64)
    x, y, z, w = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    xx, yy, zz, ww = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    return np.stack(
        [
            np.stack([xx - yy - zz + ww, 2 * (xy - zw), 2 * (xz + yw)], -1),
            np.stack([2 * (xy + zw), -xx + yy - zz + ww, 2 * (yz - xw)], -1),
            np.stack([2 * (xz - yw), 2 * (yz + xw), -xx - yy + zz + ww], -1),
        ],
        -2,
    )


def pose(
    positions: np.ndarray,
    atom_to_token: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
    *,
    translate_first: bool,
) -> np.ndarray:
    """Each residue's centred conformer moved by its own translation `[T, 3]` and rotation
    `[T, 3, 3]`, in float64 then cast to float32 as upstream: `(p + t) @ R.T` (Protenix,
    Mol-1) or `p @ R.T + t` (OpenDDE)."""
    p = positions.astype(np.float64)
    t, r = translation[atom_to_token], rotation[atom_to_token]
    if translate_first:
        out = np.einsum("nij,nj->ni", r, p + t)
    else:
        out = np.einsum("nij,nj->ni", r, p) + t
    return out.astype(np.float32)


def random_pose(key: jax.Array, num_tokens: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-residue translations `U(-1, 1)^3` and rotations from normalized `N(0, 1)^4`
    quaternions (`Rotation.random()`), as upstream draws them, from a JAX key."""
    k_translation, k_rotation = jax.random.split(key)
    t = jax.random.uniform(k_translation, (num_tokens, 3), minval=-1.0, maxval=1.0)
    q = jax.random.normal(k_rotation, (num_tokens, 4))
    return np.array(t, np.float64), rotation_matrices(np.array(q, np.float64))


def ref_pos(lay: Layout, key: jax.Array | None, *, translate_first: bool) -> np.ndarray:
    positions = centred(lay)
    if key is None:
        return positions
    t, r = random_pose(key, lay.num_tokens)
    return pose(positions, lay.atom_to_token, t, r, translate_first=translate_first)


# Residue and atom features.


def one_hot(index: np.ndarray, num_classes: int, dtype=np.int64) -> np.ndarray:
    return (np.asarray(index)[..., None] == np.arange(num_classes)).astype(dtype)


def residue_features(
    system: System, lay: Layout, key: jax.Array | None, *, translate_first: bool
) -> dict[str, np.ndarray]:
    """The features every Protenix-lineage pipeline computes alike for `system` laid out as
    `lay`: tokens, ids, atoms and their reference conformers, representative atoms and
    frames."""
    num_tokens, num_atoms = lay.num_tokens, len(lay.names)
    restype = np.array([RESTYPE[c] for c in lay.sequence])
    # Every chain is its own entity, `label_entity_id = str(i + 1)`, ranked as strings.
    labels = [str(i + 1) for i in range(len(system.chains))]
    entity = np.array([sorted(labels).index(label) for label in labels], np.int64)
    ca = np.array([lay.atom(t, "CA") for t in range(num_tokens)])
    rep = np.array([lay.atom(t, "CA" if c == "G" else "CB") for t, c in enumerate(lay.sequence)])
    frame = np.array([[lay.atom(t, a) for a in ("N", "CA", "C")] for t in range(num_tokens)])
    names = np.array([[ord(c) - 32 for c in n.ljust(4)] for n in lay.names])
    atom_chain = lay.chain[lay.atom_to_token]
    chain_start = np.searchsorted(atom_chain, atom_chain)  # first atom of the atom's chain

    def atom_mask(indices):
        mask = np.zeros(num_atoms, np.int64)
        mask[indices] = 1
        return mask

    return {
        "token_index": np.arange(num_tokens, dtype=np.int64),
        "residue_index": lay.res_id,
        "asym_id": lay.chain,
        "entity_id": entity[lay.chain],
        "sym_id": np.zeros(num_tokens, np.int64),
        "restype": one_hot(restype, NUM_CLASSES, np.float32),
        "token_bonds": np.zeros((num_tokens, num_tokens), np.float32),
        "ref_pos": ref_pos(lay, key, translate_first=translate_first),
        "ref_mask": lay.ref_mask,
        "ref_element": one_hot(lay.element - 1, NUM_ELEMENTS),
        "ref_charge": lay.charge,
        "ref_atom_name_chars": one_hot(names, NUM_NAME_CHARS),
        "ref_space_uid": lay.atom_to_token,
        "has_frame": np.ones(num_tokens, np.int64),
        "frame_atom_index": frame.astype(np.int64),
        "atom_to_token_idx": lay.atom_to_token,
        "atom_to_tokatom_idx": lay.tokatom,
        "is_protein": np.ones(num_atoms, np.int64),
        "is_ligand": np.zeros(num_atoms, np.int64),
        "is_dna": np.zeros(num_atoms, np.int64),
        "is_rna": np.zeros(num_atoms, np.int64),
        "mol_id": atom_chain,
        "mol_atom_index": np.arange(num_atoms, dtype=np.int64) - chain_start,
        "entity_mol_id": atom_chain,
        "pae_rep_atom_mask": atom_mask(ca),
        "plddt_m_rep_atom_mask": atom_mask(ca),
        "distogram_rep_atom_mask": atom_mask(rep),
        "modified_res_mask": np.zeros(num_atoms, np.int64),
        "bond_mask": np.zeros((num_atoms, num_atoms), np.int64),
    }


# MSAs.


def encode(row: str) -> list[int]:
    """MSA classes of an aligned row; any other character is an error, as in Protenix."""
    try:
        return [MSA_CLASS[c] for c in row]
    except KeyError as e:
        raise ValueError(f"unsupported MSA character {e.args[0]!r} in {row!r}") from None


def _profile(msa: np.ndarray) -> np.ndarray:
    """Column class frequencies `[L, 32]` (float64)."""
    return one_hot(msa, NUM_CLASSES).sum(0) / msa.shape[0]


def _pad_rows(blocks: list[np.ndarray], value) -> np.ndarray:
    """Per-chain row blocks side by side, shallower chains padded with `value` rows
    (`dense_max`)."""
    depth = max(b.shape[0] for b in blocks)
    return np.concatenate(
        [np.pad(b, ((0, depth - b.shape[0]), (0, 0)), constant_values=value) for b in blocks],
        axis=1,
    )


def protenix_msa(system: System) -> dict[str, np.ndarray]:
    """Protenix's MSA: per distinct sequence (the first chain with it supplies the a3m, or
    the sequence alone), the a3m's rows as written, deduplicated; profile and deletion mean
    over those rows. One distinct sequence: those rows, chains side by side. Several: the
    joint query row, then each chain's rows other than its query."""
    chains = system.chains
    first: dict[str, Chain] = {}
    for c in chains:
        first.setdefault(c.sequence, c)
    per_sequence = {}
    for sequence, chain in first.items():
        if chain.msa is None:
            rows, deletions = (sequence,), np.zeros((1, len(sequence)), np.int64)
        else:
            a3m = parse_a3m(chain.msa)
            rows, deletions = a3m.rows, a3m.deletions
        seen, keep = set(), []
        for i, row in enumerate(rows):
            if row not in seen:
                seen.add(row)
                keep.append(i)
        msa = np.array([encode(rows[i]) for i in keep], np.int64)
        deletions = deletions[keep].astype(np.float32)
        per_sequence[sequence] = (msa, deletions)

    profile = np.concatenate([_profile(per_sequence[c.sequence][0]) for c in chains])
    deletion_mean = np.concatenate([per_sequence[c.sequence][1].mean(0) for c in chains])
    if len(first) == 1:
        paired = []
        unpaired = [per_sequence[c.sequence] for c in chains]
        depth = MAX_MSA_ROWS
    else:
        query = [np.array(encode(c.sequence), np.int64) for c in chains]
        paired = [(np.concatenate(query)[None], np.zeros((1, len(profile)), np.float32))]
        unpaired = []
        for c, q in zip(chains, query, strict=True):
            msa, deletions = per_sequence[c.sequence]
            keep = ~(msa == q).all(1)
            unpaired.append((msa[keep], deletions[keep]))
        depth = MAX_MSA_ROWS - 1  # less the query, the one non-gap paired row
    msa = np.concatenate(
        [p for p, _ in paired] + [_pad_rows([m[:depth] for m, _ in unpaired], GAP)]
    )[:MAX_MSA_ROWS]
    deletions = np.concatenate(
        [d for _, d in paired] + [_pad_rows([d[:depth] for _, d in unpaired], 0)]
    )[:MAX_MSA_ROWS]
    return {
        "msa": msa,
        "has_deletion": np.clip(deletions, 0, 1).astype(bool),
        "deletion_value": ((2 / np.pi) * np.arctan(deletions / 3)).astype(np.float32),
        "profile": profile,
        "deletion_mean": deletion_mean.astype(np.float64),
    }


def _mol1_rows(chain: Chain, use_a3m: bool) -> tuple[np.ndarray, np.ndarray]:
    """A chain's unpaired MSA by Mol-1's rules (`RawMsa`): the a3m's records deduplicated
    (lowercase ignored), or the sequence alone; every character outside the alphabet is an
    insertion; rows padded with 0 (ALA) or cut to the first row's length."""
    if not use_a3m or chain.msa is None:
        return np.array([encode(chain.sequence)], np.int64), np.zeros(
            (1, len(chain.sequence)), np.int64
        )
    a3m = parse_a3m(chain.msa)
    seen, rows = set(), []
    for row, counts in zip(a3m.rows, a3m.deletions, strict=True):
        if row in seen:
            continue
        seen.add(row)
        classes, deletions, pending = [], [], 0
        for c, count in zip(row, counts, strict=True):
            pending += int(count)
            if c in MSA_CLASS:
                classes.append(MSA_CLASS[c])
                deletions.append(pending)
                pending = 0
            else:
                pending += 1
        rows.append((classes, deletions))
    width = len(rows[0][0])
    if width != len(chain.sequence):
        raise ValueError(f"chain {chain.id}: the a3m's first row is not the query's length")
    msa = np.zeros((len(rows), width), np.int64)
    deletions = np.zeros((len(rows), width), np.int64)
    for i, (classes, counts) in enumerate(rows):
        n = min(len(classes), width)
        msa[i, :n], deletions[i, :n] = classes[:n], counts[:n]
    return msa, deletions


def mol1_msa(system: System, use_msa: bool) -> dict[str, np.ndarray]:
    """Mol-1's (and OpenDDE's) MSA. With `use_msa`: per chain, a paired block that is the
    query alone and its unpaired rows (`_mol1_rows`; chains of 4 residues or fewer use the
    query alone), with profile and deletion mean over the unpaired rows. The paired block
    comes first, so with one distinct sequence the query appears twice; with several, each
    chain's unpaired rows equal to its query are dropped. Deletion counts are clipped to
    int8 for the per-row features. Without `use_msa`: the sequence as the only row."""
    chains = system.chains
    if not use_msa:
        restype = np.array([RESTYPE[c] for ch in chains for c in ch.sequence])
        n = len(restype)
        return {
            "msa": restype[None].astype(np.int64),
            "has_deletion": np.zeros((1, n), np.float32),
            "deletion_value": np.zeros((1, n), np.float32),
            "profile": one_hot(restype, NUM_CLASSES, np.float32),
            "deletion_mean": np.zeros(n, np.float32),
        }
    need_pairing = len({c.sequence for c in chains}) > 1
    query = [np.array(encode(c.sequence), np.int64)[None] for c in chains]
    unpaired, profile, deletion_mean = [], [], []
    for c, q in zip(chains, query, strict=True):
        msa, deletions = _mol1_rows(c, use_a3m=len(c.sequence) > SHORT_CHAIN)
        profile.append(_profile(msa))
        deletion_mean.append(deletions.mean(0))
        if need_pairing:
            keep = ~(msa.astype(np.int8) == q.astype(np.int8)).all(1)
            msa, deletions = msa[keep], deletions[keep]
        depth = max(0, min(msa.shape[0], MAX_MSA_ROWS - min(1, MAX_PAIRED_ROWS)))
        unpaired.append((msa[:depth], deletions[:depth]))
    msa = np.concatenate([np.concatenate(query, axis=1), _pad_rows([m for m, _ in unpaired], GAP)])
    deletions = np.concatenate([np.zeros_like(msa[:1]), _pad_rows([d for _, d in unpaired], 0)])
    deletions = np.clip(deletions, -128, 127).astype(np.int8)
    return {
        "msa": np.clip(msa, -128, 127).astype(np.int8).astype(np.int64),
        "has_deletion": np.clip(deletions, 0.0, 1.0).astype(np.float32),
        "deletion_value": (np.arctan(deletions / 3.0) * (2.0 / np.pi)).astype(np.float32),
        "profile": np.concatenate(profile).astype(np.float32),
        "deletion_mean": np.concatenate(deletion_mean).astype(np.float32),
    }


# Templates.

GLY = RESTYPE["G"]
# AlphaFold's atom37 names (`residue_constants.atom_types`), in order.
ATOM37 = (
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1", "CD2",
    "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1",
    "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
)  # fmt: skip
MAX_CA_DISTANCE = 150.0  # Mol-1 drops a template with consecutive CAs farther apart


def empty_templates(num_tokens: int, aatype: list[int]) -> dict[str, np.ndarray]:
    """Template features with no templates: one slot per entry of `aatype` (the residue
    class every column of that slot gets), everything else zero."""
    n = len(aatype)
    return {
        "template_aatype": np.repeat(np.array(aatype, np.int64)[:, None], num_tokens, 1),
        "template_distogram": np.zeros((n, num_tokens, num_tokens, NUM_TEMPLATE_BINS), np.float32),
        "template_pseudo_beta_mask": np.zeros((n, num_tokens, num_tokens), np.float32),
        "template_unit_vector": np.zeros((n, num_tokens, num_tokens, 3), np.float32),
        "template_backbone_frame_mask": np.zeros((n, num_tokens, num_tokens), np.float32),
    }


# Mol-1 and OpenDDE: one gap slot, then three slots of zero padding (ALA).
MOL1_TEMPLATE_AATYPE = [GAP, 0, 0, 0]


def _positional(chain: Chain) -> Template:
    """`chain`'s template, which must have one residue per residue of the chain."""
    template = chain.template
    if len(template.sequence) != len(chain.sequence):
        raise ValueError(
            f"chain {chain.id}: the template has {len(template.sequence)} residues, the chain "
            f"{len(chain.sequence)} (each template residue stands for the chain's residue at "
            "its position)"
        )
    return template


def _gather(atoms: list[Mapping[str, np.ndarray]], names: list[str]):
    """Each token's atom `names[t]` from its template atoms `atoms[t]`, `[T, 3]` float32,
    and its mask `[T]` float32 (zero where missing)."""
    positions = np.zeros((len(atoms), 3), np.float32)
    mask = np.zeros(len(atoms), np.float32)
    for t, (residue, name) in enumerate(zip(atoms, names, strict=True)):
        if name in residue:
            positions[t], mask[t] = residue[name], 1.0
    return positions, mask


def template_pair_features(
    atoms: list[Mapping[str, np.ndarray]],
    pseudo_beta: list[str],
    backbone: list[tuple[str, str, str]],
    epsilon: float = 1e-6,
) -> dict[str, np.ndarray]:
    """One template slot's pair features from each token's template atoms (none for a
    token without a template residue): the one-hot distogram of the pseudo-beta atoms
    (`pseudo_beta[t]`; 39 bins of squared distance from 3.25 Å to 50.75 Å, the last
    open-ended), and the unit vector from each token's backbone CA to every other's in its
    frame (`backbone[t]` names its N, CA and C), each zeroed where its pair mask is.
    Protenix's `Templates.as_protenix_dict`, in float32 step for step; Mol-1's
    `Templates.as_mol_dict` computes the same bits."""
    pb, pb_mask = _gather(atoms, pseudo_beta)
    (n, n_mask), (ca, ca_mask), (c, c_mask) = (
        _gather(atoms, [names[k] for names in backbone]) for k in range(3)
    )
    pb_mask_2d = pb_mask[:, None] * pb_mask[None, :]
    lower = np.linspace(3.25, 50.75, NUM_TEMPLATE_BINS) ** 2
    upper = np.concatenate([lower[1:], [1e8]])
    dist2 = np.sum((pb[:, None, :] - pb[None, :, :]) ** 2, axis=-1, keepdims=True)
    distogram = ((dist2 > lower) & (dist2 < upper)).astype(np.float32)

    v1, v2 = c - ca, n - ca
    e1 = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + epsilon)
    e2 = v2 - np.sum(v2 * e1, axis=-1, keepdims=True) * e1
    e2 = e2 / (np.linalg.norm(e2, axis=-1, keepdims=True) + epsilon)
    e3 = np.cross(e1, e2)
    diff = ca[None, :, :] - ca[:, None, :]
    unit = np.stack([np.sum(e[:, None, :] * diff, axis=-1) for e in (e1, e2, e3)], axis=-1)
    unit = unit / (np.linalg.norm(unit, axis=-1, keepdims=True) + epsilon)
    backbone_mask = n_mask * ca_mask * c_mask
    backbone_mask_2d = backbone_mask[:, None] * backbone_mask[None, :]
    return {
        "template_distogram": distogram * pb_mask_2d[..., None],
        "template_pseudo_beta_mask": pb_mask_2d,
        "template_unit_vector": unit * backbone_mask_2d[..., None],
        "template_backbone_frame_mask": backbone_mask_2d,
    }


def protenix_templates(system: System) -> dict[str, np.ndarray]:
    """Protenix's one template slot (`build_templates_from_chains`,
    `Templates.as_protenix_dict`): each chain's `template` placed residue by residue (it
    must have the chain's length; `_extract_chain_template`), with the template's residue
    types, unresolved residues included; UNK and no atoms on chains without one.
    Pseudo-beta CB (CA for glycine), frames from N, CA and C."""
    num_tokens = sum(len(c.sequence) for c in system.chains)
    if all(c.template is None for c in system.chains):
        return empty_templates(num_tokens, [UNK])
    aatype = np.full(num_tokens, UNK, np.int64)
    atoms: list[Mapping[str, np.ndarray]] = [{}] * num_tokens
    start = 0
    for chain in system.chains:
        if chain.template is not None:
            template = _positional(chain)
            stop = start + len(template.sequence)
            aatype[start:stop] = [RESTYPE[c] for c in template.sequence]
            atoms[start:stop] = template.atoms
        start += len(chain.sequence)
    pair = template_pair_features(
        atoms,
        ["CA" if a == GLY else "CB" for a in aatype],
        [("N", "CA", "C")] * num_tokens,
    )
    return {"template_aatype": aatype[None]} | {k: v[None] for k, v in pair.items()}


def _mol1_centre(chain: Chain, template: Template) -> np.ndarray:
    """The mean of the template's atom37 atoms (residue by residue in atom37 order, an
    arginine's NH1 and NH2 swapped when NH1 is the farther from CD), which Mol-1 subtracts
    from its coordinates (`TemplateHitProcessor._get_atom_coords`); after checking, as
    `_check_residue_distances` does, that consecutive resolved CAs are within 150 Å."""
    coords = []
    for letter, residue in zip(template.sequence, template.atoms, strict=True):
        if (
            letter == "R"
            and {"CD", "NH1", "NH2"} <= residue.keys()
            and np.linalg.norm(residue["NH1"] - residue["CD"])
            > np.linalg.norm(residue["NH2"] - residue["CD"])
        ):
            residue = {**residue, "NH1": residue["NH2"], "NH2": residue["NH1"]}
        coords += [residue[name] for name in ATOM37 if name in residue]
    ca = np.array([residue["CA"] for residue in template.atoms if "CA" in residue])
    if not coords or (
        len(ca) > 1 and np.linalg.norm(ca[1:] - ca[:-1], axis=-1).max() > MAX_CA_DISTANCE
    ):
        raise ValueError(
            f"chain {chain.id}: Mol-1 drops a template without atoms or with consecutive "
            f"CAs over {MAX_CA_DISTANCE:g} Å apart"
        )
    return np.array(coords, np.float32).mean(axis=0)


def mol1_templates(system: System) -> dict[str, np.ndarray]:
    """Mol-1's four template slots (`InferenceTemplateFeaturizer.make_template_feature`,
    `Templates.as_mol_dict`) with each chain's `template` given as an AF3-style template
    JSON (`templatesPath`; `TemplateHitFeaturizer.parse_json_templates`): its resolved
    residues as the mmCIF, each mapped to the chain residue at its position (the template
    must have the chain's length). Templates of chains of 4 residues or fewer are ignored.

    Slot 0: the template's residue types, gap where unresolved; coordinates centred on the
    template's atoms (`_mol1_centre`); pseudo-beta CB (CA for glycine) and frames from N,
    CA and C, except UNK, whose dense atom slots all point at its N (pseudo-beta N, frame
    N-N-N). Chains without a template: gap and no atoms. Slots 1-3: zero (ALA)."""
    num_tokens = sum(len(c.sequence) for c in system.chains)
    out = empty_templates(num_tokens, MOL1_TEMPLATE_AATYPE)
    templated = [c.template is not None and len(c.sequence) > SHORT_CHAIN for c in system.chains]
    if not any(templated):
        return out
    aatype = out["template_aatype"][0]
    atoms: list[Mapping[str, np.ndarray]] = [{}] * num_tokens
    start = 0
    for chain, use in zip(system.chains, templated, strict=True):
        if use:
            template = _positional(chain)
            centre = _mol1_centre(chain, template)
            for i, (letter, residue) in enumerate(zip(template.sequence, template.atoms)):
                if residue:
                    aatype[start + i] = RESTYPE[letter]
                    atoms[start + i] = {name: xyz - centre for name, xyz in residue.items()}
        start += len(chain.sequence)
    pair = template_pair_features(
        atoms,
        [{GLY: "CA", UNK: "N"}.get(a, "CB") for a in aatype],
        [("N", "N", "N") if a == UNK else ("N", "CA", "C") for a in aatype],
    )
    for key, value in pair.items():
        out[key][0] = value
    return out


# The keys `jfold.models.protenix.features` reads (Mol-1 and OpenDDE included).
JFOLD_KEYS = (
    "ref_pos", "ref_charge", "ref_mask", "ref_element", "ref_atom_name_chars",
    "ref_space_uid", "atom_to_token_idx", "atom_to_tokatom_idx", "restype", "profile",
    "deletion_mean", "asym_id", "residue_index", "entity_id", "token_index", "sym_id",
    "distogram_rep_atom_mask", "token_bonds", "msa", "has_deletion", "deletion_value",
    "template_aatype", "template_distogram", "template_pseudo_beta_mask",
    "template_unit_vector", "template_backbone_frame_mask",
)  # fmt: skip
# Further keys of upstream's dict: token frames, the pLDDT and PAE representative atoms,
# molecule types and ids, and atom bonds.
EXTRA_KEYS = (
    "atom_rep_atom_idx", "has_frame", "frame_atom_index", "is_protein", "is_ligand",
    "is_dna", "is_rna", "mol_id", "mol_atom_index", "entity_mol_id", "pae_rep_atom_mask",
    "plddt_m_rep_atom_mask", "modified_res_mask", "bond_mask",
)  # fmt: skip


def features(system: System, *, key: jax.Array | None = None) -> dict[str, np.ndarray]:
    """Protenix's feature dict for `system`, chains' templates included. `key`: draw each
    residue's reference-conformer pose (translate, then rotate); None leaves the conformers
    centred and unposed."""
    lay = layout(system, residues("protenix"))
    f = residue_features(system, lay, key, translate_first=True)
    f |= protenix_msa(system)
    f |= protenix_templates(system)
    f["atom_rep_atom_idx"] = np.flatnonzero(f["distogram_rep_atom_mask"]).astype(np.int64)
    return {k: f[k] for k in JFOLD_KEYS + EXTRA_KEYS}


def atom_sites(system: System) -> AtomSites:
    """What each real atom of `features(system)` is (chains in input order, `OXT` on each chain's last residue)."""
    return _atom_sites(system.chains, residues("protenix"), oxt=True)
