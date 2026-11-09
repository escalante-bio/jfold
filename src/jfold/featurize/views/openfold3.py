"""OpenFold3's features, as its inference featurization produces them for proteins (see
`docs/upstream-featurization/openfold3.md`): the unbatched dict `jfold.models.openfold3.features` takes,
plus upstream's `num_atoms_per_token`. Upstream's
quirks are reproduced: ids ranked by string sort, the query row twice, `.` as an aligned
`X`, the (8/pi)-scaled deletion value and the scrambled profile.

Reference conformers come from a stored table (`jfold/featurize/data/openfold3.npz`): 32
ETKDGv3 conformers per residue, made with upstream's procedure, where upstream embeds a
fresh one per residue. With a `key`, each residue draws one of them and upstream's random
pose (a uniformly random rotation from a normalized Gaussian quaternion, then a unit
Gaussian translation); with `key=None`, every residue takes conformer 0, centred and
unposed. Everything else is deterministic.

Templates (`Chain.template`) are aligned to their chain with Kalign, as upstream realigns a
template named without a residue-wise alignment."""

import warnings

import jax
import jax.numpy as jnp
import kalign
import numpy as np

from jfold.featurize.a3m import parse_a3m
from jfold.featurize.residues import ResidueTable
from jfold.featurize.sites import AtomSites
from jfold.featurize.sites import atom_sites as _atom_sites
from jfold.featurize.system import AMINO_ACIDS, THREE_LETTER, System, Template

# STANDARD_RESIDUES_WITH_GAP: the 20 amino acids (AlphaFold order), UNK, RNA, DNA, GAP.
NUM_CLASSES = 32
UNK, GAP = 20, 31
TOKEN = {a: i for i, a in enumerate(AMINO_ACIDS)} | {"X": UNK}
NUM_ELEMENTS = 119
NUM_TEMPLATES = 4
DISTOGRAM_BINS = 39
# Squared distogram bin edges (`TemplateDistogramSettings`: 3.25 to 50.75 A), the last bin
# open-ended (upstream's 1e8).
DISTOGRAM_LOWER = np.linspace(3.25, 50.75, DISTOGRAM_BINS) ** 2
DISTOGRAM_UPPER = np.append(DISTOGRAM_LOWER[1:], 1e8)
MIN_TEMPLATE_TOKENS = 5
MAX_MSA_ROWS = 16384
PROFILE_CHUNK = 1000


def residues() -> ResidueTable:
    return ResidueTable.load("openfold3")


def _rank(values: list[str]) -> dict[str, int]:
    """1-based rank of each distinct string in lexicographic order."""
    return {v: i + 1 for i, v in enumerate(sorted(set(values)))}


def _classes(rows) -> np.ndarray:
    """`map_str_array_to_idx_array`: amino acids and X to 0-20, `-` to GAP, anything else
    (`.`, `*`, B, Z, ...) to UNK."""
    return np.array(
        [[GAP if c == "-" else TOKEN.get(c, UNK) for c in row] for row in rows], np.int64
    ).reshape(len(rows), -1)


def _profile(classes: np.ndarray) -> np.ndarray:
    """`calculate_profile`, with its index mixup: per chunk of columns, column j's counts are
    those of a run of R consecutive entries of the chunk's row-major flattening."""
    n_rows, n_cols = classes.shape
    counts = np.zeros((n_cols, NUM_CLASSES), np.int64)
    for start in range(0, n_cols, PROFILE_CHUNK):
        chunk = classes[:, start : start + PROFILE_CHUNK]
        cols = chunk.shape[1]
        flat = np.repeat(np.arange(cols), n_rows) * NUM_CLASSES + chunk.ravel()
        counts[start : start + cols] = np.bincount(flat, minlength=cols * NUM_CLASSES).reshape(
            cols, NUM_CLASSES
        )
    return counts / n_rows


def _chain_msa(chain):
    """A chain's MSA rows (query row, then the whole main MSA, so the query twice) as classes
    and deletion counts over its residues, with the main MSA's profile and deletion mean.
    A chain without an MSA gets upstream's dummy alignment: the sequence alone."""
    length = len(chain.sequence)
    if chain.msa is None:
        rows, deletions = [chain.sequence], np.zeros((1, length), np.int64)
    else:
        a3m = parse_a3m(chain.msa)
        rows, deletions = list(a3m.rows), a3m.deletions
        if len(rows[0]) < length:
            raise ValueError(f"chain {chain.id}: a3m has fewer columns than the sequence")
    classes = _classes(rows)
    profile = _profile(classes)[:length]
    deletion_mean = deletions.mean(0)[:length]
    main = slice(0, MAX_MSA_ROWS - 1)
    classes = np.concatenate([classes[:1], classes[main]])[:, :length]
    deletions = np.concatenate([np.zeros_like(deletions[:1]), deletions[main]])[:, :length]
    return classes, deletions, profile, deletion_mean


def _pose(table: ResidueTable, types: list[str], key) -> np.ndarray:
    """Each residue's reference conformer, in token order, `[N_atom, 3]`: with `key`, a
    random table conformer under upstream's random rotation and translation; else
    conformer 0 as stored (centred)."""
    names = sorted(set(types))
    width = max(len(table[n].atoms) for n in names)
    num_conformers = table[names[0]].conformers.shape[0]
    stacked = np.zeros((len(names), num_conformers, width, 3), np.float32)
    for i, n in enumerate(names):
        stacked[i, :, : len(table[n].atoms)] = table[n].conformers
    type_index = np.array([names.index(t) for t in types])
    token = np.repeat(np.arange(len(types)), [len(table[t].atoms) for t in types])
    within = np.concatenate([np.arange(len(table[t].atoms)) for t in types])
    if key is None:
        return stacked[type_index[token], 0, within]
    k_conf, k_rot, k_trans = jax.random.split(key, 3)
    conf = jax.random.randint(k_conf, (len(types),), 0, num_conformers)
    q = jax.random.normal(k_rot, (len(types), 4), jnp.float32)
    a, b, c, d = jnp.moveaxis(q / jnp.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    rot = jnp.stack(
        [
            jnp.stack(
                [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)], -1
            ),
            jnp.stack(
                [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)], -1
            ),
            jnp.stack(
                [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d], -1
            ),
        ],
        -2,
    )
    trans = jax.random.normal(k_trans, (len(types), 3), jnp.float32)
    x = jnp.array(stacked)[type_index[token], conf[token], within]
    posed = jnp.einsum("nij,nj->ni", rot[token], x) + trans[token]
    return np.array(posed, np.float32)


def template_alignment(query: str, template: str) -> np.ndarray | None:
    """Each query residue's template residue (0-based), as upstream realigns a template to
    its chain (`A3mParser(realign=True)`): Kalign's default mode on the query, the query
    again and the template sequence. None when a query residue faces a gap, for which
    upstream drops the template (`map_token_pos_to_template_residues`). A template with the
    query's own sequence aligns to itself without Kalign."""
    if template == query:
        return np.arange(len(query))
    query_row, _, template_row = kalign.align([query, query, template])
    residues, t = [], 0
    for q, c in zip(query_row, template_row, strict=True):
        if q != "-":
            if c == "-":
                return None
            residues.append(t)
        t += c != "-"
    return np.array(residues, np.int64)


def _template_atoms(template: Template, residues: np.ndarray):
    """The template residues `residues` as upstream reads them: residue class, pseudo-beta
    atom (CB, CA for glycine) and N, CA, C, NaN where missing (unresolved residues have no
    atoms)."""
    nan = np.full(3, np.nan)
    restype = [TOKEN[template.sequence[t]] for t in residues]
    pseudo_beta, frame = [], []
    for t in residues:
        atoms = template.atoms[t]
        pseudo_beta.append(atoms.get("CA" if template.sequence[t] == "G" else "CB", nan))
        frame.append([atoms.get(a, nan) for a in ("N", "CA", "C")])
    return np.array(restype), np.array(pseudo_beta, np.float64), np.array(frame, np.float64)


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _normalized(v: np.ndarray) -> np.ndarray:
    """`Vec3Array.normalized`: `v / sqrt(max(|v|^2, 1e-12))`."""
    return v / np.sqrt(np.maximum(_dot(v, v), np.float32(1e-12)))[..., None]


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    y = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    z = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return np.stack([x, y, z], -1)


def unit_vectors(frame: np.ndarray) -> np.ndarray:
    """`[T, T, 3]` float32: the direction of token j's CA in token i's backbone frame, as
    `create_template_unit_vector` computes it (missing atoms at the origin): the frame's
    axes are C-CA normalized, the part of N-CA orthogonal to it, normalized, and their
    cross product; the point is `R_i^T t_j + R_i^T (-t_i)`, then normalized."""
    n, ca, c = np.moveaxis(np.nan_to_num(frame).astype(np.float32), -2, 0)
    e0 = _normalized(c - ca)
    e1 = n - ca
    e1 = _normalized(e1 - _dot(e1, e0)[..., None] * e0)
    axes = np.stack([e0, e1, _cross(e0, e1)], -2)  # rows of R^T, per token
    origin = _dot(axes, -ca[:, None])  # R^T (-t)
    local = _dot(axes[:, None], ca[None, :, None]) + origin[:, None]
    return _normalized(local)


def template_features(system: System) -> dict[str, np.ndarray]:
    """The `template_*` features, as upstream builds them when each templated chain names
    its template structure in a template alignment without a residue-wise alignment (an
    `.m8` hit, say), the structure file in `structure_directory`.

    Templates go by sequence: every chain with a templated chain's sequence gets its
    template (the first, in input order, when several chains with one sequence have one).
    Each is realigned to its chain (`template_alignment`); upstream drops a template that
    leaves a query residue unaligned (a warning here), and chains shorter than 5 residues
    get none. A chain's template fills slot 0 of 4 on its own tokens; distogram and unit
    vectors are zero between chains."""
    chains = system.chains
    num_tokens = sum(len(c.sequence) for c in chains)
    restype = np.full(num_tokens, GAP)
    pseudo_beta = np.full((num_tokens, 3), np.nan)
    frame = np.full((num_tokens, 3, 3), np.nan)
    chain_of_token = np.repeat(np.arange(len(chains)), [len(c.sequence) for c in chains])
    templates: dict[str, Template] = {}
    for chain in chains:
        if chain.template is not None:
            templates.setdefault(chain.sequence, chain.template)
    placed: dict[str, np.ndarray | None] = {}
    start = 0
    for chain in chains:
        tokens = slice(start, start + len(chain.sequence))
        start = tokens.stop
        template = templates.get(chain.sequence)
        if template is None or len(chain.sequence) < MIN_TEMPLATE_TOKENS:
            continue
        if chain.sequence not in placed:
            placed[chain.sequence] = template_alignment(chain.sequence, template.sequence)
            if placed[chain.sequence] is None:
                warnings.warn(
                    f"chain {chain.id}: the template leaves residues unaligned; OpenFold3 drops it",
                    stacklevel=3,
                )
        residues = placed[chain.sequence]
        if residues is not None:
            restype[tokens], pseudo_beta[tokens], frame[tokens] = _template_atoms(
                template, residues
            )

    pseudo_beta_mask = ~np.isnan(pseudo_beta).any(-1)
    frame_mask = ~np.isnan(frame).any((-2, -1))
    same_chain = chain_of_token[:, None] == chain_of_token[None]
    distance2 = (np.nan_to_num(pseudo_beta)[:, None] - np.nan_to_num(pseudo_beta)[None]) ** 2
    distance2 = distance2.sum(-1, keepdims=True)
    distogram = (distance2 > DISTOGRAM_LOWER) & (distance2 < DISTOGRAM_UPPER)
    distogram &= (pseudo_beta_mask[:, None] & pseudo_beta_mask[None] & same_chain)[..., None]
    pair_frame_mask = (frame_mask[:, None] & frame_mask[None] & same_chain)[..., None]

    def slots(first: np.ndarray, empty=0) -> np.ndarray:
        out = np.full((NUM_TEMPLATES, *first.shape), empty, first.dtype)
        out[0] = first
        return out

    return {
        "template_pseudo_beta_mask": slots(pseudo_beta_mask.astype(np.float32)),
        "template_backbone_frame_mask": slots(frame_mask.astype(np.float32)),
        "template_restype": np.eye(NUM_CLASSES, dtype=np.int32)[slots(restype, GAP)],
        "template_distogram": slots(distogram.astype(np.float32)),
        "template_unit_vector": slots(unit_vectors(frame) * pair_frame_mask.astype(np.float32)),
    }


def features(system: System, *, key=None) -> dict[str, np.ndarray]:
    """The feature dict for `system`; `key` (a JAX PRNG key) draws the reference conformers
    and their poses, `None` gives every residue conformer 0, unposed."""
    table = residues()
    chains = system.chains

    # Tokens: chains in input order, one per residue.
    asym = _rank([c.id for c in chains])
    entity = _rank([c.sequence for c in chains])
    copies: dict[str, int] = {}
    seq, residue_index, asym_id, entity_id, sym_id = [], [], [], [], []
    for chain in chains:
        copies[chain.sequence] = copies.get(chain.sequence, 0) + 1
        length = len(chain.sequence)
        seq += chain.sequence
        residue_index += range(1, length + 1)
        asym_id += [asym[chain.id]] * length
        entity_id += [entity[chain.sequence]] * length
        sym_id += [copies[chain.sequence]] * length
    num_tokens = len(seq)
    types = [THREE_LETTER[c] for c in seq]

    # Atoms: each residue's table entry (CCD order, no OXT), in token order.
    num_atoms = np.array([len(table[t].atoms) for t in types], np.int32)
    start = np.concatenate([[0], np.cumsum(num_atoms)[:-1]]).astype(np.int32)
    atom_to_token = np.repeat(np.arange(num_tokens, dtype=np.int32), num_atoms)
    total = int(num_atoms.sum())
    names = [n for t in types for n in table[t].atoms]
    elements = np.concatenate([table[t].elements for t in types])
    charges = np.concatenate([table[t].charges for t in types])
    name_chars = np.array([[ord(ch) - 32 for ch in n.ljust(4)] for n in names])

    # MSA: per chain, written into its token columns; shorter chains padded with gap rows.
    per_chain = [_chain_msa(chain) for chain in chains]
    depth = max(len(c) for c, *_ in per_chain)
    msa = np.full((depth, num_tokens), GAP, np.int64)
    deletions = np.zeros((depth, num_tokens), np.int64)
    profile = np.zeros((num_tokens, NUM_CLASSES))
    deletion_mean = np.zeros(num_tokens)
    offset = 0
    for chain, (c, d, p, m) in zip(chains, per_chain, strict=True):
        cols = slice(offset, offset + len(chain.sequence))
        msa[: len(c), cols], deletions[: len(d), cols] = c, d
        profile[cols], deletion_mean[cols] = p, m
        offset = cols.stop
    scale = np.float32(2.0) / np.float32(np.arccos(np.float32(0))) * np.float32(2)
    deletion_value = np.arctan(deletions.astype(np.float32) / np.float32(3)) * scale

    ints = lambda x: np.array(x, np.int32)
    return {
        "token_index": np.arange(num_tokens, dtype=np.int32),
        "restype": np.eye(NUM_CLASSES, dtype=np.int32)[[TOKEN[c] for c in seq]],
        "is_protein": np.ones(num_tokens, np.int32),
        "is_rna": np.zeros(num_tokens, np.int32),
        "is_dna": np.zeros(num_tokens, np.int32),
        "is_ligand": np.zeros(num_tokens, np.int32),
        "is_atomized": np.zeros(num_tokens, np.int32),
        "token_mask": np.ones(num_tokens, np.float32),
        "num_atoms_per_token": num_atoms,
        "start_atom_index": start,
        "atom_mask": np.ones(total, np.float32),
        "residue_index": ints(residue_index),
        "asym_id": ints(asym_id),
        "entity_id": ints(entity_id),
        "sym_id": ints(sym_id),
        "token_bonds": np.zeros((num_tokens, num_tokens), np.int32),
        "atom_to_token_index": atom_to_token,
        "ref_pos": _pose(table, types, key),
        "ref_mask": np.ones(total, np.int32),
        "ref_element": np.eye(NUM_ELEMENTS, dtype=np.int32)[elements - 1],
        "ref_charge": charges.astype(np.float32),
        "ref_atom_name_chars": np.eye(64, dtype=np.int32)[name_chars],
        "ref_space_uid": atom_to_token.copy(),
        "msa": np.eye(NUM_CLASSES, dtype=np.int32)[msa],
        "has_deletion": (deletions != 0).astype(np.float32),
        "deletion_value": deletion_value.astype(np.float32),
        "deletion_mean": deletion_mean.astype(np.float32),
        "profile": profile.astype(np.float32),
        "num_paired_seqs": np.ones(1, np.int32),
        "msa_mask": np.ones((depth, num_tokens), np.float32),
        **template_features(system),
    }


def atom_sites(system: System) -> AtomSites:
    """What each real atom of `features(system)` is (chains in input order)."""
    return _atom_sites(system.chains, residues())
