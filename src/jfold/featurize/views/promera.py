"""ProMERA's features, as tinyprot's `AF3Featurizer`, `promera.inference.utils.finalize_feats`
and `promera.data.utils.collate` produce them for proteins (see `docs/upstream-featurization/promera.md`):
the dict `jfold.models.promera.features` takes, plus upstream's `frames_mask` and
`msa_paired`, unbatched.

The only random step upstream is the reference conformer: each residue's centred CCD ideal
coordinates get a N(0, 1) translation, then a uniform random rotation, from numpy's global
RNG. Here `key=None` gives the canonical (unposed) conformers, and a `jax.random` key draws
the same distributions in the same order."""

import io
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import jax
import numpy as np

from jfold.featurize.padding import pad_to, padded_length
from jfold.featurize.residues import ResidueTable
from jfold.featurize.sites import AtomSites
from jfold.featurize.sites import atom_sites as _atom_sites
from jfold.featurize.system import THREE_LETTER, Chain, System

# tinyprot's `_tokens`: <pad>, <gap>, the 20 amino acids by three-letter code, UNK, RNA, DNA,
# <other>.
NUM_TOKEN_TYPES = 34
_AMINO_ACIDS = sorted(name for name in THREE_LETTER.values() if name != "UNK")
TOKEN = {k: 2 + _AMINO_ACIDS.index(v) for k, v in THREE_LETTER.items() if k != "X"}
TOKEN |= {"-": 1, "X": 22}
# Letters an a3m row keeps (`_letter_to_std_letter`); any other aligned character is X.
_MSA_LETTERS = frozenset(THREE_LETTER) | {"-"}
MAX_PAIRS = 8192
MAX_TOTAL = 16384
ATOM_MULTIPLE = 32


def residues() -> ResidueTable:
    return ResidueTable.load("promera")


@dataclass(frozen=True)
class MSA:
    """One chain's MSA as tinyprot holds it: `seqs` `[S, L]` letters, `dels` `[S, L]`
    insertion counts, `taxs` `[S]` taxonomy ids (-1 for none)."""

    seqs: np.ndarray
    dels: np.ndarray
    taxs: np.ndarray


def dummy_msa(sequence: str) -> MSA:
    """A chain without an a3m: the query alone (`make_dummy_msa`)."""
    return MSA(np.array([list(sequence)]), np.zeros((1, len(sequence)), np.int64), np.array([-1]))


def parse_msa(text: str, taxonomy: Callable[[str], int] | None = None) -> MSA:
    """An a3m as `tinyprot.msa.MSA.from_a3m` reads it, line by line (a wrapped sequence is
    several rows). A row whose ungapped, upper-cased text was already seen is dropped, so a
    hit identical to the query goes. Lowercase letters are insertions, counted onto the
    next column; trailing ones are dropped. Other characters outside the standard letters,
    `X` and `-` become `X`.

    A row's taxonomy is `taxonomy(id)` for a header whose first word is `>UniRef100...`
    (`id` without the `>`), else -1. tinyprot looks it up in its taxonomy database; pass
    that lookup here, since UniRef100 headers change the row order (see `pair_rows`)."""
    seen, seqs, dels, taxs = set(), [], [], []
    tax = None
    for line in io.StringIO(text, newline=None).readlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            header = line.split()[0]
            if header.startswith(">UniRef100"):
                if taxonomy is None:
                    raise ValueError(
                        f"a3m header {header!r}: UniRef100 rows need a `taxonomy` lookup"
                    )
                tax = int(taxonomy(header[1:]))
            else:
                tax = -1
            continue
        if tax is None:
            raise ValueError("a3m sequence before any header")
        key = line.replace("-", "").upper()
        if key in seen:
            continue
        seen.add(key)
        row, row_dels, count = [], [], 0
        for c in line:
            if c != "-" and c.islower():
                count += 1
                continue
            row.append(c if c in _MSA_LETTERS else "X")
            row_dels.append(count)
            count = 0
        seqs.append(row)
        dels.append(row_dels)
        taxs.append(tax)
    if len({len(row) for row in seqs}) > 1:
        raise ValueError("a3m rows have different numbers of aligned columns")
    return MSA(np.array(seqs), np.array(dels, np.int64), np.array(taxs))


def pair_rows(msas: list[MSA]) -> tuple[np.ndarray, np.ndarray]:
    """The MSA row layout, `tinyprot.msa.construct_paired_msa` with the "protenix" strategy:
    `(pairing [S, C], is_paired [S, C])`, each row's MSA row index per chain (-1: absent).

    Row 0 is every chain's query. Hits whose taxonomy (not -1) is shared by two or more
    (chain, row) entries, even within one chain, come next (paired), by coverage and then
    log rank. The remaining rows of each chain follow side by side, in order."""
    num_chains = len(msas)
    by_taxonomy: dict[int, list[tuple[int, int]]] = {}
    for c, msa in enumerate(msas):
        for i, tax in enumerate(msa.taxs.tolist()):
            if tax != -1:
                by_taxonomy.setdefault(tax, []).append((c, i))
    shared = sorted(
        (pairs for pairs in by_taxonomy.values() if len(pairs) > 1),
        key=lambda pairs: len({c for c, _ in pairs}),
        reverse=True,
    )
    paired = {entry for pairs in shared for entry in pairs}
    available = [
        deque(i for i in range(1, len(msa.seqs)) if (c, i) not in paired)
        for c, msa in enumerate(msas)
    ]
    pairing, is_paired = [[0] * num_chains], [[1] * num_chains]

    tiers: dict[int, list[dict[int, list[int]]]] = {}
    for pairs in shared:
        occurrences: dict[int, list[int]] = {}
        for c, i in pairs:
            occurrences.setdefault(c, []).append(i)
        tiers.setdefault(len(occurrences), []).append(occurrences)
    full = False
    for coverage in sorted(tiers, reverse=True):
        rows = []
        for occurrences in tiers[coverage]:
            for k in range(min(len(v) for v in occurrences.values())):
                rows.append(
                    (
                        [occurrences[c][k] if c in occurrences else -1 for c in range(num_chains)],
                        [int(c in occurrences) for c in range(num_chains)],
                    )
                )
        rows.sort(key=lambda row: sum(np.log(abs(i) + 1e-10) for i in row[0]))
        for row_pairing, row_is_paired in rows:
            pairing.append(row_pairing)
            is_paired.append(row_is_paired)
            if len(pairing) >= MAX_PAIRS:
                full = True
                break
        if full:
            break

    max_left = max((len(a) for a in available), default=0)
    for _ in range(min(MAX_TOTAL - len(pairing), max_left)):
        pairing.append([a.popleft() if a else -1 for a in available])
        is_paired.append([0] * num_chains)
        if len(pairing) >= MAX_TOTAL:
            break
    return np.array(pairing, np.int64), np.array(is_paired, bool)


def reference_conformers(system: System) -> tuple[np.ndarray, np.ndarray]:
    """The canonical conformers of every residue in token order, `[N, 3]` float64 (centred
    CCD ideal coordinates), and each atom's residue `[N]`."""
    table = residues()
    letters = [c for chain in system.chains for c in chain.sequence]
    coords = [table[THREE_LETTER[c]].extra["ref_pos"] for c in letters]
    uid = np.repeat(np.arange(len(letters)), [len(x) for x in coords])
    return np.concatenate(coords), uid


def random_pose(key: jax.Array, num_residues: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-residue translations `[R, 3]` ~ N(0, I) and uniform rotation matrices
    `[R, 3, 3]`, as `tinyprot.ccd._get_conformer` draws them (`np.random.randn(3)`, then
    scipy's `Rotation.random()`: a normalised N(0, I) quaternion, scalar last)."""
    k_translation, k_rotation = jax.random.split(key)
    translation = np.array(jax.random.normal(k_translation, (num_residues, 3)), np.float64)
    q = np.array(jax.random.normal(k_rotation, (num_residues, 4)), np.float64)
    x, y, z, w = (q / np.linalg.norm(q, axis=-1, keepdims=True)).T
    rotation = np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )
    return translation, rotation


def pose(
    canonical: np.ndarray, uid: np.ndarray, translation: np.ndarray, rotation: np.ndarray
) -> np.ndarray:
    """Each residue's conformer translated, then rotated: `(x + t) @ R.T`."""
    return np.einsum("aj,aij->ai", canonical + translation[uid], rotation[uid])


def _chain_msa(chain: Chain, taxonomy) -> MSA:
    if chain.msa is None:
        return dummy_msa(chain.sequence)
    msa = parse_msa(chain.msa, taxonomy)
    if msa.seqs.shape[1] < len(chain.sequence):
        raise ValueError(
            f"chain {chain.id}: the a3m has {msa.seqs.shape[1]} columns, the chain "
            f"{len(chain.sequence)} residues"
        )
    # Columns beyond the chain are dropped; the query row is not checked, as upstream.
    return MSA(msa.seqs[:, : len(chain.sequence)], msa.dels[:, : len(chain.sequence)], msa.taxs)


def features(
    system: System,
    *,
    key: jax.Array | None = None,
    taxonomy: Callable[[str], int] | None = None,
) -> dict[str, np.ndarray]:
    """The feature dict for `system`. `key`: a `jax.random` key to draw the per-residue
    reference-conformer pose, as upstream does with numpy's global RNG; None leaves the
    canonical conformers. `taxonomy`: UniRef100 id to taxonomy id, for a3ms with
    UniRef100 headers (see `parse_msa`)."""
    system = system.without_templates("ProMERA")
    table = residues()
    chains = system.chains

    # Tokens: one per residue, chains in input order, each chain its own entity.
    letters = [c for chain in chains for c in chain.sequence]
    num_tokens = len(letters)
    asym_id = np.repeat(np.arange(len(chains)), [len(c.sequence) for c in chains])
    residue_index = np.concatenate([np.arange(1, len(c.sequence) + 1) for c in chains])

    # Atoms: each residue's table entry, in token order.
    element, charge, hydrogens, names, rep_atom = [], [], [], [], []
    for letter in letters:
        name = THREE_LETTER[letter]
        r = table[name]
        first = len(names)
        rep_atom.append(first + r.atoms.index("CA" if name in ("GLY", "UNK") else "CB"))
        element.append(r.elements)
        charge.append(r.charges)
        hydrogens.append(r.extra["ref_hydrogens"])
        names += r.atoms
    canonical, atom_to_token = reference_conformers(system)
    if key is None:
        ref_pos = canonical
    else:
        ref_pos = pose(canonical, atom_to_token, *random_pose(key, num_tokens))
    num_atoms = len(names)
    padded = padded_length(num_atoms, ATOM_MULTIPLE)
    pad = lambda x, dtype: pad_to(x, padded, dtype=dtype)

    name_chars = [[ord(c) - 32 for c in n] + [0] * (4 - len(n)) for n in names]

    # MSA.
    msas = [_chain_msa(chain, taxonomy) for chain in chains]
    pairing, is_paired = pair_rows(msas)
    depth = len(pairing)
    msa = np.zeros((depth, num_tokens), np.int64)
    deletions = np.zeros((depth, num_tokens), np.int64)
    msa_paired = np.zeros((depth, num_tokens), bool)
    profile = np.zeros((num_tokens, NUM_TOKEN_TYPES))
    deletion_mean = np.zeros(num_tokens)
    for c, chain_msa in enumerate(msas):
        tokens = asym_id == c
        rows = pairing[:, c]
        present = rows != -1
        classes = np.vectorize(TOKEN.__getitem__, otypes=[np.int64])(chain_msa.seqs)
        msa[np.ix_(present, tokens)] = classes[rows[present]]
        deletions[np.ix_(present, tokens)] = chain_msa.dels[rows[present]]
        msa_paired[:, tokens] = is_paired[:, c, None]
        deletion_mean[tokens] = chain_msa.dels.mean(0)
        profile[tokens] = np.eye(NUM_TOKEN_TYPES, dtype=np.int64)[classes].mean(0)

    restype = np.array([TOKEN[c] for c in letters], np.int64)
    return {
        "restype": restype,
        "profile": profile.astype(np.float32),
        "deletion_mean": deletion_mean.astype(np.float32),
        "is_std": np.ones(num_tokens, bool),
        "is_epitope": np.zeros(num_tokens, np.int64),
        "token_pad_mask": np.ones(num_tokens, np.int64),
        "token_to_rep_atom": np.array(rep_atom, np.int64),
        "token_bonds": np.zeros((num_tokens, num_tokens), np.int64),
        "residue_index": residue_index.astype(np.int64),
        "token_index": np.arange(num_tokens, dtype=np.int64),
        "asym_id": asym_id.astype(np.int64),
        "entity_id": asym_id.astype(np.int64),
        "sym_id": np.zeros(num_tokens, np.int64),
        "frames_mask": np.ones(num_tokens, bool),
        "ref_pos": pad(ref_pos.astype(np.float32), np.float32),
        "ref_element": pad(np.concatenate(element), np.int64),
        "ref_charge": pad(np.concatenate(charge), np.int64),
        "ref_hydrogens": pad(np.concatenate(hydrogens), np.int64),
        "ref_atom_name_chars": pad(name_chars, np.int64),
        "ref_space_uid": pad(atom_to_token, np.int64),
        "atom_to_token": pad(atom_to_token, np.int64),
        "atom_pad_mask": pad(np.ones(num_atoms), np.int64),
        "atom_is_std": pad(np.ones(num_atoms), bool),
        "msa": msa,
        "msa_mask": np.ones((depth, num_tokens), bool),
        "msa_paired": msa_paired,
        "has_deletion": deletions > 0,
        "deletion_value": ((2 / np.pi) * np.arctan(deletions / 3)).astype(np.float32),
    }


def atom_sites(system: System) -> AtomSites:
    """What each real atom of `features(system)` is (chains in input order)."""
    return _atom_sites(system.chains, residues())
