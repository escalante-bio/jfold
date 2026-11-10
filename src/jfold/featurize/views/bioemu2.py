"""BioEmu2's features, as `bioemu2`'s featurization produces them for proteins (see
`docs/upstream-featurization/bioemu2.md`): the dict `jfold.models.bioemu2.features` takes.

Deterministic: reference conformers are the CCD ideal coordinates, unposed."""

import numpy as np

from jfold.featurize.a3m import parse_a3m
from jfold.featurize.padding import pad_to, padded_length
from jfold.featurize.residues import ResidueTable
from jfold.featurize.sites import AtomSites
from jfold.featurize.sites import atom_sites as _atom_sites
from jfold.featurize.system import THREE_LETTER, System

# The Boltz-1 alphabet: PAD, GAP, the 20 amino acids by three-letter code, UNK_PROT, ...
NUM_TOKEN_TYPES = 33
_AMINO_ACIDS = sorted(name for name in THREE_LETTER.values() if name != "UNK")
TOKEN = {"-": 1, **{k: 2 + _AMINO_ACIDS.index(v) for k, v in THREE_LETTER.items() if k != "X"}}
TOKEN |= dict.fromkeys("XJBZOU", 22)
METHOD = {"structure": 7, "equilibrium": 4}  # "non-md", "md"
MAX_MSA_ROWS = 4092
ATOM_WINDOW = 32


def residues() -> ResidueTable:
    return ResidueTable.load("bioemu2")


def chain_order(system: System) -> list[tuple[int, int, int]]:
    """(chain index, entity, copy) in upstream's order (`System.chain_order`)."""
    return system.chain_order()


def _msa_rows(chain) -> tuple[np.ndarray, np.ndarray]:
    """A chain's MSA as residue classes `[S, L]` and deletion counts: its a3m (row 0 as
    written, after the query check), or the sequence alone."""
    if chain.msa is None:
        return np.array([[TOKEN[c] for c in chain.sequence]]), np.zeros(
            (1, len(chain.sequence)), int
        )
    a3m = parse_a3m(chain.msa)
    query = a3m.rows[0].replace("-", "").replace(".", "").upper()
    if query != chain.sequence:
        raise ValueError(f"chain {chain.id}: the a3m's first row is not the chain's sequence")
    rows = a3m.rows[:MAX_MSA_ROWS]
    classes = np.array([[TOKEN[c] for c in row] for row in rows])  # '.' raises, as upstream
    return classes, a3m.deletions[: len(rows)]


def _msa(chains, paired_queries: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Residue classes, deletion counts and paired flags `[S, T]`, as `tokenize_msa` builds
    them. With unpaired MSAs only, every chain is cut to the shallowest chain's depth, row 0
    paired (every row for a homomer). With `paired_queries` (each chain's query as its paired
    MSA) a heteromer instead keeps every chain's rows: the queries (paired), then each
    chain's other rows side by side, gaps once a chain runs out."""
    per_chain = [_msa_rows(chain) for chain in chains]
    num_tokens = sum(len(chain.sequence) for chain in chains)
    heteromer = len({chain.sequence for chain in chains}) > 1
    if paired_queries and heteromer:
        depth = min(max(len(c) for c, _ in per_chain), MAX_MSA_ROWS)
        msa = np.full((depth, num_tokens), TOKEN["-"])
        deletions = np.zeros((depth, num_tokens), int)
        start = 0
        for chain, (classes, counts) in zip(chains, per_chain, strict=True):
            stop = start + len(chain.sequence)
            rows = min(len(classes), depth)
            msa[1:rows, start:stop] = classes[1:rows]
            deletions[1:rows, start:stop] = counts[1:rows]
            msa[0, start:stop] = [TOKEN[c] for c in chain.sequence]
            start = stop
        paired = np.zeros((depth, num_tokens), np.int64)
        paired[0] = 1
        return msa, deletions, paired
    depth = min(min(len(c) for c, _ in per_chain), MAX_MSA_ROWS)
    msa = np.concatenate([c[:depth] for c, _ in per_chain], axis=1)
    deletions = np.concatenate([d[:depth] for _, d in per_chain], axis=1)
    paired = np.zeros((depth, num_tokens), np.int64)
    paired[: depth if len(chains) > 1 and not heteromer else 1] = 1
    return msa, deletions, paired


def features(
    system: System, *, mode: str = "structure", paired_queries: bool = False
) -> dict[str, np.ndarray]:
    """The feature dict for `system`, in `mode` ("structure" or "equilibrium").
    `paired_queries`: as upstream builds it when each chain's paired MSA is its query alone
    (as mosaic passes them), so a heteromer's chains keep all their MSA rows (`_msa`)."""
    system = system.without_templates("BioEmu2")
    table = residues()
    order = chain_order(system)
    chains = [system.chains[i] for i, _, _ in order]

    res_type, asym_id, entity_id, sym_id = [], [], [], []
    for asym, (chain, (_, entity, copy)) in enumerate(zip(chains, order, strict=True)):
        res_type += [TOKEN[c] for c in chain.sequence]
        asym_id += [asym] * len(chain.sequence)
        entity_id += [entity] * len(chain.sequence)
        sym_id += [copy] * len(chain.sequence)
    num_tokens = len(res_type)

    # Atoms: each residue's table entry, in token order.
    coords, charge, element, names, token_of_atom, center = [], [], [], [], [], []
    for token, residue in enumerate(c for chain in chains for c in chain.sequence):
        r = table[THREE_LETTER[residue]]
        center.append(len(token_of_atom) + r.atoms.index("CA"))
        coords.append(r.conformers[0])
        charge.append(r.charges)
        element.append(r.elements)
        names += r.atoms
        token_of_atom += [token] * len(r.atoms)
    num_atoms = len(token_of_atom)
    padded = padded_length(num_atoms, ATOM_WINDOW)
    pad = lambda x, value=0: pad_to(x, padded, value)

    name_encoding = np.array([[ord(c) - 32 for c in n] + [0] * (4 - len(n)) for n in names])
    real = pad(np.ones(num_atoms, bool), False)

    msa, deletions, paired = _msa(chains, paired_queries)
    depth = len(msa)
    msa_one_hot = np.eye(NUM_TOKEN_TYPES, dtype=np.int64)[msa]
    deletion_value = (
        np.float32(np.pi / 2) * np.arctan(deletions.astype(np.float32) / np.float32(3))
    ).astype(np.float32)

    contacts = np.zeros((num_tokens, num_tokens, 5), np.int64)
    contacts[..., 0] = 1
    return {
        "template_coords": pad(np.concatenate(coords)).astype(np.float32),
        "charge": pad(np.concatenate(charge)).astype(np.float32),
        "element": pad(np.concatenate(element)).astype(np.int64),
        "name_encoding": pad(name_encoding).astype(np.int64),
        "ref_space_uid": pad(np.array(token_of_atom), -1).astype(np.int64),
        "token_idx_at_atom": pad(np.array(token_of_atom), -1).astype(np.int64),
        "atom_pad_mask": real,
        "atom_mask": real,
        "mol_type": np.zeros(padded, np.int64),
        "token_pad_mask": np.ones(num_tokens, bool),
        "token_center_atom_index": np.array(center, np.int64),
        "res_type": np.array(res_type, np.int64),
        "asym_id": np.array(asym_id, np.int64),
        "entity_id": np.array(entity_id, np.int64),
        "sym_id": np.array(sym_id, np.int64),
        "residue_index": np.arange(num_tokens, dtype=np.int64),  # a global counter
        "token_bonds": np.zeros((num_tokens, num_tokens, 1), np.float32),
        "msa": msa_one_hot,
        "msa_mask": np.ones((depth, num_tokens), np.int64),
        "msa_paired": paired,
        "has_deletion": deletions > 0,
        "deletion_value": deletion_value,
        "profile": msa_one_hot.astype(np.float32).mean(0),
        "deletion_mean": deletion_value.mean(0),
        "pocket_feature": np.zeros(num_tokens, np.int64),
        "method_feature": np.full(num_tokens, METHOD[mode], np.int64),
        "contact_conditioning": contacts,
        "contact_threshold": np.zeros((num_tokens, num_tokens), np.float32),
        "coords_cond": np.zeros((padded, 3), np.float32),
        "dt": np.zeros(1, np.float32),
        "is_dynamics_conditioned": np.zeros(1, bool),
    }


def atom_sites(system: System) -> AtomSites:
    """What each real atom of `features(system)` is (chains in BioEmu2's entity-grouped order)."""
    return _atom_sites([system.chains[i] for i, _, _ in chain_order(system)], residues())
