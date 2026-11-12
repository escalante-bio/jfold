"""Featurize proteins for ESMFold2 as `esm`'s `ESMFold2InputBuilder.prepare_input` does,
plus ESMC's packed token inputs. See `docs/upstream-featurization/esmfold2.md`."""

import re

import numpy as np

from jfold.featurize.a3m import parse_a3m
from jfold.featurize.padding import pad_to, padded_length
from jfold.featurize.residues import ResidueTable
from jfold.featurize.sites import AtomSites
from jfold.featurize.sites import atom_sites as _atom_sites
from jfold.featurize.system import THREE_LETTER, System

# Boltz's 33 token classes.
GAP, UNK = 1, 22
_AMINO_ACIDS = sorted(name for name in THREE_LETTER.values() if name != "UNK")
TOKEN = {k: 2 + _AMINO_ACIDS.index(v) for k, v in THREE_LETTER.items() if k != "X"} | {"X": UNK}
# ESMC's token ids.
BOS, EOS = 0, 2
ESMC_TOKEN = {c: 4 + i for i, c in enumerate("LAGVSERTIDPKQNFYMHWC")} | {"X": 24}
ATOM_WINDOW = 32
MAX_PAIRS, MAX_TOTAL, MAX_SEQS = 8192, 16384, 16384
_KEY = re.compile(r"key=(-?\d+)")


def residues() -> ResidueTable:
    return ResidueTable.load("esmfold2")


def _chain_msa(chain, max_sequences: int | None) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if chain.msa is None:
        classes = np.array([[TOKEN[c] for c in chain.sequence]], np.int64)
        return classes, np.zeros(classes.shape, np.float32), [-1]
    # Upstream counts `.` as an insertion, like a lowercase letter.
    lines = [
        line if line.strip().startswith(">") else line.replace(".", "x")
        for line in chain.msa.splitlines()
    ]
    a3m = parse_a3m("\n".join(lines))
    rows = a3m.rows[:max_sequences]
    classes = np.array(
        [[GAP if c == "-" else TOKEN.get(c.upper(), UNK) for c in row] for row in rows],
        np.int64,
    ).reshape(len(rows), -1)
    taxa = [int(m.group(1)) if (m := _KEY.search(h.strip())) else -1 for h in a3m.headers]
    return classes, a3m.deletions[: len(rows)].astype(np.float32), taxa[: len(rows)]


def _msa_rows(taxa: list[list[int]]) -> list[list[int]]:
    by_taxon: dict[int, list[tuple[int, int]]] = {}
    for c, chain_taxa in enumerate(taxa):
        for row, taxon in enumerate(chain_taxa):
            if row and taxon != -1:
                by_taxon.setdefault(taxon, []).append((c, row))
    by_taxon = {k: v for k, v in by_taxon.items() if len(v) > 1}
    paired = {item for items in by_taxon.values() for item in items}
    available = [
        [row for row in range(1, len(t)) if (c, row) not in paired] for c, t in enumerate(taxa)
    ]

    def unpaired(c: int) -> int:
        return available[c].pop(0) if available[c] else -1

    rows = [[0] * len(taxa)]
    for items in sorted(by_taxon.values(), key=lambda v: len({c for c, _ in v}), reverse=True):
        per_chain: dict[int, list[int]] = {}
        for c, row in items:
            per_chain.setdefault(c, []).append(row)
        for i in range(max(len(v) for v in per_chain.values())):
            rows.append(
                [
                    per_chain[c][i % len(per_chain[c])] if c in per_chain else unpaired(c)
                    for c in range(len(taxa))
                ]
            )
            if len(rows) >= MAX_PAIRS:
                break
        if len(rows) >= MAX_PAIRS:
            break
    remaining = max((len(v) for v in available), default=0)
    for _ in range(min(MAX_TOTAL - len(rows), remaining)):
        rows.append([unpaired(c) for c in range(len(taxa))])
    return rows[:MAX_SEQS]


def lm_inputs(asym_id: np.ndarray, input_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Pack ESMC's input with each chain wrapped in BOS and EOS, as
    `compute_lm_hidden_states` does."""
    ids, positions = [], np.zeros(len(asym_id), np.int64)
    for chain in np.unique(asym_id):
        tokens = np.flatnonzero(asym_id == chain)
        positions[tokens] = len(ids) + 1 + np.arange(len(tokens))
        ids += [BOS, *input_ids[tokens].tolist(), EOS]
    ids = np.array(ids, np.int64)
    return {
        "lm_input_ids": ids,
        "lm_sequence_id": np.cumsum(ids == BOS) - 1,
        "lm_positions": positions,
    }


def features(system: System, *, max_sequences: int | None = None) -> dict[str, np.ndarray]:
    """Return ESMFold2's feature dict, keeping only each a3m's first `max_sequences` records
    if given."""
    system = system.without_templates("ESMFold2")
    table = residues()
    chains = system.chains

    entities: dict[str, int] = {}
    copies: dict[str, int] = {}
    res_type, input_ids, asym_id, entity_id, sym_id, residue_index = [], [], [], [], [], []
    for asym, chain in enumerate(chains):
        entity = entities.setdefault(chain.sequence, len(entities))
        copy = copies[chain.sequence] = copies.get(chain.sequence, -1) + 1
        n = len(chain.sequence)
        res_type += [TOKEN[c] for c in chain.sequence]
        input_ids += [ESMC_TOKEN[c] for c in chain.sequence]
        asym_id += [asym] * n
        entity_id += [entity] * n
        sym_id += [copy] * n
        residue_index += range(n)
    num_tokens = len(res_type)
    asym_id = np.array(asym_id, np.int64)
    residue_index = np.array(residue_index, np.int64)

    coords, charge, element, names, token_of_atom = [], [], [], [], []
    distogram_atom, frames = [], []
    for token, residue in enumerate(c for chain in chains for c in chain.sequence):
        r = table[THREE_LETTER[residue]]
        first = len(token_of_atom)
        index = {a: first + i for i, a in enumerate(r.atoms)}
        distogram_atom.append(index.get("CB", index.get("CA", first)))
        # Upstream uses the first atom three times as the unknown residue's frame.
        frames.append([first] * 3 if residue == "X" else [index[a] for a in ("N", "CA", "C")])
        coords.append(r.conformers[0])
        charge.append(r.charges)
        element.append(r.elements)
        names += r.atoms
        token_of_atom += [token] * len(r.atoms)
    num_atoms = len(token_of_atom)
    padded = padded_length(num_atoms, ATOM_WINDOW, minimum=ATOM_WINDOW)
    pad = lambda x, dtype: pad_to(x, padded, dtype=dtype)

    name_chars = [[ord(c) - 32 for c in n.ljust(4)[:4]] for n in names]  # space -> 0

    # Upstream bonds only unknown residues, to their neighbours in the chain.
    bonds = np.zeros((num_tokens, num_tokens, 1), np.float32)
    start = 0
    for chain in chains:
        for i, c in enumerate(chain.sequence):
            if c == "X":
                for j in (i - 1, i + 1):
                    if 0 <= j < len(chain.sequence):
                        bonds[start + i, start + j, 0] = bonds[start + j, start + i, 0] = 1.0
        start += len(chain.sequence)

    # Chains that run out of rows, or have no a3m, are filled with gaps.
    per_chain = [_chain_msa(chain, max_sequences) for chain in chains]
    rows = _msa_rows([taxa for _, _, taxa in per_chain])
    msa = np.full((len(rows), num_tokens), GAP, np.int64)
    deletions = np.zeros((len(rows), num_tokens), np.float32)
    for c, (chain, (classes, counts, _)) in enumerate(zip(chains, per_chain, strict=True)):
        tokens = np.flatnonzero(asym_id == c)
        # A query shorter than the chain repeats its last column.
        columns = np.minimum(residue_index[tokens], classes.shape[1] - 1)
        source = np.array([row[c] for row in rows])
        valid = np.flatnonzero(source >= 0)
        msa[np.ix_(valid, tokens)] = classes[source[valid]][:, columns]
        deletions[np.ix_(valid, tokens)] = counts[source[valid]][:, columns]
        if chain.msa is None:
            msa[:, tokens] = GAP
            msa[0, tokens] = np.array(res_type)[tokens]
    deletion_value = (np.float32(np.pi / 2) * np.arctan(deletions / np.float32(3))).astype(
        np.float32
    )

    real = pad(np.ones(num_atoms, bool), bool)
    token_of_atom = pad(token_of_atom, np.int64)
    input_ids = np.array(input_ids, np.int64)
    return {
        "token_index": np.arange(num_tokens, dtype=np.int64),
        "residue_index": residue_index,
        "asym_id": asym_id,
        "entity_id": np.array(entity_id, np.int64),
        "sym_id": np.array(sym_id, np.int64),
        "mol_type": np.zeros(num_tokens, np.int64),
        "res_type": np.array(res_type, np.int64),
        "input_ids": input_ids,
        "token_bonds": bonds,
        "token_attention_mask": np.ones(num_tokens, bool),
        "pocket_feature": np.zeros(num_tokens, np.int64),
        "ref_pos": pad(np.concatenate(coords), np.float32),
        "ref_element": pad(np.concatenate(element), np.int64),
        "ref_charge": pad(np.concatenate(charge), np.int8),
        "ref_atom_name_chars": pad(name_chars, np.int64),
        "ref_space_uid": token_of_atom,  # one conformer per residue
        "gt_coords": np.zeros((1, padded, 3), np.float32),
        "atom_attention_mask": real,
        "atom_to_token": token_of_atom,
        "is_resolved": np.zeros(padded, bool),
        "distogram_atom_idx": np.array(distogram_atom, np.int64),
        "frames_idx": np.array(frames, np.int64).reshape(num_tokens, 3),
        "disto_cond": np.zeros((num_tokens, num_tokens), np.int64),
        "disto_cond_mask": np.zeros((num_tokens, num_tokens), bool),
        "msa": msa,
        "deletion_value": deletion_value,
        "has_deletion": deletions > 0,
        "deletion_mean": deletion_value.mean(0),
        "msa_attention_mask": np.ones(msa.shape, bool),
    } | lm_inputs(asym_id, input_ids)


def atom_sites(system: System) -> AtomSites:
    """Describe each real atom of `features(system)`, with chains in input order."""
    return _atom_sites(system.chains, residues())
