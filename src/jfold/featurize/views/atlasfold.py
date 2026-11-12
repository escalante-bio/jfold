"""Featurize proteins for AtlasFold and AtlasFold-M as their runners do, with one sequence
per chain and no MSA or templates. See `docs/upstream-featurization/atlasfold.md`."""

import numpy as np

from jfold.featurize.residues import ResidueTable
from jfold.featurize.sites import AtomSites
from jfold.featurize.sites import atom_sites as _atom_sites
from jfold.featurize.system import THREE_LETTER, System

RESTYPES = "ARNDCQEGHILKMFPSTWYVX"
# AtlasLM's token ids.
BOS, PAD, EOS = 0, 1, 2
LM_TOKEN = {c: 4 + i for i, c in enumerate("LAGVSERTIDPKQNFYMHWCX")}
# The runners pad to `atlasfold.common.featurize.DEFAULT_BUCKETS`, then to multiples of 128.
BUCKETS = (32, 64, 128, 192, 256, 384, 512, 640, 768, 896, 1024)
BUCKETS += tuple(range(1152, 2049, 128))
CB, CA = 4, 1  # atom14 slots
ATOM14, ATOM37 = 14, 37


def residues() -> ResidueTable:
    return ResidueTable.load("atlasfold")


def bucket(length: int) -> int:
    """Return the runners' padded residue count for `length` residues."""
    return next((b for b in BUCKETS if length <= b), -(-length // 128) * 128)


def _chain(sequence: str, entity: int, asym: int, sym: int) -> dict[str, np.ndarray]:
    table = residues()
    length = len(sequence)
    aatype = np.array([RESTYPES.index(c) for c in sequence], np.int64)
    atom14 = np.zeros((length, ATOM14), bool)
    atom37 = np.zeros((length, ATOM37), bool)
    for i, c in enumerate(sequence):
        slots = table[THREE_LETTER[c]].extra["atom37"]
        atom14[i, : len(slots)] = True
        atom37[i, slots] = True
    res_idx = np.arange(1, length + 1, dtype=np.int64)
    return {
        "lm.input_ids": np.array([BOS, *(LM_TOKEN[c] for c in sequence), EOS], np.int64),
        "lm.pos_id": np.concatenate([[0], res_idx, [length + 1]]),
        "lm.seq_id": np.full(length + 2, asym, np.int64),
        "entity_id": np.full(length, entity, np.int64),
        "asym_id": np.full(length, asym, np.int64),
        "sym_id": np.full(length, sym, np.int64),
        "aatype": np.eye(len(RESTYPES), dtype=np.float32)[aatype],
        "aatype_int": aatype,
        "res_idx": res_idx,
        "seq_tok_idx": res_idx.copy(),
        "seq_mask": np.ones(length, bool),
        "atom14_mask": atom14,
        "atom37_mask": atom37,
        "pseudo_beta": np.where(aatype == RESTYPES.index("G"), CA, CB),
    }


def _pad(feat: dict[str, np.ndarray], residues: int, lm_tokens: int) -> dict[str, np.ndarray]:
    out = {}
    for k, v in feat.items():
        length = lm_tokens if k.startswith("lm.") else residues
        value = PAD if k == "lm.input_ids" else 0
        out[k] = np.concatenate([v, np.full((length - len(v), *v.shape[1:]), value, v.dtype)])
    return out


def features(system: System, *, multimer: bool = True) -> dict[str, np.ndarray]:
    """Return AtlasFold's feature dict for `system`, padded to the runner's length bucket.
    """
    system = system.without_templates("AtlasFold")
    chains = system.chains
    if any(c.msa is not None for c in chains):
        raise NotImplementedError("AtlasFold featurization takes no MSAs (single sequence)")
    if not multimer and len(chains) != 1:
        raise ValueError(f"AtlasFold (monomer) takes one chain, got {len(chains)}")
    entities: dict[str, int] = {}
    copies: dict[str, int] = {}
    per_chain = []
    for asym, chain in enumerate(chains, 1):
        entity = entities.setdefault(chain.sequence, len(entities) + 1)
        copies[chain.sequence] = copies.get(chain.sequence, 0) + 1
        per_chain.append(_chain(chain.sequence, entity, asym, copies[chain.sequence]))
    offset = 0
    for feat in per_chain:
        feat["seq_tok_idx"] = feat["seq_tok_idx"] + offset
        offset += len(feat["lm.input_ids"])
    feat = {k: np.concatenate([f[k] for f in per_chain]) for k in per_chain[0]}
    length = bucket(len(feat["aatype"]))
    return _pad(feat, length, length + (32 if multimer else 2))


def atom_sites(system: System) -> AtomSites:
    """Describe the real atoms of `system`'s features in atom14 order, at
    `coords[f["atom14_mask"].reshape(-1)]`."""
    return _atom_sites(system.chains, residues())
