"""Boltz-2's residue table (`jfold/featurize/data/boltz2.npz`) from its canonical residue
mols (`~/.boltz/mols/<RES>.pkl`, the Boltz-2 `mols.tar` download): for the 20 amino acids
and UNK, the atoms of `const.ref_atoms` in that order, with each atom's element and formal
charge as stored, and all 10 stored conformers (`[10, A, 3]`, in conformer-id order).
Needs the `boltz` package and its downloaded mols; run once:

    uv run python tools/tables/extract_boltz2.py
"""

import pickle
from pathlib import Path

import numpy as np
from boltz.data import const

from jfold.featurize.residues import DATA, Residue, ResidueTable
from jfold.featurize.system import THREE_LETTER

MOLS = Path.home() / ".boltz" / "mols"

residues = {}
for name in THREE_LETTER.values():
    with (MOLS / f"{name}.pkl").open("rb") as fh:
        mol = pickle.load(fh)
    by_name = {a.GetProp("name"): a for a in mol.GetAtoms()}
    atoms = [by_name[a] for a in const.ref_atoms[name]]
    conf_ids = [int(c.GetId()) for c in mol.GetConformers()]
    # The view draws an index into this list, as upstream draws from `conf_ids`.
    assert conf_ids == list(range(10)), (name, conf_ids)
    conformers = np.array(
        [[list(mol.GetConformer(i).GetAtomPosition(a.GetIdx())) for a in atoms] for i in conf_ids]
    )
    residues[name] = Residue(
        atoms=tuple(const.ref_atoms[name]),
        elements=np.array([a.GetAtomicNum() for a in atoms]),
        charges=np.array([a.GetFormalCharge() for a in atoms]),
        conformers=conformers.astype(np.float32),
    )
ResidueTable(residues).save(DATA / "boltz2.npz")
print({k: (len(v.atoms), v.conformers.shape[0]) for k, v in residues.items()})
