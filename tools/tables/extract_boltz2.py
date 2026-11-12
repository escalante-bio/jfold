"""Write `jfold/featurize/data/boltz2.npz` from Boltz-2's `~/.boltz/mols`. Run once:

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
    # jfold indexes conformers by position, upstream by `conf_ids`, so they must match.
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
