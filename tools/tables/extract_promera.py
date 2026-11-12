"""Write `jfold/featurize/data/promera.npz` from tinyprot's CCD. Run once:

    uv run python tools/tables/extract_promera.py
"""

import numpy as np
from tinyprot.ccd import _get_attrs, _get_conformer

from jfold.featurize.residues import DATA, Residue, ResidueTable
from jfold.featurize.system import THREE_LETTER


def attrs(name, func):
    return _get_attrs(name, func, leaving=False)


residues = {}
for name in THREE_LETTER.values():
    canonical = _get_conformer(name, leaving=False, center=True, jitter=0.0, randrot=False)
    residues[name] = Residue(
        atoms=tuple(attrs(name, lambda at: at.GetProp("name"))),
        elements=np.array(attrs(name, lambda at: at.GetAtomicNum()), np.int64),
        charges=np.array(attrs(name, lambda at: at.GetFormalCharge()), np.float32),
        conformers=canonical.astype(np.float32)[None],
        extra={
            "ref_hydrogens": np.array(
                attrs(name, lambda at: at.GetTotalNumHs(includeNeighbors=True)), np.int64
            ),
            "ref_pos": canonical.astype(np.float64),
        },
    )
ResidueTable(residues).save(DATA / "promera.npz")
print({k: len(v.atoms) for k, v in residues.items()})
