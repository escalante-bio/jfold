"""ProMERA's residue table (`jfold/featurize/data/promera.npz`) from tinyprot's CCD, as
`tinyprot.feature.AF3Featurizer` reads it for polymer residues (`leaving=False`): heavy
atoms in CCD order without leaving atoms (so no OXT), their elements, formal charges and
bonded-hydrogen counts (leaving hydrogens included), and the CCD ideal coordinates centred
on those atoms. X uses the CCD `UNK` entry (6 atoms). Needs the `tinyprot` package and its
CCD database (`tinyprot.init.download_pretrained()`); run once:

    uv run python tools/tables/extract_promera.py

Stored per residue: `conformers` `[1, A, 3]` float32 (the canonical conformer), and in
`extra`, `ref_hydrogens` `[A]` int64 and `ref_pos` `[A, 3]` float64, the same centred
coordinates before tinyprot's float32 cast, so that posing them reproduces upstream's
float64 arithmetic.
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
