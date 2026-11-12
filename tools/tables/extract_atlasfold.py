"""Write `jfold/featurize/data/atlasfold.npz` from AtlasFold's residue constants. Run once:

    uv run python tools/tables/extract_atlasfold.py
"""

import numpy as np
from atlasfold.common import residue_constants as rc

from jfold.featurize.residues import DATA, Residue, ResidueTable
from jfold.featurize.system import THREE_LETTER

ELEMENTS = {"C": 6, "N": 7, "O": 8, "S": 16}

residues = {}
for name in THREE_LETTER.values():
    atoms = rc.residue_atoms[name]
    residues[name] = Residue(
        atoms=atoms,
        elements=np.array([ELEMENTS[a[0]] for a in atoms], np.int64),
        charges=np.zeros(len(atoms), np.float32),
        conformers=rc.restype_atom14_positions[name][None, : len(atoms)],
        extra={"atom37": np.array([rc.atom37_order[a] for a in atoms], np.int64)},
    )
    index = rc.restype_orders[rc.restype_3to1[name]]
    assert rc.restype_atom14_mask[index].sum() == len(atoms)
    assert rc.restype_atom37_mask[index].sum() == len(atoms)

ResidueTable(residues).save(DATA / "atlasfold.npz")
print(f"wrote {DATA / 'atlasfold.npz'}: {len(residues)} residues")
