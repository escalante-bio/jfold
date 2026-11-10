"""Mol-1's residue table (`jfold/featurize/data/mol1.npz`) from its own CCD RDKit pickle
(`$MOL_ROOT_DIR/common/components.cif.rdkit_mol.pkl`, default `~/.cache/jfold/mol1`, where
`tests/checkpoints/test_checkpoint_mol1.py` downloads it): the same layout as
`extract_protenix.py`, with Mol-1's conformers. Needs the `molence` package; run once:

    LAYERNORM_TYPE=torch uv run python tools/tables/extract_mol1.py
"""

import os
from pathlib import Path

os.environ.setdefault("LAYERNORM_TYPE", "torch")

from extract_protenix import residue_table
from mol.data.constants import RES_ATOMS_DICT
from mol.data.core import ccd

from jfold.featurize.residues import DATA

ROOT = Path(os.environ.get("MOL_ROOT_DIR", "~/.cache/jfold/mol1")).expanduser()

ccd.RKDIT_MOL_PKL = ROOT / "common" / "components.cif.rdkit_mol.pkl"
ccd._ccd_rdkit_mols = {}
table = residue_table(ccd.get_component_rdkit_mol, RES_ATOMS_DICT)
table.save(DATA / "mol1.npz")
print({k: len(v.atoms) for k, v in table.items()})
