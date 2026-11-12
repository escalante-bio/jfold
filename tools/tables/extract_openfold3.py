"""Write `jfold/featurize/data/openfold3.npz` with OpenFold3's vendored code.

Conformer `s` is the one upstream embeds after `random.seed(s)`, so the table is
reproducible for a given RDKit build. Run once:

    uv run python tools/tables/extract_openfold3.py
"""

import random

import numpy as np
from jopenfold3._vendor.openfold3.core.data.primitives.structure.query import (
    atom_array_from_ccd_code,
    processed_reference_molecule_from_atom_array,
)
from jopenfold3._vendor.openfold3.core.data.resources.residues import (
    MOLECULE_TYPE_TO_LEAVING_ATOMS,
    MoleculeType,
)
from rdkit import RDLogger

from jfold.featurize.residues import DATA, Residue, ResidueTable
from jfold.featurize.system import THREE_LETTER

NUM_CONFORMERS = 32
LEAVING = MOLECULE_TYPE_TO_LEAVING_ATOMS[MoleculeType.PROTEIN]  # ["OXT"]

RDLogger.DisableLog("rdApp.*")


def conformer(atom_array, seed: int):
    """Return upstream's reference molecule for one residue after `random.seed(seed)`, and its
    kept-atom mask."""
    random.seed(seed)
    ref = processed_reference_molecule_from_atom_array(atom_array, atoms_to_mask=LEAVING)
    return ref.mol, np.asarray(ref.in_crop_mask, bool)


residues = {}
for name in THREE_LETTER.values():
    atom_array = atom_array_from_ccd_code(name, chain_id="A", molecule_type=MoleculeType.PROTEIN)
    mol, keep = conformer(atom_array, 0)
    atoms = [a for a, k in zip(atom_array.atom_name, keep, strict=True) if k]
    mol_atoms = [a for a, k in zip(mol.GetAtoms(), keep, strict=True) if k]
    assert [a.GetProp("annot_atom_name") for a in mol_atoms] == atoms
    coords = []
    for seed in range(NUM_CONFORMERS):
        mol, _ = conformer(atom_array, seed)
        x = mol.GetConformer().GetPositions()[keep].astype(np.float32)
        coords.append(x - x.mean(0))
    residues[name] = Residue(
        atoms=tuple(str(a) for a in atoms),
        elements=np.array([a.GetAtomicNum() for a in mol_atoms]),
        charges=np.array([a.GetFormalCharge() for a in mol_atoms], np.float32),
        conformers=np.stack(coords),
    )
ResidueTable(residues).save(DATA / "openfold3.npz")
print({k: len(v.atoms) for k, v in residues.items()})
