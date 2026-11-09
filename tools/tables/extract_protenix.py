"""Protenix's residue table (`jfold/featurize/data/protenix.npz`) from its CCD RDKit pickle
(`components.v20240608.cif.rdkit_mol.pkl`): for each of the 21 residue types (X is `UNK`),
the heavy atoms in CCD order (`RES_ATOMS_DICT`, `OXT` last), their elements and formal
charges, `ref_mask` (in `extra`), and the reference conformer `ref_conf_id` (2, the RDKit
one), as float32, as `json_parser.add_reference_features` stores it. Views drop `OXT`
except on a chain's last residue. Needs the `protenix` package; run once:

    LAYERNORM_TYPE=torch uv run python tools/tables/extract_protenix.py

`extract_mol1.py` and `extract_opendde.py` reuse `residue_table` with their own pickles.
"""

import os

os.environ.setdefault("LAYERNORM_TYPE", "torch")

import numpy as np

from jfold.featurize.residues import DATA, Residue, ResidueTable
from jfold.featurize.system import THREE_LETTER


def residue_table(get_mol, res_atoms: dict[str, dict[str, int]]) -> ResidueTable:
    """The table from `get_mol(ccd_code)` (an upstream `get_component_rdkit_mol`) and the
    pipeline's `RES_ATOMS_DICT` (atom name to position within the residue)."""
    residues = {}
    for name in THREE_LETTER.values():
        mol = get_mol(name)
        assert mol.ref_conf_id == 2 and mol.ref_conf_type == "rdkit", name
        order = sorted(res_atoms[name], key=res_atoms[name].get)
        assert [res_atoms[name][a] for a in order] == list(range(len(order))), name
        assert order[-1] == "OXT", name
        index = [mol.atom_map[a] for a in order]
        # The heavy atoms of the RDKit mol are exactly these, in this (CCD) order.
        heavy = [i for i, atom in enumerate(mol.GetAtoms()) if atom.GetAtomicNum() > 1]
        assert index == heavy, (name, index, heavy)
        atoms = [mol.GetAtomWithIdx(i) for i in index]
        coords = mol.GetConformer(mol.ref_conf_id).GetPositions()[index]
        residues[name] = Residue(
            atoms=tuple(order),
            elements=np.array([a.GetAtomicNum() for a in atoms], np.int64),
            charges=np.array([a.GetFormalCharge() for a in atoms], np.float32),
            conformers=coords.astype(np.float32)[None],
            extra={"ref_mask": np.asarray(mol.ref_mask, np.int64)[index]},
        )
    return ResidueTable(residues)


if __name__ == "__main__":
    from protenix.data import ccd
    from protenix.data.constants import RES_ATOMS_DICT

    table = residue_table(ccd.get_component_rdkit_mol, RES_ATOMS_DICT)
    table.save(DATA / "protenix.npz")
    print({k: len(v.atoms) for k, v in table.items()})
