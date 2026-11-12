"""Write `jfold/featurize/data/protenix.npz` from Protenix's CCD RDKit pickle. Run once:

    LAYERNORM_TYPE=torch uv run python tools/tables/extract_protenix.py
"""

import os

os.environ.setdefault("LAYERNORM_TYPE", "torch")

import numpy as np

from jfold.featurize.residues import DATA, Residue, ResidueTable
from jfold.featurize.system import THREE_LETTER


def residue_table(get_mol, res_atoms: dict[str, dict[str, int]]) -> ResidueTable:
    """Build the residue table from an upstream `get_component_rdkit_mol` and `RES_ATOMS_DICT`."""
    residues = {}
    for name in THREE_LETTER.values():
        mol = get_mol(name)
        assert mol.ref_conf_id == 2 and mol.ref_conf_type == "rdkit", name
        order = sorted(res_atoms[name], key=res_atoms[name].get)
        assert [res_atoms[name][a] for a in order] == list(range(len(order))), name
        assert order[-1] == "OXT", name
        index = [mol.atom_map[a] for a in order]
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
