"""ESMFold2's residue table (`jfold/featurize/data/esmfold2.npz`) from ESMFold2's own CCD
pickle (`biohub/ESMFold2`'s `ccd.pkl`, not Boltz's): for each standard residue and UNK, the
heavy atoms in upstream's order (`PROTEIN_HEAVY_ATOMS`: N CA C O first, no OXT; UNK as N CA C
O), their elements and formal charges, and the raw "Computed" conformer (else "Ideal", else
conformer 0) after `RemoveHs(sanitize=False)`, with no centring. Checks every atom against
upstream's own lookup (`get_idealized_atom_pos`, `_infer_element`, `CHARGED_ATOMS`). Needs
the `esm` package, RDKit and the checkpoint in the Hugging Face cache; run once:

    uv run python tools/tables/extract_esmfold2.py
"""

import os
import pickle

import numpy as np
from huggingface_hub import hf_hub_download
from rdkit import Chem

ccd_path = os.environ.get("ESMCFOLD_CCD_PATH") or hf_hub_download(
    repo_id="biohub/ESMFold2", filename="ccd.pkl"
)
os.environ["ESMCFOLD_CCD_PATH"] = ccd_path  # upstream's lookup reads the same pickle

from esm.models.esmfold2 import conformers
from esm.models.esmfold2.constants import (
    CHARGED_ATOMS,
    ELEMENT_TO_ATOMIC_NUM,
    PROTEIN_HEAVY_ATOMS,
    PROTEIN_RESIDUE_TO_RES_TYPE,
    PROTEIN_UNK_RES_TYPE,
)
from esm.models.esmfold2.prepare_input import _infer_element

from jfold.featurize.residues import DATA, Residue, ResidueTable
from jfold.featurize.system import THREE_LETTER


def conformer_index(mol) -> int:
    names = [c.GetPropsAsDict().get("name") for c in mol.GetConformers()]
    for wanted in ("Computed", "Ideal"):
        if wanted in names:
            return names.index(wanted)
    return 0


with open(ccd_path, "rb") as f:
    ccd = pickle.load(f)

residues = {}
for name in THREE_LETTER.values():
    mol = ccd[name]
    index = conformer_index(mol)
    mol = Chem.RemoveHs(mol, sanitize=False)
    conformer = mol.GetConformer(min(index, mol.GetNumConformers() - 1))
    by_name = {a.GetProp("name"): a for a in mol.GetAtoms() if a.HasProp("name")}
    atoms = PROTEIN_HEAVY_ATOMS[name]
    coords = np.array(
        [list(conformer.GetAtomPosition(by_name[a].GetIdx())) for a in atoms], np.float32
    )
    elements = np.array([by_name[a].GetAtomicNum() for a in atoms], np.int64)
    charges = np.array([by_name[a].GetFormalCharge() for a in atoms], np.int64)

    # Upstream's featurizer takes elements from the atom names and charges from a fixed
    # table; both agree with the CCD for these residues.
    res_type = PROTEIN_RESIDUE_TO_RES_TYPE.get(name, PROTEIN_UNK_RES_TYPE)
    expected = [conformers.get_idealized_atom_pos(res_type, a) for a in atoms]
    assert np.array_equal(coords, np.stack(expected)), name
    assert elements.tolist() == [ELEMENT_TO_ATOMIC_NUM[_infer_element(a)] for a in atoms], name
    assert charges.tolist() == [CHARGED_ATOMS.get((name, a), 0) for a in atoms], name

    residues[name] = Residue(
        atoms=tuple(atoms), elements=elements, charges=charges, conformers=coords[None]
    )

ResidueTable(residues).save(DATA / "esmfold2.npz")
print(ccd_path)
print({k: len(v.atoms) for k, v in residues.items()})
