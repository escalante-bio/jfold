"""Write `jfold/featurize/data/opendde.npz` from OpenDDE's CCD RDKit pickle. Run once:

    LAYERNORM_TYPE=torch uv run python tools/tables/extract_opendde.py
"""

import os

os.environ.setdefault("LAYERNORM_TYPE", "torch")

from extract_protenix import residue_table
from opendde.config.inference import build_inference_config
from opendde.data.constants import RES_ATOMS_DICT
from opendde.data.core import ccd

from jfold.featurize.residues import DATA

configs = build_inference_config(model_name="opendde_v1", fill_required_with_null=True)
ccd.set_ccd_cache_paths(
    components_file=configs.data.ccd_components_file,
    rdkit_mol_pkl=configs.data.ccd_components_rdkit_mol_file,
)
table = residue_table(ccd.get_component_rdkit_mol, RES_ATOMS_DICT)
table.save(DATA / "opendde.npz")
print({k: len(v.atoms) for k, v in table.items()})
