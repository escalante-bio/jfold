"""Upstream OpenDDE feature dicts for the shared fixture systems.

Runs OpenDDE's inference featurizer (`opendde.data.inference.infer_dataloader
.InferenceDataset.process_one`, which `get_inference_dataloader` calls per entry, as in
OpenDDE's inference runner and jfold's OpenDDE test) with the `opendde_v1` inference config,
templates off, on every system in `tests/featurize/data/systems.json`, and saves
`tests/featurize/data/opendde_<system>.npz` (numpy arrays only).

Run from `~/jfold`:

    LAYERNORM_TYPE=torch uv run python tools/fixtures/dump_opendde.py

Needs OpenDDE's CCD data (`configs.data.ccd_components_file` and its RDKit pickle,
`~/.cache/opendde/common/` by default; `opendde.utils.download.download_inference_cache`
fetches them, as `tests/checkpoints/test_checkpoint_opendde.py` does).

MSA: a chain with an `msa` passes the a3m as `proteinChain.unpairedMsaPath` and the
config runs with `use_msa=True`; systems without any MSA run with `use_msa=False`
(OpenDDE's single-sequence path, a dummy MSA from `restype`), as jfold's test does.

Determinism: the per-residue `ref_pos` augmentation draws from numpy's global RNG
(`np.random.uniform` translation, scipy `Rotation.random()` rotation). numpy, `random`
and torch are seeded with `SEED` right before each system, and `process_one` runs in this
process (no DataLoader workers), so the files are reproducible bit-for-bit.

Keys: every tensor of `input_feature_dict` as a numpy array under its own name, plus
`aux_*` extras (not upstream features): `aux_ref_pos_conformer` (the reference conformer
per atom before centring and augmentation, `atom_array.ref_pos`), `aux_atom_name`,
`aux_res_name`, `aux_element`, `aux_seed`, and `aux_relp` / `aux_structural_relp`
(the residue- and structural-token relative-position features the model computes from
the dict, `RelativePositionEncoding.generate_relp`, r_max=32, s_max=2).
"""

import json
import os
import random
import tempfile
from pathlib import Path

os.environ.setdefault("LAYERNORM_TYPE", "torch")

import numpy as np

SEED = 0
HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parents[1] / "tests" / "featurize" / "data"


def biotite_compat():
    """OpenDDE's featurizer reaches into `BondList` internals that biotite 1.7 (a compiled
    extension) does not expose; restore the few it uses (as jfold's OpenDDE test does)."""
    import biotite.structure as struc

    bonds = struc.BondList
    if hasattr(bonds, "_bonds"):
        return

    def set_bonds(self, value):
        self.remove_bonds(self.copy())
        for i, j, bond_type in value:
            self.add_bond(int(i), int(j), int(bond_type))

    def max_bonds_per_atom(self):
        return int(np.bincount(self.as_array()[:, :2].ravel(), minlength=1).max())

    bonds._bonds = property(lambda self: self.as_array(), set_bonds)
    bonds._remove_redundant_bonds = lambda self: None
    bonds._get_max_bonds_per_atom = max_bonds_per_atom
    bonds._max_bonds_per_atom = property(max_bonds_per_atom, lambda self, value: None)
    bonds._atom_count = property(lambda self: self.get_atom_count())


def configs(use_msa: bool):
    from opendde.config.inference import build_inference_config

    cfg = build_inference_config(model_name="opendde_v1", fill_required_with_null=True)
    cfg.use_msa = use_msa
    cfg.use_template = cfg.use_rna_msa = False
    return cfg


def sample(name: str, system: dict) -> tuple[dict, bool]:
    """An OpenDDE input-JSON entry and whether any chain has an MSA."""
    sequences, use_msa = [], False
    for chain in system["chains"]:
        entry = {"sequence": chain["sequence"], "count": 1}
        if "msa" in chain:
            entry["unpairedMsaPath"] = str(FIXTURES / chain["msa"])
            use_msa = True
        sequences.append({"proteinChain": entry})
    return {"name": name, "modelSeeds": [SEED], "sequences": sequences}, use_msa


def relp(feats: dict) -> tuple[np.ndarray, np.ndarray]:
    """The model's residue and structural-token relative-position features."""
    import types

    import torch
    from opendde.model.opendde import RelativePositionEncoding

    module = types.SimpleNamespace(r_max=32, s_max=2)
    ids = ("asym_id", "residue_index", "entity_id", "sym_id")
    residue = {k: feats[k] for k in (*ids, "token_index")}
    # As `OpenDDE.expand_to_structural_tokens` builds the structural dict.
    parent = feats["parent_residue_idx"].long()
    structural = {k: feats[k].index_select(-1, parent) for k in ids}
    structural["token_index"] = feats["structural_token_index"].long()
    with torch.no_grad():
        residue = RelativePositionEncoding.generate_relp(module, residue, lazy=False)
        structural = RelativePositionEncoding.generate_relp(module, structural, lazy=False)
    return residue["relp"].numpy(), structural["relp"].numpy()


def featurize_system(name: str, system: dict) -> dict[str, np.ndarray]:
    import torch
    from opendde.data.inference.infer_dataloader import InferenceDataset

    biotite_compat()
    entry, use_msa = sample(name, system)
    cfg = configs(use_msa)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "input.json"
        path.write_text(json.dumps([entry]))
        cfg.input_json_path, cfg.seeds, cfg.dump_dir = str(path), [SEED], tmp
        dataset = InferenceDataset(configs=cfg)
        np.random.seed(SEED)
        random.seed(SEED)
        torch.manual_seed(SEED)
        data, atom_array, _ = dataset.process_one(dataset.inputs[0])
    feats = data["input_feature_dict"]
    out = {}
    for key, value in feats.items():
        if torch.is_tensor(value):
            out[key] = value.numpy()
        elif isinstance(value, np.ndarray):
            out[key] = value
        else:
            raise TypeError(f"{key}: {type(value)}")
    residue_relp, structural_relp = relp(feats)
    out["aux_relp"] = residue_relp
    if structural_relp is not None:
        out["aux_structural_relp"] = structural_relp
    out["aux_ref_pos_conformer"] = np.asarray(atom_array.ref_pos, np.float32)
    out["aux_atom_name"] = np.asarray(atom_array.atom_name, str)
    out["aux_res_name"] = np.asarray(atom_array.res_name, str)
    out["aux_element"] = np.asarray(atom_array.element, str)
    out["aux_seed"] = np.asarray(SEED)
    return out


def main():
    systems = json.loads((FIXTURES / "systems.json").read_text())
    for name, system in systems.items():
        out = featurize_system(name, system)
        path = FIXTURES / f"opendde_{name}.npz"
        np.savez_compressed(path, **out)
        print(
            f"{path}: {len(out)} arrays, {out['asym_id'].shape[0]} tokens, "
            f"{out['parent_residue_idx'].shape[0]} structural tokens, "
            f"{out['ref_pos'].shape[0]} atoms, msa {out['msa'].shape}"
        )


if __name__ == "__main__":
    main()
