"""Upstream Mol-1 feature dicts for the shared fixture systems.

Runs Mol-1's inference featurizer (`mol.data.inference.infer_dataloader.InferenceDataset
.process_one`, as `mol pred` and jfold's Mol-1 test call it, with templates off) on every
system in `tests/featurize/data/systems.json` and saves `tests/featurize/data/mol1_<system>.npz` (numpy arrays
only).

Run from `~/jfold`:

    LAYERNORM_TYPE=torch uv run python tools/fixtures/dump_mol1.py

Needs the Mol-1 reference data (`common/components.cif` and
`common/components.cif.rdkit_mol.pkl`) under `$MOL_ROOT_DIR` (default
`~/.cache/jfold/mol1`, where `tests/test_checkpoint_mol1.py` downloads it).

MSA: a chain with an `msa` passes the a3m as `proteinChain.unpairedMsaPath` and the
dataset runs with `use_msa=True` (Mol-1's own way to give a precomputed unpaired MSA);
systems without any MSA run with `use_msa=False`, Mol-1's single-sequence path (a dummy
MSA built from `restype`). No pairing happens: no `pairedMsa`/`pairedMsaPath` is given.

Determinism: the only random step for protein-only inputs is the per-residue `ref_pos`
augmentation (a translation from `np.random.uniform` and a rotation from scipy's
`Rotation.random()`, both on numpy's global RNG). numpy, `random` and torch are seeded
with `SEED` right before each system, so the files are reproducible bit-for-bit.

Templates (the systems in `template_systems.json`, PDB files from `make_templates.py`): a
templated chain's `templatesPath` is an AF3-style template JSON (`[{"mmcif",
"queryIndices", "templateIndices"}]`, which `TemplateHitFeaturizer.parse_json_templates`
reads), and the dataset runs with `use_template=True` and the `TemplateHitFeaturizer`
`InferenceDataset.__init__` builds. The mmCIF is the PDB file's chain (gemmi); the indices
map its residues to the chain's positions as jfold's view does: residue number = position
when the system gives the template's `sequence`, file order otherwise. Systems whose
template does not have its chain's length have no such mapping and get no file.

    LAYERNORM_TYPE=torch uv run python tools/fixtures/dump_mol1.py [system ...]

dumps the named systems only.

Keys: every tensor of `input_feature_dict` as a numpy array under its own name, plus
`aux_*` extras (not model inputs): `aux_ref_pos_conformer` (the reference conformer per
atom before centring and augmentation, `atom_array.ref_pos`), `aux_atom_name`,
`aux_res_name`, `aux_element`, `aux_seed`, and `aux_relp` (the relative-position
feature Mol-1's model computes from the dict, `generate_relp`, for checking the jfold
relative-position features).
"""

import json
import os
import random
import sys
import tempfile
from pathlib import Path
from unittest import mock

os.environ.setdefault("LAYERNORM_TYPE", "torch")

import numpy as np

SEED = 0
HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parents[1] / "tests" / "featurize" / "data"
ROOT = Path(os.environ.get("MOL_ROOT_DIR", "~/.cache/jfold/mol1")).expanduser()


def template_json(spec: dict, length: int) -> list[dict] | None:
    """The template `spec` as Mol-1's template JSON for a chain of `length` residues: the
    PDB file's chain as mmCIF, each of its residues mapped to its position in the template
    (its residue number with a `sequence`, its order without); None if the template does
    not have `length` residues."""
    import gemmi

    st = gemmi.read_structure(str(FIXTURES / spec["pdb"]))
    st.setup_entities()
    residues = list(st[0][0])
    if "sequence" in spec:
        size, positions = len(spec["sequence"]), [r.seqid.num - 1 for r in residues]
    else:
        size, positions = len(residues), list(range(len(residues)))
    if size != length:
        return None
    return [
        {
            "mmcif": st.make_mmcif_document().as_string(),
            "queryIndices": positions,
            "templateIndices": list(range(len(residues))),
        }
    ]


def sample(name: str, system: dict, directory: Path) -> tuple[dict, bool] | None:
    """A `mol pred` input entry (template JSONs written to `directory`) and whether any
    chain has an MSA; None if a template has no mapping to its chain."""
    sequences, use_msa = [], False
    for chain in system["chains"]:
        entry = {"sequence": chain["sequence"], "count": 1}
        if "msa" in chain:
            entry["unpairedMsaPath"] = str(FIXTURES / chain["msa"])
            use_msa = True
        if "template" in chain:
            templates = template_json(chain["template"], len(chain["sequence"]))
            if templates is None:
                return None
            path = directory / f"{chain['id']}.json"
            path.write_text(json.dumps(templates))
            entry["templatesPath"] = str(path)
        sequences.append({"proteinChain": entry})
    return {"name": name, "sequences": sequences}, use_msa


def template_featurizer():
    """The `TemplateHitFeaturizer` `InferenceDataset.__init__` builds (the template JSONs
    read only its centring setting)."""
    from mol.data.template.template_utils import TemplateHitFeaturizer

    return TemplateHitFeaturizer(
        mmcif_dir=None,
        max_hits=4,
        max_template_date="2021-09-30",
        _shuffle_top_k_prefiltered=None,
        _max_template_candidates_num=20,
        fetch_remote=False,
    )


def featurize_system(name: str, system: dict) -> dict[str, np.ndarray] | None:
    import torch
    from mol.data.core import ccd
    from mol.data.inference.infer_dataloader import InferenceDataset

    common = ROOT / "common"
    use_template = any("template" in chain for chain in system["chains"])
    dataset = InferenceDataset.__new__(InferenceDataset)  # skip the config-driven setup
    dataset.msa_pair_as_unpair = True
    dataset.use_rna_msa = False
    dataset.use_template = use_template
    dataset.online_template_featurizer = template_featurizer() if use_template else None
    with (
        tempfile.TemporaryDirectory() as tmp,
        mock.patch.object(ccd, "COMPONENTS_FILE", str(common / "components.cif")),
        mock.patch.object(ccd, "RKDIT_MOL_PKL", common / "components.cif.rdkit_mol.pkl"),
    ):
        sampled = sample(name, system, Path(tmp))
        if sampled is None:
            return None
        entry, dataset.use_msa = sampled
        np.random.seed(SEED)
        random.seed(SEED)
        torch.manual_seed(SEED)
        data, atom_array, _ = dataset.process_one(entry)
    feats = data["input_feature_dict"]
    out = {}
    for key, value in feats.items():
        if torch.is_tensor(value):
            out[key] = value.numpy()
        elif isinstance(value, np.ndarray):
            out[key] = value
        else:
            raise TypeError(f"{key}: {type(value)}")
    out["aux_relp"] = relp(feats)
    out["aux_ref_pos_conformer"] = np.asarray(atom_array.ref_pos, np.float32)
    out["aux_atom_name"] = np.asarray(atom_array.atom_name, str)
    out["aux_res_name"] = np.asarray(atom_array.res_name, str)
    out["aux_element"] = np.asarray(atom_array.element, str)
    out["aux_seed"] = np.asarray(SEED)
    return out


def relp(feats: dict) -> np.ndarray:
    """Mol-1's relative-position feature (`RelativePositionEncoding.generate_relp`,
    r_max=32, s_max=2) for the dict."""
    import types

    import torch
    from mol.model.modules.embedders import RelativePositionEncoding

    module = types.SimpleNamespace(r_max=32, s_max=2)
    with torch.no_grad():
        out = RelativePositionEncoding.generate_relp(module, dict(feats))
    return np.asarray(out["relp"] if isinstance(out, dict) else out)


def main():
    systems = json.loads((FIXTURES / "systems.json").read_text())
    systems |= json.loads((FIXTURES / "template_systems.json").read_text())
    if sys.argv[1:]:
        systems = {name: systems[name] for name in sys.argv[1:]}
    for name, system in systems.items():
        out = featurize_system(name, system)
        if out is None:
            print(f"{name}: a template without its chain's length, skipped")
            continue
        path = FIXTURES / f"mol1_{name}.npz"
        np.savez_compressed(path, **out)
        print(
            f"{path}: {len(out)} arrays, {out['asym_id'].shape[0]} tokens, "
            f"{out['ref_pos'].shape[0]} atoms, msa {out['msa'].shape}"
        )


if __name__ == "__main__":
    main()
