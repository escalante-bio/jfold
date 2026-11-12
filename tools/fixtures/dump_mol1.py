"""Write Mol-1's upstream features for the test systems to `tests/featurize/data/`.

    LAYERNORM_TYPE=torch uv run python tools/fixtures/dump_mol1.py [system ...]

The cases in `mol1_constraint_systems.json` are featurized as `mol cpred` does, with their
constraints.
"""

import json
import os
import random
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

os.environ.setdefault("LAYERNORM_TYPE", "torch")

import numpy as np

SEED = 0
HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parents[1] / "tests" / "featurize" / "data"
ROOT = Path(os.environ.get("MOL_ROOT_DIR", "~/.cache/jfold/mol1")).expanduser()


def template_json(spec: dict, length: int) -> list[dict] | None:
    """Return `spec` as Mol-1's template JSON, or None if it does not have `length` residues."""
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


def sample(
    name: str, system: dict, directory: Path, constraints: dict | None = None
) -> tuple[dict, bool] | None:
    """Return a `mol pred` input entry and whether any chain has an MSA, or None if a template
    does not fit its chain. With `constraints`, chains keep their own ids."""
    sequences, use_msa = [], False
    for chain in system["chains"]:
        entry = {"sequence": chain["sequence"], "count": 1}
        if constraints is not None:
            entry["id"] = [chain["id"]]
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
    task = {"name": name, "sequences": sequences}
    if constraints is not None:
        task["constraints"] = constraints
    return task, use_msa


def template_featurizer():
    """Build the `TemplateHitFeaturizer` that `InferenceDataset.__init__` builds."""
    from mol.data.template.template_utils import TemplateHitFeaturizer

    return TemplateHitFeaturizer(
        mmcif_dir=None,
        max_hits=4,
        max_template_date="2021-09-30",
        _shuffle_top_k_prefiltered=None,
        _max_template_candidates_num=20,
        fetch_remote=False,
    )


def featurize_system(
    name: str, system: dict, constraints: dict | None = None
) -> dict[str, np.ndarray] | None:
    import torch
    from mol.constraint.inference import ConstraintInferenceDataset
    from mol.constraint.inputs import prepare_inputs
    from mol.data.core import ccd
    from mol.data.inference.infer_dataloader import InferenceDataset

    common = ROOT / "common"
    use_template = any("template" in chain for chain in system["chains"])
    cls = InferenceDataset if constraints is None else ConstraintInferenceDataset
    dataset = cls.__new__(cls)  # Skip the config-driven setup.
    dataset.msa_pair_as_unpair = True
    dataset.use_rna_msa = False
    dataset.use_template = use_template
    dataset.online_template_featurizer = template_featurizer() if use_template else None
    with (
        tempfile.TemporaryDirectory() as tmp,
        mock.patch.object(ccd, "COMPONENTS_FILE", str(common / "components.cif")),
        mock.patch.object(ccd, "RKDIT_MOL_PKL", common / "components.cif.rdkit_mol.pkl"),
    ):
        sampled = sample(name, system, Path(tmp), constraints)
        if sampled is None:
            return None
        entry, dataset.use_msa = sampled
        if constraints is not None:
            enabled = {"constraint_exclusion_enable": True, "constraint_contact_enable": True}
            (entry,), (row,) = prepare_inputs([entry], config=enabled)
            dataset.configs = types.SimpleNamespace(**enabled)
            dataset.conditions = {name: row}
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
    """Compute Mol-1's relative-position feature."""
    import torch
    from mol.model.modules.embedders import RelativePositionEncoding

    module = types.SimpleNamespace(r_max=32, s_max=2)
    with torch.no_grad():
        out = RelativePositionEncoding.generate_relp(module, dict(feats))
    return np.asarray(out["relp"] if isinstance(out, dict) else out)


def main():
    systems = json.loads((FIXTURES / "systems.json").read_text())
    systems |= json.loads((FIXTURES / "template_systems.json").read_text())
    cases = {name: (system, None) for name, system in systems.items()}
    for name, case in json.loads((FIXTURES / "mol1_constraint_systems.json").read_text()).items():
        cases[name] = (systems[case["system"]], case["constraints"])
    if sys.argv[1:]:
        cases = {name: cases[name] for name in sys.argv[1:]}
    for name, (system, constraints) in cases.items():
        out = featurize_system(name, system, constraints)
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
