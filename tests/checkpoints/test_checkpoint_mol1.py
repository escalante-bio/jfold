"""Compare jfold with upstream Mol-1 on the released weights.

Both `mol-1-base` and `mol-1-constraint` are tested on a small two-chain complex
featurized by Mol-1, the constraint model with SITE, EXCLUSION and CONTACT all set. The
test compares the trunk, one denoising step, confidence and distogram. It downloads the
checkpoints and about 3 GB of reference data into `$MOL_ROOT_DIR` (by default
`~/.cache/jfold/mol1`). Run it with `pytest -m checkpoint`."""

import os
import shutil
import tarfile
import urllib.request
from pathlib import Path
from unittest import mock

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from reference import STEPS, assert_round_trip, assert_schedule, rel_err

pytestmark = pytest.mark.checkpoint

SEQUENCES = ("MKTAYIAKQRQISFVKSHFSRQ", "GSHMLEDPV")
NUM_RECYCLES = 2
NOISE_LEVEL = 10.0
KEY = jax.random.key(0)
RELEASE_URL = "https://mol-1-release.oss-cn-shanghai.aliyuncs.com"


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as f:
        shutil.copyfileobj(response, f)
    partial.rename(path)


CONSTRAINTS = {
    "site": {"source_chains": ["A"], "partner_chains": ["B"], "residue_ids": [["A", 3], ["A", 7]]},
    "exclusion": {
        "source_chains": ["A"],
        "partner_chains": ["B"],
        "residue_ids": [["A", 15], ["A", 18]],
    },
    "contact": [(("A", 10), ("B", 4))],
}


def _release_dir(name: str) -> Path:
    """Download the checkpoint and the reference data if missing, and return their root."""
    root = Path(os.environ.get("MOL_ROOT_DIR", "~/.cache/jfold/mol1")).expanduser()
    checkpoint = root / "checkpoint" / f"{name}.pt"
    if not checkpoint.exists():
        _download(f"{RELEASE_URL}/checkpoint/{name}.pt", checkpoint)
    if not (root / "common" / "components.cif").exists():
        archive = root / "common.tar.gz"
        _download(f"{RELEASE_URL}/common.tar.gz", archive)
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")
        archive.unlink()
    return root


def _reference_model(release: str, checkpoint: Path):
    """Build the upstream model the way `jfold.pretrained` does."""
    from jfold import pretrained

    return pretrained.upstream(release, checkpoint_path=checkpoint)


def _featurize(common: Path, constraints: dict | None = None) -> dict:
    """Featurize `SEQUENCES` with Mol-1, without MSAs or templates.

    If `constraints` is given, add their tensors the way `mol cpred` does."""
    from mol.data.core import ccd
    from mol.data.inference.infer_dataloader import InferenceDataset

    sample = {
        "name": "complex",
        "sequences": [{"proteinChain": {"sequence": s, "count": 1}} for s in SEQUENCES],
    }
    dataset = InferenceDataset.__new__(InferenceDataset)  # skip the config-driven setup
    dataset.use_msa = False
    dataset.msa_pair_as_unpair = True
    dataset.use_rna_msa = False
    dataset.use_template = False
    dataset.online_template_featurizer = None
    with (
        mock.patch.object(ccd, "COMPONENTS_FILE", str(common / "components.cif")),
        mock.patch.object(ccd, "RKDIT_MOL_PKL", common / "components.cif.rdkit_mol.pkl"),
    ):
        data, atoms, _ = dataset.process_one(sample)
    features = data["input_feature_dict"]
    if constraints is not None:
        from mol.constraint import features as constraint_features

        # Upstream's input parser adds the task name to each constraint.
        site, exclusion = (
            {**constraints[k], "target_id": "complex"} for k in ("site", "exclusion")
        )
        features, _ = constraint_features.project_sites(features, atoms, site, site["residue_ids"])
        features, _ = constraint_features.exclusion_project(
            features, atoms, exclusion, exclusion["residue_ids"]
        )
        features, _ = constraint_features.contact_project(
            features, atoms, [list(pair) for pair in constraints["contact"]]
        )
    return {k: v.numpy() for k, v in features.items()}


def _trunk(model, features):
    inputs = model.embed(features, key=KEY)
    state = model.trunk.init_state(inputs)
    for _ in range(NUM_RECYCLES):
        state = model.trunk(features, inputs, state, key=KEY)
    return inputs, state


@pytest.mark.parametrize(
    "release, name, constraints",
    [("mol1_base", "mol-1-base", None), ("mol1_constraint", "mol-1-constraint", CONSTRAINTS)],
)
def test_mol1(release, name, constraints):
    from mol.model.mol import update_input_feature_dict

    from jfold.convert import protenix
    from jfold.models import mol1 as mol1_features
    from jfold.slots import TrunkState

    root = _release_dir(name)
    reference = _reference_model(release, root / "checkpoint" / f"{name}.pt")
    feats = _featurize(root / "common", constraints)
    t = {k: torch.from_numpy(v) for k, v in feats.items()}
    if constraints is not None:
        assert feats["site_mask"] and feats["exclusion_mask"] and feats["contact_pair_mask"].any()
        with torch.no_grad():
            update = reference.get_constraint_update(t)
        assert update.abs().max() > 0
    t = update_input_feature_dict(reference.relative_position_encoding.generate_relp(t))
    n_atoms = feats["ref_pos"].shape[0]
    x_noisy = 16 * np.random.default_rng(0).standard_normal((n_atoms, 3), dtype=np.float32)
    with torch.no_grad():
        s_inputs, s, z = reference.get_pairformer_output(t, N_cycle=NUM_RECYCLES)
        denoised = reference.diffusion_module(
            torch.from_numpy(x_noisy)[None],
            torch.tensor([NOISE_LEVEL]),
            t,
            s_inputs,
            s,
            z,
            pair_z=None,
            p_lm=None,
            c_l=None,
        )[0]
        plddt, pae, pde, resolved = reference.confidence_head(
            input_feature_dict=t,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=denoised[None],
        )
        distogram = reference.distogram_head(z)

    model = protenix.structure_model(reference)

    model = assert_round_trip(model)
    assert_schedule(model.sampling, reference.inference_noise_scheduler(N_step=STEPS, device="cpu"))
    features = mol1_features.features(feats)
    inputs, state = eqx.filter_jit(_trunk)(model, features)
    errors = {}
    if constraints is not None:
        errors["constraint"] = rel_err(model.trunk_init.constraints(features.pair_features), update)
    errors |= {
        "s_inputs": rel_err(inputs.s_inputs, s_inputs),
        "s": rel_err(state.s, s),
        "z": rel_err(state.z, z),
    }

    # Later stages from the reference trunk, so errors do not compound.
    ref_state = TrunkState(s=jnp.array(s.numpy()), z=jnp.array(z.numpy()))
    context = model.diffusion.prepare(features, inputs, ref_state)
    ours = model.diffusion(context, jnp.array(x_noisy), jnp.array(NOISE_LEVEL))
    errors["denoised"] = rel_err(ours, denoised)
    conf = model.confidence_logits(
        features, inputs, ref_state, jnp.array(denoised.numpy()), key=KEY
    )
    errors["pae"] = rel_err(conf.pae, pae[0])
    errors["pde"] = rel_err(conf.pde, pde[0])
    errors["plddt"] = rel_err(conf.plddt, plddt[0])
    errors["resolved"] = rel_err(conf.resolved, resolved[0])
    errors["distogram"] = rel_err(model.distogram(ref_state.z), distogram)
    print(" ".join(f"{k}={v:.1e}" for k, v in errors.items()))

    tolerances = {"constraint": 1e-5, "s_inputs": 1e-4, "denoised": 1e-4, "distogram": 1e-4}
    for stage, err in errors.items():
        assert err < tolerances.get(stage, 1e-3), (stage, err)
