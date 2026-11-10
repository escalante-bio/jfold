"""End-to-end parity with the released Mol-1 weights (`mol-1-base`): the trunk, one denoising
step, confidence and distogram on a small two-chain complex, featurized by Mol-1. Downloads
the checkpoint and the reference data (`common/`, ~3 GB) into `$MOL_ROOT_DIR` (default
`~/.cache/jfold/mol1`); run with `pytest -m checkpoint`."""

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


def _release_dir() -> Path:
    """The checkpoint and the reference data featurization reads, downloaded if missing."""
    root = Path(os.environ.get("MOL_ROOT_DIR", "~/.cache/jfold/mol1")).expanduser()
    checkpoint = root / "checkpoint" / "mol-1-base.pt"
    if not checkpoint.exists():
        _download(f"{RELEASE_URL}/checkpoint/mol-1-base.pt", checkpoint)
    if not (root / "common" / "components.cif").exists():
        archive = root / "common.tar.gz"
        _download(f"{RELEASE_URL}/common.tar.gz", archive)
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")
        archive.unlink()
    return root


def _reference_model(checkpoint: Path):
    """`mol-1-base` as `jfold.pretrained` builds it (as `mol pred` does, with plain-PyTorch
    triangle kernels, on CPU)."""
    from jfold import pretrained

    return pretrained.upstream("mol1_base", checkpoint_path=checkpoint)


def _featurize(common: Path) -> dict:
    """Mol-1's feature dict (numpy) for `SEQUENCES`, single-sequence MSAs, no templates."""
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
        data, _, _ = dataset.process_one(sample)
    return {k: v.numpy() for k, v in data["input_feature_dict"].items()}


def _trunk(model, features):
    inputs = model.embed(features, key=KEY)
    state = model.trunk.init_state(inputs)
    for _ in range(NUM_RECYCLES):
        state = model.trunk(features, inputs, state, key=KEY)
    return inputs, state


def test_mol1_base():
    from mol.model.mol import update_input_feature_dict

    from jfold.convert import protenix
    from jfold.models import mol1 as mol1_features
    from jfold.slots import TrunkState

    root = _release_dir()
    reference = _reference_model(root / "checkpoint" / "mol-1-base.pt")
    feats = _featurize(root / "common")
    t = {k: torch.from_numpy(v) for k, v in feats.items()}
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
    errors = {
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

    tolerances = {"s_inputs": 1e-4, "denoised": 1e-4, "distogram": 1e-4}
    for name, err in errors.items():
        assert err < tolerances.get(name, 1e-3), (name, err)
