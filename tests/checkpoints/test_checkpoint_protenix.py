"""End-to-end parity with the released Protenix checkpoints: the trunk, one denoising step,
confidence and distogram on a small real complex. Downloads weights; run with
`pytest -m checkpoint`."""

from pathlib import Path

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


def _trunk(model, features):
    inputs = model.embed(features, key=KEY)
    state = model.trunk.init_state(inputs)
    for _ in range(NUM_RECYCLES):
        state = model.trunk(features, inputs, state, key=KEY)
    return inputs, state


def _reference(name: str):
    """The released model as `jfold.pretrained` builds it (as `protenix predict` does), from
    where that command keeps its checkpoints (`~/.protenix` for the larger ones)."""
    from jfold import pretrained
    from jfold.pretrained.recipes.protenix import NAMES, configs

    upstream = NAMES[name]
    checkpoint = Path.home() / ".protenix" / f"{upstream}.pt"
    if not checkpoint.exists():
        checkpoint = Path(configs(upstream).load_checkpoint_dir) / f"{upstream}.pt"
    return pretrained.upstream(name, checkpoint_path=checkpoint)


@pytest.mark.parametrize(
    "name",
    ["protenix_mini", "protenix_tiny", "protenix_base", "protenix_base_20250630", "protenix_v2"],
)
def test_protenix(name):
    from protenix.data.template import ChainInput, featurize

    from jfold.convert import protenix
    from jfold.models import protenix as protenix_features
    from jfold.slots import TrunkState

    reference = _reference(name)
    feats, _, _ = featurize([ChainInput(s, compute_msa=False) for s in SEQUENCES])
    t = {
        k: torch.as_tensor(v.astype(np.float32) if v.dtype == np.float64 else v)
        for k, v in feats.items()
        if isinstance(v, np.ndarray)
    }
    t["constraint_feature"] = feats["constraint_feature"]
    n_atoms = feats["ref_pos"].shape[0]
    x_noisy = 16 * np.random.default_rng(0).standard_normal((n_atoms, 3), dtype=np.float32)
    with torch.no_grad():
        s_inputs, s, z = reference.get_pairformer_output(t, N_cycle=NUM_RECYCLES)
        denoised = reference.diffusion_module(
            torch.from_numpy(x_noisy)[None], torch.tensor([NOISE_LEVEL]), t, s_inputs, s, z
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
    features = protenix_features.features(feats)
    inputs, state = eqx.filter_jit(_trunk)(model, features)
    assert rel_err(inputs.s_inputs, s_inputs) < 1e-4
    assert rel_err(state.s, s) < 1e-3
    assert rel_err(state.z, z) < 1e-3

    # Later stages from the reference trunk, so errors do not compound.
    ref_state = TrunkState(s=jnp.array(s.numpy()), z=jnp.array(z.numpy()))
    ref_inputs = jax.tree.map(lambda x: x, inputs)
    context = model.diffusion.prepare(features, ref_inputs, ref_state)
    ours = model.diffusion(context, jnp.array(x_noisy), jnp.array(NOISE_LEVEL))
    assert rel_err(ours, denoised) < 1e-4
    conf = model.confidence_logits(
        features, ref_inputs, ref_state, jnp.array(denoised.numpy()), key=KEY
    )
    assert rel_err(conf.pae, pae[0]) < 1e-3
    assert rel_err(conf.pde, pde[0]) < 1e-3
    assert rel_err(conf.plddt, plddt[0]) < 1e-3
    assert rel_err(conf.resolved, resolved[0]) < 1e-3
    assert rel_err(model.distogram(ref_state.z), distogram) < 1e-4
