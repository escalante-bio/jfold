"""End-to-end parity with the released AtlasFold-M and AtlasFold checkpoints on AtlasLM-3B:
AtlasLM's features, two trunk passes, one denoising step, confidence and distogram, on a
small complex (AtlasFold-M) and a monomer (AtlasFold), featurized and padded by the
upstream runners. Downloads weights; run with `pytest -m checkpoint`. The reference runs on
the CPU in float32."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from reference import STEPS, assert_round_trip, assert_schedule, rel_err

pytestmark = pytest.mark.checkpoint

SYSTEMS = {
    "atlasfold_m": ("MKTAYIAKQRQISFVKSHFSRQ", "GSHMLEDPV"),
    "atlasfold": ("MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPD",),
}
NOISE_LEVEL = 10.0


def _batch(name: str) -> dict[str, np.ndarray]:
    """The runner's features for one target, in its length bucket (unbatched)."""
    from atlasfold.common.protein import ProteinMultimer
    from atlasfold.runner import FoldingRunner
    from atlasfold.runner_multimer import MultimerFoldingRunner

    chains = SYSTEMS[name]
    length = sum(map(len, chains))
    if name == "atlasfold_m":
        runner = MultimerFoldingRunner
        target = [ProteinMultimer.get_empty(name, list(chains))]
    else:
        runner, target = FoldingRunner, list(chains)
    batch = runner._make_batch_features(target, runner._get_length_bucket(length))
    return {k: v[0] for k, v in batch.items()}


@pytest.mark.parametrize("name", ["atlasfold_m", "atlasfold"])
def test_atlasfold(name):
    from atlasfold.model.network.diffusion_head import SamplingConfig

    from jfold import pretrained
    from jfold.models import atlasfold
    from jfold.pretrained.recipes.atlasfold import MLM_RATE, NUM_STEPS, _convert, _convert_lm

    reference = pretrained.upstream(name)
    feat = _batch(name)
    real = feat["seq_mask"].astype(bool)
    assert not real.all()  # padding is exercised
    atoms = feat["atom14_mask"].astype(bool)
    pair = real[:, None] & real[None, :]
    rng = np.random.default_rng(0)
    masks = rng.random((2, feat["lm.input_ids"].shape[0])) < MLM_RATE[name]
    x_noisy = 16 * rng.standard_normal((*atoms.shape, 3)).astype(np.float32)

    batch = {k: torch.as_tensor(v)[None] for k, v in feat.items()}
    with torch.no_grad():
        reference.compute_rel_pos_encoding(batch)
        s_lm, z_lm = reference.run_lm_embedder(batch, torch.as_tensor(masks[:1]))
        s_lm, z_lm = reference.proj_s_lm(s_lm), reference.proj_z_lm(z_lm)
        draws = iter(torch.as_tensor(masks[:, None]))
        reference.sample_mlm_mask = lambda *args, **kwargs: next(draws)
        s, z = reference.run_trunk(batch, num_recycles=1, mlm_prob=MLM_RATE[name])
        head = reference.diffusion_head
        pair_bias = head.pair_conditioning(batch, z)
        c_noise = torch.tensor(head.c_noise(NOISE_LEVEL)).view(1, 1)
        single_cond = head.single_conditioning(batch, s, c_noise)
        denoised = head.inference_step(
            batch, torch.as_tensor(x_noisy)[None, None], NOISE_LEVEL, single_cond, pair_bias
        )
        confidence = reference.confidence_head(batch, s, z, denoised)
        distogram = reference.distogram_head(z)["logits"]
    ref = {
        "s_lm": s_lm[0].numpy(),
        "z_lm": z_lm[0].numpy(),
        "s": s[0].numpy(),
        "z": z[0].numpy(),
        "denoised": denoised[0, 0].numpy(),
        "plddt": confidence["plddt"]["logits"][0, 0].numpy(),
        "resolved": confidence["experimentally_resolved"]["logits"][0, 0].numpy(),
        "pae": confidence["pae"]["logits"][0, 0].numpy(),
        "distogram": distogram[0].numpy(),
    }
    if "pde" in confidence:
        ref["pde"] = confidence["pde"]["logits"][0, 0].numpy()
    del batch, s_lm, z_lm, s, z, pair_bias, single_cond, denoised, confidence, distogram

    model = _convert(name, reference)
    assert_schedule(model.sampling, SamplingConfig(num_steps=STEPS).get_sigmas())
    model = assert_round_trip(model)  # as stored: without AtlasLM
    model = eqx.tree_at(
        lambda m: m.trunk.atlaslm,
        model,
        _convert_lm(reference.lm),
        is_leaf=lambda x: x is None,
    )
    del reference
    assert model.trunk.mlm_rate == MLM_RATE[name] and model.sampling.num_steps == NUM_STEPS[name]

    features = atlasfold.features(feat)
    lm = model.trunk.language_model
    ours_s_lm, ours_z_lm = eqx.filter_jit(lm)(
        model.trunk.atlaslm,
        features.lm,
        jnp.array(masks[0]),
        features.asym_id,
        features.token_mask,
    )

    @eqx.filter_jit
    def trunk(model, features, masks):
        inputs = model.embed(features)
        state = model.trunk.init_state(inputs)
        for mask_rows in masks:
            state = model.trunk.masked(features, inputs, state, mask_rows)
        return inputs, state

    inputs, state = trunk(model, features, jnp.array(masks))
    errors = {
        "s_lm": rel_err(ours_s_lm[real], ref["s_lm"][real]),
        "z_lm": rel_err(ours_z_lm[pair], ref["z_lm"][pair]),
        "s": rel_err(state.s[real], ref["s"][real]),
        "z": rel_err(state.z[pair], ref["z"][pair]),
    }

    # Later stages from the reference trunk, so errors do not compound.
    ref_state = type(state)(s=jnp.array(ref["s"]), z=jnp.array(ref["z"]))

    @eqx.filter_jit
    def heads(model, features, inputs, state, x_noisy, denoised):
        context = model.diffusion.prepare(features, inputs, state)
        ours = model.diffusion(context, x_noisy, jnp.array(NOISE_LEVEL))
        conf = model.confidence_logits(features, inputs, state, denoised)
        return ours, conf, model.distogram(state.z)

    ours, conf, distogram = heads(
        model,
        features,
        inputs,
        ref_state,
        jnp.array(x_noisy.reshape(-1, 3)),
        jnp.array(ref["denoised"].reshape(-1, 3)),
    )
    errors["denoised"] = rel_err(atlasfold.atom14(ours)[atoms], ref["denoised"][atoms])
    errors["plddt"] = rel_err(conf.plddt[real], ref["plddt"][real])
    errors["resolved"] = rel_err(conf.resolved[real], ref["resolved"][real])
    errors["pae"] = rel_err(conf.pae[pair], ref["pae"][pair])
    if "pde" in ref:
        errors["pde"] = rel_err(conf.pde[pair], ref["pde"][pair])
    else:
        assert conf.pde is None
    errors["distogram"] = rel_err(distogram[pair], ref["distogram"][pair])
    print(errors)
    jax.clear_caches()

    for part, err in errors.items():
        assert err < 1e-3, (part, err)
