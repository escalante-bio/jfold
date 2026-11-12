"""Test small AtlasFold and AtlasFold-M models with random weights against the upstream."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from reference import STEPS, assert_finite, assert_round_trip, assert_schedule, randomize, rel_err

from jfold.slots import TrunkState

atlasfold_model = pytest.importorskip("atlasfold.model")

# Two chains of 8 and 4 residues, unpadded (12 is a multiple of the 4-residue atom windows),
# and three chains, one repeated, padded as the runner pads.
UNPADDED = (("MKTAYIGK", "GSXM"), (1, 2), (1, 2), (1, 1), None)
PADDED = (("MKTAYIGK", "GSHMW", "MKTAYIGK"), (1, 2, 1), (1, 2, 3), (1, 1, 2), 28)
NUM_TEMPLATES = 2


def _upstream(multimer: bool, seed: int):
    from atlasfold.model import AtlasFold, AtlasFold_Multimer
    from atlasfold.model import model as monomer
    from atlasfold.model import model_multimer as multi
    from atlaslm.model import AtlasLM

    torch.manual_seed(seed)
    lm = randomize(AtlasLM(d_model=64, n_heads=4, n_layers=3), seed)
    configs = multi if multimer else monomer
    cfg = (multi.AtlasFoldMultimerConfig if multimer else monomer.AtlasFoldConfig)(
        channel_s=32,
        channel_s_lm=48,
        channel_z=16,
        trunk=configs.TrunkConfig(
            num_heads=4, num_tri_heads=2, num_lm_blocks=2, num_blocks=3, num_pair_to_single_blocks=1
        ),
        diffusion_head=configs.DiffusionHeadConfig(
            channel_a=32,
            channel_atom=16,
            channel_cond=24,
            num_heads=4,
            num_blocks=2,
            num_atom_heads=2,
            num_atom_blocks=2,
        ),
        distogram_head=configs.DistogramHeadConfig(),
        confidence_head=configs.ConfidenceHeadConfig(num_heads=4, num_tri_heads=2, num_blocks=1),
        **(
            {"template_module": multi.TemplateModuleConfig(channel_template=8, num_tri_heads=2)}
            if multimer
            else {}
        ),
    )
    model = (AtlasFold_Multimer if multimer else AtlasFold)(cfg, lm=lm)
    return randomize(model, seed + 1)


def _batch(system, *, templates: bool, seed: int = 0) -> dict[str, np.ndarray]:
    """Featurize the complex with upstream's `featurize_complex`, adding random templates."""
    from atlasfold.common import featurize

    sequences, entity, asym, sym, length = system
    feat = featurize.featurize_complex(list(sequences), list(entity), list(asym), list(sym))
    if length is not None:
        feat = _pad(feat, length, length + 7)
    if templates:
        rng = np.random.default_rng(seed)
        num_residues = feat["aatype"].shape[0]
        shape = (NUM_TEMPLATES, num_residues)
        types = rng.integers(0, 21, shape)
        feat |= {
            "template.mask": np.array([True, False]),
            "template.aatype": np.eye(21, dtype=np.float32)[types],
            "template.pseudo_beta": 8 * rng.standard_normal((*shape, 3)).astype(np.float32),
            "template.pseudo_beta_mask": rng.random(shape) < 0.8,
            "template.backbone_coords": 8 * rng.standard_normal((*shape, 3, 3)).astype(np.float32),
            "template.backbone_frame_mask": rng.random(shape) < 0.8,
        }
    return feat


def _pad(feat, residues: int, lm_tokens: int):
    from atlasfold.runner_multimer import MultimerFoldingRunner

    return MultimerFoldingRunner._pad_to_lengths(feat, residue_length=residues, lm_length=lm_tokens)


def _torch_batch(feat):
    return {k: torch.as_tensor(v)[None] for k, v in feat.items()}


def _convert(m):
    from atlasfold.model.network.diffusion_head import SamplingConfig

    from jfold.convert import atlasfold as convert

    model = convert.structure_model(m, SamplingConfig(), mlm_rate=0.2, num_steps=200)
    model = eqx.tree_at(
        lambda x: x.trunk.atlaslm, model, convert.atlaslm(m.lm), is_leaf=lambda x: x is None
    )
    return model


def _masks(feat, num: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((num, feat["lm.input_ids"].shape[0])) < 0.3


@pytest.mark.parametrize("multimer", [True, False])
@pytest.mark.parametrize("system", [UNPADDED, PADDED], ids=["unpadded", "padded"])
def test_trunk(multimer, system):
    """Check the language model's single and pair, and two trunk passes with their own MLM masks."""
    from jfold.models import atlasfold

    m = _upstream(multimer, seed=1)
    feat = _batch(system, templates=multimer)
    batch = _torch_batch(feat)
    real = feat["seq_mask"].astype(bool)
    masks = _masks(feat, 2, seed=2)
    with torch.no_grad():
        m.compute_rel_pos_encoding(batch)
        s_lm, z_lm = m.run_lm_embedder(batch, torch.as_tensor(masks[:1]))
        s_lm, z_lm = m.proj_s_lm(s_lm)[0].numpy(), m.proj_z_lm(z_lm)[0].numpy()
        draws = iter(torch.as_tensor(masks[:, None]))
        m.sample_mlm_mask = lambda *args, **kwargs: next(draws)
        s, z = (x[0].numpy() for x in m.run_trunk(batch, num_recycles=1, mlm_prob=0.2))

    model = _convert(m)
    features = atlasfold.features(feat)
    lm_s, lm_z = model.trunk.language_model(
        model.trunk.atlaslm,
        features.lm,
        jnp.array(masks[0]),
        features.asym_id,
        features.token_mask,
    )
    inputs = model.embed(features)
    state = model.trunk.init_state(inputs)
    for mask_rows in masks:
        state = model.trunk.masked(features, inputs, state, mask_rows)
    assert_finite(state)
    errors = {
        "s_lm": rel_err(lm_s[real], s_lm[real]),
        "z_lm": rel_err(lm_z[real][:, real], z_lm[real][:, real]),
        "s": rel_err(state.s[real], s[real]),
        "z": rel_err(state.z[real][:, real], z[real][:, real]),
    }
    print(errors)
    for name, err in errors.items():
        assert err < 1e-4, (name, err)


@pytest.mark.parametrize("multimer", [True, False])
@pytest.mark.parametrize("system", [UNPADDED, PADDED], ids=["unpadded", "padded"])
def test_heads(multimer, system):
    """Check a denoising step, the confidence logits and the distogram from a random trunk."""
    from jfold.models import atlasfold

    m = _upstream(multimer, seed=3)
    feat = _batch(system, templates=False)
    batch = _torch_batch(feat)
    real = feat["seq_mask"].astype(bool)
    atoms = feat["atom14_mask"].astype(bool)
    num = real.shape[0]
    rng = np.random.default_rng(4)
    s = rng.standard_normal((num, 32)).astype(np.float32)
    z = rng.standard_normal((num, num, 16)).astype(np.float32)
    x_noisy = 12 * rng.standard_normal((num, 14, 3)).astype(np.float32)
    t = 7.5
    with torch.no_grad():
        m.compute_rel_pos_encoding(batch)
        head = m.diffusion_head
        s_t, z_t = torch.as_tensor(s)[None], torch.as_tensor(z)[None]
        pair_bias = head.pair_conditioning(batch, z_t)
        c_noise = torch.tensor(head.c_noise(t)).view(1, 1)
        single_cond = head.single_conditioning(batch, s_t, c_noise)
        denoised = head.inference_step(
            batch, torch.as_tensor(x_noisy)[None, None], t, single_cond, pair_bias
        )[0, 0].numpy()
        confidence = m.confidence_head(batch, s_t, z_t, torch.as_tensor(denoised)[None, None])
        distogram = m.distogram_head(z_t)["logits"][0].numpy()

    model = _convert(m)
    features = atlasfold.features(feat)
    inputs = model.embed(features)
    state = TrunkState(s=jnp.array(s), z=jnp.array(z))
    context = model.diffusion.prepare(features, inputs, state)
    ours = model.diffusion(context, jnp.array(x_noisy.reshape(-1, 3)), jnp.array(t))
    ours = atlasfold.atom14(ours)
    conf = model.confidence_logits(features, inputs, state, jnp.array(denoised.reshape(-1, 3)))
    pair = real[:, None] & real[None, :]
    errors = {
        "denoised": rel_err(ours[atoms], denoised[atoms]),
        "plddt": rel_err(conf.plddt[real], confidence["plddt"]["logits"][0, 0][real]),
        "resolved": rel_err(
            conf.resolved[real], confidence["experimentally_resolved"]["logits"][0, 0][real]
        ),
        "pae": rel_err(conf.pae[pair], confidence["pae"]["logits"][0, 0][pair]),
        "distogram": rel_err(model.distogram(state.z)[pair], distogram[pair]),
    }
    if multimer:
        errors["pde"] = rel_err(conf.pde[pair], confidence["pde"]["logits"][0, 0][pair])
    else:
        assert conf.pde is None
    print(errors)
    for name, err in errors.items():
        assert err < 1e-4, (name, err)


def test_padding_is_ignored():
    """Check that garbage in padded residues, atoms and language-model rows changes no output."""
    from jfold.models import atlasfold

    m = _upstream(True, seed=5)
    model = _convert(m)
    feat = _batch(PADDED, templates=True)
    features = atlasfold.features(feat)
    real = np.asarray(features.token_mask)
    rng = np.random.default_rng(6)

    def garbage(x):
        if not jnp.issubdtype(x.dtype, jnp.floating):
            return x
        return jnp.where(
            np.broadcast_to(_padding(x, real), x.shape), rng.standard_normal(x.shape), x
        )

    def _padding(x, real):
        if x.ndim >= 1 and x.shape[0] == real.shape[0]:
            pad = ~real.reshape(-1, *[1] * (x.ndim - 1))
            if x.ndim >= 2 and x.shape[1] == real.shape[0]:
                pad = pad | ~real.reshape(1, -1, *[1] * (x.ndim - 2))
            return pad
        return np.zeros((1,) * x.ndim, bool)

    lm_padding = (feat["lm.seq_id"] == 0)[:, None]
    noisy = dataclasses.replace(
        features,
        residues=garbage(features.residues),
        relative_positions=garbage(features.relative_positions),
        template_features=jnp.where(
            np.broadcast_to(
                _padding(features.template_features[0], real),
                (1, *features.template_features.shape[1:]),
            ),
            rng.standard_normal(features.template_features.shape),
            features.template_features,
        ),
        lm=dataclasses.replace(
            features.lm,
            tokens=jnp.where(lm_padding, rng.random(features.lm.tokens.shape), features.lm.tokens),
        ),
    )
    mask_rows = jnp.array(_masks(feat, 1, seed=7)[0])

    def run(f):
        inputs = model.embed(f)
        state = model.trunk.masked(f, inputs, model.trunk.init_state(inputs), mask_rows)
        context = model.diffusion.prepare(f, inputs, state)
        x = jnp.array(rng.standard_normal((real.shape[0] * 14, 3)), jnp.float32)
        denoised = model.diffusion(context, x, jnp.array(3.0))
        return state, denoised, model.confidence_logits(f, inputs, state, denoised)

    rng_state = rng.bit_generator.state
    clean = run(features)
    rng.bit_generator.state = rng_state
    dirty = run(noisy)
    atoms = np.asarray(features.flat_atom_mask)
    pair = real[:, None] & real[None, :]
    np.testing.assert_allclose(dirty[0].z[pair], clean[0].z[pair], atol=1e-4)
    np.testing.assert_allclose(dirty[1][atoms], clean[1][atoms], atol=1e-4)
    np.testing.assert_allclose(dirty[2].pae[pair], clean[2].pae[pair], atol=1e-4)


def test_atlaslm():
    """Check AtlasLM's embeddings with packed chains and rotary positions restarting per chain."""
    from atlaslm.model import AtlasLM

    from jfold.convert import atlasfold as convert
    from jfold.models import atlasfold

    lm = randomize(AtlasLM(d_model=64, n_heads=4, n_layers=3), 8)
    feat = _batch(PADDED, templates=False)
    with torch.no_grad():
        ref = (
            lm(
                torch.as_tensor(feat["lm.input_ids"])[None],
                seq_id=torch.as_tensor(feat["lm.seq_id"])[None],
                pos_id=torch.as_tensor(feat["lm.pos_id"])[None],
            )
            .embeddings[0]
            .numpy()
        )
    features = atlasfold.features(feat)
    ours = convert.atlaslm(lm)(features.lm.tokens, features.lm.positions, features.lm.sequence_id)
    real = feat["lm.seq_id"] > 0
    assert rel_err(ours[real], ref[real]) < 1e-5


def test_sampling_and_round_trip():
    from atlasfold.model.network.diffusion_head import SamplingConfig

    model = _convert(_upstream(True, seed=9))
    assert_schedule(model.sampling, SamplingConfig(num_steps=STEPS).get_sigmas())
    assert_round_trip(model)


def test_structure_model():
    """Check that the converted model runs as a `StructureModel`, also after a save and load."""
    from jfold.models import atlasfold
    from jfold.pretrained.recipes.atlasfold import _attach_lm, _detach_lm

    model = _convert(_upstream(True, seed=12))
    features = atlasfold.features(_batch(UNPADDED, templates=False))
    k_trunk, k_sample = jax.random.split(jax.random.key(13))
    inputs, state = model.run_trunk(features, num_passes=2, key=k_trunk)
    coords = model.structure_sampler(features, inputs, state, num_steps=4).sample(k_sample)
    assert_finite((state, coords, model.confidence_logits(features, inputs, state, coords)))

    stored = _detach_lm(model)
    assert stored.trunk.atlaslm is None
    with pytest.raises(ValueError, match="no AtlasLM"):
        stored.trunk(features, inputs, state, key=k_trunk)
    restored = _attach_lm(stored, lambda name: model.trunk.atlaslm)
    assert jax.tree.structure(restored) == jax.tree.structure(model)


def test_soft_sequence_reaches_the_language_model():
    from jfold.models import atlasfold

    model = _convert(_upstream(True, seed=10))
    feat = _batch(UNPADDED, templates=False)
    real = feat["seq_mask"].astype(bool)
    soft = jax.nn.softmax(np.random.default_rng(11).standard_normal(feat["aatype"].shape))
    f = atlasfold.features(feat | {"aatype": soft})
    rows = feat["seq_tok_idx"][real]
    np.testing.assert_allclose(f.lm.tokens[rows], atlasfold.lm_residues(soft[real]), atol=1e-6)
    hard = atlasfold.features(feat)
    np.testing.assert_array_equal(hard.lm.tokens, jax.nn.one_hot(feat["lm.input_ids"], 64))
    mask_rows = jnp.zeros(feat["lm.input_ids"].shape[0], bool)

    def loss(residues):
        f = atlasfold.features(feat | {"aatype": residues})
        inputs = model.embed(f)
        state = model.trunk.masked(f, inputs, model.trunk.init_state(inputs), mask_rows)
        return state.z.sum()

    grad = np.asarray(jax.grad(loss)(soft))
    assert np.isfinite(grad).all() and np.abs(grad[real]).sum() > 0
