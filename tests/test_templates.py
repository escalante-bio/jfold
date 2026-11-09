"""Template embedder parity against each upstream PyTorch implementation."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from reference import assert_finite, randomize, rel_err

from jfold.convert import boltz, openfold3, protenix
from jfold.models import boltz2 as boltz2_features
from jfold.models import openfold3 as openfold3_features
from jfold.models import protenix as protenix_features

NUM_TEMPLATES, NUM_TOKENS = 2, 9
C_T, C_Z = 8, 16
t = torch.from_numpy


@dataclasses.dataclass
class Templates:
    """Templates in every upstream's vocabulary: geometry (Boltz-2 derives its pair features
    from it) and precomputed pair features (the others')."""

    aatype: np.ndarray  # [N, T]
    distogram: np.ndarray  # [N, T, T, 39] one-hot
    pseudo_beta_mask: np.ndarray  # [N, T]
    backbone_frame_mask: np.ndarray  # [N, T]
    unit_vector: np.ndarray  # [N, T, T, 3]
    frame_rot: np.ndarray  # [N, T, 3, 3]
    frame_t: np.ndarray
    ca: np.ndarray
    cb: np.ndarray
    visibility_ids: np.ndarray  # [N, T]
    asym_id: np.ndarray  # [T]
    z: np.ndarray  # [T, T, Cz]

    def pair(self, x):
        return x[:, :, None] * x[:, None, :]


def make_templates(num_templates, num_tokens, seed) -> Templates:
    rng = np.random.default_rng(seed)
    f32 = lambda *shape: rng.standard_normal(shape, dtype=np.float32)
    rot, _ = np.linalg.qr(rng.standard_normal((num_templates, num_tokens, 3, 3)))
    unit = f32(num_templates, num_tokens, num_tokens, 3)
    return Templates(
        aatype=rng.integers(0, 32, (num_templates, num_tokens)),
        distogram=np.eye(39, dtype=np.float32)[
            rng.integers(0, 39, (num_templates, num_tokens, num_tokens))
        ],
        pseudo_beta_mask=(rng.random((num_templates, num_tokens)) < 0.9).astype(np.float32),
        backbone_frame_mask=(rng.random((num_templates, num_tokens)) < 0.9).astype(np.float32),
        unit_vector=unit / np.linalg.norm(unit, axis=-1, keepdims=True),
        frame_rot=rot.astype(np.float32),
        frame_t=10 * f32(num_templates, num_tokens, 3),
        ca=10 * f32(num_templates, num_tokens, 3),
        cb=10 * f32(num_templates, num_tokens, 3),
        visibility_ids=rng.integers(0, 2, (num_templates, num_tokens)),
        asym_id=(np.arange(num_tokens) >= num_tokens // 2).astype(np.int64),
        z=f32(num_tokens, num_tokens, C_Z),
    )


def padded(x: Templates, templates: int, tokens: int) -> Templates:
    """`x` with `templates` garbage templates and `tokens` garbage tokens appended."""
    n, k = x.aatype.shape
    g = make_templates(n + templates, k + tokens, seed=9)
    for f in dataclasses.fields(x):
        big, small = getattr(g, f.name), getattr(x, f.name)
        if f.name == "asym_id":
            big[:k] = small
        elif f.name == "z":
            big[:k, :k] = small
        elif small.ndim >= 3 and small.shape[2] == k:  # [N, T, T, ...]
            big[:n, :k, :k] = small
        else:
            big[:n, :k] = small
    return g


def _protenix_lineage(module, **kwargs):
    m = module.TemplateEmbedder(n_blocks=2, c=C_T, c_z=C_Z, **kwargs)

    def run(x: Templates):
        feats = {
            "template_aatype": t(x.aatype),
            "template_distogram": t(x.distogram),
            "template_pseudo_beta_mask": t(x.pair(x.pseudo_beta_mask)),
            "template_unit_vector": t(x.unit_vector),
            "template_backbone_frame_mask": t(x.pair(x.backbone_frame_mask)),
            "asym_id": t(x.asym_id),
        }
        n = x.asym_id.shape[0]
        return m(feats, t(x.z), torch.ones(n, n))

    def features(x: Templates):
        return protenix_features.template_features(
            x.distogram,
            x.pair(x.pseudo_beta_mask),
            x.aatype,
            x.unit_vector,
            x.pair(x.backbone_frame_mask),
            x.asym_id,
        )

    return m, run, protenix.template_embedder, features


def _protenix():
    import protenix.model.modules.pairformer as module

    return _protenix_lineage(module)


def _opendde():
    import opendde.model.modules.pairformer as module

    return _protenix_lineage(module)


def _mol1():
    import mol.model.modules.pairformer as module

    return _protenix_lineage(module)


def _boltz2():
    from boltz.model.modules.trunkv2 import TemplateV2Module

    m = TemplateV2Module(token_z=C_Z, template_dim=C_T, template_blocks=2)

    def restype(x):
        return np.eye(33, dtype=np.float32)[x.aatype]

    def run(x: Templates):
        n, k = x.aatype.shape
        feats = {
            "template_restype": t(restype(x)),
            "template_frame_rot": t(x.frame_rot),
            "template_frame_t": t(x.frame_t),
            "template_mask_frame": t(x.backbone_frame_mask),
            "template_cb": t(x.cb),
            "template_ca": t(x.ca),
            "template_mask_cb": t(x.pseudo_beta_mask),
            "visibility_ids": t(x.visibility_ids),
            "template_mask": torch.ones(n, k),
        }
        return m(t(x.z)[None], {k: v[None] for k, v in feats.items()}, torch.ones(1, k, k))[0]

    def features(x: Templates):
        return boltz2_features.template_features(
            restype(x),
            x.frame_rot,
            x.frame_t,
            x.backbone_frame_mask,
            x.cb,
            x.ca,
            x.pseudo_beta_mask,
            x.visibility_ids,
        )

    return m, run, boltz.template_embedder, features


def _openfold3():
    import ml_collections
    from jopenfold3._vendor.openfold3.core.model.latent.template_module import (
        TemplateEmbedderAllAtom,
    )

    config = ml_collections.ConfigDict(
        {
            "c_t": C_T,
            "c_z": C_Z,
            "template_pair_embedder": {"c_in": C_Z, "c_dgram": 39, "c_aatype": 32, "c_out": C_T},
            "template_pair_stack": {
                "c_t": C_T,
                "c_hidden_tri_att": 4,
                "c_hidden_tri_mul": 8,
                "no_blocks": 2,
                "no_heads": 2,
                "transition_type": "swiglu",
                "pair_transition_n": 2,
                "dropout_rate": 0.25,
                "tri_mul_first": True,
                "fuse_projection_weights": False,
                "blocks_per_ckpt": None,
                "ckpt_per_template": False,
                "inf": 1e9,
            },
        }
    )
    m = TemplateEmbedderAllAtom(config)

    def restype(x):
        return np.eye(32, dtype=np.float32)[x.aatype]

    def run(x: Templates):
        k = x.asym_id.shape[0]
        batch = {
            "template_restype": t(restype(x)),
            "template_distogram": t(x.distogram),
            "template_pseudo_beta_mask": t(x.pseudo_beta_mask),
            "template_unit_vector": t(x.unit_vector),
            "template_backbone_frame_mask": t(x.backbone_frame_mask),
            "asym_id": t(x.asym_id),
        }
        return m({k: v[None] for k, v in batch.items()}, t(x.z)[None], torch.ones(1, k, k))[0]

    def features(x: Templates):
        return openfold3_features.template_features(
            x.distogram,
            x.pseudo_beta_mask,
            restype(x),
            x.unit_vector,
            x.backbone_frame_mask,
            x.asym_id,
        )

    return m, run, openfold3.template_embedder, features


CASES = {
    "boltz2": _boltz2,
    "protenix": _protenix,
    "opendde": _opendde,
    "mol1": _mol1,
    "openfold3": _openfold3,
}


@eqx.filter_jit
def _forward(module, z, features, template_mask, mask):
    return module(z, features, template_mask, mask, key=jax.random.key(0))


@pytest.mark.parametrize("padding", [(0, 0), (1, 2)], ids=["unpadded", "padded"])
@pytest.mark.parametrize("case", CASES)
def test_template_embedder_matches_reference(case, padding):
    """The template embedding matches the upstream on the real tokens, with a masked garbage
    template and garbage tokens appended; outputs and gradients stay finite."""
    module, run_reference, convert, features = CASES[case]()
    randomize(module, seed=0)
    ours = convert(module)
    x = make_templates(NUM_TEMPLATES, NUM_TOKENS, seed=1)
    with torch.no_grad():
        ref = run_reference(x)

    templates, tokens = padding
    x_in = padded(x, templates, tokens) if templates or tokens else x
    n, k = x_in.aatype.shape
    args = (jnp.array(features(x_in)), jnp.arange(n) < NUM_TEMPLATES, jnp.arange(k) < NUM_TOKENS)
    out = _forward(ours, jnp.array(x_in.z), *args)

    assert rel_err(out[:NUM_TOKENS, :NUM_TOKENS], ref) < 1e-5
    grads = jax.grad(lambda z: _forward(ours, z, *args)[:NUM_TOKENS, :NUM_TOKENS].sum())(
        jnp.array(x_in.z)
    )
    assert_finite((out, grads))


@pytest.mark.checkpoint
def test_constant_template_mask_under_jit():
    """A released template embedder jitted with an all-real template mask made inside the
    jitted function (as `jfold.models.protenix.features` makes it) gives its eager output.
    XLA's GPU GEMM fusion once miscompiled exactly this case."""
    from test_featurize import system

    from jfold import pretrained
    from jfold.featurize.views import mol1 as view
    from jfold.models import mol1

    embedder = pretrained.load("mol1_base").trunk.templates
    f = mol1.features(view.features(system("monomer_template_unresolved")))
    tf, mask = f.template_features, f.token_mask
    n = mask.shape[0]
    z = jax.random.normal(jax.random.key(0), (n, n, embedder.norm_z.weight.shape[0]))
    all_real = lambda: jnp.ones(tf.shape[0], bool)
    eager = embedder(z, tf, all_real(), mask)
    jitted = eqx.filter_jit(lambda e, z, tf: e(z, tf, all_real(), mask))(embedder, z, tf)
    assert rel_err(jitted, eager) < 1e-5
