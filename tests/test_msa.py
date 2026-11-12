"""Test the MSA module against each upstream PyTorch implementation."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from reference import assert_finite, randomize, rel_err

from jfold.blocks.msa import subsample_rows
from jfold.convert import boltz, openfold3, protenix
from jfold.models import boltz2 as boltz2_features
from jfold.models import openfold3 as openfold3_features
from jfold.models import promera as promera_features
from jfold.models import protenix as protenix_features

NUM_ROWS, NUM_TOKENS = 5, 9
C_M, C_Z, C_S = 8, 32, 12  # ProMERA's cuequivariance triangle kernel needs C_Z % 32 == 0
t = torch.from_numpy


@dataclasses.dataclass
class MSA:
    msa: np.ndarray  # [S, T] residue classes
    has_deletion: np.ndarray
    deletion_value: np.ndarray
    is_paired: np.ndarray
    z: np.ndarray
    s_inputs: np.ndarray


def make_msa(num_rows, num_tokens, num_classes, seed) -> MSA:
    rng = np.random.default_rng(seed)
    f32 = lambda *shape: rng.standard_normal(shape, dtype=np.float32)
    return MSA(
        msa=rng.integers(0, num_classes, (num_rows, num_tokens)),
        has_deletion=(rng.random((num_rows, num_tokens)) < 0.2).astype(np.float32),
        deletion_value=rng.random((num_rows, num_tokens), dtype=np.float32),
        is_paired=(rng.random((num_rows, num_tokens)) < 0.5).astype(np.float32),
        z=f32(num_tokens, num_tokens, C_Z),
        s_inputs=f32(num_tokens, C_S),
    )


def padded(x: MSA, rows: int, tokens: int, num_classes: int) -> MSA:
    """Append `rows` garbage MSA rows and `tokens` garbage tokens to `x`."""
    s, n = x.msa.shape
    g = make_msa(s + rows, n + tokens, num_classes, seed=7)
    for f in ("msa", "has_deletion", "deletion_value", "is_paired"):
        getattr(g, f)[:s, :n] = getattr(x, f)
    g.z[:n, :n] = x.z
    g.s_inputs[:n] = x.s_inputs
    return g


def _boltz_feats(x: MSA):
    s, n = x.msa.shape
    feats = {
        "msa": t(x.msa),
        "has_deletion": t(x.has_deletion),
        "deletion_value": t(x.deletion_value),
        "msa_paired": t(x.is_paired),
        "msa_mask": torch.ones(s, n),
        "token_pad_mask": torch.ones(n),
    }
    return {k: v[None] for k, v in feats.items()}


def _boltz2():
    from boltz.model.modules.trunkv2 import MSAModule

    m = MSAModule(
        msa_s=C_M, token_z=C_Z, token_s=C_S, msa_blocks=2, msa_dropout=0.15, z_dropout=0.25
    )

    def run(x: MSA):
        return m(t(x.z)[None], t(x.s_inputs)[None], _boltz_feats(x))[0], None

    def features(x: MSA):
        return boltz2_features.msa_features(
            jax.nn.one_hot(x.msa, 33), x.has_deletion, x.deletion_value, x.is_paired
        )

    return m, run, boltz.msa_module, features, 33


def _promera():
    from promera.model.trunk import MSAModule

    m = MSAModule(
        msa_s=C_M, token_z=C_Z, s_input_dim=C_S, msa_blocks=2, msa_dropout=0.15, z_dropout=0.25
    )

    def run(x: MSA):
        # ProMERA's triangle updates exist only as cuequivariance CUDA kernels.
        m.cuda()
        feats = {k: v.cuda() for k, v in _boltz_feats(x).items()}
        s, z = m(t(x.z)[None].cuda(), t(x.s_inputs)[None].cuda(), feats)
        m.cpu()
        return z[0].cpu(), s[0].cpu()  # s: the mean of the final MSA over rows

    def features(x: MSA):
        return promera_features.msa_features(
            jax.nn.one_hot(x.msa, 34), x.has_deletion, x.deletion_value
        )

    return m, run, boltz.msa_module, features, 34


def _protenix_lineage(module, configs, outer_product_first=True):
    m = module.MSAModule(n_blocks=2, c_m=C_M, c_z=C_Z, c_s_inputs=C_S, msa_configs=configs)

    def run(x: MSA):
        feats = {
            "msa": t(x.msa),
            "has_deletion": t(x.has_deletion),
            "deletion_value": t(x.deletion_value),
        }
        n = x.msa.shape[1]
        return m(feats, t(x.z), t(x.s_inputs), torch.ones(n, n)), None

    def features(x: MSA):
        return protenix_features.msa_features(
            jax.nn.one_hot(x.msa, 32), x.has_deletion, x.deletion_value
        )

    convert = lambda m: protenix.msa_module(m, outer_product_first=outer_product_first)
    return m, run, convert, features, 32


# Keep every row (their order does not change the pair).
_ALL_ROWS = {
    "enable": True,
    "strategy": "random",
    "sample_cutoff": {"train": 64, "test": 64},
    "min_size": {"train": 64, "test": 64},  # the upstream draws a random row count above this
}


def _protenix():
    import protenix.model.modules.pairformer as module

    return _protenix_lineage(module, _ALL_ROWS)


def _mol1():
    import mol.model.modules.pairformer as module

    return _protenix_lineage(module, _ALL_ROWS)


def _opendde():
    import opendde.model.modules.pairformer as module

    return _protenix_lineage(module, {"msa_depth": 64}, outer_product_first=False)


def _openfold3():
    from jopenfold3._vendor.openfold3.core.model.feature_embedders.input_embedders import (
        MSAModuleEmbedder,
    )
    from jopenfold3._vendor.openfold3.core.model.latent.msa_module import MSAModuleStack

    embedder = MSAModuleEmbedder(
        c_m_feats=34,
        c_m=C_M,
        c_s_input=C_S,
        subsample_main_msa=False,
        subsample_all_msa=False,
        min_subsampled_all_msa=1,
        max_subsampled_all_msa=64,
    )
    stack = MSAModuleStack(
        c_m=C_M,
        c_z=C_Z,
        c_hidden_msa_att=4,
        c_hidden_opm=4,
        c_hidden_mul=8,
        c_hidden_pair_att=4,
        no_heads_msa=2,
        no_heads_pair=2,
        no_blocks=2,
        transition_type="swiglu",
        transition_n=4,
        msa_dropout=0.15,
        pair_dropout=0.25,
        opm_first=True,
        fuse_projection_weights=False,
        blocks_per_ckpt=None,
        inf=1e9,
        eps=1e-3,
    )

    def run(x: MSA):
        s, n = x.msa.shape
        batch = {
            "msa": t(np.eye(32, dtype=np.float32)[x.msa]),
            "has_deletion": t(x.has_deletion),
            "deletion_value": t(x.deletion_value),
            "msa_mask": torch.ones(s, n),
        }
        m, msa_mask = embedder({k: v[None] for k, v in batch.items()}, t(x.s_inputs)[None])
        z = stack(m, t(x.z)[None], msa_mask, torch.ones(1, n, n))
        return z[0], None

    def features(x: MSA):
        return openfold3_features.msa_features(
            jax.nn.one_hot(x.msa, 32), x.has_deletion, x.deletion_value
        )

    modules = torch.nn.ModuleList([embedder, stack])
    return modules, run, lambda m: openfold3.msa_module(m[0], m[1]), features, 32


CASES = {
    "boltz2": _boltz2,
    "promera": _promera,
    "protenix": _protenix,
    "opendde": _opendde,
    "mol1": _mol1,
    "openfold3": _openfold3,
}


@eqx.filter_jit
def _forward(module, z, features, s_inputs, msa_mask, mask):
    return module(z, features, s_inputs, msa_mask, mask, key=jax.random.key(0))


@pytest.mark.parametrize("padding", [(0, 0), (3, 2)], ids=["unpadded", "padded"])
@pytest.mark.parametrize("case", CASES)
def test_msa_module_matches_reference(case, padding):
    """Check that the pair after the MSA module matches the upstream on the real tokens."""
    module, run_reference, convert, features, num_classes = CASES[case]()
    randomize(module, seed=0)
    ours = convert(module)
    x = make_msa(NUM_ROWS, NUM_TOKENS, num_classes, seed=1)
    with torch.no_grad():
        z_ref, msa_mean_ref = run_reference(x)

    rows, tokens = padding
    x_in = padded(x, rows, tokens, num_classes) if rows or tokens else x
    s, n = x_in.msa.shape
    mask = jnp.arange(n) < NUM_TOKENS
    msa_mask = (jnp.arange(s) < NUM_ROWS)[:, None] & mask[None, :]
    args = (jnp.array(features(x_in)), jnp.array(x_in.s_inputs), msa_mask, mask)
    m, z = _forward(ours, jnp.array(x_in.z), *args)

    assert rel_err(z[:NUM_TOKENS, :NUM_TOKENS], z_ref) < 1e-5
    if msa_mean_ref is not None:
        mean = (m * msa_mask[..., None]).sum(0) / (msa_mask.sum(0)[:, None] + 1e-5)
        assert rel_err(mean[:NUM_TOKENS], msa_mean_ref) < 1e-5
    grads = jax.grad(lambda z: _forward(ours, z, *args)[1][:NUM_TOKENS, :NUM_TOKENS].sum())(
        jnp.array(x_in.z)
    )
    assert_finite((z, grads))


def test_subsample_rows_prefers_real_rows():
    msa_mask = jnp.zeros((40, 3), bool).at[:25].set(True)
    for keep_query in (True, False):
        draws = set()
        for seed in range(5):
            key = jax.random.key(seed)
            rows = subsample_rows(msa_mask, 10, key, keep_query=keep_query).tolist()
            assert len(set(rows)) == 10 and max(rows) < 25
            assert rows[0] == 0 or not keep_query
            draws.add(tuple(sorted(rows)))
        assert len(draws) > 1
        # Asking for more rows than are real gives every real row, then padding.
        rows = subsample_rows(msa_mask, 30, jax.random.key(0), keep_query=keep_query)
        assert set(range(25)) <= set(rows.tolist())


def _module_and_msa(num_rows):
    module, _, convert, features, num_classes = _protenix()
    randomize(module, seed=0)
    x = make_msa(num_rows, NUM_TOKENS, num_classes, seed=1)
    msa = (jnp.array(features(x)), jnp.array(x.s_inputs), jnp.array(x.z))
    full = dataclasses.replace(convert(module), max_rows=None, min_rows=None)
    return full, msa


@pytest.mark.parametrize("min_rows", [None, 2], ids=["fixed", "random_depth"])
def test_msa_module_draws_on_every_call(min_rows):
    """Check that a module drawing rows equals the full module run on the rows it reads."""
    full, (f, s, z0) = _module_and_msa(12)
    drawing = dataclasses.replace(full, max_rows=8, min_rows=min_rows)
    mask = jnp.ones(NUM_TOKENS, bool)
    msa_mask = jnp.ones((12, NUM_TOKENS), bool)

    key = jax.random.key(3)
    _, z = drawing(z0, f, s, msa_mask, mask, key=key)
    rows, drawn = drawing.draw_rows(msa_mask, jax.random.split(key, 3)[0])
    rows = rows[drawn.any(-1)]
    assert len(rows) == 8 if min_rows is None else 2 <= len(rows) <= 8
    _, z_rows = full(z0, f[rows], s, msa_mask[rows], mask, key=key)
    assert rel_err(z, z_rows) < 1e-5
    _, z_other = drawing(z0, f, s, msa_mask, mask, key=jax.random.key(4))
    assert rel_err(z_other, z) > 1e-3


def test_msa_module_draws_only_from_a_deeper_msa():
    """Check that a row draw needs a key only for an MSA deeper than it reads."""
    full, (f, s, z0) = _module_and_msa(6)
    mask = jnp.ones(NUM_TOKENS, bool)
    msa_mask = jnp.ones((6, NUM_TOKENS), bool)
    _, z = full(z0, f, s, msa_mask, mask)
    for fields in ({"max_rows": 6}, {"max_rows": 8, "min_rows": 6}):
        _, z_all = dataclasses.replace(full, **fields)(z0, f, s, msa_mask, mask)
        assert rel_err(z_all, z) < 1e-6
    for fields in ({"max_rows": 5}, {"min_rows": 5}):
        with pytest.raises(ValueError, match="pass a key"):
            dataclasses.replace(full, **fields)(z0, f, s, msa_mask, mask)


# Compare upstream's row draws at inference with the converted module's (`MSAModule.draw_rows`)
# on an MSA deeper than a pass reads. Each upstream run reports which rows it read.
DRAWS, DEEP = 2000, 20


class _Read(Exception):
    """Raised by a hook once it has seen the rows a pass reads."""


def _rows_reaching(layer, channel: int, run) -> np.ndarray:
    """Return `[DRAWS, DEEP]` flags for the rows reaching `layer` in each call of `run`."""
    seen = np.zeros((DRAWS, DEEP), bool)

    def hook(_, args):
        rows = args[0].reshape(-1, *args[0].shape[-2:])[:, 0, channel]
        raise _Read(rows.round().long().numpy())

    handle = layer.register_forward_pre_hook(hook)
    try:
        for i in range(DRAWS):
            try:
                run()
            except _Read as read:
                seen[i, read.args[0]] = True
    finally:
        handle.remove()
    return seen


def _deep_msa() -> dict:
    """Build a `[DEEP, 3]` MSA without gaps whose deletion values are the row indices."""
    x = make_msa(DEEP, 3, 20, seed=2)
    rows = np.broadcast_to(np.arange(DEEP, dtype=np.float32)[:, None], x.msa.shape)
    return {
        "msa": t(x.msa),
        "has_deletion": t(x.has_deletion),
        "deletion_value": t(rows.copy()),
        "msa_paired": t(x.is_paired),
        "msa_mask": torch.ones(DEEP, 3),
        "token_pad_mask": torch.ones(3),
    }


def _ours(module) -> np.ndarray:
    """Return `[DRAWS, DEEP]` flags for the rows `module` reads in each of `DRAWS` passes."""
    msa_mask = jnp.ones((DEEP, 3), bool)

    def read(key):
        draw = module.draw_rows(msa_mask, key)
        if draw is None:
            return jnp.ones(DEEP, bool)
        rows, drawn = draw
        return jnp.zeros(DEEP, bool).at[rows].set(drawn.any(-1))

    return np.asarray(jax.vmap(read)(jax.random.split(jax.random.key(0), DRAWS)))


def _boltz2_draws():
    from boltz.model.modules.trunkv2 import MSAModule

    from jfold.convert import boltz2

    m = MSAModule(
        msa_s=C_M, token_z=C_Z, token_s=C_S, msa_blocks=1, msa_dropout=0, z_dropout=0,
        subsample_msa=True, num_subsampled_msa=8,
    ).eval()  # fmt: skip
    feats = {k: v[None] for k, v in _deep_msa().items()}
    z, s = torch.zeros(1, 3, 3, C_Z), torch.zeros(1, 3, C_S)
    theirs = _rows_reaching(m.msa_proj, 34, lambda: m(z, s, feats))
    return theirs, boltz.msa_module(m, max_rows=boltz2.msa_max_rows(m))


def _promera_draws():
    from types import SimpleNamespace

    from promera.model.model import PromeraModel
    from promera.model.trunk import MSAModule

    feats = {k: v[None] for k, v in _deep_msa().items()}
    model = SimpleNamespace(cfg=SimpleNamespace(model=SimpleNamespace(msas_per_trunk_iter=8)))
    theirs = np.zeros((DRAWS, DEEP), bool)
    np.random.seed(0)
    for i in range(DRAWS):
        rows = PromeraModel.subsample_msa(model, feats)["deletion_value"][0, :, 0]
        theirs[i, rows.long().numpy()] = True
    m = MSAModule(msa_s=C_M, token_z=C_Z, s_input_dim=C_S, msa_blocks=1, msa_dropout=0, z_dropout=0)
    return theirs, boltz.msa_module(m, max_rows=8)


def _protenix_lineage_draws(module, configs, outer_product_first=True):
    m = module.MSAModule(n_blocks=1, c_m=C_M, c_z=C_Z, c_s_inputs=C_S, msa_configs=configs)
    feats = {k: _deep_msa()[k] for k in ("msa", "has_deletion", "deletion_value")}
    z, s = torch.zeros(3, 3, C_Z), torch.zeros(3, C_S)
    theirs = _rows_reaching(m.linear_no_bias_m, 33, lambda: m.eval()(feats, z, s, None))
    return theirs, protenix.msa_module(m, outer_product_first=outer_product_first)


def _protenix_release_draws():
    import protenix.model.modules.pairformer as module

    from jfold.pretrained.recipes.protenix import configs

    return _protenix_lineage_draws(module, configs("protenix_base_default_v1.0.0").data.msa)


def _protenix_capped_draws():
    import protenix.model.modules.pairformer as module

    configs = {
        "enable": True,
        "strategy": "random",
        "sample_cutoff": {"train": 512, "test": 8},
        "min_size": {"train": 1, "test": 4},
    }
    return _protenix_lineage_draws(module, configs)


def _mol1_draws():
    import mol.model.modules.pairformer as module
    from mol.config.data import data_configs

    return _protenix_lineage_draws(module, data_configs["msa"])


def _opendde_draws():
    import opendde.model.modules.pairformer as module

    return _protenix_lineage_draws(module, {"msa_depth": 8}, outer_product_first=False)


def _openfold3_draws():
    module, _, _, _, _ = _openfold3()
    embedder, stack = module
    embedder.subsample_all_msa = True
    embedder.min_subsampled_all_msa = embedder.max_subsampled_all_msa = 8
    feats = _deep_msa()
    batch = {
        "msa": torch.nn.functional.one_hot(feats["msa"], 32).float()[None],
        "has_deletion": feats["has_deletion"][None],
        "deletion_value": feats["deletion_value"][None],
        "msa_mask": feats["msa_mask"][None],
    }
    s = torch.zeros(1, 3, C_S)
    theirs = _rows_reaching(embedder.linear_m, 33, lambda: embedder(batch, s))
    return theirs, openfold3.msa_module(embedder, stack)


def _esmfold2_draws():
    from esm.models.esmfold2.layers import maybe_subsample_msa
    from esm.models.esmfold2.model import MSAEncoder

    from jfold.convert import esmfold2

    rows = torch.arange(DEEP)[None, :, None].expand(1, DEEP, 3)
    theirs = np.zeros((DRAWS, DEEP), bool)
    for i in range(DRAWS):
        drawn, *_ = maybe_subsample_msa(rows, None, None, None, max_depth=8, enabled=True)
        theirs[i, drawn[0, :, 0].numpy()] = True
    return theirs, esmfold2.msa_module(MSAEncoder(C_M, C_Z, C_S, 4, 3, 2, 4), max_rows=8)


DRAW_CASES = {
    "boltz2": _boltz2_draws,
    "promera": _promera_draws,
    "protenix": _protenix_release_draws,
    "protenix_capped": _protenix_capped_draws,
    "mol1": _mol1_draws,
    "opendde": _opendde_draws,
    "openfold3": _openfold3_draws,
    "esmfold2": _esmfold2_draws,
}


@pytest.mark.parametrize("case", DRAW_CASES)
def test_rows_drawn_as_upstream(case):
    """Check that over many passes the converted module draws rows as upstream's inference does."""
    torch.manual_seed(0)
    theirs, module = DRAW_CASES[case]()
    ours = _ours(module)
    a, b = ours.sum(1), theirs.sum(1)
    assert (a.min(), a.max()) == (b.min(), b.max())
    assert abs(a.mean() - b.mean()) <= 5 * np.sqrt((a.var() + b.var()) / DRAWS)
    assert np.abs(ours.mean(0) - theirs.mean(0)).max() < 0.07
