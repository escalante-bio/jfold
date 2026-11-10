"""Dropout: each converted model's dropout shares its mask along the same axes as the
upstream's own, at the upstream's rate; off by default, on with `eqx.nn.inference_mode`.
Masks are random, so only their structure is compared."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from reference import randomize

from jfold.convert import boltz, openfold3, protenix
from jfold.nn import Dropout

T, C = 6, 32
ROWS = 64  # enough rows that a mask never keeps (or drops) all of them by chance


def constant_axes(x) -> tuple[int, ...]:
    """The axes along which `x` does not vary."""
    x = np.asarray(x)
    return tuple(i for i in range(x.ndim) if np.all(x == np.take(x, [0], axis=i)))


def ours(dropout, shape, seed=0):
    return eqx.nn.inference_mode(dropout, False)(jnp.ones(shape), key=jax.random.key(seed))


def _openfold_row(rate):
    from protenix.openfold_local.model.dropout import DropoutRowwise

    m = DropoutRowwise(rate).train()
    return lambda shape: m(torch.ones(shape))


def _boltz_mask(rate, columnwise=False):
    from boltz.model.layers.dropout import get_dropout_mask

    return lambda shape: (
        get_dropout_mask(rate, torch.ones((1, *shape)), True, columnwise) * torch.ones((1, *shape))
    )[0]


def _protenix_block():
    from protenix.model.modules.pairformer import PairformerBlock

    return protenix.pairformer_layer(PairformerBlock(n_heads=2, c_z=C, c_s=16, dropout=0.25)), 0.25


def _boltz_layer():
    from boltz.model.layers.pairformer import PairformerLayer

    return boltz.pairformer_layer(PairformerLayer(16, C, num_heads=2, dropout=0.25, v2=True)), 0.25


def _openfold3_block():
    from jopenfold3._vendor.openfold3.core.model.latent.pairformer import PairFormerBlock

    block = PairFormerBlock(
        c_s=16,
        c_z=C,
        c_hidden_pair_bias=8,
        no_heads_pair_bias=2,
        c_hidden_mul=8,
        c_hidden_pair_att=4,
        no_heads_pair=2,
        transition_type="swiglu",
        transition_n=4,
        pair_dropout=0.25,
        fuse_projection_weights=False,
        inf=1e9,
    )
    return openfold3.pairformer_layer(block), 0.25


@pytest.mark.parametrize(
    "case, reference",
    [
        (
            _protenix_block,
            lambda r: (_openfold_row(r), lambda s: _openfold_row(r)(s).transpose(0, 1)),
        ),
        (
            _openfold3_block,
            lambda r: (_openfold_row(r), lambda s: _openfold_row(r)(s).transpose(0, 1)),
        ),
        (_boltz_layer, lambda r: (_boltz_mask(r), _boltz_mask(r, columnwise=True))),
    ],
    ids=["protenix", "openfold3", "boltz2"],
)
def test_pair_dropout_matches_upstream_structure(case, reference):
    """Row and column dropout on the pair share their masks along the upstream's axes, at
    its rate. (OpenFold-style blocks drop the ending-node update rowwise on the transposed
    pair, so its column mask is the transposed row mask.)"""
    layer, rate = case()
    row, column = reference(rate)
    shape = (ROWS, ROWS, C)
    assert layer.dropout_row.rate == rate and layer.dropout_column.rate == rate
    assert constant_axes(ours(layer.dropout_row, shape)) == constant_axes(row(shape))
    assert constant_axes(ours(layer.dropout_column, shape)) == constant_axes(column(shape))


def test_msa_dropout_matches_upstream_structure():
    """MSA dropout: OpenFold-style shares the mask across MSA rows; Boltz drops whole rows."""
    import protenix.model.modules.pairformer as protenix_pairformer
    from boltz.model.modules.trunkv2 import MSALayer

    configs = {
        "enable": True,
        "strategy": "random",
        "sample_cutoff": {"train": 8, "test": 8},
        "min_size": {"train": 8, "test": 8},
    }
    module = protenix_pairformer.MSAModule(
        n_blocks=2, c_m=8, c_z=C, c_s_inputs=12, msa_configs=configs, msa_dropout=0.15
    )
    protenix_msa = protenix.msa_module(module).layers.msa.dropout
    boltz_msa = boltz.msa_layer(
        MSALayer(msa_s=8, token_z=C, msa_dropout=0.15, z_dropout=0.25)
    ).msa.dropout
    shape = (ROWS, ROWS, 8)
    assert protenix_msa.rate == boltz_msa.rate == 0.15
    assert constant_axes(ours(protenix_msa, shape)) == constant_axes(_openfold_row(0.15)(shape))
    assert constant_axes(ours(boltz_msa, shape)) == constant_axes(_boltz_mask(0.15)(shape))


def test_dropout_is_off_by_default_and_unbiased():
    d = Dropout(rate=0.25, shared_axes=(0,))
    x = jnp.arange(T * T * C, dtype=jnp.float32).reshape(T, T, C)
    assert jnp.array_equal(d(x, key=jax.random.key(0)), x)
    samples = jax.vmap(lambda k: ours(d, (T, T, C), k))(jnp.arange(4000))
    np.testing.assert_allclose(np.unique(np.asarray(samples)), [0.0, 1 / 0.75], rtol=1e-6)
    np.testing.assert_allclose(samples.mean(), 1.0, atol=0.02)


def _boltz_stack():
    from boltz.model.layers.pairformer import PairformerModule

    torch_stack = PairformerModule(
        16,
        C,
        num_blocks=2,
        num_heads=2,
        dropout=0.25,
        pairwise_head_width=4,
        pairwise_num_heads=2,
        v2=True,
    )
    return boltz.pairformer_stack(randomize(torch_stack, seed=0))


def test_inference_mode_reaches_scanned_layers():
    """Turning dropout on reaches every layer of a scanned stack; outputs and gradients stay
    finite, and a different key gives a different output."""
    stack = _boltz_stack()
    s, z = jnp.ones((T, 16)), jax.random.normal(jax.random.key(0), (T, T, C))
    mask = jnp.ones(T, bool)
    run = eqx.filter_jit(lambda m, z, k: m(s, z, mask, key=k)[1])
    off = run(stack, z, jax.random.key(1))
    assert jnp.array_equal(off, run(stack, z, jax.random.key(2)))
    on = eqx.nn.inference_mode(stack, False)
    a, b = run(on, z, jax.random.key(1)), run(on, z, jax.random.key(2))
    assert not jnp.allclose(a, off) and not jnp.allclose(a, b)
    grads = jax.grad(lambda z: run(on, z, jax.random.key(1)).sum())(z)
    assert np.isfinite(np.asarray(grads)).all()


def test_keys_are_optional_while_dropout_is_off():
    """Without a key a stack runs as it does with any key while its dropout is off; once
    dropout is on it refuses to run without one, so no mask is reused."""
    stack = _boltz_stack()
    s, z = jnp.ones((T, 16)), jax.random.normal(jax.random.key(0), (T, T, C))
    mask = jnp.ones(T, bool)
    run = eqx.filter_jit(lambda m, z, k: m(s, z, mask, key=k)[1])
    assert jnp.array_equal(run(stack, z, None), run(stack, z, jax.random.key(1)))
    with pytest.raises(ValueError, match="needs a key"):
        run(eqx.nn.inference_mode(stack, False), z, None)


def test_esmfold2_lm_dropout_is_on_at_inference():
    """ESMFold2's dropout on the language model's pair is elementwise, as upstream's
    `F.dropout(..., training=True)`, and on unless switched off."""
    from jfold.convert.esmfold2 import lm_dropout

    d = lm_dropout(0.25)
    x = jnp.ones((ROWS, ROWS, C))
    y = d(x, key=jax.random.key(0))
    reference = torch.nn.functional.dropout(torch.ones(ROWS, ROWS, C), 0.25, training=True)
    assert d.rate == 0.25 and constant_axes(y) == constant_axes(reference) == ()
    assert jnp.array_equal(eqx.nn.inference_mode(d)(x, key=jax.random.key(0)), x)
    assert isinstance(lm_dropout(0.0), eqx.nn.Identity)
