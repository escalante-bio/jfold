"""Diffusion transformer parity against each upstream PyTorch implementation."""

import equinox as eqx
import jax
import pytest
import torch
from reference import assert_finite, mask, pad, randomize, rel_err

from jfold.convert import boltz, openfold3, protenix

NUM_TOKENS = 10
C_A, C_S, C_Z = 16, 12, 8
HEADS, DEPTH = 2, 2


def _boltz2(post_layer_norm):
    from boltz.model.modules.transformersv2 import DiffusionTransformer

    m = DiffusionTransformer(
        depth=DEPTH, heads=HEADS, dim=C_A, dim_single_cond=C_S, post_layer_norm=post_layer_norm
    )
    # Boltz-2's diffusion conditioning holds each layer's pair-bias projection.
    m.proj_z = torch.nn.ModuleList(
        torch.nn.Sequential(torch.nn.LayerNorm(C_Z), torch.nn.Linear(C_Z, HEADS, bias=False))
        for _ in range(DEPTH)
    )

    def run(a, s, z, mask):
        bias = torch.cat([p(z) for p in m.proj_z], dim=-1)
        return m(a, s, bias=bias, mask=mask)

    return m, run, lambda m: boltz.diffusion_transformer(m, proj_z=m.proj_z)


def _promera():
    from promera.model.transformers import DiffusionTransformer

    m = DiffusionTransformer(
        depth=DEPTH, heads=HEADS, dim=C_A, dim_single_cond=C_S, dim_pairwise=C_Z
    )
    return m, lambda a, s, z, mask: m(a, s, z, mask=mask), boltz.diffusion_transformer


def _protenix_lineage(module):
    m = module.DiffusionTransformer(c_a=C_A, c_s=C_S, c_z=C_Z, n_blocks=DEPTH, n_heads=HEADS)
    return m, lambda a, s, z, mask: m(a, s, z), protenix.diffusion_transformer


def _openfold3():
    from jopenfold3._vendor.openfold3.core.model.layers.diffusion_transformer import (
        DiffusionTransformer,
    )

    m = DiffusionTransformer(
        c_a=C_A,
        c_s=C_S,
        c_z=C_Z,
        c_hidden=C_A // HEADS,
        no_heads=HEADS,
        no_blocks=DEPTH,
        n_transition=2,
        use_ada_layer_norm=True,
        n_query=None,
        n_key=None,
        inf=1e9,
    )
    return m, lambda a, s, z, mask: m(a, s, z, mask=mask), openfold3.diffusion_transformer


def _protenix():
    import protenix.model.modules.transformer as module

    return _protenix_lineage(module)


def _opendde():
    import opendde.model.modules.transformer as module

    return _protenix_lineage(module)


def _mol1():
    import mol.model.modules.transformer as module

    return _protenix_lineage(module)


CASES = {
    "boltz2": lambda: _boltz2(post_layer_norm=False),
    "boltz2-post-norm": lambda: _boltz2(post_layer_norm=True),
    "promera": _promera,
    "protenix": _protenix,
    "opendde": _opendde,
    "mol1": _mol1,
    "openfold3": _openfold3,
}


@eqx.filter_jit
def _forward(model, a, s, z, mask):
    return model(a, s, model.pair_biases(z), mask)


@eqx.filter_jit
def _input_grads(model, a, s, z, mask, num_real):
    """Gradients of the real tokens' outputs with respect to every input."""

    def loss(a, s, z):
        return _forward(model, a, s, z, mask)[:num_real].sum()

    return jax.grad(loss, argnums=(0, 1, 2))(a, s, z)


@pytest.mark.parametrize("padding", [0, 3], ids=["unpadded", "padded"])
@pytest.mark.parametrize("case", CASES)
def test_diffusion_transformer_matches_reference(case, padding):
    """jfold on inputs with `padding` masked garbage tokens appended matches the upstream
    on the real tokens alone; outputs and gradients stay finite everywhere."""
    module, run_reference, convert = CASES[case]()
    randomize(module, seed=0)
    gen = torch.Generator().manual_seed(1)
    a = torch.randn(1, NUM_TOKENS, C_A, generator=gen)
    s = torch.randn(1, NUM_TOKENS, C_S, generator=gen)
    z = torch.randn(1, NUM_TOKENS, NUM_TOKENS, C_Z, generator=gen)
    with torch.no_grad():
        ref = run_reference(a, s, z, torch.ones(1, NUM_TOKENS))

    model = convert(module)
    a_in = pad(a[0].numpy(), padding, axes=(0,))
    s_in = pad(s[0].numpy(), padding, axes=(0,))
    z_in = pad(z[0].numpy(), padding, axes=(0, 1))
    token_mask = mask(NUM_TOKENS, padding)
    out = _forward(model, a_in, s_in, z_in, token_mask)

    assert rel_err(out[:NUM_TOKENS], ref[0]) < 1e-5
    assert_finite((out, _input_grads(model, a_in, s_in, z_in, token_mask, NUM_TOKENS)))
