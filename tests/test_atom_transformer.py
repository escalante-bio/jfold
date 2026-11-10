"""Atom transformer (windowed diffusion transformer) parity against each upstream."""

import functools

import equinox as eqx
import jax
import numpy as np
import pytest
import torch
from reference import assert_finite, mask, pad, randomize, rel_err

from jfold.blocks.windows import CenteredWindows, ShiftedWindows
from jfold.convert import boltz, openfold3, protenix

NUM_ATOMS = 70  # not a multiple of the query block: the last block is partial
C_A, C_P = 16, 8  # the single conditioning has the atom width, as in every upstream
HEADS, DEPTH = 2, 2
W, H = 32, 128


def windowed_pair(pair, windows, atom_mask):
    """A dense atom pair `[N, N, C]` in `windows`' layout `[NW, W, H, C]`."""
    n = pair.shape[0]
    index, _ = windows.key_index(atom_mask)
    rows = np.minimum(np.arange(windows.num_blocks(n) * W).reshape(-1, W), n - 1)
    return pair[rows[:, :, None], np.asarray(index)[:, None, :]]


def _boltz_lineage(m, keys_module, proj_in_layers):
    def run(q, c, p, atom_mask):
        # Boltz needs a whole number of query blocks: pad, and mask the padding.
        n = -(-q.shape[1] // W) * W
        q, c = (torch.nn.functional.pad(x, (0, 0, 0, n - x.shape[1])) for x in (q, c))
        atom_mask = torch.nn.functional.pad(atom_mask, (0, n - atom_mask.shape[1]))
        index = keys_module.get_indexing_matrix(n // W, W, H, q.device)
        to_keys = functools.partial(keys_module.single_to_keys, indexing_matrix=index, W=W, H=H)
        if proj_in_layers:
            return m(q, c, p, to_keys=to_keys, mask=atom_mask)
        bias = torch.cat([proj(p) for proj in m.proj_z], dim=-1)
        return m(q, c, bias, to_keys, atom_mask)

    return run


def _boltz2():
    from boltz.model.modules import encodersv2
    from boltz.model.modules.transformersv2 import AtomTransformer

    m = AtomTransformer(
        attn_window_queries=W,
        attn_window_keys=H,
        depth=DEPTH,
        heads=HEADS,
        dim=C_A,
        dim_single_cond=C_A,
    )
    # Boltz-2's diffusion conditioning holds each layer's pair-bias projection.
    m.proj_z = torch.nn.ModuleList(
        torch.nn.Sequential(torch.nn.LayerNorm(C_P), torch.nn.Linear(C_P, HEADS, bias=False))
        for _ in range(DEPTH)
    )
    run = _boltz_lineage(m, encodersv2, proj_in_layers=False)
    return m, run, lambda m: boltz.atom_transformer(m, proj_z=m.proj_z), "padded-to-block"


def _promera():
    from promera.model import encoders
    from promera.model.transformers import AtomTransformer

    m = AtomTransformer(
        attn_window_queries=W,
        attn_window_keys=H,
        depth=DEPTH,
        heads=HEADS,
        dim=C_A,
        dim_single_cond=C_A,
        dim_pairwise=C_P,
    )
    return (
        m,
        _boltz_lineage(m, encoders, proj_in_layers=True),
        boltz.atom_transformer,
        "padded-to-block",
    )


def _protenix_lineage(module):
    m = module.AtomTransformer(
        c_atom=C_A, c_atompair=C_P, n_blocks=DEPTH, n_heads=HEADS, n_queries=W, n_keys=H
    )
    return m, lambda q, c, p, atom_mask: m(q, c, p), protenix.atom_transformer, "unpadded"


def _protenix():
    import protenix.model.modules.transformer as module

    return _protenix_lineage(module)


def _opendde():
    import opendde.model.modules.transformer as module

    return _protenix_lineage(module)


def _mol1():
    import mol.model.modules.transformer as module

    return _protenix_lineage(module)


def _openfold3():
    from jopenfold3._vendor.openfold3.core.model.layers.diffusion_transformer import (
        DiffusionTransformer,
    )

    m = DiffusionTransformer(
        c_a=C_A,
        c_s=C_A,
        c_z=C_P,
        c_hidden=C_A // HEADS,
        no_heads=HEADS,
        no_blocks=DEPTH,
        n_transition=2,
        use_ada_layer_norm=True,
        n_query=W,
        n_key=H,
        inf=1e9,
    )
    return (
        m,
        lambda q, c, p, atom_mask: m(q, c, p, mask=atom_mask),
        openfold3.atom_transformer,
        "unpadded",
    )


CASES = {
    "boltz2": _boltz2,
    "promera": _promera,
    "protenix": _protenix,
    "opendde": _opendde,
    "mol1": _mol1,
    "openfold3": _openfold3,
}


@eqx.filter_jit
def _forward(model, q, c, p, atom_mask):
    return model(q, c, model.pair_biases(p), atom_mask)


@eqx.filter_jit
def _input_grads(model, q, c, p, atom_mask, num_real):
    """Gradients of the real atoms' outputs with respect to every input."""

    def loss(q, c, p):
        return _forward(model, q, c, p, atom_mask)[:num_real].sum()

    return jax.grad(loss, argnums=(0, 1, 2))(q, c, p)


@pytest.mark.parametrize("padding", [0, 40], ids=["unpadded", "padded"])
@pytest.mark.parametrize("case", CASES)
def test_atom_transformer_matches_reference(case, padding):
    """jfold on atoms with `padding` masked garbage atoms appended (here, a whole extra
    query block) matches the upstream on the real atoms; outputs and gradients stay finite."""
    module, run_reference, convert, reference_layout = CASES[case]()
    randomize(module, seed=0)
    model = convert(module)
    windows = model.layers.attention.attention.windows
    assert isinstance(windows, ShiftedWindows if case == "openfold3" else CenteredWindows)

    rng = np.random.default_rng(1)
    q = rng.standard_normal((NUM_ATOMS, C_A), dtype=np.float32)
    c = rng.standard_normal((NUM_ATOMS, C_A), dtype=np.float32)
    pair = rng.standard_normal((NUM_ATOMS, NUM_ATOMS, C_P), dtype=np.float32)

    # The reference, on the real atoms (Boltz pads to whole query blocks itself).
    real = np.ones(NUM_ATOMS, bool)
    if reference_layout == "padded-to-block":
        n = -(-NUM_ATOMS // W) * W
        ref_pair = windowed_pair(
            pad(pair, n - NUM_ATOMS, axes=(0, 1)), windows, mask(NUM_ATOMS, n - NUM_ATOMS)
        )
    else:
        ref_pair = windowed_pair(pair, windows, real)
    with torch.no_grad():
        ref = run_reference(
            torch.from_numpy(q)[None],
            torch.from_numpy(c)[None],
            torch.from_numpy(np.ascontiguousarray(ref_pair))[None],
            torch.ones(1, NUM_ATOMS),
        )

    atom_mask = mask(NUM_ATOMS, padding)
    q_in, c_in = pad(q, padding, axes=(0,)), pad(c, padding, axes=(0,))
    p_in = windowed_pair(pad(pair, padding, axes=(0, 1)), windows, atom_mask)
    out = _forward(model, q_in, c_in, p_in, atom_mask)

    assert rel_err(out[:NUM_ATOMS], ref[0, :NUM_ATOMS]) < 1e-5
    assert_finite((out, _input_grads(model, q_in, c_in, p_in, atom_mask, NUM_ATOMS)))
