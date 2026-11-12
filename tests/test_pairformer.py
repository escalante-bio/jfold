"""Test the Pairformer against each upstream PyTorch implementation with random weights."""

import types

import equinox as eqx
import jax
import pytest
import torch
from reference import assert_finite, mask, pad, randomize, rel_err

from jfold.blocks.pairformer import TriangleAttention, with_triangle_chunks
from jfold.convert import boltz, openfold3, protenix

NUM_TOKENS = 7
C_S, C_Z = 16, 32  # ProMERA's cuequivariance triangle kernel needs C_Z % 32 == 0


def _boltz2(post_layer_norm):
    from boltz.model.layers.pairformer import PairformerModule

    m = PairformerModule(
        C_S,
        C_Z,
        num_blocks=2,
        num_heads=2,
        pairwise_head_width=4,
        pairwise_num_heads=2,
        post_layer_norm=post_layer_norm,
        v2=True,
    )
    return (
        m,
        lambda s, z, mask: m(s, z, mask, mask[:, :, None] * mask[:, None, :]),
        boltz.pairformer_stack,
    )


def _promera(no_update_s):
    from promera.model.trunk import PairformerModule

    cfg = types.SimpleNamespace(
        num_blocks=2,
        dims=types.SimpleNamespace(token_s=C_S, token_z=C_Z),
        num_heads=2,
        dropout=0.25,
        pairwise_head_width=4,
        pairwise_num_heads=2,
        no_update_s=no_update_s,
        activation_checkpointing=False,
        offload_to_cpu=False,
    )
    m = PairformerModule(cfg)

    def run(s, z, mask):
        # ProMERA's triangle updates exist only as cuequivariance CUDA kernels.
        m.cuda()
        s, z, mask = s.cuda(), z.cuda(), mask.cuda()
        s, z = m(s, z, mask, mask[:, :, None] * mask[:, None, :])
        m.cpu()
        return s.cpu(), z.cpu()

    return m, run, boltz.pairformer_stack


def _protenix():
    from protenix.model.modules.pairformer import PairformerStack

    m = PairformerStack(n_blocks=2, n_heads=2, c_z=C_Z, c_s=C_S)
    return (
        m,
        lambda s, z, mask: m(s, z, mask[:, :, None] * mask[:, None, :]),
        protenix.pairformer_stack,
    )


def _opendde():
    from opendde.model.modules.pairformer import PairformerStack

    m = PairformerStack(n_blocks=2, n_heads=2, c_z=C_Z, c_s=C_S)
    return (
        m,
        lambda s, z, mask: m(s, z, mask[:, :, None] * mask[:, None, :]),
        protenix.pairformer_stack,
    )


def _mol1():
    from mol.model.modules.pairformer import PairformerStack

    # One PairMixer block at the resident width, then two Pairformer blocks behind adapters.
    m = PairformerStack(
        n_blocks=3, n_heads=2, c_z=C_Z, c_s=C_S, n_pairmixer_blocks=1, c_z_pairformer=C_Z // 2
    )
    return (
        m,
        lambda s, z, mask: m(s, z, mask[:, :, None] * mask[:, None, :]),
        protenix.mol1_pairformer,
    )


def _openfold3(fused):
    from jopenfold3._vendor.openfold3.core.model.latent.pairformer import PairFormerStack

    m = PairFormerStack(
        c_s=C_S,
        c_z=C_Z,
        c_hidden_pair_bias=8,
        no_heads_pair_bias=2,
        c_hidden_mul=8,
        c_hidden_pair_att=4,
        no_heads_pair=2,
        no_blocks=2,
        transition_type="swiglu",
        transition_n=4,
        pair_dropout=0.25,
        fuse_projection_weights=fused,
        blocks_per_ckpt=None,
        inf=1e9,
    )
    return (
        m,
        lambda s, z, mask: m(s, z, mask, mask[:, :, None] * mask[:, None, :]),
        openfold3.pairformer_stack,
    )


CASES = {
    "boltz2": lambda: _boltz2(post_layer_norm=False),
    "boltz2-post-norm": lambda: _boltz2(post_layer_norm=True),
    "promera": lambda: _promera(no_update_s=False),
    "promera-pair-only": lambda: _promera(no_update_s=True),
    "protenix": _protenix,
    "opendde": _opendde,
    "mol1": _mol1,
    "openfold3": lambda: _openfold3(fused=False),
    "openfold3-fused": lambda: _openfold3(fused=True),
}


@eqx.filter_jit
def _forward(model, s, z, mask):
    return model(s, z, mask, key=jax.random.key(0))


@eqx.filter_jit
def _input_grads(model, s, z, mask, num_real):
    """Return the gradients of the real tokens' outputs with respect to every input."""

    def loss(s, z):
        s_out, z_out = model(s, z, mask, key=jax.random.key(0))
        return s_out[:num_real].sum() + z_out[:num_real, :num_real].sum()

    return jax.grad(loss, argnums=(0, 1))(s, z)


@pytest.mark.parametrize("padding", [0, 3], ids=["unpadded", "padded"])
@pytest.mark.parametrize("case", CASES)
def test_pairformer_matches_reference(case, padding):
    """Check that jfold matches the upstream on the real tokens with masked garbage appended."""
    module, run_reference, convert = CASES[case]()
    randomize(module, seed=0)
    gen = torch.Generator().manual_seed(1)
    s = torch.randn(1, NUM_TOKENS, C_S, generator=gen)
    z = torch.randn(1, NUM_TOKENS, NUM_TOKENS, C_Z, generator=gen)
    with torch.no_grad():
        s_ref, z_ref = run_reference(s, z, torch.ones(1, NUM_TOKENS))

    model = convert(module)
    s_in = pad(s[0].numpy(), padding, axes=(0,))
    z_in = pad(z[0].numpy(), padding, axes=(0, 1))
    token_mask = mask(NUM_TOKENS, padding)
    s_out, z_out = _forward(model, s_in, z_in, token_mask)

    assert rel_err(s_out[:NUM_TOKENS], s_ref[0]) < 1e-5
    assert rel_err(z_out[:NUM_TOKENS, :NUM_TOKENS], z_ref[0]) < 1e-5
    assert_finite((s_out, z_out, _input_grads(model, s_in, z_in, token_mask, NUM_TOKENS)))


def _chunk_inputs(padding=3):
    model = protenix.pairformer_stack(randomize(_protenix()[0], seed=0))
    num_tokens = NUM_TOKENS + padding
    s = jax.random.normal(jax.random.key(1), (num_tokens, C_S))
    z = jax.random.normal(jax.random.key(2), (num_tokens, num_tokens, C_Z))
    return model, s, z, mask(NUM_TOKENS, padding)


@pytest.mark.parametrize("chunk_size", [1, 4, 16])
def test_triangle_chunks_match_unchunked(chunk_size):
    """Check that triangle attention in row chunks matches all rows at once, with padding."""
    model, s, z, token_mask = _chunk_inputs()
    chunked = with_triangle_chunks(model, chunk_size)
    s_ref, z_ref = _forward(model, s, z, token_mask)
    s_out, z_out = _forward(chunked, s, z, token_mask)
    assert rel_err(s_out, s_ref) < 1e-5
    assert rel_err(z_out, z_ref) < 1e-5
    grad_ref = _input_grads(model, s, z, token_mask, NUM_TOKENS)[1]
    grad = _input_grads(chunked, s, z, token_mask, NUM_TOKENS)[1]
    assert rel_err(grad, grad_ref) < 1e-5


@eqx.filter_jit
def _forward_many(model, s, zs, mask):
    return jax.vmap(lambda z: model(s, z, mask)[1])(zs)


def test_triangle_chunks_under_vmap():
    """Check a chunked stack mapped over several pairs, as the template embedder maps its own."""
    model, s, z, token_mask = _chunk_inputs()
    zs = jax.numpy.stack([z, z.swapaxes(0, 1), 2 * z])
    out = _forward_many(with_triangle_chunks(model, 4), s, zs, token_mask)
    assert rel_err(out, _forward_many(model, s, zs, token_mask)) < 1e-5


def _chunk_sizes(tree):
    is_attention = lambda x: isinstance(x, TriangleAttention)
    return [x.chunk_size for x in jax.tree.leaves(tree, is_leaf=is_attention) if is_attention(x)]


def test_with_triangle_chunks_sets_every_triangle_attention():
    """Check that every nested `TriangleAttention` gets the chunk size and nothing else changes."""
    stack = protenix.pairformer_stack(randomize(_protenix()[0], seed=0))
    mol1 = protenix.mol1_pairformer(randomize(_mol1()[0], seed=0))
    model = {"trunk": (stack, mol1), "other": [stack.layers.tri_att_end, None, 3]}
    num = len(_chunk_sizes(model))
    assert num >= 5 and _chunk_sizes(model) == [None] * num
    chunked = with_triangle_chunks(model, 4)
    assert _chunk_sizes(chunked) == [4] * num
    assert eqx.tree_equal(jax.tree.leaves(chunked), jax.tree.leaves(model))
    assert not eqx.tree_equal(chunked, model)
    assert eqx.tree_equal(with_triangle_chunks(chunked, None), model)
