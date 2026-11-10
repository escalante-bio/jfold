"""ESMFold2 parity against `esm.models.esmfold2` (and ESMC, `esm.models.esmc`), block by
block, with random weights: each jfold block, on inputs with masked garbage appended,
matches the upstream on the real inputs alone, with finite outputs and gradients."""

import contextlib
import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from reference import (
    STEPS,
    assert_finite,
    assert_round_trip,
    assert_schedule,
    block_features,
    mask,
    pad,
    randomize,
    rel_err,
)

from jfold.blocks.atoms import Atoms
from jfold.blocks.inputs import relative_position_features
from jfold.convert import esmfold2 as convert
from jfold.models import esmfold2
from jfold.slots import TrunkInputs, TrunkState

NUM_TOKENS = 7
NUM_ATOMS = 23
PADDING = 3
C_Z, C_M, C_S_INPUTS, C_TOKEN, C_ATOM = 16, 8, 24, 32, 32


@pytest.fixture(autouse=True)
def _jax_on_cpu():
    """ESMFold2 rounds its RoPE tables to bfloat16, as upstream does. XLA's GPU cos/sin
    differ from torch's CPU ones in the last bit often enough to flip some of those
    roundings (about 1e-3 downstream), so compare on JAX's CPU device."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


KEY = jax.random.key(0)


def _layers():
    from esm.models.esmfold2 import layers

    return layers


def _model():
    from esm.models.esmfold2 import model

    return model


@contextlib.contextmanager
def _float32_attention():
    """The upstream's atom attention casts to bfloat16 (its flash kernels' dtype); keep
    float32 so tolerances measure implementation differences."""
    original = torch.Tensor.bfloat16
    torch.Tensor.bfloat16 = lambda self, *args, **kwargs: self
    try:
        yield
    finally:
        torch.Tensor.bfloat16 = original


def _randn(gen, *shape):
    return torch.randn(*shape, generator=gen)


def _unchunk(module):
    for m in module.modules():
        if hasattr(m, "_chunk_size"):
            m._chunk_size = None
    return module


@eqx.filter_jit
def _call(f, *args):
    return f(*args)


@eqx.filter_jit
def _grads(f, args, num_real):
    """Gradients of the real part of `f(*args)` (its first output if a tuple) with respect
    to every floating-point argument."""

    def loss(float_args):
        out = f(*eqx.combine(float_args, rest))
        out = out[0] if isinstance(out, tuple) else out
        return out[(slice(0, num_real),) * min(out.ndim, 2)].sum()

    float_args, rest = eqx.partition(args, eqx.is_inexact_array)
    return jax.grad(loss)(float_args)


def _check(ours, ref, tol=1e-5):
    assert rel_err(ours, ref) < tol, rel_err(ours, ref)


# --- Pair trunk ---------------------------------------------------------------------------


def test_pair_stack():
    """`FoldingTrunk` on `PairformerStack` of `PairMixerLayer`s."""
    m = _unchunk(randomize(_layers().FoldingTrunk(n_layers=2, d_pair=C_Z, expansion_ratio=2), 0))
    gen = torch.Generator().manual_seed(1)
    z = _randn(gen, 1, NUM_TOKENS, NUM_TOKENS, C_Z)
    with torch.no_grad():
        ref = m(z, torch.ones(1, NUM_TOKENS, NUM_TOKENS))[0]

    stack = convert.pair_stack(m)
    z_in = pad(z[0].numpy(), PADDING, axes=(0, 1))
    token_mask = mask(NUM_TOKENS, PADDING)
    f = lambda stack, z, mask: stack(None, z, mask, key=KEY)[1]
    out = _call(f, stack, z_in, token_mask)
    _check(out[:NUM_TOKENS, :NUM_TOKENS], ref)
    assert_finite((out, _grads(f, (stack, z_in, token_mask), NUM_TOKENS)))


# --- MSA ----------------------------------------------------------------------------------


NUM_ROWS = 5


@pytest.mark.parametrize("divide_first", [False, True], ids=["project-first", "divide-first"])
def test_outer_product_mean(divide_first):
    m = randomize(_layers().OuterProductMean(C_M, 4, C_Z, divide_first), 0)
    gen = torch.Generator().manual_seed(1)
    msa = _randn(gen, 1, NUM_TOKENS, NUM_ROWS, C_M)
    msa_mask = torch.rand(1, NUM_TOKENS, NUM_ROWS, generator=gen) > 0.3
    msa_mask[..., 0] = True
    with torch.no_grad():
        ref = m(msa, msa_mask)[0]

    opm = convert.outer_product_mean(m)
    m_in = pad(pad(msa[0].numpy().swapaxes(0, 1), PADDING, axes=(0,)), PADDING, axes=(1,), seed=1)
    mask_in = np.pad(msa_mask[0].numpy().T, [(0, PADDING), (0, PADDING)])
    out = _call(lambda opm, m, msa_mask: opm(m, msa_mask), opm, m_in, mask_in)
    _check(out[:NUM_TOKENS, :NUM_TOKENS], ref)
    assert_finite(out)


def test_msa_module():
    """`MSAEncoder` on `MSAModule`: its blocks are `MSALayer`s around `PairMixerLayer`s."""
    model = _model()
    m = _unchunk(randomize(model.MSAEncoder(C_M, C_Z, C_S_INPUTS, 4, 3, 2, 4), 0))
    gen = torch.Generator().manual_seed(1)
    z = _randn(gen, 1, NUM_TOKENS, NUM_TOKENS, C_Z)
    s_inputs = _randn(gen, 1, NUM_TOKENS, C_S_INPUTS)
    msa = torch.randint(0, 33, (NUM_ROWS, NUM_TOKENS), generator=gen)
    has_deletion = (torch.rand(NUM_ROWS, NUM_TOKENS, generator=gen) > 0.7).float()
    deletion_value = torch.rand(NUM_ROWS, NUM_TOKENS, generator=gen)
    msa_mask = torch.ones(NUM_ROWS, NUM_TOKENS, dtype=torch.bool)
    msa_mask[2:, 4:] = False
    features = esmfold2.msa_features(
        jax.nn.one_hot(msa.numpy(), 33),
        jnp.array(has_deletion.numpy()),
        jnp.array(deletion_value.numpy()),
        jnp.array(msa_mask.numpy()),
    )
    with torch.no_grad():
        ref = m(
            z,
            s_inputs,
            torch.from_numpy(np.asarray(features[..., :33]).swapaxes(0, 1))[None],
            torch.from_numpy(np.asarray(features[..., 33]).T)[None],
            torch.from_numpy(np.asarray(features[..., 34]).T)[None],
            msa_mask.T[None].float(),
        )[0]

    module = convert.msa_module(m)
    z_in = pad(z[0].numpy(), PADDING, axes=(0, 1))
    s_in = pad(s_inputs[0].numpy(), PADDING, axes=(0,))
    f_in = pad(pad(np.asarray(features), 2, axes=(0,)), PADDING, axes=(1,), seed=1)
    mask_in = np.pad(msa_mask.numpy(), [(0, 2), (0, PADDING)])
    token_mask = mask(NUM_TOKENS, PADDING)
    f = lambda module, z, feats, s, msa_mask, mask: module(z, feats, s, msa_mask, mask, key=KEY)[1]
    args = (module, z_in, f_in, s_in, mask_in, token_mask)
    out = _call(f, *args)
    _check(out[:NUM_TOKENS, :NUM_TOKENS], ref)
    assert_finite((out, _grads(f, args, NUM_TOKENS)))


# --- Diffusion transformer and conditioning ----------------------------------------------


def test_diffusion_transformer():
    """The token `DiffusionTransformer` on jfold's."""
    m = randomize(
        _layers().DiffusionTransformer(
            d_model=C_TOKEN, d_pair=C_Z, num_heads=4, num_blocks=2, d_cond=C_TOKEN
        ),
        0,
    )
    gen = torch.Generator().manual_seed(1)
    a = _randn(gen, 1, NUM_TOKENS, C_TOKEN)
    s = _randn(gen, 1, NUM_TOKENS, C_TOKEN)
    z = _randn(gen, 1, NUM_TOKENS, NUM_TOKENS, C_Z)
    with torch.no_grad():
        ref = m(a, s, z, attention_mask=torch.ones(1, NUM_TOKENS, dtype=torch.bool))[0][0]

    transformer = convert.diffusion_transformer(m)
    a_in = pad(a[0].numpy(), PADDING, axes=(0,))
    s_in = pad(s[0].numpy(), PADDING, axes=(0,), seed=1)
    z_in = pad(z[0].numpy(), PADDING, axes=(0, 1), seed=2)
    token_mask = mask(NUM_TOKENS, PADDING)

    def f(t, a, s, z, mask):
        return t(a, s, t.pair_biases(z), mask)

    args = (transformer, a_in, s_in, z_in, token_mask)
    out = _call(f, *args)
    _check(out[:NUM_TOKENS], ref)
    assert_finite((out, _grads(f, args, NUM_TOKENS)))


def test_diffusion_conditioning():
    """`DiffusionConditioning` on jfold's, with a zero-width trunk single."""
    m = randomize(
        _layers().DiffusionConditioning(
            c_z=C_Z, c_s=C_TOKEN, c_s_inputs=C_S_INPUTS, sigma_data=16.0, fourier_dim=8
        ),
        0,
    )
    gen = torch.Generator().manual_seed(1)
    s_inputs = _randn(gen, 1, NUM_TOKENS, C_S_INPUTS)
    z = _randn(gen, 1, NUM_TOKENS, NUM_TOKENS, C_Z)
    rel = _randn(gen, 1, NUM_TOKENS, NUM_TOKENS, C_Z)
    t = 3.7
    with torch.no_grad():
        s_ref, z_ref = m(torch.tensor([t]), s_inputs, None, z, rel)

    conditioning = convert.diffusion_conditioning(m)
    s_out = conditioning.single(jnp.zeros((NUM_TOKENS, 0)), s_inputs[0].numpy(), jnp.array(t))
    z_out = conditioning.pair(z[0].numpy(), rel[0].numpy())
    _check(s_out, s_ref[0])
    _check(z_out, z_ref[0])


def test_relative_positions():
    """`ResIdxAsymIdSymIdEntityIdEncoding` on `relative_position_features` with chain
    offsets between chains."""
    m = randomize(_layers().ResIdxAsymIdSymIdEntityIdEncoding(4, 2, C_Z), 0)
    asym = torch.tensor([[0, 0, 0, 1, 1, 2, 2]])
    residue = torch.tensor([[0, 1, 1, 0, 9, 3, 4]])
    entity = torch.tensor([[0, 0, 0, 1, 1, 0, 0]])
    token = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
    sym = torch.tensor([[0, 0, 0, 0, 0, 1, 1]])
    with torch.no_grad():
        ref = m(residue, asym, sym, entity, token)[0]
    asym_j = jnp.array(asym[0].numpy())
    features = relative_position_features(
        asym_j,
        jnp.array(residue[0].numpy()),
        jnp.array(entity[0].numpy()),
        jnp.array(token[0].numpy()),
        jnp.array(sym[0].numpy()),
        r_max=4,
        s_max=2,
        chain_offset=asym_j[:, None] != asym_j[None, :],
    )
    _check(convert.relative_positions(m)(features), ref)


# --- Atoms ---------------------------------------------------------------------------------


def _atom_inputs(gen, num_atoms=NUM_ATOMS):
    ref_pos = 3 * _randn(gen, num_atoms, 3)
    uid = torch.sort(torch.randint(0, 4, (num_atoms,), generator=gen)).values
    charge = torch.randint(-1, 2, (num_atoms,), generator=gen).float()
    element = torch.randint(0, 128, (num_atoms,), generator=gen)
    chars = torch.randint(0, 64, (num_atoms, 4), generator=gen)
    atom_to_token = torch.sort(torch.randint(0, NUM_TOKENS, (num_atoms,), generator=gen)).values
    return ref_pos, uid, charge, element, chars, atom_to_token


def _pad_atoms(ref_pos, uid, charge, element, chars, atom_to_token):
    """jfold `Atoms` with `PADDING` garbage atoms appended."""
    n = ref_pos.shape[0] + PADDING
    rng = np.random.default_rng(3)
    atom_mask = mask(ref_pos.shape[0], PADDING)
    ref_pos = pad(ref_pos.numpy(), PADDING, axes=(0,))
    uid = np.concatenate([uid.numpy(), rng.integers(0, 4, PADDING)])
    features = esmfold2.atom_features(
        jnp.array(ref_pos),
        jnp.array(pad(charge.numpy(), PADDING, axes=(0,))),
        jnp.array(atom_mask),
        jnp.array(np.concatenate([element.numpy(), rng.integers(0, 128, PADDING)])),
        jnp.array(np.concatenate([chars.numpy(), rng.integers(0, 64, (PADDING, 4))])),
    )
    return Atoms(
        features=features,
        ref_pos=jnp.array(ref_pos),
        ref_space_uid=jnp.array(uid),
        atom_to_token=jnp.array(
            np.concatenate([atom_to_token.numpy(), rng.integers(0, NUM_TOKENS, PADDING)])
        ),
        mask=jnp.array(atom_mask),
    ), n


def _torch_atoms(ref_pos, uid, charge, element, chars, atom_to_token):
    """The upstream encoder's keyword inputs (batched, one-hot)."""
    n = ref_pos.shape[0]
    return {
        "ref_pos": ref_pos[None],
        "atom_attention_mask": torch.ones(1, n, dtype=torch.bool),
        "ref_space_uid": uid[None],
        "ref_charge": charge[None],
        "ref_element": torch.nn.functional.one_hot(element, 128).float()[None],
        "ref_atom_name_chars": torch.nn.functional.one_hot(chars, 64).float()[None],
        "atom_to_token": atom_to_token[None],
    }


SWA = {
    "n_blocks": 2,
    "n_heads": 2,
    "swa_window_size": 8,
    "expansion_ratio": 2,
    "n_spatial_rope_pairs_per_axis": 2,
    "n_uid_rope_pairs": 1,
}


@pytest.mark.parametrize("num_atoms", [NUM_ATOMS, 70], ids=["short", "long"])
def test_sliding_window_transformer(num_atoms):
    """`SWAAtomTransformer`: windows over the real atoms, 3D RoPE, adaLN-Zero."""
    m = randomize(_layers().SWAAtomTransformer(d_atom=C_ATOM, **SWA), 0)
    gen = torch.Generator().manual_seed(1)
    ref_pos, uid, *_ = _atom_inputs(gen, num_atoms)
    q = _randn(gen, 1, num_atoms, C_ATOM)
    c = _randn(gen, 1, num_atoms, C_ATOM)
    with torch.no_grad(), _float32_attention():
        rope = m._build_3d_rope(ref_pos[None], uid[None])
        ref = m(q, c, rope)[0]

    transformer = convert.sliding_window_transformer(m)
    atom_mask = mask(num_atoms, PADDING)
    # Garbage atoms interleaved with the real ones: windows count real atoms only.
    order = np.argsort(np.concatenate([np.arange(num_atoms), [2.5, 9.5, 11.5]]), kind="stable")
    pos_in = pad(ref_pos.numpy(), PADDING, axes=(0,))[order]
    uid_in = np.concatenate([uid.numpy(), [7, 8, 9]])[order]
    q_in = pad(q[0].numpy(), PADDING, axes=(0,))[order]
    c_in = pad(c[0].numpy(), PADDING, axes=(0,), seed=1)[order]
    mask_in = atom_mask[order]

    def f(t, q, c, pos, uid, mask):
        return t(q, t.modulations(c), t.rope(pos, uid), mask)

    args = (transformer, q_in, c_in, pos_in, uid_in, mask_in)
    out = _call(f, *args)
    _check(np.asarray(out)[mask_in], ref)
    assert_finite((out, _grads(f, args, 2)))


def test_input_embedder():
    """`InputsEmbedder`: the sliding-window atom encoder's aggregates and token
    features."""
    from esm.models.esmfold2.config import EsmFold2Config

    config = EsmFold2Config(
        sliding_window=8,
        atom_encoder={
            "hidden_size": C_ATOM,
            "output_dim": 2 * C_ATOM,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "n_spatial_rope_pairs_per_axis": 2,
            "n_uid_rope_pairs": 2,
        },
    )
    m = randomize(_layers().InputsEmbedder(config), 0)
    gen = torch.Generator().manual_seed(1)
    atoms = _atom_inputs(gen)
    res_type = torch.softmax(3 * _randn(gen, NUM_TOKENS, 33), -1)  # a soft sequence
    profile = torch.softmax(_randn(gen, NUM_TOKENS, 33), -1)
    deletion_mean = torch.rand(NUM_TOKENS, generator=gen)
    with torch.no_grad(), _float32_attention():
        ref = m(
            aatype=res_type[None],
            profile=profile[None],
            deletion_mean=deletion_mean[None],
            **_torch_atoms(*atoms),
        )[0]

    embedder = esmfold2.InputEmbedder(encoder=convert.atom_encoder(m.atom_attention_encoder))
    atoms_in, _ = _pad_atoms(*atoms)
    token_mask = mask(NUM_TOKENS, PADDING)
    token_features = esmfold2.token_features(
        jnp.array(pad(res_type.numpy(), PADDING, axes=(0,))),
        jnp.array(pad(profile.numpy(), PADDING, axes=(0,))),
        jnp.array(pad(deletion_mean.numpy(), PADDING, axes=(0,))),
        jnp.array(token_mask),
    )
    f = lambda e, atoms, tf: e(
        block_features(tf.shape[0], atoms.mask.shape[0], atoms=atoms, token_features=tf)
    )
    args = (embedder, atoms_in, token_features)
    out = _call(f, *args)
    _check(out[:NUM_TOKENS], ref)
    assert_finite((out, _grads(f, args, NUM_TOKENS)))


def test_diffusion_module():
    """One denoising step of `DiffusionModule`."""
    m = randomize(
        _layers().DiffusionModule(
            c_atom=C_ATOM,
            c_token=C_TOKEN,
            c_z=C_Z,
            c_s_inputs=C_S_INPUTS,
            fourier_dim=8,
            atom_num_blocks=2,
            atom_num_heads=2,
            token_num_blocks=2,
            token_num_heads=4,
            swa_window_size=8,
            n_spatial_rope_pairs_per_axis=2,
            n_uid_rope_pairs=2,
        ),
        0,
    )
    gen = torch.Generator().manual_seed(1)
    atoms = _atom_inputs(gen)
    s_inputs = _randn(gen, 1, NUM_TOKENS, C_S_INPUTS)
    z = _randn(gen, 1, NUM_TOKENS, NUM_TOKENS, C_Z)
    rel = _randn(gen, 1, NUM_TOKENS, NUM_TOKENS, C_Z)
    x_noisy = 10 * _randn(gen, 1, NUM_ATOMS, 3)
    t = 7.5
    kw = _torch_atoms(*atoms)
    with torch.no_grad(), _float32_attention():
        ref = m(
            x_noisy=x_noisy,
            t_hat=torch.tensor([t]),
            ref_pos=kw["ref_pos"],
            ref_charge=kw["ref_charge"],
            ref_mask=kw["atom_attention_mask"],
            ref_element=kw["ref_element"],
            ref_atom_name_chars=kw["ref_atom_name_chars"],
            ref_space_uid=kw["ref_space_uid"],
            tok_idx=kw["atom_to_token"],
            s_inputs=s_inputs,
            s_trunk=None,
            z_trunk=z,
            relative_position_encoding=rel,
            asym_id=None,
            residue_index=None,
            entity_id=None,
            token_index=None,
            sym_id=None,
            token_attention_mask=torch.ones(1, NUM_TOKENS, dtype=torch.bool),
        )["x_denoised"][0]

    module = convert.diffusion_module(m)
    atoms_in, _ = _pad_atoms(*atoms)
    s_in = pad(s_inputs[0].numpy(), PADDING, axes=(0,))
    z_in = pad(z[0].numpy(), PADDING, axes=(0, 1), seed=1)
    rel_in = pad(rel[0].numpy(), PADDING, axes=(0, 1), seed=2)
    x_in = pad(x_noisy[0].numpy(), PADDING, axes=(0,), seed=4)
    token_mask = mask(NUM_TOKENS, PADDING)

    def f(module, x, atoms, s, z, rel, mask):
        features = block_features(mask.shape[0], atoms.mask.shape[0], atoms=atoms, token_mask=mask)
        context = module.prepare(features, _trunk_inputs(s, z, rel), TrunkState(s=s, z=z))
        return module(context, x, jnp.array(t))

    args = (module, x_in, atoms_in, s_in, z_in, rel_in, token_mask)
    out = _call(f, *args)
    _check(out[:NUM_ATOMS], ref)
    assert_finite((out, _grads(f, args, NUM_ATOMS)))


# --- Confidence ----------------------------------------------------------------------------


def _trunk_inputs(s_inputs, z, relative_positions) -> TrunkInputs:
    """Trunk inputs carrying only what the diffusion and confidence modules read."""
    extras = esmfold2.TrunkExtras(
        relative_positions=relative_positions,
        bonds=jnp.zeros_like(relative_positions),
        lm_pair=None,
        msa_features=None,
        msa_mask=None,
    )
    return TrunkInputs(s_inputs=s_inputs, s_init=s_inputs, z_init=z, extras=extras)


def _confidence(module, s_inputs, z, rep_coords, atom_to_token, atom_position, mask, pair_features):
    """The confidence logits, the structure given as each token's representative atom."""
    num_tokens = mask.shape[0]
    features = block_features(
        num_tokens,
        atom_to_token.shape[0],
        token_mask=mask,
        rep_atom=jnp.arange(num_tokens),
        atom_position=atom_position,
    )
    features = eqx.tree_at(lambda f: f.atoms.atom_to_token, features, atom_to_token)
    inputs = _trunk_inputs(s_inputs, z, pair_features)
    return module(features, inputs, TrunkState(s=s_inputs, z=z), rep_coords, key=KEY)


def test_confidence():
    """`ConfidenceHead`'s logits."""
    from esm.models.esmfold2.config import EsmFold2Config

    config = EsmFold2Config(
        hidden_size=C_TOKEN,
        pairwise_hidden_size=C_Z,
        single_inputs_size=C_S_INPUTS,
        confidence_head={
            "num_hidden_layers": 2,
            "num_plddt_bins": 6,
            "num_pde_bins": 5,
            "num_pae_bins": 4,
            "distogram_bins": 9,
        },
    )
    m = _unchunk(randomize(_model().ConfidenceHead(config), 0))
    gen = torch.Generator().manual_seed(1)
    atoms = _atom_inputs(gen)
    atom_to_token = atoms[-1]
    s_inputs = _randn(gen, 1, NUM_TOKENS, C_S_INPUTS)
    z = _randn(gen, 1, NUM_TOKENS, NUM_TOKENS, C_Z)
    pair_features = _randn(gen, 1, NUM_TOKENS, NUM_TOKENS, C_Z)
    coords = 8 * _randn(gen, 1, NUM_ATOMS, 3)
    rep_atom = torch.tensor(
        [
            int(torch.nonzero(atom_to_token == i)[0]) if (atom_to_token == i).any() else 0
            for i in range(NUM_TOKENS)
        ]
    )
    asym = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    with torch.no_grad():
        ref = m(
            s_inputs=s_inputs,
            z=z,
            x_pred=coords,
            distogram_atom_idx=rep_atom[None],
            token_attention_mask=torch.ones(1, NUM_TOKENS, dtype=torch.bool),
            atom_to_token=atom_to_token[None],
            atom_attention_mask=torch.ones(1, NUM_ATOMS, dtype=torch.bool),
            asym_id=asym[None],
            mol_type=torch.zeros(1, NUM_TOKENS, dtype=torch.long),
            relative_position_encoding=pair_features,
        )

    module = convert.confidence_module(m)
    first = np.searchsorted(atom_to_token.numpy(), np.arange(NUM_TOKENS))
    position = np.arange(NUM_ATOMS) - first[atom_to_token.numpy()]
    token_mask = mask(NUM_TOKENS, PADDING)
    s_in = pad(s_inputs[0].numpy(), PADDING, axes=(0,))
    z_in = pad(z[0].numpy(), PADDING, axes=(0, 1), seed=1)
    pf_in = pad(pair_features[0].numpy(), PADDING, axes=(0, 1), seed=2)
    rep_in = pad(coords[0].numpy()[rep_atom.numpy()], PADDING, axes=(0,), seed=3)
    a2t = np.concatenate([atom_to_token.numpy(), [NUM_TOKENS] * PADDING])
    pos_in = np.concatenate([position, [0] * PADDING])

    def f(module, s, z, rep, a2t, pos, mask, pf):
        out = _confidence(module, s, z, rep, a2t, pos, mask, pf)
        return out.pae, out.pde, out.plddt, out.resolved

    args = (module, s_in, z_in, rep_in, a2t, pos_in, token_mask, pf_in)
    pae, pde, plddt, resolved = _call(f, *args)
    _check(pae[:NUM_TOKENS, :NUM_TOKENS], ref["pae_logits"][0])
    _check(pde[:NUM_TOKENS, :NUM_TOKENS], ref["pde_logits"][0])
    _check(plddt[:NUM_ATOMS], ref["plddt_logits"][0])
    _check(resolved[:NUM_ATOMS], ref["resolved_logits"][0])
    assert_finite((pae, pde, plddt, resolved, _grads(f, args, NUM_TOKENS)))


# --- Language model ------------------------------------------------------------------------


def test_language_model_pair():
    """`LanguageModelShim`: ESMC's layers combined and projected to the pair."""
    m = randomize(_layers().LanguageModelShim(d_z=C_Z, d_model=24, num_layers=3), 0)
    gen = torch.Generator().manual_seed(1)
    hidden = _randn(gen, 1, NUM_TOKENS, 4, 24)
    with torch.no_grad():
        ref = m(hidden)[0]
    out = convert.language_model_pair(m)(jnp.array(hidden[0].numpy().swapaxes(0, 1)))
    _check(out, ref)


def test_esmc():
    """`EsmcModel`'s hidden states (every layer's input and the final output), two chains
    attending separately."""
    from esm.models.esmc import EsmcConfig, EsmcModel

    m = randomize(
        EsmcModel(EsmcConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=3)), 0
    )
    tokens = torch.tensor([[0, 5, 9, 13, 2, 0, 7, 21, 2]])
    sequence_id = (tokens == 0).cumsum(-1) - 1
    with torch.no_grad():
        ref = m(input_ids=tokens, sequence_id=sequence_id, output_hidden_states=True).hidden_states
    model = convert.esmc(m)
    n = tokens.shape[1]
    tokens_in = np.concatenate([tokens[0].numpy(), [1] * PADDING])
    seq_in = np.concatenate([sequence_id[0].numpy(), [0] * PADDING])
    f = lambda model, tokens, seq, mask: model(tokens, seq, mask, key=KEY)
    vectors = jax.nn.one_hot(tokens_in, 64)
    out = _call(f, model, vectors, seq_in, mask(n, PADDING))
    _check(out[:, :n], ref[:, 0])
    grads = _call(
        lambda model, v: jax.grad(lambda v: f(model, v, seq_in, mask(n, PADDING))[:, :n].sum())(v),
        model,
        vectors,
    )
    assert_finite((out, grads))


def test_esmc_for_masked_lm():
    """`EsmcForMaskedLM`'s logits (ESMC and its head) on a sequence with masked positions,
    and the vocabulary jfold names its tokens with."""
    from esm.models.esmc import EsmcConfig, EsmcForMaskedLM
    from esm.tokenization import EsmSequenceTokenizer

    from jfold.models.esmfold2 import ESMC_VOCAB

    assert ESMC_VOCAB == EsmSequenceTokenizer().get_vocab()
    m = randomize(
        EsmcForMaskedLM(
            EsmcConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=3)
        ),
        0,
    )
    tokens = torch.tensor([[0, 5, 32, 13, 9, 32, 21, 2]])  # <cls> ... <mask> ... <eos>
    with torch.no_grad():
        ref = m(input_ids=tokens).logits[0]
    model = convert.esmc_for_masked_lm(m)
    vectors = jax.nn.one_hot(tokens[0].numpy(), 64)
    out = _call(lambda model, v: model(v, key=KEY), model, vectors)
    _check(out, ref)
    assert_finite(out)


def test_esmc_residues_match_upstream():
    """ESMC's input ids for each protein residue, X and unusual letters included, are the
    ones upstream's input builder gives."""
    from esm.models.esmfold2.processor import ESMFold2InputBuilder
    from esm.utils.structure.input_builder import ProteinInput, StructurePredictionInput

    chains = [
        ProteinInput(id="A", sequence="MKTXAYIAKQRX"),
        ProteinInput(id="B", sequence="GSHBZUOXLEDPV"),
    ]
    batched, _ = ESMFold2InputBuilder().prepare_input(
        StructurePredictionInput(sequences=chains), seed=0
    )
    f = {k: v[0].numpy() for k, v in batched.items() if isinstance(v, torch.Tensor)}
    protein = (f["mol_type"] == 0) & f["token_attention_mask"].astype(bool)
    assert (f["res_type"][protein] == 22).any()  # the unknown residue is exercised
    residues = esmfold2.esmc_residues(jax.nn.one_hot(f["res_type"], 33))
    np.testing.assert_array_equal(np.argmax(residues, -1)[protein], f["input_ids"][protein])


def test_lm_layout():
    """`lm_layout` and `lm_input` pack chains as upstream's `compute_lm_hidden_states`:
    collapsed multi-token residues, chains in `asym_id` order, BOS/EOS rows."""
    asym = np.array([1, 1, 1, 0, 0, 2])
    residue = np.array([0, 1, 1, 0, 1, 0])
    protein = np.array([True, True, True, True, True, False])
    res_type = np.array([2, 3, 3, 12, 22, 9])
    layout = esmfold2.lm_layout(asym, residue, protein)
    vectors = esmfold2.lm_input(layout, esmfold2.esmc_residues(jax.nn.one_hot(res_type, 33)))
    # chain 0: BOS L X EOS; chain 1: BOS A R EOS (ESMC ids: L=4, X=24, A=5, R=10)
    np.testing.assert_array_equal(np.argmax(vectors, -1), [0, 4, 24, 2, 0, 5, 10, 2])
    np.testing.assert_array_equal(vectors.sum(-1), 1)
    np.testing.assert_array_equal(layout.sequence_id, [0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(layout.positions, [5, 6, 6, 1, 2, -1])


def test_soft_features():
    """`features` takes the residue type and MSA as probability vectors (a soft sequence)
    under `jit`: one-hot vectors give the indices' features; soft ones reach the residue
    channels and, where only the query row is real, the profile."""
    gen = torch.Generator().manual_seed(1)
    f = _system(gen, True)
    f["msa_attention_mask"][1:, :2] = False  # tokens 0 and 1: query row only
    vectors = f | {
        "res_type": np.eye(33, dtype=np.float32)[f["res_type"]],
        "msa": np.eye(33, dtype=np.float32)[f["msa"]],
    }
    soft_features = eqx.filter_jit(esmfold2.features)
    ref = esmfold2.features(f)
    for ours, theirs in zip(jax.tree.leaves(soft_features(vectors)), jax.tree.leaves(ref)):
        np.testing.assert_array_equal(ours, theirs)

    soft = jax.nn.softmax(_randn(gen, NUM_TOKENS, 33).numpy())
    msa = vectors["msa"].copy()
    msa[0] = soft
    out = soft_features(vectors | {"res_type": soft, "msa": msa})
    np.testing.assert_allclose(out.token_features[:, :33], soft, rtol=1e-6)
    np.testing.assert_allclose(out.token_features[:2, 33:66], soft[:2], rtol=1e-6)
    np.testing.assert_allclose(out.msa_features[0, :, :33], soft, rtol=1e-6)


# --- Experimental model type ----------------------------------------------------------------


def _experimental():
    from esm.models.esmfold2 import experimental

    return experimental


@pytest.mark.parametrize("padding", [0, PADDING], ids=["unpadded", "padded"])
@pytest.mark.parametrize("rows", [NUM_ROWS, 1], ids=["msa", "query-only"])
def test_experimental_msa(rows, padding):
    """The experimental `MSAEncoder` on `TrackedMSA` around `MSAModule`: MSA-first
    `MSALayer`s, the outer product divided before its projection; no update without MSA
    rows past the query."""
    m = _unchunk(randomize(_experimental().MSAEncoder(C_M, C_Z, C_S_INPUTS, 4, 2, 2, 4), 0))
    gen = torch.Generator().manual_seed(1)
    z = _randn(gen, 1, NUM_TOKENS, NUM_TOKENS, C_Z)
    s_inputs = _randn(gen, 1, NUM_TOKENS, C_S_INPUTS)
    msa = torch.randint(0, 33, (rows, NUM_TOKENS), generator=gen)
    has_deletion = (torch.rand(rows, NUM_TOKENS, generator=gen) > 0.7).float()
    deletion_value = torch.rand(rows, NUM_TOKENS, generator=gen)
    msa_mask = torch.ones(rows, NUM_TOKENS, dtype=torch.bool)
    msa_mask[2:, 4:] = False
    features = esmfold2.msa_features(
        jax.nn.one_hot(msa.numpy(), 33),
        jnp.array(has_deletion.numpy()),
        jnp.array(deletion_value.numpy()),
        jnp.array(msa_mask.numpy()),
    )
    with torch.no_grad():
        ref = m(
            z,
            s_inputs,
            torch.from_numpy(np.array(features[..., :33]).swapaxes(0, 1))[None],
            torch.from_numpy(np.array(features[..., 33]).T)[None],
            torch.from_numpy(np.array(features[..., 34]).T)[None],
            msa_mask.T[None].float(),
        )[0]

    module = convert.experimental_msa(m)
    extra_rows = 2 if padding else 0
    z_in = pad(z[0].numpy(), padding, axes=(0, 1))
    s_in = pad(s_inputs[0].numpy(), padding, axes=(0,))
    f_in = pad(pad(np.array(features), extra_rows, axes=(0,)), padding, axes=(1,), seed=1)
    mask_in = np.pad(msa_mask.numpy(), [(0, extra_rows), (0, padding)])
    token_mask = mask(NUM_TOKENS, padding)
    f = lambda module, z, feats, s, msa_mask, mask: module(z, feats, s, msa_mask, mask, key=KEY)
    args = (module, z_in, f_in, s_in, mask_in, token_mask)
    out = _call(f, *args)
    if rows == 1:
        assert np.all(np.asarray(out) == 0) and np.all(ref.numpy() == 0)
    else:
        _check(out[:NUM_TOKENS, :NUM_TOKENS], ref)
    assert_finite((out, _grads(f, args, NUM_TOKENS)))


@pytest.mark.parametrize("padding", [0, PADDING], ids=["unpadded", "padded"])
def test_experimental_confidence(padding):
    """The experimental `ConfidenceHead` (PAE without a norm, pLDDT; no PDE or resolved) on
    `ConfidenceModule`."""
    from esm.models.esmfold2.config import EsmFold2Config

    config = EsmFold2Config(
        type="experimental",
        hidden_size=C_TOKEN,
        pairwise_hidden_size=C_Z,
        single_inputs_size=C_S_INPUTS,
        confidence_head={
            "num_hidden_layers": 2,
            "num_plddt_bins": 6,
            "num_pae_bins": 4,
            "distogram_bins": 9,
        },
    )
    m = _unchunk(randomize(_experimental().ConfidenceHead(config), 0))
    gen = torch.Generator().manual_seed(1)
    atom_to_token = _atom_inputs(gen)[-1]
    s_inputs = _randn(gen, 1, NUM_TOKENS, C_S_INPUTS)
    z = _randn(gen, 1, NUM_TOKENS, NUM_TOKENS, C_Z)
    pair_features = _randn(gen, 1, NUM_TOKENS, NUM_TOKENS, C_Z)
    coords = 8 * _randn(gen, 1, NUM_ATOMS, 3)
    first = np.searchsorted(atom_to_token.numpy(), np.arange(NUM_TOKENS))
    rep_atom = torch.from_numpy(np.minimum(first, NUM_ATOMS - 1))
    with torch.no_grad():
        ref = m(
            s_inputs=s_inputs,
            z=z,
            x_pred=coords,
            distogram_atom_idx=rep_atom[None],
            token_attention_mask=torch.ones(1, NUM_TOKENS, dtype=torch.bool),
            atom_to_token=atom_to_token[None],
            atom_attention_mask=torch.ones(1, NUM_ATOMS, dtype=torch.bool),
            asym_id=torch.tensor([[0, 0, 0, 0, 1, 1, 1]]),
            mol_type=torch.zeros(1, NUM_TOKENS, dtype=torch.long),
            relative_position_encoding=pair_features,
        )

    module = convert.confidence_module(m)
    assert module.pde is None and module.resolved is None
    position = np.arange(NUM_ATOMS) - first[atom_to_token.numpy()]
    token_mask = mask(NUM_TOKENS, padding)
    s_in = pad(s_inputs[0].numpy(), padding, axes=(0,))
    z_in = pad(z[0].numpy(), padding, axes=(0, 1), seed=1)
    pf_in = pad(pair_features[0].numpy(), padding, axes=(0, 1), seed=2)
    rep_in = pad(coords[0].numpy()[rep_atom.numpy()], padding, axes=(0,), seed=3)
    a2t = np.concatenate([atom_to_token.numpy(), np.full(padding, NUM_TOKENS)])
    pos_in = np.concatenate([position, np.zeros(padding, int)])

    def f(module, s, z, rep, a2t, pos, mask, pf):
        out = _confidence(module, s, z, rep, a2t, pos, mask, pf)
        return out.pae, out.plddt

    args = (module, s_in, z_in, rep_in, a2t, pos_in, token_mask, pf_in)
    pae, plddt = _call(f, *args)
    _check(pae[:NUM_TOKENS, :NUM_TOKENS], ref["pae_logits"][0])
    _check(plddt[:NUM_ATOMS], ref["plddt_logits"][0])
    assert_finite((pae, plddt, _grads(f, args, NUM_TOKENS)))


# --- Whole model ---------------------------------------------------------------------------


def _small_model(msa: bool, experimental: bool = False):
    """A small release model, or experimental one (its MSA features disabled without an
    MSA, as in the -Fast checkpoints)."""
    from esm.models.esmfold2.config import EsmFold2Config

    config = EsmFold2Config(
        type="experimental" if experimental else "release",
        disable_msa_features=experimental and not msa,
        hidden_size=C_TOKEN,
        pairwise_hidden_size=C_Z,
        single_inputs_size=C_ATOM + 67,
        sliding_window=8,
        folding_trunk_num_hidden_layers=2,
        parcae_num_coda_layers=1,
        n_relative_residx_bins=4,
        lm_d_model=24,
        lm_num_layers=3,
        atom_encoder={
            "hidden_size": C_ATOM,
            "output_dim": 2 * C_ATOM,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "n_spatial_rope_pairs_per_axis": 2,
            "n_uid_rope_pairs": 2,
        },
        structure_head={
            "distogram_bins": 6,
            "diffusion_module": {
                "token_hidden_size": C_TOKEN,
                "fourier_dim": 8,
                "token_num_blocks": 1,
                "token_num_heads": 2,
                "c_z": C_Z,
                "c_s_inputs": C_ATOM + 67,
                "atom_encoder": {
                    "hidden_size": C_ATOM,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                },
            },
        },
        confidence_head={
            "num_hidden_layers": 1,
            "num_plddt_bins": 6,
            "num_pde_bins": 5,
            "num_pae_bins": 4,
            "distogram_bins": 9,
        },
        lm_encoder={"num_hidden_layers": 1, "lm_dropout": 0.0},
        msa_encoder={
            "enabled": msa,
            "hidden_size": C_M,
            "outer_hidden_size": 4,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "head_width": 4,
            "max_depth": None,
            "column_mask_rate": 0.0,
        },
    )
    if experimental:
        from esm.models.esmfold2.experimental import EsmFold2ExperimentalModel

        return _unchunk(randomize(EsmFold2ExperimentalModel(config), 0))
    return _unchunk(randomize(_model().EsmFold2Model(config), 0))


def _system(gen, msa: bool):
    """An unbatched upstream input dict: two chains, one with a two-token residue."""
    ref_pos, uid, charge, element, chars, atom_to_token = _atom_inputs(gen)
    first = np.searchsorted(atom_to_token.numpy(), np.arange(NUM_TOKENS))
    f = {
        "token_index": np.arange(NUM_TOKENS),
        "residue_index": np.array([0, 1, 2, 2, 0, 1, 2]),
        "asym_id": np.array([0, 0, 0, 0, 1, 1, 1]),
        "sym_id": np.array([0, 0, 0, 0, 0, 0, 0]),
        "entity_id": np.array([0, 0, 0, 0, 1, 1, 1]),
        "mol_type": np.zeros(NUM_TOKENS, np.int64),
        "res_type": torch.randint(0, 33, (NUM_TOKENS,), generator=gen).numpy(),
        "token_bonds": (torch.rand(NUM_TOKENS, NUM_TOKENS, 1, generator=gen) > 0.8).numpy(),
        "token_attention_mask": np.ones(NUM_TOKENS, bool),
        "ref_pos": ref_pos.numpy(),
        "ref_element": element.numpy(),
        "ref_charge": charge.numpy(),
        "ref_atom_name_chars": chars.numpy(),
        "ref_space_uid": uid.numpy(),
        "atom_attention_mask": np.ones(NUM_ATOMS, bool),
        "atom_to_token": atom_to_token.numpy(),
        "distogram_atom_idx": np.minimum(first, NUM_ATOMS - 1),
        "deletion_mean": torch.rand(NUM_TOKENS, generator=gen).numpy(),
        "lm_hidden_states": _randn(gen, NUM_TOKENS, 4, 24).numpy(),
    }
    if msa:
        msa_mask = np.ones((NUM_ROWS, NUM_TOKENS), bool)
        msa_mask[3:, 5:] = False
        f |= {
            "msa": torch.randint(0, 33, (NUM_ROWS, NUM_TOKENS), generator=gen).numpy(),
            "has_deletion": (torch.rand(NUM_ROWS, NUM_TOKENS, generator=gen) > 0.7).numpy(),
            "deletion_value": torch.rand(NUM_ROWS, NUM_TOKENS, generator=gen).numpy(),
            "msa_attention_mask": msa_mask,
        }
    return f


def _pad_system(f, padding=PADDING):
    """`f` with `padding` masked garbage tokens and atoms (and two MSA rows) appended."""
    if padding == 0:
        return dict(f)
    rng = np.random.default_rng(5)
    g = dict(f)
    for k in ("token_index", "residue_index", "asym_id", "sym_id", "entity_id", "res_type"):
        g[k] = np.concatenate([f[k], rng.integers(0, 3, padding)])
    g["mol_type"] = np.concatenate([f["mol_type"], [0] * padding])
    g["token_attention_mask"] = mask(NUM_TOKENS, padding)
    g["token_bonds"] = np.pad(
        f["token_bonds"], [(0, padding), (0, padding), (0, 0)], constant_values=True
    )
    g["deletion_mean"] = pad(f["deletion_mean"], padding, axes=(0,))
    g["lm_hidden_states"] = pad(f["lm_hidden_states"], padding, axes=(0,))
    g["distogram_atom_idx"] = np.concatenate([f["distogram_atom_idx"], [NUM_ATOMS] * padding])
    g["ref_pos"] = pad(f["ref_pos"], padding, axes=(0,))
    g["ref_charge"] = pad(f["ref_charge"], padding, axes=(0,))
    g["ref_element"] = np.concatenate([f["ref_element"], rng.integers(0, 128, padding)])
    g["ref_atom_name_chars"] = np.concatenate(
        [f["ref_atom_name_chars"], rng.integers(0, 64, (padding, 4))]
    )
    g["ref_space_uid"] = np.concatenate([f["ref_space_uid"], [9] * padding])
    g["atom_attention_mask"] = mask(NUM_ATOMS, padding)
    g["atom_to_token"] = np.concatenate([f["atom_to_token"], [NUM_TOKENS] * padding])
    if "msa" in f:
        g["msa"] = np.pad(
            np.pad(f["msa"], [(0, 0), (0, padding)], constant_values=4),
            [(0, 2), (0, 0)],
            constant_values=7,
        )
        g["has_deletion"] = pad(pad(f["has_deletion"] * 1.0, padding, (1,)), 2, (0,))
        g["deletion_value"] = pad(pad(f["deletion_value"], padding, (1,)), 2, (0,))
        g["msa_attention_mask"] = np.pad(f["msa_attention_mask"], [(0, 2), (0, padding)])
    return g


def _upstream(m, f, z0, coords):
    """The upstream forward from state `z0`, with the sampler replaced by `coords`; returns
    its outputs and what the sampler was given."""
    seen = {}

    def sample(**kw):
        seen.update(kw)
        return {"sample_atom_coords": coords[None]}

    batched = {k: torch.as_tensor(v)[None] for k, v in f.items()}
    m._init_pair_state = lambda ref: z0[None]  # the experimental model starts from zeros
    m.structure_head.sample = sample
    with _float32_attention():
        out = m(
            **batched,
            num_loops=NUM_LOOPS - 1,
            num_diffusion_samples=1,
            msa_column_mask_rate=0.0,
            msa_max_depth=None,
        )
    return out, seen


NUM_LOOPS = 3


def _upstream_schedule(head, cap: float = 256.0):
    """`DiffusionStructureHead.sample`'s noise levels, `fold`'s default cap applied."""
    schedule = head.inference_noise_schedule(STEPS)
    schedule = schedule[schedule <= cap]
    return torch.nn.functional.pad(schedule, (1, 0), value=cap)


@pytest.mark.parametrize("msa", [False, True], ids=["single-sequence", "msa"])
def test_esmfold2(msa):
    """The whole model from the same initial trunk state: `s_inputs`, the trunk's pair,
    the distogram, a denoising step from the upstream's trunk, and the confidence logits
    for a structure."""
    m = _small_model(msa)
    gen = torch.Generator().manual_seed(1)
    f = _system(gen, msa)
    z0 = _randn(gen, NUM_TOKENS, NUM_TOKENS, C_Z)
    coords = 8 * _randn(gen, NUM_ATOMS, 3)
    out, seen = _upstream(m, f, z0, coords)
    x_noisy = 10 * _randn(gen, 1, NUM_ATOMS, 3)
    t = 7.5
    with torch.no_grad(), _float32_attention():
        denoised = m.structure_head.diffusion_module(
            x_noisy=x_noisy,
            t_hat=torch.tensor([t]),
            ref_pos=seen["ref_pos"],
            ref_charge=seen["ref_charge"],
            ref_mask=seen["ref_mask"],
            ref_element=seen["ref_element"],
            ref_atom_name_chars=seen["ref_atom_name_chars"],
            ref_space_uid=seen["ref_space_uid"],
            tok_idx=seen["tok_idx"],
            s_inputs=seen["s_inputs"],
            s_trunk=None,
            z_trunk=seen["z_trunk"],
            relative_position_encoding=seen["relative_position_encoding"],
            asym_id=None,
            residue_index=None,
            entity_id=None,
            token_index=None,
            sym_id=None,
            token_attention_mask=seen["token_attention_mask"],
        )["x_denoised"][0]

    model = convert.esmfold2(m)
    model = assert_round_trip(model)
    assert_schedule(model.sampling, _upstream_schedule(m.structure_head))
    features = esmfold2.features(_pad_system(f), r_max=4)
    z0_in = jnp.array(pad(z0.numpy(), PADDING, axes=(0, 1)))
    x_in = jnp.array(pad(x_noisy[0].numpy(), PADDING, axes=(0,)))
    coords_in = jnp.array(pad(coords.numpy(), PADDING, axes=(0,)))
    run = eqx.filter_jit(_run)
    inputs, z, distogram, ours, conf = run(model, features, z0_in, x_in, coords_in, t)
    n = NUM_TOKENS
    _check(inputs.s_inputs[:n], seen["s_inputs"][0])
    _check(z[:n, :n], seen["z_trunk"][0])
    _check(distogram[:n, :n], out["distogram_logits"][0])

    # Later stages from the upstream's trunk, so errors do not compound.
    z_ref = jnp.array(pad(seen["z_trunk"][0].numpy(), PADDING, axes=(0, 1)))
    _, _, _, ours, conf = run(model, features, z0_in, x_in, coords_in, t, z_ref)
    _check(ours[:NUM_ATOMS], denoised)
    _check(conf.pae[:n, :n], out["pae_logits"][0])
    _check(conf.pde[:n, :n], out["pde_logits"][0])
    _check(conf.plddt[:NUM_ATOMS], out["plddt_logits"][0])
    _check(conf.resolved[:NUM_ATOMS], out["resolved_logits"][0])
    assert_finite((z, distogram, ours, conf))


def _run(model, features, z0, x_noisy, coords, t, z_trunk=None):
    """The trunk from the pair `z0`, then the heads (on the pair `z_trunk` if given)."""
    inputs = model.embed(features, key=KEY)
    state = TrunkState(s=inputs.s_init, z=z0)
    for _ in range(NUM_LOOPS):
        state = model.trunk(features, inputs, state, key=KEY)
    state = model.trunk.finish(features, state, key=KEY)
    used = state if z_trunk is None else TrunkState(s=state.s, z=z_trunk)
    context = model.diffusion.prepare(features, inputs, used)
    denoised = model.diffusion(context, x_noisy, jnp.array(t))
    conf = None
    if model.confidence is not None:
        conf = model.confidence_logits(features, inputs, used, coords, key=KEY)
    return inputs, state.z, model.distogram(state.z), denoised, conf


@pytest.mark.parametrize("padding", [0, PADDING], ids=["unpadded", "padded"])
@pytest.mark.parametrize("msa", [False, True], ids=["single-sequence", "msa"])
def test_esmfold2_experimental(msa, padding):
    """The whole experimental model, as `test_esmfold2` (both start the trunk from zeros
    here)."""
    m = _small_model(msa, experimental=True)
    gen = torch.Generator().manual_seed(1)
    f = _system(gen, msa)
    coords = 8 * _randn(gen, NUM_ATOMS, 3)
    out, seen = _upstream(m, f, None, coords)
    _check_experimental(m, f, out, seen, padding, gen, r_max=4)


def _check_experimental(m, f, out, seen, padding, gen, tol=1e-5, **feature_kwargs):
    """Compare a converted experimental model with the upstream run `out`/`seen` on `f`."""
    num_tokens, num_atoms = f["token_attention_mask"].shape[0], f["atom_attention_mask"].shape[0]
    x_noisy = 10 * _randn(gen, 1, num_atoms, 3)
    t = 7.5
    with torch.no_grad(), _float32_attention():
        denoised = m.structure_head.diffusion_module(
            x_noisy=x_noisy,
            t_hat=torch.tensor([t]),
            ref_pos=seen["ref_pos"],
            ref_charge=seen["ref_charge"],
            ref_mask=seen["ref_mask"],
            ref_element=seen["ref_element"],
            ref_atom_name_chars=seen["ref_atom_name_chars"],
            ref_space_uid=seen["ref_space_uid"],
            tok_idx=seen["tok_idx"],
            s_inputs=seen["s_inputs"],
            s_trunk=None,
            z_trunk=seen["z_trunk"],
            relative_position_encoding=seen["relative_position_encoding"],
            asym_id=None,
            residue_index=None,
            entity_id=None,
            token_index=None,
            sym_id=None,
            token_attention_mask=seen["token_attention_mask"],
        )["x_denoised"][0]

    # Column masking off, as in the upstream run.
    model = _column_mask_rate(convert.esmfold2_experimental(m), 0.0)
    model = assert_round_trip(model)
    assert_schedule(model.sampling, _upstream_schedule(m.structure_head))
    features = esmfold2.features(_pad_system(f, padding), **feature_kwargs)
    x_in = jnp.array(pad(x_noisy[0].numpy(), padding, axes=(0,)))
    coords_in = jnp.array(pad(out["sample_atom_coords"][0].numpy(), padding, axes=(0,)))
    run = eqx.filter_jit(_run_from_init)
    inputs, z, distogram, _, _ = run(model, features, x_in, coords_in, t)
    n, a = num_tokens, num_atoms
    real = f["atom_attention_mask"].astype(bool)
    _check(inputs.s_inputs[:n], seen["s_inputs"][0], tol)
    _check(z[:n, :n], seen["z_trunk"][0], tol)
    _check(distogram[:n, :n], out["distogram_logits"][0], tol)

    z_ref = jnp.array(pad(seen["z_trunk"][0].numpy(), padding, axes=(0, 1)))
    _, _, _, ours, conf = run(model, features, x_in, coords_in, t, z_ref)
    _check(np.asarray(ours[:a])[real], denoised[real], tol)
    if m.confidence_head is not None:
        _check(conf.pae[:n, :n], out["pae_logits"][0], tol)
        _check(np.asarray(conf.plddt[:a])[real], out["plddt_logits"][0][real], tol)
        assert conf.pde is None and conf.resolved is None
    assert_finite((z, distogram, ours, conf))


def _run_from_init(model, features, x_noisy, coords, t, z_trunk=None):
    inputs = model.embed(features, key=KEY)
    return _run(model, features, model.trunk.init_state(inputs, key=KEY).z, x_noisy, coords, t, z_trunk)


def _column_mask_rate(model, rate: float):
    """`model` with its trunk init masking `rate` of the MSA columns."""
    return dataclasses.replace(
        model, trunk_init=dataclasses.replace(model.trunk_init, msa_column_mask_rate=rate)
    )


# --- Released weights ----------------------------------------------------------------------


def _reference(name: str):
    """The released model as `jfold.pretrained` builds it (as `from_pretrained` loads it for
    inference), from the Hugging Face cache."""
    from jfold import pretrained
    from jfold.pretrained.recipes import esmfold2 as releases

    return pretrained.upstream(name, checkpoint_path=releases.snapshot(name))


@pytest.mark.checkpoint
def test_esmfold2_checkpoint():
    """The released ESMFold2 (`biohub/ESMFold2`, ESMC-6B bundled) on a two-chain complex:
    ESMC's hidden states, then the whole model from the same initial state and ESMC
    states, as in `test_esmfold2`. The language model's per-loop dropout is off."""
    from esm.models.esmfold2.processor import ESMFold2InputBuilder
    from esm.utils.structure.input_builder import ProteinInput, StructurePredictionInput

    m = _reference("esmfold2")
    m.config.lm_encoder.lm_dropout = 0.0
    chains = [
        ProteinInput(id="A", sequence="MKTAYIAKQRQISFVKSHFSRQ"),
        ProteinInput(id="B", sequence="GSHMLEDPV"),
    ]
    batched, _ = ESMFold2InputBuilder().prepare_input(
        StructurePredictionInput(sequences=chains), seed=0
    )
    f = {k: v[0].numpy() for k, v in batched.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        hidden = m._compute_lm_hidden_states(
            batched["input_ids"],
            batched["asym_id"],
            batched["residue_index"],
            batched["mol_type"],
            batched["token_attention_mask"].bool(),
        )[0]

    protein = (f["mol_type"] == 0) & f["token_attention_mask"]
    layout = esmfold2.lm_layout(f["asym_id"], f["residue_index"], protein)
    residues = esmfold2.esmc_residues(jax.nn.one_hot(f["res_type"], 33))
    np.testing.assert_array_equal(np.argmax(residues, -1)[protein], f["input_ids"][protein])
    lm = convert.esmc(m.esmc)
    ours = eqx.filter_jit(lambda lm, *a: lm(*a, key=KEY))(
        lm,
        esmfold2.lm_input(layout, residues),
        layout.sequence_id,
        np.ones(len(layout.source), bool),
    )
    _check(esmfold2.token_hidden_states(ours, jnp.array(layout.positions)), hidden, 1e-4)
    del lm, ours
    m.esmc = None

    f["lm_hidden_states"] = hidden.numpy()
    num_tokens, num_atoms = f["token_attention_mask"].shape[0], f["atom_attention_mask"].shape[0]
    gen = torch.Generator().manual_seed(1)
    z0 = _randn(gen, num_tokens, num_tokens, m.config.pairwise_hidden_size)
    coords = 8 * _randn(gen, num_atoms, 3)
    out, seen = _upstream(m, f, z0, coords)
    x_noisy = 10 * _randn(gen, 1, num_atoms, 3)
    t = 7.5
    with torch.no_grad(), _float32_attention():
        denoised = m.structure_head.diffusion_module(
            x_noisy=x_noisy,
            t_hat=torch.tensor([t]),
            ref_pos=seen["ref_pos"],
            ref_charge=seen["ref_charge"],
            ref_mask=seen["ref_mask"],
            ref_element=seen["ref_element"],
            ref_atom_name_chars=seen["ref_atom_name_chars"],
            ref_space_uid=seen["ref_space_uid"],
            tok_idx=seen["tok_idx"],
            s_inputs=seen["s_inputs"],
            s_trunk=None,
            z_trunk=seen["z_trunk"],
            relative_position_encoding=seen["relative_position_encoding"],
            asym_id=None,
            residue_index=None,
            entity_id=None,
            token_index=None,
            sym_id=None,
            token_attention_mask=seen["token_attention_mask"],
        )["x_denoised"][0]

    model = _column_mask_rate(convert.esmfold2(m), 0.0)  # as upstream

    model = assert_round_trip(model)
    assert_schedule(model.sampling, _upstream_schedule(m.structure_head))
    features = esmfold2.features(f)
    run = eqx.filter_jit(_run)
    inputs, z, distogram, _, _ = run(
        model, features, jnp.array(z0.numpy()), x_noisy[0].numpy(), coords.numpy(), t
    )
    atom_mask = f["atom_attention_mask"]
    _check(inputs.s_inputs, seen["s_inputs"][0], 1e-4)
    _check(z, seen["z_trunk"][0], 1e-3)
    _check(distogram, out["distogram_logits"][0], 1e-3)
    z_ref = jnp.array(seen["z_trunk"][0].numpy())
    _, _, _, ours, conf = run(
        model, features, jnp.array(z0.numpy()), x_noisy[0].numpy(), coords.numpy(), t, z_ref
    )
    _check(np.asarray(ours)[atom_mask], denoised[atom_mask], 1e-4)
    _check(conf.pae, out["pae_logits"][0], 1e-3)
    _check(conf.pde, out["pde_logits"][0], 1e-3)
    _check(np.asarray(conf.plddt)[atom_mask], out["plddt_logits"][0][atom_mask], 1e-3)
    _check(np.asarray(conf.resolved)[atom_mask], out["resolved_logits"][0][atom_mask], 1e-3)


EXPERIMENTAL = (
    "esmfold2_experimental_fast",
    "esmfold2_experimental_fast_cutoff2025",
    "esmfold2_experimental",
    "esmfold2_experimental_cutoff2025",
    "esmfold2_experimental_fast_base300m",
    "esmfold2_experimental_fast_base600m",
    "esmfold2_experimental_fast_base6b",
)


@pytest.mark.checkpoint
@pytest.mark.parametrize(
    "name", ["esmfold2_experimental_fast_base300m", "esmfold2_experimental_fast_base600m"]
)
def test_esmc_sizes_checkpoint(name):
    """The smaller ESMCs of the scaling checkpoints (300M, 600M): hidden states match the
    upstream's, both in float32 from the same (bfloat16-valued) weights; and the shared
    entry `jfold.pretrained` attaches is that same ESMC."""
    from esm.models.esmfold2.processor import ESMFold2InputBuilder
    from esm.utils.structure.input_builder import ProteinInput, StructurePredictionInput

    from jfold.pretrained.recipes import esmfold2 as releases

    m = _reference(name)
    m.esmc = m.esmc.float()
    chains = [
        ProteinInput(id="A", sequence="MKTAYIAKQRQISFVKSHFSRQ"),
        ProteinInput(id="B", sequence="GSHMLEDPV"),
    ]
    batched, _ = ESMFold2InputBuilder().prepare_input(
        StructurePredictionInput(sequences=chains), seed=0
    )
    f = {k: v[0].numpy() for k, v in batched.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        hidden = m._compute_lm_hidden_states(
            batched["input_ids"],
            batched["asym_id"],
            batched["residue_index"],
            batched["mol_type"],
            batched["token_attention_mask"].bool(),
        )[0]
    protein = (f["mol_type"] == 0) & f["token_attention_mask"]
    layout = esmfold2.lm_layout(f["asym_id"], f["residue_index"], protein)
    residues = esmfold2.esmc_residues(jax.nn.one_hot(f["res_type"], 33))
    lm = releases.RELEASES[releases.ESMC[name]].convert(m)
    ours = eqx.filter_jit(lambda lm, *a: lm(*a, key=KEY))(
        lm,
        esmfold2.lm_input(layout, residues),
        layout.sequence_id,
        np.ones(len(layout.source), bool),
    )
    _check(esmfold2.token_hidden_states(ours, jnp.array(layout.positions)), hidden, 1e-4)


@pytest.mark.checkpoint
@pytest.mark.parametrize("name", EXPERIMENTAL)
def test_esmfold2_experimental_checkpoint(name):
    """A released experimental checkpoint on a two-chain complex (with an MSA for those
    with an MSA encoder), as `test_esmfold2_experimental`. ESMC is not loaded (its parity
    is `test_esmfold2_checkpoint`'s); random states stand in for its hidden states."""
    from esm.models.esmfold2.processor import ESMFold2InputBuilder
    from esm.utils.structure.input_builder import ProteinInput, StructurePredictionInput

    m = _reference(name)
    m.esmc = None
    chains = [
        ProteinInput(id="A", sequence="MKTAYIAKQRQISFVKSHFSRQ"),
        ProteinInput(id="B", sequence="GSHMLEDPV"),
    ]
    batched, _ = ESMFold2InputBuilder().prepare_input(
        StructurePredictionInput(sequences=chains), seed=0
    )
    f = {k: v[0].numpy() for k, v in batched.items() if isinstance(v, torch.Tensor)}
    f.pop("input_ids")
    num_tokens = f["token_attention_mask"].shape[0]
    gen = torch.Generator().manual_seed(1)
    layers = m.config.lm_num_layers + 1
    hidden = _randn(gen, num_tokens, layers, m.config.lm_d_model).numpy()
    f["lm_hidden_states"] = hidden * (f["mol_type"] == 0)[:, None, None]
    if m.msa_encoder is not None:
        rows = 6
        msa = np.where(
            torch.rand(rows, num_tokens, generator=gen).numpy() < 0.3,
            torch.randint(0, 20, (rows, num_tokens), generator=gen).numpy(),
            f["res_type"][None],
        )
        msa[0] = f["res_type"]
        msa_mask = np.ones((rows, num_tokens), bool)
        msa_mask[4:, :5] = False
        f |= {
            "msa": msa,
            "msa_attention_mask": msa_mask,
            "has_deletion": (torch.rand(rows, num_tokens, generator=gen) > 0.9).numpy(),
            "deletion_value": torch.rand(rows, num_tokens, generator=gen).numpy() * 0.1,
            "deletion_mean": torch.rand(num_tokens, generator=gen).numpy() * 0.1,
        }
    coords = 8 * _randn(gen, f["atom_attention_mask"].shape[0], 3)
    out, seen = _upstream(m, f, None, coords)
    _check_experimental(m, f, out, seen, 0, gen, 1e-4)


@pytest.mark.parametrize("experimental", [False, True], ids=["release", "experimental"])
def test_msa_column_mask(experimental, monkeypatch):
    """Column masking leaves the query row and padding alone and masks a column in every
    other row; the trunk with it matches the upstream's, its own draw replaced by ours."""
    import esm.models.esmfold2.experimental as upstream_experimental
    import esm.models.esmfold2.model as upstream_model

    m = _small_model(True, experimental=experimental)
    gen = torch.Generator().manual_seed(1)
    f = _system(gen, True)
    z0 = None if experimental else _randn(gen, NUM_TOKENS, NUM_TOKENS, C_Z)
    coords = 8 * _randn(gen, NUM_ATOMS, 3)
    to_jfold = convert.esmfold2_experimental if experimental else convert.esmfold2
    model = _column_mask_rate(to_jfold(m), 0.5)
    features = esmfold2.features(_pad_system(f), r_max=4)

    real, masked = features.msa_mask, model.embed(features, key=KEY).extras.msa_mask
    dropped = (real & ~masked).any(0)
    assert jnp.array_equal(masked[0], real[0]) and not (masked & ~real).any()
    assert dropped.any() and not (masked[1:] & dropped[None]).any()

    s = f["msa"].shape[0]
    ours = torch.as_tensor(np.asarray(masked[:s, :NUM_TOKENS]))[None]
    calls = []

    def column_mask(mask, rate):
        calls.append(rate)
        return ours

    for module in (upstream_model, upstream_experimental):
        monkeypatch.setattr(module, "maybe_apply_msa_column_masking", column_mask)
    _, seen = _upstream(m, f, z0, coords)
    assert len(calls) == 1
    n = NUM_TOKENS + PADDING
    z0_in = jnp.zeros((n, n, C_Z)) if z0 is None else jnp.array(pad(z0.numpy(), PADDING, (0, 1)))
    x_in = jnp.zeros((NUM_ATOMS + PADDING, 3))
    _, z, *_ = eqx.filter_jit(_run)(model, features, z0_in, x_in, x_in, 7.5)
    _check(z[:NUM_TOKENS, :NUM_TOKENS], seen["z_trunk"][0])


@pytest.mark.parametrize("experimental", [False, True], ids=["release", "experimental"])
def test_run_trunk(experimental):
    """`run_trunk` is `embed`, `init_state`, the passes and `finish` with its keys split as
    documented (dropout on, as ESMFold2's language-model pair runs, so the keys matter)."""
    gen = torch.Generator().manual_seed(1)
    m = _small_model(msa=True, experimental=experimental)
    model = (convert.esmfold2_experimental if experimental else convert.esmfold2)(m)
    features = esmfold2.features(_pad_system(_system(gen, msa=True)), r_max=4)
    key = jax.random.key(3)

    def by_hand(model, features, key):
        k_embed, k_init, k_passes, k_finish = jax.random.split(key, 4)
        inputs = model.embed(features, key=k_embed)
        state = model.trunk.init_state(inputs, key=k_init)
        for k in jax.random.split(k_passes, NUM_LOOPS):
            state = model.trunk(features, inputs, state, key=k)
        return inputs, model.trunk.finish(features, state, key=k_finish)

    run = eqx.filter_jit(lambda m, f, k: m.run_trunk(f, num_passes=NUM_LOOPS, key=k))
    (inputs, z), (inputs_ref, z_ref) = (
        run(model, features, key),
        eqx.filter_jit(by_hand)(model, features, key),
    )
    _check(inputs.s_inputs, inputs_ref.s_inputs, tol=1e-6)
    _check(z.z, z_ref.z, tol=1e-5)
    with pytest.raises(ValueError):
        model.run_trunk(features, num_passes=0, key=key)


def test_model_runs_esmc():
    """Without precomputed states the model runs its ESMC on the features' `lm_tokens`: the
    same trunk inputs as the states computed outside (`lm_layout`, `lm_input`,
    `token_hidden_states`), and a soft sequence's gradient reaches the language model."""
    from esm.models.esmc import EsmcConfig, EsmcModel

    gen = torch.Generator().manual_seed(1)
    f = _system(gen, msa=False)
    del f["lm_hidden_states"]
    esmc = convert.esmc(
        randomize(
            EsmcModel(EsmcConfig(hidden_size=24, num_attention_heads=4, num_hidden_layers=3)), 0
        )
    )
    model = convert.esmfold2(_small_model(msa=False))
    model = dataclasses.replace(model, trunk_init=dataclasses.replace(model.trunk_init, esmc=esmc))
    features = esmfold2.features(f, r_max=4)
    assert features.extras.hidden is None and features.extras.tokens is not None

    protein = (f["mol_type"] == 0) & f["token_attention_mask"]
    layout = esmfold2.lm_layout(f["asym_id"], f["residue_index"], protein)
    residues = esmfold2.esmc_residues(jax.nn.one_hot(f["res_type"], 33))
    hidden = esmc(
        esmfold2.lm_input(layout, residues),
        layout.sequence_id,
        np.ones(len(layout.source), bool),
        key=KEY,
    )
    outside = esmfold2.features(
        f
        | {"lm_hidden_states": np.asarray(esmfold2.token_hidden_states(hidden, layout.positions))},
        r_max=4,
    )
    embed = eqx.filter_jit(lambda m, x: m.embed(x, key=KEY))
    ours, ref = embed(model, features), embed(model, outside)
    _check(ours.extras.lm_pair, ref.extras.lm_pair, tol=1e-6)

    def lm_pair_sum(model, res_type):
        soft = esmfold2.features(f | {"res_type": res_type}, r_max=4)
        return model.embed(soft, key=KEY).extras.lm_pair.sum()

    grad = eqx.filter_jit(jax.grad(lm_pair_sum, argnums=1))(
        model, jax.nn.one_hot(f["res_type"], 33)
    )
    assert jnp.abs(grad).sum() > 0
    assert_finite(grad)


def test_language_model_required():
    """ESMFold2 never runs without its language model: a model without ESMC needs
    precomputed states, and ESMC's input needs concrete ids."""
    gen = torch.Generator().manual_seed(1)
    f = _system(gen, msa=False)
    del f["lm_hidden_states"]
    model = convert.esmfold2(_small_model(msa=False))
    features = esmfold2.features(f, r_max=4)
    with pytest.raises(ValueError, match="language model"):
        model.embed(features, key=KEY)
    with pytest.raises(ValueError, match="concrete"):
        jax.jit(lambda f: esmfold2.features(f, r_max=4).token_mask)(f)
    without_ids = {k: v for k, v in f.items() if k != "mol_type"}
    with pytest.raises(ValueError, match="mol_type"):
        esmfold2.features(without_ids, r_max=4)
