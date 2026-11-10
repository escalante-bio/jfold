"""Atom attention encoder and decoder parity against each upstream."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from reference import assert_finite, randomize, rel_err

from jfold.blocks.atoms import Atoms
from jfold.convert import boltz, openfold3, protenix
from jfold.models import boltz2 as boltz2_features
from jfold.models import openfold3 as openfold3_features
from jfold.models import promera as promera_features
from jfold.models import protenix as protenix_features

NUM_TOKENS, ATOMS_PER_TOKEN = 14, 5
NUM_ATOMS = NUM_TOKENS * ATOMS_PER_TOKEN  # 70: the last query block is partial
C_ATOM, C_PAIR, C_TOKEN = 16, 8, 24
HEADS, DEPTH, W, H = 2, 1, 32, 128


@dataclasses.dataclass
class Inputs:
    """One system's atoms in every upstream's vocabulary (numpy, unbatched)."""

    ref_pos: np.ndarray
    ref_charge: np.ndarray
    element: np.ndarray  # element index
    name_chars: np.ndarray  # [N, 4] character indices
    hydrogens: np.ndarray
    is_std: np.ndarray
    ref_space_uid: np.ndarray
    atom_to_token: np.ndarray
    s_trunk: np.ndarray
    z_trunk: np.ndarray
    r: np.ndarray  # noisy positions
    a: np.ndarray  # token activations into the decoder

    @property
    def num_atoms(self):
        return self.ref_pos.shape[0]

    @property
    def num_tokens(self):
        return self.s_trunk.shape[0]

    def element_onehot(self):
        return np.eye(128, dtype=np.float32)[self.element]

    def name_chars_onehot(self):
        return np.eye(64, dtype=np.float32)[self.name_chars]


def make_inputs(c_s, c_z, num_atoms, num_tokens, atom_to_token, seed) -> Inputs:
    rng = np.random.default_rng(seed)
    f32 = lambda *shape: rng.standard_normal(shape, dtype=np.float32)
    return Inputs(
        ref_pos=3 * f32(num_atoms, 3),
        ref_charge=rng.integers(-1, 2, num_atoms).astype(np.float32),
        element=rng.integers(0, 128, num_atoms),
        name_chars=rng.integers(0, 64, (num_atoms, 4)),
        hydrogens=rng.integers(0, 8, num_atoms),
        is_std=rng.random(num_atoms) < 0.8,
        ref_space_uid=atom_to_token // 2,  # conformers spanning two tokens
        atom_to_token=atom_to_token,
        s_trunk=f32(num_tokens, c_s),
        z_trunk=f32(num_tokens, num_tokens, c_z),
        r=f32(num_atoms, 3),
        a=f32(num_tokens, C_TOKEN),
    )


def padded(x: Inputs, num_atoms: int, seed: int) -> Inputs:
    """`x` with `num_atoms` garbage atoms appended, on one extra garbage token."""
    g = make_inputs(
        x.s_trunk.shape[1],
        x.z_trunk.shape[2],
        num_atoms,
        x.num_tokens + 1,
        np.full(num_atoms, x.num_tokens),
        seed,
    )
    z = g.z_trunk.copy()
    z[: x.num_tokens, : x.num_tokens] = x.z_trunk
    cat = lambda f: np.concatenate([getattr(x, f), getattr(g, f)[:num_atoms]])
    atom_fields = ["ref_pos", "ref_charge", "element", "name_chars", "hydrogens", "is_std", "r"]
    return dataclasses.replace(
        x,
        **{f: cat(f) for f in atom_fields},
        ref_space_uid=np.concatenate([x.ref_space_uid, g.ref_space_uid + 1000]),
        atom_to_token=np.concatenate([x.atom_to_token, g.atom_to_token]),
        s_trunk=np.concatenate([x.s_trunk, g.s_trunk[-1:]]),
        z_trunk=z,
        a=np.concatenate([x.a, g.a[-1:]]),
    )


t = torch.from_numpy


# --- upstream references: (module to randomize, run(Inputs) -> (a, r_update), convert,
# feature vector, trunk widths)


def _protenix_lineage(module, explicit_args):
    enc = module.AtomAttentionEncoder(
        has_coords=True,
        c_token=C_TOKEN,
        c_atom=C_ATOM,
        c_atompair=C_PAIR,
        c_s=20,
        c_z=12,
        n_blocks=DEPTH,
        n_heads=HEADS,
    )
    dec = module.AtomAttentionDecoder(
        n_blocks=DEPTH, n_heads=HEADS, c_token=C_TOKEN, c_atom=C_ATOM, c_atompair=C_PAIR
    )

    def run(x: Inputs):
        feats = {
            "ref_pos": t(x.ref_pos),
            "ref_charge": t(x.ref_charge),
            "ref_mask": torch.ones(x.num_atoms),
            "ref_element": t(x.element_onehot()),
            "ref_atom_name_chars": t(x.name_chars_onehot()),
            "ref_space_uid": t(x.ref_space_uid),
            "atom_to_token_idx": t(x.atom_to_token),
        }
        s, z, r = t(x.s_trunk), t(x.z_trunk), t(x.r)[None]
        if explicit_args:  # Mol-1 and OpenDDE take the windowed offsets precomputed
            from mol.model.mol import update_input_feature_dict

            feats = update_input_feature_dict(feats)
            args = {
                k: feats[k]
                for k in (
                    "atom_to_token_idx",
                    "ref_pos",
                    "ref_charge",
                    "ref_mask",
                    "ref_atom_name_chars",
                    "ref_element",
                    "d_lm",
                    "v_lm",
                    "pad_info",
                )
            }
            a, q, c, p = enc(**args, r_l=r, s=s, z=z)
            r_update = dec(feats["atom_to_token_idx"], t(x.a)[None], q, c, p)
        else:
            a, q, c, p = enc(feats, r_l=r, s=s, z=z)
            r_update = dec(feats, t(x.a)[None], q, c, p)
        return a[0], r_update[0]

    def features(x: Inputs):
        return protenix_features.atom_features(
            x.ref_pos,
            x.ref_charge,
            np.ones(x.num_atoms, np.float32),
            x.element_onehot(),
            x.name_chars_onehot(),
        )

    convert = lambda m: (protenix.atom_encoder(m[0]), protenix.atom_decoder(m[1]))
    return torch.nn.ModuleList([enc, dec]), run, convert, features, (20, 12)


def _protenix():
    import protenix.model.modules.transformer as module

    return _protenix_lineage(module, explicit_args=False)


def _opendde():
    import opendde.model.modules.transformer as module

    return _protenix_lineage(module, explicit_args=True)


def _mol1():
    import mol.model.modules.transformer as module

    return _protenix_lineage(module, explicit_args=True)


def _boltz_padded_batch(x: Inputs):
    """Boltz takes a batch of whole query blocks: pad atoms (masked, on no token)."""
    n = -(-x.num_atoms // W) * W
    pad = lambda v: np.concatenate([v, np.zeros((n - len(v),) + v.shape[1:], v.dtype)])
    mask = pad(np.ones(x.num_atoms, np.float32))
    one_hot = np.eye(x.num_tokens, dtype=np.float32)[x.atom_to_token]
    return pad, mask, one_hot


def _boltz2():
    from boltz.model.modules import encodersv2 as module

    feats_module = module.AtomEncoder(
        atom_s=C_ATOM,
        atom_z=C_PAIR,
        token_s=12,
        token_z=12,
        atoms_per_window_queries=W,
        atoms_per_window_keys=H,
        atom_feature_dim=388,
    )
    enc = module.AtomAttentionEncoder(
        atom_s=C_ATOM,
        token_s=12,
        atoms_per_window_queries=W,
        atoms_per_window_keys=H,
        atom_encoder_depth=DEPTH,
        atom_encoder_heads=HEADS,
    )
    dec = module.AtomAttentionDecoder(
        atom_s=C_ATOM,
        token_s=12,
        attn_window_queries=W,
        attn_window_keys=H,
        atom_decoder_depth=DEPTH,
        atom_decoder_heads=HEADS,
    )
    proj = lambda: torch.nn.ModuleList(
        torch.nn.Sequential(torch.nn.LayerNorm(C_PAIR), torch.nn.Linear(C_PAIR, HEADS, bias=False))
        for _ in range(DEPTH)
    )
    enc_proj, dec_proj = proj(), proj()

    def run(x: Inputs):
        pad, mask, one_hot = _boltz_padded_batch(x)
        feats = {
            k: t(v)[None]
            for k, v in {
                "ref_pos": pad(x.ref_pos),
                "atom_pad_mask": mask,
                "ref_space_uid": pad(x.ref_space_uid),
                "ref_charge": pad(x.ref_charge),
                "ref_element": pad(x.element_onehot()),
                "ref_atom_name_chars": pad(x.name_chars_onehot()),
                "atom_to_token": pad(one_hot),
            }.items()
        }
        q, c, p, to_keys = feats_module(feats, s_trunk=t(x.s_trunk)[None], z=t(x.z_trunk)[None])
        bias = lambda projections: torch.cat([f(p) for f in projections], dim=-1)
        a, q, c, to_keys = enc(feats, q, c, bias(enc_proj), to_keys, r=t(pad(x.r))[None])
        r_update = dec(t(x.a)[None], q, c, bias(dec_proj), feats, to_keys)
        return a[0], r_update[0, : x.num_atoms]

    def features(x: Inputs):
        return boltz2_features.atom_features(
            x.ref_pos, x.ref_charge, x.element_onehot(), x.name_chars_onehot()
        )

    def convert(m):
        feats_module, enc, dec, enc_proj, dec_proj = m
        return boltz.atom_encoder(feats_module, enc, enc_proj), boltz.atom_decoder(dec, dec_proj)

    modules = torch.nn.ModuleList([feats_module, enc, dec, enc_proj, dec_proj])
    return modules, run, convert, features, (12, 12)


def _promera():
    from promera.model import encoders as module

    enc = module.AtomAttentionEncoder(
        atom_s=C_ATOM,
        atom_z=C_PAIR,
        token_s=12,
        token_z=12,
        atoms_per_window_queries=W,
        atoms_per_window_keys=H,
        atom_feature_dim=389,
        atom_encoder_depth=DEPTH,
        atom_encoder_heads=HEADS,
    )
    dec = module.AtomAttentionDecoder(
        atom_s=C_ATOM,
        atom_z=C_PAIR,
        token_s=12,
        attn_window_queries=W,
        attn_window_keys=H,
        atom_decoder_depth=DEPTH,
        atom_decoder_heads=HEADS,
    )

    def run(x: Inputs):
        pad, mask, _ = _boltz_padded_batch(x)
        feats = {
            k: t(v)[None]
            for k, v in {
                "ref_pos": pad(x.ref_pos),
                "atom_pad_mask": mask,
                "ref_space_uid": pad(x.ref_space_uid),
                "ref_charge": pad(x.ref_charge),
                "ref_element": pad(x.element),
                "ref_atom_name_chars": pad(x.name_chars),
                "atom_is_std": pad(x.is_std.astype(np.int64)),
                "ref_hydrogens": pad(x.hydrogens),
                "atom_to_token": pad(x.atom_to_token),
                "restype": np.zeros(x.num_tokens, np.int64),
            }.items()
        }
        a, q, c, p, to_keys = enc(
            feats, s_trunk=t(x.s_trunk)[None], z=t(x.z_trunk)[None], r=t(pad(x.r))[None]
        )
        r_update = dec(t(x.a)[None], q, c, p, feats, to_keys)
        return a[0], r_update[0, : x.num_atoms]

    def features(x: Inputs):
        return promera_features.atom_features(
            x.ref_pos,
            x.ref_charge,
            np.ones(x.num_atoms, bool),
            x.element,
            x.name_chars,
            x.is_std,
            x.hydrogens,
        )

    convert = lambda m: (boltz.atom_encoder(m[0]), boltz.atom_decoder(m[1]))
    return torch.nn.ModuleList([enc, dec]), run, convert, features, (12, 12)


def _openfold3():
    import ml_collections
    from jopenfold3._vendor.openfold3.core.model.layers import (
        sequence_local_atom_attention as module,
    )

    widths = {
        "c_hidden": C_ATOM // HEADS,
        "no_heads": HEADS,
        "no_blocks": DEPTH,
        "n_transition": 2,
        "n_query": W,
        "n_key": H,
        "use_ada_layer_norm": True,
    }
    enc = module.AtomAttentionEncoder(
        c_atom_ref=ml_collections.ConfigDict({"element": 128, "name_chars": 256}),
        c_atom=C_ATOM,
        c_atom_pair=C_PAIR,
        c_token=C_TOKEN,
        add_noisy_pos=True,
        c_s=20,
        c_z=12,
        **widths,
    )
    dec = module.AtomAttentionDecoder(c_atom=C_ATOM, c_atom_pair=C_PAIR, c_token=C_TOKEN, **widths)

    def run(x: Inputs):
        batch = {
            "ref_pos": t(x.ref_pos),
            "ref_charge": t(x.ref_charge),
            "ref_mask": torch.ones(x.num_atoms),
            "ref_element": t(x.element_onehot()),
            "ref_atom_name_chars": t(x.name_chars_onehot()),
            "ref_space_uid": t(x.ref_space_uid),
            "atom_mask": torch.ones(x.num_atoms),
            "token_mask": torch.ones(x.num_tokens),
            "num_atoms_per_token": t(np.bincount(x.atom_to_token, minlength=x.num_tokens)),
            "atom_to_token_index": t(x.atom_to_token),
        }
        batch = {k: v[None] for k, v in batch.items()}
        a, q, c, p = enc(
            batch, rl=t(x.r)[None], si_trunk=t(x.s_trunk)[None], zij_trunk=t(x.z_trunk)[None]
        )
        r_update = dec(batch, ai=t(x.a)[None], ql=q, cl=c, plm=p)
        return a[0], r_update[0]

    def features(x: Inputs):
        return openfold3_features.atom_features(
            x.ref_pos,
            x.ref_charge,
            np.ones(x.num_atoms, np.float32),
            x.element_onehot(),
            x.name_chars_onehot(),
        )

    convert = lambda m: (openfold3.atom_encoder(m[0]), openfold3.atom_decoder(m[1]))
    return torch.nn.ModuleList([enc, dec]), run, convert, features, (20, 12)


CASES = {
    "boltz2": _boltz2,
    "promera": _promera,
    "protenix": _protenix,
    "opendde": _opendde,
    "mol1": _mol1,
    "openfold3": _openfold3,
}


@eqx.filter_jit
def _forward(encoder, decoder, features, x: dict, mask, num_tokens):
    atoms = Atoms(features, x["ref_pos"], x["ref_space_uid"], x["atom_to_token"], mask)
    q, c, p = encoder.condition(atoms, x["s_trunk"], x["z_trunk"])
    a, q = encoder(
        q,
        c,
        encoder.transformer.pair_biases(p),
        x["atom_to_token"],
        mask,
        num_tokens,
        x["r"],
    )
    biases = decoder.transformer.pair_biases(p)
    r_update = decoder(x["a"], q, c, biases, x["atom_to_token"], mask)
    return a, r_update


def _ours(encoder, decoder, features, x: Inputs, mask):
    arrays = {f.name: jnp.array(getattr(x, f.name)) for f in dataclasses.fields(x)}
    return _forward(encoder, decoder, jnp.array(features), arrays, jnp.array(mask), x.num_tokens)


@pytest.mark.parametrize("padding", [0, 40], ids=["unpadded", "padded"])
@pytest.mark.parametrize("case", CASES)
def test_atom_encoder_decoder_match_reference(case, padding):
    """The encoder's token aggregates and the decoder's position updates match the
    upstream on the real atoms, with `padding` masked garbage atoms appended (on a garbage
    token); outputs and gradients stay finite."""
    module, run_reference, convert, features, (c_s, c_z) = CASES[case]()
    randomize(module, seed=0)
    encoder, decoder = convert(module)
    x = make_inputs(c_s, c_z, NUM_ATOMS, NUM_TOKENS, np.arange(NUM_ATOMS) // ATOMS_PER_TOKEN, 1)
    with torch.no_grad():
        a_ref, r_ref = run_reference(x)

    x_in = padded(x, padding, seed=2) if padding else x
    mask = np.arange(x_in.num_atoms) < NUM_ATOMS
    a, r_update = _ours(encoder, decoder, features(x_in), x_in, mask)

    assert rel_err(a[:NUM_TOKENS], a_ref) < 1e-5
    assert rel_err(r_update[:NUM_ATOMS], r_ref) < 1e-5
    grads = jax.grad(
        lambda f: sum(o[:NUM_TOKENS].sum() for o in _ours(encoder, decoder, f, x_in, mask))
    )(jnp.array(features(x_in)))
    assert_finite((a, r_update, grads))
