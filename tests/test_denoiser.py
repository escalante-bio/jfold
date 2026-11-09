"""Full diffusion module (one denoising step) parity against each upstream."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from reference import assert_finite, randomize, rel_err
from test_atoms import ATOMS_PER_TOKEN, NUM_ATOMS, NUM_TOKENS, Inputs, W, make_inputs, padded

from jfold.blocks.atoms import Atoms
from jfold.convert import boltz, openfold3, protenix
from jfold.models import boltz2 as boltz2_features
from jfold.models import openfold3 as openfold3_features
from jfold.models import promera as promera_features
from jfold.models import protenix as protenix_features

C_ATOM, C_PAIR, C_TOKEN, HEADS = 16, 8, 24, 2
NOISE_LEVEL = 7.0
t = torch.from_numpy


@dataclasses.dataclass
class Tokens:
    """Token-level inputs: features for relative positions, and `s_inputs`."""

    asym_id: np.ndarray
    residue_index: np.ndarray
    entity_id: np.ndarray
    token_index: np.ndarray
    sym_id: np.ndarray
    s_inputs: np.ndarray


def make_tokens(num_tokens, c_inputs, seed) -> Tokens:
    rng = np.random.default_rng(seed)
    chain = (np.arange(num_tokens) >= num_tokens // 2).astype(np.int64)
    return Tokens(
        asym_id=chain,
        residue_index=np.arange(num_tokens) - chain * (num_tokens // 2),
        entity_id=chain,
        token_index=np.arange(num_tokens),
        sym_id=np.zeros(num_tokens, np.int64),
        s_inputs=rng.standard_normal((num_tokens, c_inputs), dtype=np.float32),
    )


def _protenix_lineage(DiffusionModule, precomputed_features):
    """Mol-1 and OpenDDE take the windowed atom offsets and the relative-position features
    (`relp`) precomputed."""
    blocks = {"n_blocks": 1, "n_heads": HEADS}
    m = DiffusionModule(
        sigma_data=16.0,
        c_atom=C_ATOM,
        c_atompair=C_PAIR,
        c_token=C_TOKEN,
        c_s=20,
        c_z=12,
        c_s_inputs=10,
        atom_encoder=blocks,
        transformer=blocks,
        atom_decoder=blocks,
    )

    def feats(x: Inputs, tok: Tokens):
        feats = dict(
            ref_pos=t(x.ref_pos),
            ref_charge=t(x.ref_charge),
            ref_mask=torch.ones(x.num_atoms),
            ref_element=t(x.element_onehot()),
            ref_atom_name_chars=t(x.name_chars_onehot()),
            ref_space_uid=t(x.ref_space_uid),
            atom_to_token_idx=t(x.atom_to_token),
            **{
                k: t(getattr(tok, k))
                for k in ("asym_id", "residue_index", "entity_id", "token_index", "sym_id")
            },
        )
        if precomputed_features:
            from mol.model.mol import update_input_feature_dict

            feats = m.diffusion_conditioning.relpe.generate_relp(update_input_feature_dict(feats))
        return feats

    def run(x: Inputs, tok: Tokens, x_noisy):
        out = m(
            t(x_noisy)[None],
            torch.tensor([NOISE_LEVEL]),
            feats(x, tok),
            t(tok.s_inputs),
            t(x.s_trunk),
            t(x.z_trunk),
            **({"pair_z": None, "p_lm": None, "c_l": None} if precomputed_features else {}),
        )
        return out[0]

    def relative_positions(x: Inputs, tok: Tokens):
        with torch.no_grad():
            f = feats(x, tok)
            return m.diffusion_conditioning.relpe(f["relp"] if precomputed_features else f).numpy()

    def features(x: Inputs):
        return protenix_features.atom_features(
            x.ref_pos,
            x.ref_charge,
            np.ones(x.num_atoms, np.float32),
            x.element_onehot(),
            x.name_chars_onehot(),
        )

    return m, run, protenix.diffusion_module, features, relative_positions, (20, 12, 10)


def _protenix():
    from protenix.model.modules.diffusion import DiffusionModule

    return _protenix_lineage(DiffusionModule, precomputed_features=False)


def _opendde():
    from opendde.model.modules.diffusion import DiffusionModule

    return _protenix_lineage(DiffusionModule, precomputed_features=True)


def _mol1():
    from mol.model.modules.diffusion import DiffusionModule

    return _protenix_lineage(DiffusionModule, precomputed_features=True)


def _boltz2():
    from boltz.model.modules.diffusion_conditioning import DiffusionConditioning
    from boltz.model.modules.diffusionv2 import DiffusionModule

    depths = {
        "atom_encoder_depth": 1,
        "atom_encoder_heads": HEADS,
        "token_transformer_depth": 1,
        "token_transformer_heads": HEADS,
        "atom_decoder_depth": 1,
        "atom_decoder_heads": HEADS,
    }
    conditioning = DiffusionConditioning(
        token_s=12, token_z=12, atom_s=C_ATOM, atom_z=C_PAIR, atom_feature_dim=388, **depths
    )
    module = DiffusionModule(token_s=12, atom_s=C_ATOM, sigma_data=16, **depths)
    rng = np.random.default_rng(3)
    relpe = {}

    def relative_positions(x: Inputs, tok: Tokens):
        # Boltz-2's relative positions come from its trunk; any pair tensor will do here.
        key = x.num_tokens
        if key not in relpe:
            relpe[key] = rng.standard_normal((key, key, 12), dtype=np.float32)
        return relpe[key][: x.num_tokens, : x.num_tokens]

    def run(x: Inputs, tok: Tokens, x_noisy):
        n = -(-x.num_atoms // W) * W
        pad = lambda v: np.concatenate([v, np.zeros((n - len(v),) + v.shape[1:], v.dtype)])
        feats = {
            k: t(v)[None]
            for k, v in {
                "ref_pos": pad(x.ref_pos),
                "atom_pad_mask": pad(np.ones(x.num_atoms, np.float32)),
                "ref_space_uid": pad(x.ref_space_uid),
                "ref_charge": pad(x.ref_charge),
                "ref_element": pad(x.element_onehot()),
                "ref_atom_name_chars": pad(x.name_chars_onehot()),
                "atom_to_token": pad(np.eye(x.num_tokens, dtype=np.float32)[x.atom_to_token]),
                "token_pad_mask": np.ones(x.num_tokens, np.float32),
            }.items()
        }
        s_trunk, s_inputs = t(x.s_trunk)[None], t(tok.s_inputs)[None]
        q, c, to_keys, enc_bias, dec_bias, token_bias = conditioning(
            s_trunk, t(x.z_trunk)[None], t(relative_positions(x, tok))[None], feats
        )
        sigma, sigma_data = torch.tensor([NOISE_LEVEL]), 16.0
        x_in = t(pad(x_noisy))[None]
        r_update = module(
            s_inputs,
            s_trunk,
            x_in / torch.sqrt(sigma**2 + sigma_data**2),
            torch.log(sigma / sigma_data) / 4,
            feats,
            {
                "q": q,
                "c": c,
                "to_keys": to_keys,
                "atom_enc_bias": enc_bias,
                "atom_dec_bias": dec_bias,
                "token_trans_bias": token_bias,
            },
        )
        scale = sigma**2 + sigma_data**2
        out = sigma_data**2 / scale * x_in + sigma * sigma_data / torch.sqrt(scale) * r_update
        return out[0, : x.num_atoms]

    def features(x: Inputs):
        return boltz2_features.atom_features(
            x.ref_pos, x.ref_charge, x.element_onehot(), x.name_chars_onehot()
        )

    modules = torch.nn.ModuleList([conditioning, module])
    convert = lambda m: boltz.diffusion_module(m[1], conditioning=m[0])
    return modules, run, convert, features, relative_positions, (12, 12, 12)


def _promera():
    from promera.model.diffusion import DiffusionModule

    module = DiffusionModule(
        token_s=12,
        token_z=12,
        atom_s=C_ATOM,
        atom_z=C_PAIR,
        atom_feature_dim=389,
        atom_encoder_depth=1,
        atom_encoder_heads=HEADS,
        token_transformer_depth=1,
        token_transformer_heads=HEADS,
        atom_decoder_depth=1,
        atom_decoder_heads=HEADS,
    )
    rng = np.random.default_rng(3)
    relpe = rng.standard_normal((NUM_TOKENS, NUM_TOKENS, 12), dtype=np.float32)

    def relative_positions(x: Inputs, tok: Tokens):
        return relpe  # ProMERA's relative positions come from its trunk

    def run(x: Inputs, tok: Tokens, x_noisy):
        n = -(-x.num_atoms // W) * W
        pad = lambda v: np.concatenate([v, np.zeros((n - len(v),) + v.shape[1:], v.dtype)])
        feats = {
            k: t(v)[None]
            for k, v in {
                "ref_pos": pad(x.ref_pos),
                "atom_pad_mask": pad(np.ones(x.num_atoms, np.float32)),
                "ref_space_uid": pad(x.ref_space_uid),
                "ref_charge": pad(x.ref_charge),
                "ref_element": pad(x.element),
                "ref_atom_name_chars": pad(x.name_chars),
                "atom_is_std": pad(x.is_std.astype(np.int64)),
                "ref_hydrogens": pad(x.hydrogens),
                "atom_to_token": pad(x.atom_to_token),
                "restype": np.zeros(x.num_tokens, np.int64),
                "token_pad_mask": np.ones(x.num_tokens, np.float32),
            }.items()
        }
        sigma, sigma_data = torch.tensor([NOISE_LEVEL]), 16.0
        x_in = t(pad(x_noisy))[None]
        r_update = module(
            t(tok.s_inputs)[None],
            t(x.s_trunk)[None],
            t(x.z_trunk)[None],
            x_in / torch.sqrt(sigma**2 + sigma_data**2),
            torch.log(sigma / sigma_data) / 4,
            t(relpe)[None],
            feats,
        )["r_update"]
        scale = sigma**2 + sigma_data**2
        out = sigma_data**2 / scale * x_in + sigma * sigma_data / torch.sqrt(scale) * r_update
        return out[0, : x.num_atoms]

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

    # ProMERA's s_inputs has a fixed width of 82.
    return module, run, boltz.diffusion_module, features, relative_positions, (12, 12, 82)


def _openfold3():
    import ml_collections
    from jopenfold3._vendor.openfold3.core.model.structure.diffusion_module import DiffusionModule
    from jopenfold3._vendor.openfold3.core.utils.relpos import relpos_complex

    atoms = {
        "c_atom": C_ATOM,
        "c_atom_pair": C_PAIR,
        "c_token": C_TOKEN,
        "c_hidden": C_ATOM // HEADS,
        "no_heads": HEADS,
        "no_blocks": 1,
        "n_transition": 2,
        "n_query": W,
        "n_key": 128,
        "use_ada_layer_norm": True,
    }
    config = ml_collections.ConfigDict(
        {
            "diffusion_module": {"c_s": 20, "c_token": C_TOKEN, "sigma_data": 16.0},
            "diffusion_conditioning": {
                "c_s_input": 10,
                "c_s": 20,
                "c_z": 12,
                "sigma_data": 16.0,
                "c_fourier_emb": 256,
                "max_relative_idx": 32,
                "max_relative_chain": 2,
            },
            "atom_attn_enc": {
                **atoms,
                "c_s": 20,
                "c_z": 12,
                "c_atom_ref": {"element": 128, "name_chars": 256},
            },
            "diffusion_transformer": {
                "c_a": C_TOKEN,
                "c_s": 20,
                "c_z": 12,
                "c_hidden": C_TOKEN // HEADS,
                "no_heads": HEADS,
                "no_blocks": 1,
                "n_transition": 2,
                "use_ada_layer_norm": True,
                "n_query": None,
                "n_key": None,
                "inf": 1e9,
            },
            "atom_attn_dec": atoms,
        }
    )
    module = DiffusionModule(config)

    def batch(x: Inputs, tok: Tokens):
        b = {
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
            **{
                k: t(getattr(tok, k))
                for k in ("asym_id", "residue_index", "entity_id", "token_index", "sym_id")
            },
        }
        return {k: v[None] for k, v in b.items()}

    def relative_positions(x: Inputs, tok: Tokens):
        return (
            relpos_complex(batch(x, tok), max_relative_idx=32, max_relative_chain=2)[0]
            .float()
            .numpy()
        )

    def run(x: Inputs, tok: Tokens, x_noisy):
        out = module(
            batch(x, tok),
            t(x_noisy)[None],
            torch.ones(1, x.num_tokens),
            torch.ones(1, x.num_atoms),
            torch.tensor([NOISE_LEVEL]),
            t(tok.s_inputs)[None],
            t(x.s_trunk)[None],
            t(x.z_trunk)[None],
            use_conditioning=True,
        )
        return out[0]

    def features(x: Inputs):
        return openfold3_features.atom_features(
            x.ref_pos,
            x.ref_charge,
            np.ones(x.num_atoms, np.float32),
            x.element_onehot(),
            x.name_chars_onehot(),
        )

    return module, run, openfold3.diffusion_module, features, relative_positions, (20, 12, 10)


CASES = {
    "protenix": _protenix,
    "opendde": _opendde,
    "mol1": _mol1,
    "boltz2": _boltz2,
    "promera": _promera,
    "openfold3": _openfold3,
}


@eqx.filter_jit
def _denoise(module, atoms, s_inputs, s_trunk, z_trunk, relative_positions, token_mask, x_noisy):
    context = module.context(atoms, s_inputs, s_trunk, z_trunk, relative_positions, token_mask)
    return module(context, x_noisy, jnp.array(NOISE_LEVEL), key=jax.random.key(0))


@pytest.mark.parametrize("padding", [0, 40], ids=["unpadded", "padded"])
@pytest.mark.parametrize("case", CASES)
def test_denoising_step_matches_reference(case, padding):
    """One denoising step matches the upstream on the real atoms, with `padding` masked
    garbage atoms (on a garbage token) appended; outputs and gradients stay finite."""
    module, run_reference, convert, features, relative_positions, (c_s, c_z, c_inputs) = CASES[
        case
    ]()
    randomize(module, seed=0)
    ours = convert(module)
    x = make_inputs(c_s, c_z, NUM_ATOMS, NUM_TOKENS, np.arange(NUM_ATOMS) // ATOMS_PER_TOKEN, 1)
    tok = make_tokens(NUM_TOKENS, c_inputs, 4)
    x_noisy = 10 * np.random.default_rng(5).standard_normal((NUM_ATOMS, 3), dtype=np.float32)
    with torch.no_grad():
        ref = run_reference(x, tok, x_noisy)

    x_in = padded(x, padding, seed=2) if padding else x
    num_tokens = x_in.num_tokens
    relpe = np.zeros((num_tokens, num_tokens, relative_positions(x, tok).shape[-1]), np.float32)
    relpe[:] = np.random.default_rng(6).standard_normal(relpe.shape)
    relpe[:NUM_TOKENS, :NUM_TOKENS] = relative_positions(x, tok)
    s_inputs = np.concatenate(
        [tok.s_inputs, np.ones((num_tokens - NUM_TOKENS, c_inputs), np.float32)]
    )
    atoms = Atoms(
        features=jnp.array(features(x_in)),
        ref_pos=jnp.array(x_in.ref_pos),
        ref_space_uid=jnp.array(x_in.ref_space_uid),
        atom_to_token=jnp.array(x_in.atom_to_token),
        mask=jnp.arange(x_in.num_atoms) < NUM_ATOMS,
    )
    token_mask = jnp.arange(num_tokens) < NUM_TOKENS
    noisy_in = np.concatenate([x_noisy, np.ones((x_in.num_atoms - NUM_ATOMS, 3), np.float32)])
    args = (atoms, s_inputs, x_in.s_trunk, x_in.z_trunk, relpe, token_mask)
    out = _denoise(ours, *args, noisy_in)

    assert rel_err(out[:NUM_ATOMS], ref) < 1e-5
    grads = jax.grad(lambda x: _denoise(ours, *args, x)[:NUM_ATOMS].sum())(jnp.array(noisy_in))
    assert_finite((out, grads))
