"""Test the atom encoder without coordinates (the trunk's input embedder) against each upstream."""

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from reference import assert_finite, block_features, randomize, rel_err
from test_atoms import (
    ATOMS_PER_TOKEN,
    C_ATOM,
    C_PAIR,
    DEPTH,
    HEADS,
    NUM_ATOMS,
    NUM_TOKENS,
    H,
    Inputs,
    W,
    make_inputs,
    padded,
)

from jfold.blocks.atoms import Atoms
from jfold.convert import boltz, openfold3, protenix
from jfold.models import boltz2 as boltz2_features
from jfold.models import openfold3 as openfold3_features
from jfold.models import promera as promera_features
from jfold.models import protenix as protenix_features

C_TOKEN = 24
t = torch.from_numpy


def _protenix_lineage(module, explicit_args):
    m = module.AtomAttentionEncoder(
        has_coords=False,
        c_token=C_TOKEN,
        c_atom=C_ATOM,
        c_atompair=C_PAIR,
        n_blocks=DEPTH,
        n_heads=HEADS,
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
        if explicit_args:  # Mol-1 and OpenDDE take the windowed offsets precomputed
            from mol.model.mol import update_input_feature_dict

            feats = update_input_feature_dict(feats)
            keys = (
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
            return m(**{k: feats[k] for k in keys})[0]
        return m(feats)[0]

    def features(x: Inputs):
        return protenix_features.atom_features(
            x.ref_pos,
            x.ref_charge,
            np.ones(x.num_atoms, np.float32),
            x.element_onehot(),
            x.name_chars_onehot(),
        )

    return m, run, protenix.atom_encoder, features


def _protenix():
    import protenix.model.modules.transformer as module

    return _protenix_lineage(module, explicit_args=False)


def _opendde():
    import opendde.model.modules.transformer as module

    return _protenix_lineage(module, explicit_args=True)


def _mol1():
    import mol.model.modules.transformer as module

    return _protenix_lineage(module, explicit_args=True)


def _boltz_feats(x: Inputs, **extra):
    n = -(-x.num_atoms // W) * W
    pad = lambda v: np.concatenate([v, np.zeros((n - len(v),) + v.shape[1:], v.dtype)])
    base = {
        "ref_pos": pad(x.ref_pos),
        "atom_pad_mask": pad(np.ones(x.num_atoms, np.float32)),
        "ref_space_uid": pad(x.ref_space_uid),
        "ref_charge": pad(x.ref_charge),
    }
    return (
        n,
        pad,
        {k: t(v)[None] for k, v in {**base, **{k: pad(v) for k, v in extra.items()}}.items()},
    )


def _boltz2():
    from boltz.model.modules import encodersv2 as module

    feats_module = module.AtomEncoder(
        atom_s=C_ATOM,
        atom_z=C_PAIR,
        token_s=C_TOKEN,
        token_z=12,
        atoms_per_window_queries=W,
        atoms_per_window_keys=H,
        atom_feature_dim=388,
        structure_prediction=False,
    )
    enc = module.AtomAttentionEncoder(
        atom_s=C_ATOM,
        token_s=C_TOKEN,
        atoms_per_window_queries=W,
        atoms_per_window_keys=H,
        atom_encoder_depth=DEPTH,
        atom_encoder_heads=HEADS,
        structure_prediction=False,
    )
    proj = torch.nn.ModuleList(
        torch.nn.Sequential(torch.nn.LayerNorm(C_PAIR), torch.nn.Linear(C_PAIR, HEADS, bias=False))
        for _ in range(DEPTH)
    )

    def run(x: Inputs):
        _, _, feats = _boltz_feats(
            x,
            ref_element=x.element_onehot(),
            ref_atom_name_chars=x.name_chars_onehot(),
            atom_to_token=np.eye(x.num_tokens, dtype=np.float32)[x.atom_to_token],
        )
        q, c, p, to_keys = feats_module(feats)
        a, *_ = enc(feats, q, c, torch.cat([f(p) for f in proj], dim=-1), to_keys)
        return a[0]

    def features(x: Inputs):
        return boltz2_features.atom_features(
            x.ref_pos, x.ref_charge, x.element_onehot(), x.name_chars_onehot()
        )

    modules = torch.nn.ModuleList([feats_module, enc, proj])
    return modules, run, lambda m: boltz.atom_encoder(m[0], m[1], m[2]), features


def _promera():
    from promera.model import encoders as module

    enc = module.AtomAttentionEncoder(
        atom_s=C_ATOM,
        atom_z=C_PAIR,
        token_s=C_TOKEN,
        token_z=12,
        atoms_per_window_queries=W,
        atoms_per_window_keys=H,
        atom_feature_dim=389,
        atom_encoder_depth=DEPTH,
        atom_encoder_heads=HEADS,
        structure_prediction=False,
    )

    def run(x: Inputs):
        _, _, feats = _boltz_feats(
            x,
            ref_element=x.element,
            ref_atom_name_chars=x.name_chars,
            atom_is_std=x.is_std.astype(np.int64),
            ref_hydrogens=x.hydrogens,
            atom_to_token=x.atom_to_token,
        )
        feats["restype"] = torch.zeros(1, x.num_tokens, dtype=torch.long)
        return enc(feats)[0][0]

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

    return enc, run, boltz.atom_encoder, features


def _openfold3():
    import ml_collections
    from jopenfold3._vendor.openfold3.core.model.layers import (
        sequence_local_atom_attention as module,
    )

    m = module.AtomAttentionEncoder(
        c_atom_ref=ml_collections.ConfigDict({"element": 128, "name_chars": 256}),
        c_atom=C_ATOM,
        c_atom_pair=C_PAIR,
        c_token=C_TOKEN,
        add_noisy_pos=False,
        c_hidden=C_ATOM // HEADS,
        no_heads=HEADS,
        no_blocks=DEPTH,
        n_transition=2,
        n_query=W,
        n_key=H,
        use_ada_layer_norm=True,
    )

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
        return m({k: v[None] for k, v in batch.items()})[0][0]

    def features(x: Inputs):
        return openfold3_features.atom_features(
            x.ref_pos,
            x.ref_charge,
            np.ones(x.num_atoms, np.float32),
            x.element_onehot(),
            x.name_chars_onehot(),
        )

    return m, run, openfold3.atom_encoder, features


CASES = {
    "boltz2": _boltz2,
    "promera": _promera,
    "protenix": _protenix,
    "opendde": _opendde,
    "mol1": _mol1,
    "openfold3": _openfold3,
}


@functools.partial(eqx.filter_jit)
def _encode(encoder, features, ref_pos, ref_space_uid, atom_to_token, mask, num_tokens):
    q, c, p = encoder.condition(Atoms(features, ref_pos, ref_space_uid, atom_to_token, mask))
    biases = encoder.transformer.pair_biases(p)
    return encoder(q, c, biases, atom_to_token, mask, num_tokens)[0]


@pytest.mark.parametrize("padding", [0, 40], ids=["unpadded", "padded"])
@pytest.mark.parametrize("case", CASES)
def test_input_atom_encoder_matches_reference(case, padding):
    """Check that per-token aggregates from the reference conformers alone match the upstream."""
    module, run_reference, convert, features = CASES[case]()
    randomize(module, seed=0)
    encoder = convert(module)
    assert encoder.trunk is None and encoder.linear_r is None
    x = make_inputs(1, 1, NUM_ATOMS, NUM_TOKENS, np.arange(NUM_ATOMS) // ATOMS_PER_TOKEN, 1)
    with torch.no_grad():
        ref = run_reference(x)

    x_in = padded(x, padding, seed=2) if padding else x
    mask = jnp.arange(x_in.num_atoms) < NUM_ATOMS
    args = (
        jnp.array(x_in.ref_pos),
        jnp.array(x_in.ref_space_uid),
        jnp.array(x_in.atom_to_token),
        mask,
        x_in.num_tokens,
    )
    a = _encode(encoder, jnp.array(features(x_in)), *args)

    assert rel_err(a[:NUM_TOKENS], ref) < 1e-5
    grads = jax.grad(lambda f: _encode(encoder, f, *args)[:NUM_TOKENS].sum())(
        jnp.array(features(x_in))
    )
    assert_finite((a, grads))


def _tokens(num_tokens, seed):
    rng = np.random.default_rng(seed)
    profile = rng.random((num_tokens, 34), dtype=np.float32)
    return {
        "restype": rng.integers(0, 20, num_tokens),
        "profile": profile / profile.sum(-1, keepdims=True),
        "deletion_mean": rng.random(num_tokens, dtype=np.float32),
        "method": rng.integers(0, 12, num_tokens),
        "modified": rng.integers(0, 2, num_tokens),
        "cyclic_period": np.where(np.arange(num_tokens) < 4, 7, 0).astype(np.float32),
        "mol_type": rng.integers(0, 4, num_tokens),
        "is_epitope": (rng.random(num_tokens) < 0.3).astype(np.float32),
        "is_std": rng.random(num_tokens) < 0.5,
    }


def _embedder_protenix_lineage(module_path, explicit_args):
    import importlib

    m = importlib.import_module(module_path).InputFeatureEmbedder(
        c_atom=C_ATOM, c_atompair=C_PAIR, c_token=C_TOKEN
    )
    m.atom_attention_encoder = importlib.import_module(
        module_path.replace("embedders", "transformer")
    ).AtomAttentionEncoder(
        has_coords=False,
        c_token=C_TOKEN,
        c_atom=C_ATOM,
        c_atompair=C_PAIR,
        n_blocks=DEPTH,
        n_heads=HEADS,
    )

    def run(x: Inputs, tok):
        feats = {
            "ref_pos": t(x.ref_pos),
            "ref_charge": t(x.ref_charge),
            "ref_mask": torch.ones(x.num_atoms),
            "ref_element": t(x.element_onehot()),
            "ref_atom_name_chars": t(x.name_chars_onehot()),
            "ref_space_uid": t(x.ref_space_uid),
            "atom_to_token_idx": t(x.atom_to_token),
            "restype": t(np.eye(32, dtype=np.float32)[tok["restype"]]),
            "profile": t(tok["profile"][:, :32]),
            "deletion_mean": t(tok["deletion_mean"]),
        }
        if explicit_args:
            from mol.model.mol import update_input_feature_dict

            feats = update_input_feature_dict(feats)
        return m(feats)

    def features(x, tok):
        return protenix_features.token_features(
            jax.nn.one_hot(tok["restype"], 32), tok["profile"][:, :32], tok["deletion_mean"]
        )

    return m, run, protenix.input_embedder, features, None


def _embedder_boltz2():
    from boltz.model.modules.trunkv2 import InputEmbedder

    m = InputEmbedder(
        atom_s=C_ATOM,
        atom_z=C_PAIR,
        token_s=C_TOKEN,
        token_z=12,
        atoms_per_window_queries=W,
        atoms_per_window_keys=H,
        atom_feature_dim=388,
        atom_encoder_depth=DEPTH,
        atom_encoder_heads=HEADS,
        add_method_conditioning=True,
        add_modified_flag=True,
        add_cyclic_flag=True,
        add_mol_type_feat=True,
    )

    def run(x: Inputs, tok):
        _, _, feats = _boltz_feats(
            x,
            ref_element=x.element_onehot(),
            ref_atom_name_chars=x.name_chars_onehot(),
            atom_to_token=np.eye(x.num_tokens, dtype=np.float32)[x.atom_to_token],
        )
        token = {
            "res_type": np.eye(33, dtype=np.float32)[tok["restype"]],
            "profile": tok["profile"][:, :33],
            "deletion_mean": tok["deletion_mean"],
            "method_feature": tok["method"],
            "modified": tok["modified"],
            "cyclic_period": tok["cyclic_period"],
            "mol_type": tok["mol_type"],
        }
        return m({**feats, **{k: t(v)[None] for k, v in token.items()}})[0]

    def features(x, tok):
        return boltz2_features.token_features(
            jax.nn.one_hot(tok["restype"], 33),
            tok["profile"][:, :33],
            tok["deletion_mean"],
            tok["method"],
            tok["modified"],
            tok["cyclic_period"],
            tok["mol_type"],
        )

    return m, run, boltz.input_embedder, features, None


def _embedder_promera():
    import types

    from promera.model.trunk import InputEmbedder

    cfg = types.SimpleNamespace(
        dims={
            "atom_s": C_ATOM,
            "atom_z": C_PAIR,
            "token_s": C_TOKEN,
            "token_z": 12,
            "atoms_per_window_queries": W,
            "atoms_per_window_keys": H,
            "atom_feature_dim": 389,
        },
        atom_encoder_depth=DEPTH,
        atom_encoder_heads=HEADS,
        feature=types.SimpleNamespace(mask_std_feats=True),
    )
    m = InputEmbedder(cfg)

    def run(x: Inputs, tok):
        _, _, feats = _boltz_feats(
            x,
            ref_element=x.element,
            ref_atom_name_chars=x.name_chars,
            atom_is_std=x.is_std.astype(np.int64),
            ref_hydrogens=x.hydrogens,
            atom_to_token=x.atom_to_token,
        )
        token = {
            "restype": tok["restype"],
            "profile": tok["profile"],
            "deletion_mean": tok["deletion_mean"],
            "is_epitope": tok["is_epitope"],
            "is_std": tok["is_std"],
        }
        return m({**feats, **{k: t(v)[None] for k, v in token.items()}})[0]

    def features(x, tok):
        return promera_features.token_features(
            jax.nn.one_hot(tok["restype"], 34),
            tok["profile"],
            tok["deletion_mean"],
            tok["is_epitope"],
        )

    return m, run, boltz.input_embedder, features, lambda tok: ~tok["is_std"]


def _embedder_openfold3():
    import ml_collections
    from jopenfold3._vendor.openfold3.core.model.feature_embedders.input_embedders import (
        InputEmbedderAllAtom,
    )

    atom_attn_enc = {
        "c_atom_ref": {"element": 128, "name_chars": 256},
        "c_atom": C_ATOM,
        "c_atom_pair": C_PAIR,
        "c_token": C_TOKEN,
        "c_hidden": C_ATOM // HEADS,
        "no_heads": HEADS,
        "no_blocks": DEPTH,
        "n_transition": 2,
        "n_query": W,
        "n_key": H,
        "use_ada_layer_norm": True,
    }
    m = InputEmbedderAllAtom(
        c_s_input=C_TOKEN + 65,
        c_s=20,
        c_z=12,
        max_relative_idx=32,
        max_relative_chain=2,
        atom_attn_enc=ml_collections.ConfigDict(atom_attn_enc),
    )

    def run(x: Inputs, tok):
        n = x.num_tokens
        batch = {
            "ref_pos": t(x.ref_pos),
            "ref_charge": t(x.ref_charge),
            "ref_mask": torch.ones(x.num_atoms),
            "ref_element": t(x.element_onehot()),
            "ref_atom_name_chars": t(x.name_chars_onehot()),
            "ref_space_uid": t(x.ref_space_uid),
            "atom_mask": torch.ones(x.num_atoms),
            "token_mask": torch.ones(n),
            "num_atoms_per_token": t(np.bincount(x.atom_to_token, minlength=n)),
            "atom_to_token_index": t(x.atom_to_token),
            "restype": t(np.eye(32, dtype=np.float32)[tok["restype"]]),
            "profile": t(tok["profile"][:, :32]),
            "deletion_mean": t(tok["deletion_mean"]),
            "token_bonds": torch.zeros(n, n),
            "residue_index": torch.arange(n),
            "token_index": torch.arange(n),
            "asym_id": torch.zeros(n, dtype=torch.long),
            "entity_id": torch.zeros(n, dtype=torch.long),
            "sym_id": torch.zeros(n, dtype=torch.long),
        }
        return m({k: v[None] for k, v in batch.items()})[0][0]

    def features(x, tok):
        return openfold3_features.token_features(
            jax.nn.one_hot(tok["restype"], 32), tok["profile"][:, :32], tok["deletion_mean"]
        )

    return m, run, openfold3.input_embedder, features, None


EMBEDDERS = {
    "protenix": lambda: _embedder_protenix_lineage("protenix.model.modules.embedders", False),
    "opendde": lambda: _embedder_protenix_lineage("opendde.model.modules.embedders", True),
    "mol1": lambda: _embedder_protenix_lineage("mol.model.modules.embedders", True),
    "boltz2": _embedder_boltz2,
    "promera": _embedder_promera,
    "openfold3": _embedder_openfold3,
}


@eqx.filter_jit
def _embed(embedder, atoms, token_features, keep):
    return embedder(
        block_features(
            token_features.shape[0],
            atoms.mask.shape[0],
            atoms=atoms,
            token_features=token_features,
            keep_aggregate=keep,
        )
    )


@pytest.mark.parametrize("padding", [0, 40], ids=["unpadded", "padded"])
@pytest.mark.parametrize("case", EMBEDDERS)
def test_input_embedder_matches_reference(case, padding):
    """Check that `s_inputs` matches the upstream on the real tokens with garbage atoms appended."""
    from jfold.blocks.atoms import Atoms

    module, run_reference, convert, features, keep_fn = EMBEDDERS[case]()
    randomize(module, seed=0)
    embedder = convert(module)
    x = make_inputs(1, 1, NUM_ATOMS, NUM_TOKENS, np.arange(NUM_ATOMS) // ATOMS_PER_TOKEN, 1)
    tok = _tokens(NUM_TOKENS, seed=3)
    with torch.no_grad():
        ref = run_reference(x, tok)

    x_in = padded(x, padding, seed=2) if padding else x
    tok_in = {k: np.concatenate([v, v[:1]]) for k, v in tok.items()} if padding else tok
    atom_features = CASES[case]()[3](x_in)
    atoms = Atoms(
        features=jnp.array(atom_features),
        ref_pos=jnp.array(x_in.ref_pos),
        ref_space_uid=jnp.array(x_in.ref_space_uid),
        atom_to_token=jnp.array(x_in.atom_to_token),
        mask=jnp.arange(x_in.num_atoms) < NUM_ATOMS,
    )
    keep = None if keep_fn is None else jnp.array(keep_fn(tok_in))
    s_inputs = _embed(embedder, atoms, jnp.array(features(x_in, tok_in)), keep)
    assert rel_err(s_inputs[:NUM_TOKENS], ref) < 1e-5


def test_openfold3_trunk_init_matches_reference():
    """Check OpenFold3's initial single and pair from `s_inputs`, relative positions and bonds."""
    from jfold.blocks.inputs import relative_position_features

    module, run_reference, _, _, _ = _embedder_openfold3()
    randomize(module, seed=0)
    x = make_inputs(1, 1, NUM_ATOMS, NUM_TOKENS, np.arange(NUM_ATOMS) // ATOMS_PER_TOKEN, 1)
    tok = _tokens(NUM_TOKENS, seed=3)
    captured = {}
    module.register_forward_hook(
        lambda m, i, out: captured.update(s_inputs=out[0][0], s=out[1][0], z=out[2][0])
    )
    with torch.no_grad():
        run_reference(x, tok)
    n = NUM_TOKENS
    zeros = jnp.zeros(n, jnp.int32)
    relpos = relative_position_features(zeros, jnp.arange(n), zeros, jnp.arange(n), zeros)
    s, z = openfold3.trunk_init(module).initial(
        jnp.array(captured["s_inputs"].numpy()), relpos, jnp.zeros((n, n, 1))
    )
    assert rel_err(s, captured["s"]) < 1e-5
    assert rel_err(z, captured["z"]) < 1e-5
