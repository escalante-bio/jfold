"""Confidence module and distogram head parity against each upstream."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from reference import assert_finite, randomize, rel_err

from jfold.convert import boltz, openfold3, protenix

NUM_TOKENS, ATOMS_PER_TOKEN = 9, 3
NUM_ATOMS = NUM_TOKENS * ATOMS_PER_TOKEN
C_S, C_Z, C_I = 32, 8, 10
t = torch.from_numpy


@dataclasses.dataclass
class Inputs:
    s_inputs: np.ndarray
    s_trunk: np.ndarray
    z_trunk: np.ndarray
    coords: np.ndarray  # [N, 3]
    atom_to_token: np.ndarray
    atom_position: np.ndarray  # index within the token
    rep_atom: np.ndarray  # [T] each token's representative atom
    asym_id: np.ndarray

    @property
    def num_tokens(self):
        return self.s_trunk.shape[0]

    @property
    def num_atoms(self):
        return self.coords.shape[0]

    def rep_mask(self):
        m = np.zeros(self.num_atoms, bool)
        m[self.rep_atom] = True
        return m


def make_inputs(num_tokens, seed, c_s=C_S, c_z=C_Z, c_i=C_I) -> Inputs:
    rng = np.random.default_rng(seed)
    f32 = lambda *shape: rng.standard_normal(shape, dtype=np.float32)
    n = num_tokens * ATOMS_PER_TOKEN
    return Inputs(
        s_inputs=f32(num_tokens, c_i),
        s_trunk=f32(num_tokens, c_s),
        z_trunk=f32(num_tokens, num_tokens, c_z),
        coords=8 * f32(n, 3),
        atom_to_token=np.arange(n) // ATOMS_PER_TOKEN,
        atom_position=np.arange(n) % ATOMS_PER_TOKEN,
        rep_atom=np.arange(num_tokens) * ATOMS_PER_TOKEN,
        asym_id=(np.arange(num_tokens) >= num_tokens // 2).astype(np.int64),
    )


def padded(x: Inputs, tokens: int) -> Inputs:
    """`x` with `tokens` garbage tokens (and their atoms) appended."""
    g = make_inputs(
        x.num_tokens + tokens,
        seed=11,
        c_s=x.s_trunk.shape[1],
        c_z=x.z_trunk.shape[2],
        c_i=x.s_inputs.shape[1],
    )
    g.s_inputs[: x.num_tokens] = x.s_inputs
    g.s_trunk[: x.num_tokens] = x.s_trunk
    g.z_trunk[: x.num_tokens, : x.num_tokens] = x.z_trunk
    g.coords[: x.num_atoms] = x.coords
    g.asym_id[: x.num_tokens] = x.asym_id
    return g


def _protenix_lineage(module):
    m = module.ConfidenceHead(n_blocks=1, c_s=C_S, c_z=C_Z, c_s_inputs=C_I)

    def run(x: Inputs):
        feats = {
            "distogram_rep_atom_mask": t(x.rep_mask()),
            "atom_to_token_idx": t(x.atom_to_token),
            "atom_to_tokatom_idx": t(x.atom_position),
        }
        n = x.num_tokens
        plddt, pae, pde, resolved = m(
            feats, t(x.s_inputs), t(x.s_trunk), t(x.z_trunk), torch.ones(n, n), t(x.coords)[None]
        )
        return {"pae": pae[0], "pde": pde[0], "plddt": plddt[0], "resolved": resolved[0]}

    return m, run, protenix.confidence_module


def _protenix():
    import protenix.model.modules.confidence as module

    return _protenix_lineage(module)


def _opendde():
    import opendde.model.modules.confidence as module

    return _protenix_lineage(module)


def _mol1():
    import mol.model.modules.confidence as module

    return _protenix_lineage(module)


def _openfold3():
    from unittest import mock

    import ml_collections
    from jopenfold3._vendor.openfold3.core.model.heads import head_modules

    pairformer = {
        "c_s": C_S,
        "c_z": C_Z,
        "c_hidden_pair_bias": 8,
        "no_heads_pair_bias": 4,
        "c_hidden_mul": 8,
        "c_hidden_pair_att": 4,
        "no_heads_pair": 2,
        "no_blocks": 1,
        "transition_type": "swiglu",
        "transition_n": 4,
        "pair_dropout": 0.25,
        "fuse_projection_weights": False,
        "blocks_per_ckpt": None,
        "inf": 1e9,
    }
    config = ml_collections.ConfigDict(
        {
            "max_atoms_per_token": 23,
            "per_sample_token_cutoff": None,
            "pairformer_embedding": {
                "pairformer": pairformer,
                "c_s_input": C_I,
                "c_z": C_Z,
                "min_bin": 3.25,
                "max_bin": 50.75,
                "no_bin": 39,
                "inf": 1e9,
            },
            "pae": {"c_z": C_Z, "c_out": 64, "enabled": True},
            "pde": {"c_z": C_Z, "c_out": 64},
            "lddt": {"c_s": C_S, "c_out": 50, "max_atoms_per_token": 23},
            "distogram": {"c_z": C_Z, "c_out": 64, "enabled": True},
            "experimentally_resolved": {"c_s": C_S, "c_out": 2, "max_atoms_per_token": 23},
        }
    )
    m = head_modules.AuxiliaryHeadsAllAtom(config)

    def run(x: Inputs):
        rep = t(x.coords[x.rep_atom])[None, None]
        batch = {
            "token_mask": torch.ones(1, x.num_tokens),
            "atom_mask": torch.ones(1, x.num_atoms),
            "num_atoms_per_token": torch.full((1, x.num_tokens), ATOMS_PER_TOKEN),
        }
        output = {
            "si_trunk": t(x.s_trunk)[None],
            "zij_trunk": t(x.z_trunk)[None],
            "atom_positions_predicted": t(x.coords)[None, None],
        }
        # Representative-atom selection is featurization: give it the test's atoms.
        with mock.patch.object(
            head_modules,
            "get_token_representative_atoms",
            return_value=(rep, torch.ones(1, 1, x.num_tokens)),
        ):
            out = m(batch, t(x.s_inputs)[None], output, use_zij_trunk_embedding=True)
        return {
            "pae": out["pae_logits"][0, 0],
            "pde": out["pde_logits"][0, 0],
            "plddt": out["plddt_logits"][0, 0],
            "resolved": out["experimentally_resolved_logits"][0, 0],
        }

    return m, run, openfold3.confidence_module


def _promera():
    import types

    from promera.model.confidence import ConfidenceModule

    pairformer = types.SimpleNamespace(
        num_blocks=1,
        dims=types.SimpleNamespace(token_s=C_S, token_z=32),
        num_heads=4,
        dropout=0.25,
        pairwise_head_width=8,
        pairwise_num_heads=2,
        no_update_s=False,
        activation_checkpointing=False,
        offload_to_cpu=False,
    )
    cfg = types.SimpleNamespace(
        dims=types.SimpleNamespace(token_s=C_S, token_z=32),
        max_dist=22.0,
        num_dist_bins=64,
        pairformer=pairformer,
    )
    m = ConfidenceModule(cfg, distogram=True)

    def run(x: Inputs):
        # ProMERA's triangle updates exist only as cuequivariance CUDA kernels.
        m.cuda()
        feats = {
            "token_to_rep_atom": t(x.rep_atom)[None].cuda(),
            "token_pad_mask": torch.ones(1, x.num_tokens).cuda(),
        }
        out = {
            "s_inputs": t(x.s_inputs)[None].cuda(),
            "s": t(x.s_trunk)[None].cuda(),
            "z": t(x.z_trunk)[None].cuda(),
            "sample_atom_coords": t(x.coords)[None].cuda(),
        }
        out = m(feats, out, multiplicity=1)
        m.cpu()
        return {
            "pae": out["pae_logits"][0].cpu(),
            "pde": out["pde_logits"][0].cpu(),
            "plddt": out["plddt_logits"][0].cpu(),
        }

    # ProMERA: pair width 32 (its triangle kernel), s_inputs 70 wider than the single.
    return m, run, boltz.confidence_module, {"c_z": 32, "c_i": C_S + 70}


def _boltz2():
    from boltz.model.modules.confidencev2 import ConfidenceModule

    m = ConfidenceModule(
        token_s=C_S,
        token_z=C_Z,
        pairformer_args={"num_blocks": 1, "num_heads": 4, "dropout": 0.25},
        num_dist_bins=64,
        max_dist=22,
        add_s_to_z_prod=True,
        add_s_input_to_s=True,
        add_z_input_to_z=True,
        confidence_args={"use_separate_heads": True},
        conditioning_cutoff_min=4.0,
        conditioning_cutoff_max=20.0,
    )

    def feats(x: Inputs):
        n = x.num_tokens
        rep = np.zeros((n, x.num_atoms), np.float32)
        rep[np.arange(n), x.rep_atom] = 1
        f = {
            "token_to_rep_atom": t(rep),
            "token_pad_mask": torch.ones(n),
            "asym_id": t(x.asym_id),
            "mol_type": torch.zeros(n, dtype=torch.long),
            "residue_index": torch.arange(n),
            "entity_id": t(x.asym_id),
            "token_index": torch.arange(n),
            "sym_id": torch.zeros(n, dtype=torch.long),
            "cyclic_period": torch.zeros(n),
            "token_bonds": torch.zeros(n, n, 1),
            "type_bonds": torch.zeros(n, n, dtype=torch.long),
            "contact_conditioning": torch.nn.functional.one_hot(
                torch.zeros(n, n, dtype=torch.long), 5
            ).float(),
            "contact_threshold": torch.zeros(n, n),
        }
        return {k: v[None] for k, v in f.items()}

    def run(x: Inputs):
        n = x.num_tokens
        out = m(
            t(x.s_inputs)[None],
            t(x.s_trunk)[None],
            t(x.z_trunk)[None],
            t(x.coords)[None],
            feats(x),
            torch.zeros(1, n, n, 1, 64),
        )
        return {
            "pae": out["pae_logits"][0],
            "pde": out["pde_logits"][0],
            "plddt": out["plddt_logits"][0],
            "resolved": out["resolved_logits"][0],
        }

    def pair_features(x: Inputs):
        """Boltz-2's relative positions, bonds and contacts, from its own modules."""
        f = feats(x)
        with torch.no_grad():
            z = m.rel_pos(f) + m.token_bonds(f["token_bonds"]) + m.contact_conditioning(f)
        return z[0].numpy()

    return m, run, boltz.confidence_module, {"c_i": C_S}, pair_features


CASES = {
    "boltz2": _boltz2,
    "protenix": _protenix,
    "opendde": _opendde,
    "mol1": _mol1,
    "openfold3": _openfold3,
    "promera": _promera,
}


@eqx.filter_jit
def _forward(module, x: dict, mask, pair_features):
    return module.logits(
        x["s_inputs"],
        x["s_trunk"],
        x["z_trunk"],
        x["coords"][x["rep_atom"]],
        x["asym_id"],
        x["atom_to_token"],
        x["atom_position"],
        mask,
        pair_features,
        key=jax.random.key(0),
    )


def _arrays(x: Inputs) -> dict:
    return {f.name: jnp.array(getattr(x, f.name)) for f in dataclasses.fields(x)}


@pytest.mark.parametrize("padding", [0, 2], ids=["unpadded", "padded"])
@pytest.mark.parametrize("case", CASES)
def test_confidence_matches_reference(case, padding):
    """Confidence logits match the upstream on the real tokens and atoms, with garbage
    tokens appended; outputs and gradients stay finite."""
    module, run_reference, convert, *extra = CASES[case]()
    widths = extra[0] if extra else {}
    pair_features_fn = extra[1] if len(extra) > 1 else None
    randomize(module, seed=0)
    ours = convert(module)
    x = make_inputs(NUM_TOKENS, seed=1, **widths)
    with torch.no_grad():
        ref = run_reference(x)

    x_in = padded(x, padding) if padding else x
    mask = jnp.arange(x_in.num_tokens) < NUM_TOKENS
    pair_features = None
    if pair_features_fn is not None:
        pair_features = np.zeros(x_in.z_trunk.shape, np.float32)
        pair_features[:NUM_TOKENS, :NUM_TOKENS] = pair_features_fn(x)
    out = _forward(ours, _arrays(x_in), mask, pair_features)

    n, a = NUM_TOKENS, NUM_ATOMS
    assert rel_err(out.pae[:n, :n], ref["pae"]) < 1e-5
    assert rel_err(out.pde[:n, :n], ref["pde"]) < 1e-5
    per_atom = out.plddt.shape[0] == x_in.num_atoms
    assert rel_err(out.plddt[: a if per_atom else n], ref["plddt"]) < 1e-5
    if out.resolved is not None:
        assert rel_err(out.resolved[: a if per_atom else n], ref["resolved"]) < 1e-5

    def loss(z):
        o = _forward(ours, {**_arrays(x_in), "z_trunk": z}, mask, pair_features)
        return o.pae[:n, :n].sum() + o.plddt.sum()

    assert_finite((out, jax.grad(loss)(jnp.array(x_in.z_trunk))))


def _distogram_cases():
    import boltz.model.modules.trunkv2 as boltz_trunk
    import mol.model.modules.head as mol_head
    import opendde.model.modules.head as opendde_head
    import promera.model.trunk as promera_trunk
    import protenix.model.modules.head as protenix_head
    from jopenfold3._vendor.openfold3.core.model.heads import prediction_heads as of3_heads

    unbatch = lambda m, z: m(z[None])[0]
    return {
        "protenix": (
            protenix_head.DistogramHead(c_z=C_Z, no_bins=16),
            lambda m, z: m(z),
            protenix.distogram_head,
        ),
        "opendde": (
            opendde_head.DistogramHead(c_z=C_Z, no_bins=16),
            lambda m, z: m(z),
            protenix.distogram_head,
        ),
        "mol1": (
            mol_head.DistogramHead(c_z=C_Z, no_bins=16),
            lambda m, z: m(z),
            protenix.distogram_head,
        ),
        "boltz2": (
            boltz_trunk.DistogramModule(token_z=C_Z, num_bins=16),
            lambda m, z: m(z[None])[0, :, :, 0],
            boltz.distogram_head,
        ),
        "promera": (
            promera_trunk.DistogramModule(token_z=C_Z, num_bins=16),
            unbatch,
            boltz.distogram_head,
        ),
        "openfold3": (
            of3_heads.DistogramHead(c_z=C_Z, c_out=16),
            lambda m, z: m(z),
            openfold3.distogram_head,
        ),
    }


@pytest.mark.parametrize("case", ["protenix", "opendde", "mol1", "boltz2", "promera", "openfold3"])
def test_distogram_head_matches_reference(case):
    module, run, convert = _distogram_cases()[case]
    randomize(module, seed=0)
    z = np.random.default_rng(1).standard_normal((NUM_TOKENS, NUM_TOKENS, C_Z), dtype=np.float32)
    with torch.no_grad():
        ref = run(module, t(z))
    assert rel_err(convert(module)(jnp.array(z)), ref) < 1e-5
