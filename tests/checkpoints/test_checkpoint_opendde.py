"""End-to-end parity with the released OpenDDE checkpoint (`opendde_v1`): the residue trunk,
the structural-token branch, one denoising step, confidence and distogram on a small real
complex. Downloads weights; run with `pytest -m checkpoint`."""

import copy
import json
import os
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from reference import STEPS, assert_round_trip, assert_schedule, rel_err

pytestmark = pytest.mark.checkpoint

SEQUENCES = ("MKTAYIAKQRQISFVKSHFSRQ", "GSHMLEDPV")
NUM_RECYCLES = 2
NOISE_LEVEL = 10.0
KEY = jax.random.key(0)


def _biotite_compat():
    """OpenDDE's featurizer reaches into `BondList` internals that biotite 1.7 (a compiled
    extension) does not expose; restore the few it uses."""
    import biotite.structure as struc

    bonds = struc.BondList
    if hasattr(bonds, "_bonds"):
        return

    def set_bonds(self, value):
        self.remove_bonds(self.copy())
        for i, j, bond_type in value:
            self.add_bond(int(i), int(j), int(bond_type))

    def max_bonds_per_atom(self):
        return int(np.bincount(self.as_array()[:, :2].ravel(), minlength=1).max())

    bonds._bonds = property(lambda self: self.as_array(), set_bonds)
    bonds._remove_redundant_bonds = lambda self: None
    bonds._get_max_bonds_per_atom = max_bonds_per_atom
    bonds._max_bonds_per_atom = property(max_bonds_per_atom, lambda self, value: None)
    bonds._atom_count = property(lambda self: self.get_atom_count())


def _configs():
    from opendde.config.inference import build_inference_config

    configs = build_inference_config(model_name="opendde_v1", fill_required_with_null=True)
    configs.use_msa = configs.use_template = configs.use_rna_msa = False
    configs.triangle_multiplicative = configs.triangle_attention = "torch"
    configs.enable_diffusion_shared_vars_cache = False
    configs.enable_efficient_fusion = False
    return configs


def _reference_model(configs):
    """The released model as `jfold.pretrained` builds it (as OpenDDE's inference runner
    does), from that runner's checkpoint path. Downloads the checkpoint and the featurizer's
    CCD data if missing."""
    from opendde.utils.download import download_inference_cache, resolve_checkpoint_path

    from jfold import pretrained

    download_inference_cache(configs)
    return pretrained.upstream("opendde_v1", checkpoint_path=resolve_checkpoint_path(configs))


def _featurize(configs) -> dict:
    from opendde.data.inference.infer_dataloader import get_inference_dataloader

    _biotite_compat()
    spec = [
        {
            "name": "complex",
            "modelSeeds": [0],
            "sequences": [{"proteinChain": {"sequence": s, "count": 1}} for s in SEQUENCES],
        }
    ]
    configs = copy.deepcopy(configs)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "input.json")
        with open(path, "w") as f:
            json.dump(spec, f)
        configs.input_json_path, configs.seeds, configs.dump_dir = path, [0], tmp
        data, _, error = next(iter(get_inference_dataloader(configs=configs)))[0]
    assert not error, error
    return data["input_feature_dict"]


def _pad_structural(features, num: int, seed: int = 0):
    """`features` with `num` masked garbage structural tokens appended."""
    import equinox as eqx

    rng = np.random.default_rng(seed)
    tokens = features.extras
    m, t = tokens.parent.shape[0], features.asym_id.shape[0]
    rel = np.array(tokens.relative_positions)
    padded_rel = rng.standard_normal((m + num, m + num, rel.shape[-1])).astype(np.float32)
    padded_rel[:m, :m] = rel

    def append(x, garbage):
        return jnp.concatenate([x, jnp.array(garbage, x.dtype)])

    return eqx.tree_at(
        lambda f: (
            f.extras.parent,
            f.extras.role,
            f.extras.prev_parent,
            f.extras.next_parent,
            f.extras.mask,
            f.extras.relative_positions,
        ),
        features,
        (
            append(tokens.parent, rng.integers(0, t, num)),
            append(tokens.role, rng.integers(0, 7, num)),
            append(tokens.prev_parent, rng.integers(-1, t, num)),
            append(tokens.next_parent, rng.integers(-1, t, num)),
            append(tokens.mask, np.zeros(num, bool)),
            jnp.array(padded_rel),
        ),
    )


def _trunk(model, features):
    inputs = model.embed(features, key=KEY)
    state = model.trunk.init_state(inputs)
    for _ in range(NUM_RECYCLES):
        state = model.trunk(features, inputs, state, key=KEY)
    return inputs, state


def test_opendde():
    from opendde.model.opendde import update_input_feature_dict

    from jfold.convert import opendde as convert
    from jfold.models import opendde
    from jfold.slots import TrunkState

    configs = _configs()
    reference = _reference_model(configs)
    feats = _featurize(configs)
    t = update_input_feature_dict(
        reference.relative_position_encoding.generate_relp(copy.deepcopy(feats))
    )
    n_atoms = feats["ref_pos"].shape[0]
    x_noisy = 16 * np.random.default_rng(0).standard_normal((n_atoms, 3), dtype=np.float32)
    with torch.no_grad():
        s_inputs, s, z = reference.get_pairformer_output(t, N_cycle=NUM_RECYCLES)
        structural, s_inputs_st, s_st, z_st = reference.expand_to_structural_tokens(
            t, s_inputs, s, z
        )
        denoised = reference.diffusion_module(
            torch.from_numpy(x_noisy)[None],
            torch.tensor([NOISE_LEVEL]),
            structural,
            s_inputs_st,
            s_st,
            z_st,
            pair_z=None,
            p_lm=None,
            c_l=None,
        )[0]
        plddt, pae, pde, resolved = reference.confidence_head(
            input_feature_dict=t,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=denoised[None],
        )
        distogram = reference.distogram_head(z)

    model = convert.structure_model(reference)

    model = assert_round_trip(model)
    assert_schedule(model.sampling, reference.inference_noise_scheduler(N_step=STEPS, device="cpu"))
    features = opendde.features({k: v.numpy() for k, v in feats.items() if torch.is_tensor(v)})
    inputs, state = eqx.filter_jit(_trunk)(model, features)
    errors = {
        "s_inputs": rel_err(inputs.s_inputs, s_inputs),
        "s": rel_err(state.s, s),
        "z": rel_err(state.z, z),
    }

    # Later stages from the reference trunk, so errors do not compound.
    ref_state = TrunkState(s=jnp.array(s.numpy()), z=jnp.array(z.numpy()))
    ref_inputs = jax.tree.map(lambda x: x, inputs)
    ours_s_inputs, ours_structural, _ = eqx.filter_jit(type(model.diffusion).structural_trunk)(
        model.diffusion, features, ref_inputs, ref_state
    )
    errors |= {
        "structural s_inputs": rel_err(ours_s_inputs, s_inputs_st),
        "structural s": rel_err(ours_structural.s, s_st),
        "structural z": rel_err(ours_structural.z, z_st),
    }
    context = eqx.filter_jit(lambda m, *a: m.diffusion.prepare(*a))(model, features, ref_inputs, ref_state)
    ours = model.diffusion(context, jnp.array(x_noisy), jnp.array(NOISE_LEVEL))
    errors["denoised"] = rel_err(ours, denoised)
    # Masked structural tokens change nothing.
    padded = _pad_structural(features, 9)
    context = eqx.filter_jit(lambda m, *a: m.diffusion.prepare(*a))(model, padded, ref_inputs, ref_state)
    padded_out = model.diffusion(context, jnp.array(x_noisy), jnp.array(NOISE_LEVEL))
    errors["denoised, padded"] = rel_err(padded_out, denoised)
    conf = model.confidence_logits(
        features, ref_inputs, ref_state, jnp.array(denoised.numpy()), key=KEY
    )
    errors |= {
        "pae": rel_err(conf.pae, pae[0]),
        "pde": rel_err(conf.pde, pde[0]),
        "plddt": rel_err(conf.plddt, plddt[0]),
        "resolved": rel_err(conf.resolved, resolved[0]),
        "distogram": rel_err(model.distogram(ref_state.z), distogram),
    }
    print(errors)
    tolerances = {"s_inputs": 1e-4, "denoised": 1e-4, "denoised, padded": 1e-4, "distogram": 1e-4}
    for name, error in errors.items():
        assert error < tolerances.get(name, 1e-3), (name, error)
