"""Trunk input parity: relative position encodings."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from reference import randomize, rel_err

from jfold.blocks.inputs import relative_position_features
from jfold.convert import boltz, protenix

t = torch.from_numpy


def make_tokens(seed):
    """Three chains (two copies of one entity), a residue split into several tokens, and
    one cyclic chain."""
    rng = np.random.default_rng(seed)
    asym_id = np.repeat([0, 1, 2], [12, 12, 8])
    entity_id = np.repeat([0, 0, 1], [12, 12, 8])
    sym_id = np.repeat([0, 1, 0], [12, 12, 8])
    residue_index = np.concatenate(
        [np.arange(12), np.arange(12), np.array([0, 0, 0, 1, 2, 3, 4, 5])]
    )
    residue_index = residue_index + rng.integers(0, 3) * 40  # offsets past r_max matter too
    token_index = np.arange(len(asym_id))
    cyclic_period = np.where(asym_id == 1, 12, 0)
    return {
        k: v.astype(np.int64)
        for k, v in {
            "asym_id": asym_id,
            "residue_index": residue_index,
            "entity_id": entity_id,
            "token_index": token_index,
            "sym_id": sym_id,
            "cyclic_period": cyclic_period,
        }.items()
    }


def _protenix_lineage(module):
    m = module.RelativePositionEncoding(c_z=8)

    def run(f):
        f = {k: t(v) for k, v in f.items() if k != "cyclic_period"}
        if hasattr(m, "generate_relp"):  # Mol-1, OpenDDE: features first, then the projection
            return m(m.generate_relp(f)["relp"])
        return m(f)

    return m, run, protenix.relative_positions, {}


def _protenix():
    import protenix.model.modules.embedders as module

    return _protenix_lineage(module)


def _mol1():
    import mol.model.modules.embedders as module

    return _protenix_lineage(module)


def _opendde():
    import opendde.model.modules.embedders as module

    return _protenix_lineage(module)


def _boltz2():
    from boltz.model.modules.encodersv2 import RelativePositionEncoder

    m = RelativePositionEncoder(token_z=8, fix_sym_check=True, cyclic_pos_enc=True)
    run = lambda f: m({k: t(v)[None] for k, v in f.items()})[0]
    return m, run, boltz.relative_positions, {"cyclic": True}


def _promera():
    from promera.model.encoders import RelativePositionEncoder

    m = RelativePositionEncoder(token_z=8)
    run = lambda f: m({k: t(v)[None] for k, v in f.items()})[0]
    return m, run, boltz.relative_positions, {"chain_offset_on_different_chains": True}


CASES = {
    "protenix": _protenix,
    "mol1": _mol1,
    "opendde": _opendde,
    "boltz2": _boltz2,
    "promera": _promera,
}


def _features(f, cyclic=False, chain_offset_on_different_chains=False):
    asym = jnp.array(f["asym_id"])
    return relative_position_features(
        asym,
        jnp.array(f["residue_index"]),
        jnp.array(f["entity_id"]),
        jnp.array(f["token_index"]),
        jnp.array(f["sym_id"]),
        cyclic_period=jnp.array(f["cyclic_period"]) if cyclic else None,
        chain_offset=(asym[:, None] != asym[None, :]) if chain_offset_on_different_chains else None,
    )


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("case", CASES)
def test_relative_positions_match_reference(case, seed):
    module, run, convert, options = CASES[case]()
    randomize(module, seed=0)
    f = make_tokens(seed)
    with torch.no_grad():
        ref = run(f)
    assert rel_err(convert(module)(_features(f, **options)), ref) < 1e-6


def test_openfold3_relative_position_features_match_reference():
    from jopenfold3._vendor.openfold3.core.utils.relpos import relpos_complex

    f = make_tokens(0)
    ref = relpos_complex({k: t(v) for k, v in f.items()}, max_relative_idx=32, max_relative_chain=2)
    np.testing.assert_array_equal(np.asarray(_features(f)), ref.numpy())


@pytest.mark.parametrize("seed", [0, 1])
def test_mol1_constraints(seed):
    """Compare Mol-1-constraint's pair conditioning with upstream's, using random weights."""
    from types import SimpleNamespace

    from mol.constraint import encoders
    from mol.constraint.model import ConstraintMol

    from jfold.models import mol1

    c_z, n = 16, 20
    conditioner = encoders.ResidueSiteConstraintEncoder(c_z=c_z)
    conditioner.exclusion = encoders.ResidueSiteConstraintEncoder(c_z=c_z)
    conditioner.contact = encoders.ContactMaskEncoder(c_z)
    randomize(conditioner, seed)
    owner = SimpleNamespace(
        constraint_conditioner=conditioner,
        exclusion_enabled=True,
        contact_enabled=True,
        configs={"constraint_exclusion_d_gap": 1.5},
    )

    rng = np.random.default_rng(seed)
    source = np.arange(n) < 12
    site, excluded = np.zeros(n, bool), np.zeros(n, bool)
    picked = rng.permutation(12)
    site[picked[:3]], excluded[picked[3:5]] = True, True
    pairs = [(int(picked[5]), 13), (int(picked[6]), 17), (2, 9)]
    f = {
        "token_index": np.arange(n),
        "protein_token_mask": np.ones(n, bool),
        "site_mask": np.array(True),
        "site_residue_mask": site,
        "site_partner_token_mask": ~source,
        "exclusion_mask": np.array(True),
        "exclusion_residue_mask": excluded,
        "exclusion_partner_token_mask": ~source,
        "contact_pair_indices": np.array(sorted(tuple(sorted(p)) for p in pairs) + [(0, 0)]),
        "contact_pair_mask": np.array([True, True, True, False]),
    }
    with torch.no_grad():
        ref = ConstraintMol.get_constraint_update(owner, {k: t(v) for k, v in f.items()})

    pair = mol1.constraint_pair_features(f, n)
    ours = protenix.mol1_constraints(conditioner)(pair)
    assert rel_err(ours, ref) < 1e-5

    assert mol1.constraint_pair_features({"token_index": np.arange(n)}, n) is None
