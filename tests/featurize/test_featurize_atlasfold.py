"""`jfold.featurize.views.atlasfold` against AtlasFold's runners on the shared fixture
systems without MSAs and a homomer (`tests/featurize/data/atlasfold[_m]_<system>.npz`)."""

import numpy as np
import pytest
from test_featurize import FIXTURES, SYSTEMS, assert_matches, system

from jfold.featurize import Chain, System
from jfold.featurize.views import atlasfold as view
from jfold.models import atlasfold

HOMOMER = {"A": "MKTXGYIAK", "B": "GSHMW", "C": "MKTXGYIAK"}  # as in dump_atlasfold.py
SINGLE_SEQUENCE = sorted(n for n, s in SYSTEMS.items() if not any("msa" in c for c in s["chains"]))


def _system(name: str) -> System:
    if name == "homomer":
        return System(tuple(Chain(id=k, sequence=v) for k, v in HOMOMER.items()))
    return system(name)


@pytest.mark.parametrize("name", [*SINGLE_SEQUENCE, "homomer"])
def test_atlasfold_m(name):
    ours = view.features(_system(name))
    path = FIXTURES / f"atlasfold_m_{name}.npz"
    with np.load(path) as upstream:
        assert set(ours) == set(upstream.files)
    assert_matches(ours, path)
    atlasfold.features(ours)  # the model reads it


def test_atlasfold():
    ours = view.features(system("monomer"), multimer=False)
    path = FIXTURES / "atlasfold_monomer.npz"
    with np.load(path) as upstream:
        assert set(ours) == set(upstream.files)
    assert_matches(ours, path)


def test_refuses_msas():
    with pytest.raises(NotImplementedError, match="no MSAs"):
        view.features(system("monomer_msa"))


def test_atom_sites():
    """One site per atom14 slot in use, in slot order."""
    s = _system("homomer")
    f = view.features(s)
    sites = view.atom_sites(s)
    assert len(sites) == f["atom14_mask"].sum()
    flat = np.flatnonzero(f["atom14_mask"].reshape(-1))
    assert (flat // 14 == sites.token).all()
    assert sites.atom_name[:5] == ("N", "CA", "C", "O", "CB")
