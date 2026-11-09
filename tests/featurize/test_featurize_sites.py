"""Each view's `atom_sites` describes its features' atoms: the real atoms first, with the
names, elements and tokens the features encode."""

import importlib

import numpy as np
import pytest
from test_featurize import SYSTEMS, system

VIEWS = ["bioemu2", "boltz2", "esmfold2", "mol1", "opendde", "openfold3", "promera", "protenix"]
NAME_KEYS = ("ref_atom_name_chars", "name_encoding")
ELEMENT_KEYS = ("ref_element", "element")
MASK_KEYS = ("atom_pad_mask", "atom_attention_mask", "atom_mask")
TOKEN_KEYS = ("atom_to_token_idx", "atom_to_token", "atom_to_token_index", "token_idx_at_atom")
# Views whose element one-hot has class Z - 1 for atomic number Z (the rest: class Z).
ELEMENT_FROM_ZERO = {"mol1", "opendde", "openfold3", "protenix"}


def _first(f: dict, keys: tuple[str, ...]) -> np.ndarray | None:
    return next((np.asarray(f[k]) for k in keys if k in f), None)


def _indices(x: np.ndarray, ndim: int) -> np.ndarray:
    """Class indices, from indices or one-hots."""
    return x.argmax(-1) if x.ndim > ndim else x


@pytest.mark.parametrize("name", sorted(SYSTEMS))
@pytest.mark.parametrize("view_name", VIEWS)
def test_atom_sites(view_name, name):
    view = importlib.import_module(f"jfold.featurize.views.{view_name}")
    s = system(name)
    f = view.features(s)
    sites = view.atom_sites(s)
    n = len(sites)

    mask = _first(f, MASK_KEYS)
    if mask is not None:
        assert mask[:n].all() and not mask[n:].any()  # real atoms first
    names = _indices(_first(f, NAME_KEYS), 2)[:n]
    decoded = ["".join(chr(c + 32) for c in row).strip() for row in names]
    assert decoded == list(sites.atom_name)
    elements = _indices(_first(f, ELEMENT_KEYS), 1)[:n] + (view_name in ELEMENT_FROM_ZERO)
    np.testing.assert_array_equal(elements, sites.element)
    np.testing.assert_array_equal(_indices(_first(f, TOKEN_KEYS), 1)[:n], sites.token)


def test_to_pdb():
    from jfold.featurize.views import protenix

    s = system("dimer")
    sites = protenix.atom_sites(s)
    coords = np.arange(3 * len(sites), dtype=float).reshape(-1, 3)
    lines = sites.to_pdb(coords, b_factor=np.full(len(sites), 50.0)).splitlines()
    atoms = [line for line in lines if line.startswith("ATOM")]
    assert len(atoms) == len(sites) and lines[-1] == "END"
    first = atoms[0]
    assert first[12:16] == " N  " and first[17:20] == "MET" and first[21] == "A"
    assert int(first[22:26]) == 1 and float(first[60:66]) == 50.0 and first[76:78] == " N"
    assert atoms[-1][21] == "B" and atoms[-1][12:16] == " OXT"
    models = sites.to_pdb(np.stack([coords, coords])).splitlines()
    assert models.count("ENDMDL") == 2
