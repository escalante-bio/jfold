"""The Protenix-lineage `jfold.featurize` views (Protenix, Mol-1, OpenDDE) against the upstream
featurizations on the shared fixture systems (Protenix and Mol-1 also on the templated ones),
their reference-conformer poses, and the jfold adapters on their output."""

import jax
import numpy as np
import pytest
from test_featurize import FIXTURES, SYSTEMS, TEMPLATE_SYSTEMS, assert_matches, system

from jfold.featurize import Chain, System, Template
from jfold.featurize.views import mol1, opendde, protenix
from jfold.models import mol1 as mol1_model
from jfold.models import opendde as opendde_model
from jfold.models import protenix as protenix_model

VIEWS = {"protenix": protenix, "mol1": mol1, "opendde": opendde}
ADAPTERS = {
    "protenix": protenix_model.features,
    "mol1": mol1_model.features,
    "opendde": opendde_model.features,
}
TRANSLATE_FIRST = {"protenix": True, "mol1": True, "opendde": False}
# Templates of another length than their chain, which Protenix and Mol-1 refuse.
UNEQUAL = ("dimer_template", "dimer_template_covering", "monomer_template_gap")
TEMPLATED = sorted(set(TEMPLATE_SYSTEMS) - set(UNEQUAL))
CASES = [(model, name) for model in VIEWS for name in sorted(SYSTEMS)] + [
    (model, name) for model in ("protenix", "mol1") for name in TEMPLATED
]


def fixture(model: str, name: str) -> dict[str, np.ndarray]:
    with np.load(FIXTURES / f"{model}_{name}.npz", allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def centred_conformer(upstream: dict) -> np.ndarray:
    """Upstream's reference conformer, centred per residue (its pose with no augmentation)."""
    conformer, uid = upstream["aux_ref_pos_conformer"], upstream["ref_space_uid"]
    return np.concatenate(
        [conformer[uid == u] - conformer[uid == u].mean(0) for u in np.unique(uid)]
    )


@pytest.mark.parametrize("model,name", CASES)
def test_matches_upstream(model, name):
    """Every key equals upstream's; unposed, `ref_pos` is upstream's centred conformer."""
    ours = VIEWS[model].features(system(name))
    upstream = fixture(model, name)
    np.testing.assert_array_equal(ours["ref_pos"], centred_conformer(upstream))
    assert ours["ref_pos"].dtype == np.float32
    assert_matches(
        {k: v for k, v in ours.items() if k != "ref_pos"}, FIXTURES / f"{model}_{name}.npz"
    )


@pytest.mark.parametrize("model,name", CASES)
def test_pose_replays_upstream(model, name):
    """Upstream's posed `ref_pos` from the canonical conformer and its numpy draws (one
    uniform(-1, 1, 3) and one normal(4) per residue): checks the pose composition order."""
    upstream = fixture(model, name)
    rng = np.random.RandomState(int(upstream["aux_seed"]))
    draws = [(rng.uniform(-1.0, 1.0, size=3), rng.normal(size=4)) for _ in upstream["asym_id"]]
    translation = np.array([t for t, _ in draws])
    rotation = protenix.rotation_matrices(np.array([q for _, q in draws]))
    posed = protenix.pose(
        centred_conformer(upstream),
        upstream["ref_space_uid"],
        translation,
        rotation,
        translate_first=TRANSLATE_FIRST[model],
    )
    np.testing.assert_allclose(posed, upstream["ref_pos"], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("model", VIEWS)
def test_pose_is_rigid(model):
    """With a key, each residue's `ref_pos` is a proper rigid motion of its centred
    conformer, with the translation in upstream's range; the other keys don't change."""
    view = VIEWS[model]
    canonical = view.features(system("dimer"))
    posed = view.features(system("dimer"), key=jax.random.key(0))
    assert not np.allclose(posed["ref_pos"], canonical["ref_pos"])
    assert not np.allclose(
        posed["ref_pos"], view.features(system("dimer"), key=jax.random.key(1))["ref_pos"]
    )
    for k in canonical.keys() - {"ref_pos"}:
        np.testing.assert_array_equal(posed[k], canonical[k], err_msg=k)
    a2t = canonical["atom_to_token_idx"]
    for t in np.unique(a2t):
        p, x = canonical["ref_pos"][a2t == t], posed["ref_pos"][a2t == t]
        centroid = x.mean(0)
        u, _, vt = np.linalg.svd((x - centroid).T @ p)
        r = u @ vt
        assert np.linalg.det(r) > 0
        np.testing.assert_allclose(p @ r.T + centroid, x, atol=1e-4)
        # Protenix/Mol-1: centroid = R t with t in [-1, 1]^3; OpenDDE: centroid = t.
        if TRANSLATE_FIRST[model]:
            assert np.linalg.norm(centroid) <= np.sqrt(3) + 1e-5
        else:
            assert np.all(np.abs(centroid) <= 1 + 1e-5)


@pytest.mark.parametrize("model,name", CASES)
def test_adapter(model, name):
    """The jfold adapter takes the view's output; its relative positions equal upstream's
    model-side ones where the fixture has them."""
    ours = VIEWS[model].features(system(name), key=jax.random.key(0))
    upstream = fixture(model, name)
    features = ADAPTERS[model](ours)
    num_tokens, num_atoms = ours["restype"].shape[0], ours["ref_pos"].shape[0]
    num_rows, num_templates = ours["msa"].shape[0], ours["template_aatype"].shape[0]
    assert features.atoms.features.shape == (num_atoms, 389)
    assert features.token_features.shape == (num_tokens, 65)
    assert features.msa_features.shape == (num_rows, num_tokens, 34)
    assert features.template_features.shape == (num_templates, num_tokens, num_tokens, 108)
    np.testing.assert_array_equal(
        features.rep_atom,
        ours["atom_rep_atom_idx"]
        if model == "protenix"
        else np.flatnonzero(ours["distogram_rep_atom_mask"]),
    )
    if "aux_relp" in upstream:
        np.testing.assert_array_equal(features.relative_positions, upstream["aux_relp"])
    if model == "opendde":
        np.testing.assert_array_equal(
            features.extras.relative_positions, upstream["aux_structural_relp"]
        )
        assert features.extras.atoms.atom_to_token.shape == (num_atoms,)


@pytest.mark.parametrize("model", ["protenix", "mol1"])
@pytest.mark.parametrize("name", UNEQUAL)
def test_template_length_must_match(model, name):
    with pytest.raises(ValueError, match="the template has"):
        VIEWS[model].features(system(name))


def test_mol1_refuses_template_it_would_drop():
    """Upstream drops (silently) a template with consecutive CAs over 150 Å apart."""
    atoms = tuple({"N": (x, 0, 0), "CA": (x + 1, 0, 0), "C": (x + 2, 0, 0)} for x in (0, 4, 200))
    chain = Chain("A", "AAAAA", template=Template("AAAAA", (*atoms, {}, {})))
    with pytest.raises(ValueError, match="150"):
        mol1.features(System((chain,)))


def test_opendde_refuses_templates():
    with pytest.raises(NotImplementedError, match="takes no templates"):
        opendde.features(system("monomer_template"))
