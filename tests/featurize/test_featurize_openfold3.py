"""`jfold.featurize.views.openfold3` against OpenFold3's upstream featurization on the shared
fixture systems and the template systems. Every feature but `ref_pos` matches exactly (unit
vectors to float32 rounding); upstream embeds a fresh random RDKit conformer per residue, so
`ref_pos` is checked structurally: the same atoms with the same bond lengths, and each
residue a rigid motion of a table conformer."""

import jax
import numpy as np
import pytest
from test_featurize import FIXTURES, SYSTEMS, TEMPLATE_SYSTEMS, assert_matches, system

from jfold.featurize.system import THREE_LETTER
from jfold.featurize.views import openfold3
from jfold.models.openfold3 import features as model_features

BOND = 1.9  # heavy-atom bonds are shorter (S-C 1.82), 1-3 distances longer (> 2.1)
# Templates that leave a residue of their chain unaligned, which upstream drops.
DROPPED = {"dimer_template", "monomer_template_gap"}


def fixture(name: str) -> dict[str, np.ndarray]:
    with np.load(FIXTURES / f"openfold3_{name}.npz", allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def residue_coords(f: dict, token: int) -> np.ndarray:
    start, n = f["start_atom_index"][token], f["num_atoms_per_token"][token]
    return f["ref_pos"][start : start + n].astype(np.float64)


def bonds(x: np.ndarray) -> np.ndarray:
    """Bonded atom pairs `[B, 2]` of a heavy-atom conformer, by distance."""
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    i, j = np.triu_indices(len(x), 1)
    bonded = d[i, j] < BOND
    return np.stack([i[bonded], j[bonded]], -1)


def bond_lengths(x: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x[..., pairs[:, 0], :] - x[..., pairs[:, 1], :], axis=-1)


def rigid_rmsd(x: np.ndarray, y: np.ndarray) -> float:
    """RMSD of `y` to the best proper rigid motion of `x` (Kabsch, no reflection)."""
    x, y = x - x.mean(0), y - y.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    s = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    return float(np.sqrt(((x @ u @ s @ vt - y) ** 2).sum(-1).mean()))


def sequence(name: str) -> str:
    return "".join(c["sequence"] for c in SYSTEMS[name]["chains"])


@pytest.mark.parametrize("name", sorted(SYSTEMS) + sorted(TEMPLATE_SYSTEMS))
@pytest.mark.filterwarnings("ignore:chain")
def test_openfold3(name):
    ours = openfold3.features(system(name))
    assert ours.keys() == fixture(name).keys()
    assert_matches(
        {k: v for k, v in ours.items() if k != "ref_pos"}, FIXTURES / f"openfold3_{name}.npz"
    )


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_openfold3_ref_pos(name):
    """Atoms, names and elements match exactly (above); per residue, bond lengths agree
    with upstream's conformer, and the posed coordinates are a rigid motion of one of the
    table's conformers (conformer 0, centred and unposed, without a key)."""
    table = openfold3.residues()
    upstream = fixture(name)
    unposed = openfold3.features(system(name))
    posed = openfold3.features(system(name), key=jax.random.key(0))
    assert posed["ref_pos"].shape == upstream["ref_pos"].shape
    assert posed["ref_pos"].dtype == np.float32
    for token, residue in enumerate(sequence(name)):
        conformers = table[THREE_LETTER[residue]].conformers.astype(np.float64)
        theirs = residue_coords(upstream, token)
        pairs = bonds(conformers[0])
        assert np.array_equal(bonds(theirs), pairs), token
        # ETKDG does not minimize, so bond lengths vary by up to ~0.1 A between draws:
        # upstream's draw is within 0.05 A of some table conformer, and of every one to 0.1 A.
        diff = np.abs(bond_lengths(conformers, pairs) - bond_lengths(theirs, pairs)).max(-1)
        assert diff.min() < 0.05 and diff.max() < 0.1, (token, diff.min(), diff.max())

        np.testing.assert_allclose(residue_coords(unposed, token), conformers[0], atol=1e-6)
        x = residue_coords(posed, token)
        assert min(rigid_rmsd(c, x) for c in conformers) < 1e-4, token


def test_openfold3_conformer_draws():
    """Different keys give different draws: conformers and poses vary across residues."""
    f = openfold3.features(system("dimer"), key=jax.random.key(1))
    g = openfold3.features(system("dimer"), key=jax.random.key(2))
    assert not np.allclose(f["ref_pos"], g["ref_pos"])
    centres = [residue_coords(f, t).mean(0) for t in range(len(sequence("dimer")))]
    assert np.std(centres) > 0.3  # unit-Gaussian translations


def test_openfold3_table_follows_upstream():
    """The table's conformer s is upstream's conformer with `random.seed(s)`: the fixtures
    seed `random` with 0 before featurizing, so their first residue is conformer 0."""
    table = openfold3.residues()
    for name in SYSTEMS:
        first = THREE_LETTER[sequence(name)[0]]
        x = residue_coords(fixture(name), 0)
        assert rigid_rmsd(table[first].conformers[0].astype(np.float64), x) < 1e-4


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_openfold3_model_features(name):
    f = model_features(openfold3.features(system(name), key=jax.random.key(0)))
    assert f.atoms.ref_pos.shape == (fixture(name)["ref_pos"].shape[0], 3)
    assert bool(np.isfinite(np.array(f.atoms.features)).all())


@pytest.mark.parametrize("name", sorted(DROPPED))
def test_openfold3_partial_templates_are_dropped(name):
    with pytest.warns(UserWarning, match="OpenFold3 drops it"):
        f = openfold3.features(system(name))
    assert not f["template_pseudo_beta_mask"].any()
    np.testing.assert_array_equal(f["template_restype"][..., openfold3.GAP], 1)


def test_openfold3_templates_by_sequence():
    """A chain without a template takes the one of a chain with its sequence; distogram and
    unit vectors stay within chains."""
    f = openfold3.features(system("homodimer_template"))
    n = len(TEMPLATE_SYSTEMS["homodimer_template"]["chains"][0]["sequence"])
    np.testing.assert_array_equal(f["template_restype"][0, :n], f["template_restype"][0, n:])
    np.testing.assert_array_equal(
        f["template_distogram"][0, :n, :n], f["template_distogram"][0, n:, n:]
    )
    assert not f["template_distogram"][0, :n, n:].any()
    assert not f["template_unit_vector"][0, :n, n:].any()
    assert not f["template_pseudo_beta_mask"][1:].any()


def test_openfold3_template_model_features():
    f = model_features(openfold3.features(system("dimer_template_covering")))
    assert bool(np.isfinite(np.array(f.template_features)).all())


def test_openfold3_identical_template_skips_kalign():
    """The shortcut for a template with its chain's sequence is Kalign's alignment."""
    kalign = pytest.importorskip("kalign")
    sequence = TEMPLATE_SYSTEMS["monomer_template"]["chains"][0]["sequence"]
    rows = kalign.align([sequence] * 3)
    assert rows == [sequence] * 3
    np.testing.assert_array_equal(
        openfold3.template_alignment(sequence, sequence), np.arange(len(sequence))
    )
