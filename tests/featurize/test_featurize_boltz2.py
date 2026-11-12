"""Test the Boltz-2 view against upstream's featurization on the shared fixture systems."""

import jax
import numpy as np
import pytest
from test_featurize import FIXTURES, SYSTEMS, TEMPLATE_SYSTEMS, assert_matches, system

from jfold.featurize.views import boltz2
from jfold.models.boltz2 import features as model_features


@pytest.mark.parametrize("name", sorted(SYSTEMS) + sorted(TEMPLATE_SYSTEMS))
def test_boltz2(name):
    ours = boltz2.features(system(name))
    fixture = FIXTURES / f"boltz2_{name}.npz"
    with np.load(fixture) as upstream:
        np.testing.assert_array_equal(
            boltz2.conformer_ids(len(ours["res_type"])), upstream["ref_conformer_id"]
        )
    # Without a key, the conformers are upstream's before it applies its random rigid motion.
    ours["ref_pos_raw"] = ours.pop("ref_pos")
    assert_matches(ours, fixture)


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_boltz2_posed(name):
    """Check that a key rotates and translates each residue, leaving the last one as stored."""
    canonical = boltz2.features(system(name))
    posed = boltz2.features(system(name), key=jax.random.key(0))
    for k, v in canonical.items():
        if k != "ref_pos":
            np.testing.assert_array_equal(posed[k], v, err_msg=k)
    real = canonical["atom_pad_mask"] > 0
    uid = canonical["ref_space_uid"][real]
    x, y = canonical["ref_pos"][real], posed["ref_pos"][real]
    assert not posed["ref_pos"][~real].any()
    last = uid.max()
    np.testing.assert_array_equal(y[uid == last], x[uid == last])
    for r in range(last):
        a, b = x[uid == r], y[uid == r]
        a = a - a.mean(0)
        b = b - b.mean(0)
        # The best rotation (Kabsch) aligns them exactly, and is proper.
        u, _, vt = np.linalg.svd(a.T @ b)
        rotation = u @ vt
        assert np.linalg.det(rotation) > 0
        np.testing.assert_allclose(a @ rotation, b, atol=1e-4)
        assert not np.allclose(a, b, atol=1e-3)


def test_boltz2_model_features():
    f = boltz2.features(system("dimer"), key=jax.random.key(1))
    out = model_features(f)
    assert out.token_mask.shape == (len(f["res_type"]),)


def test_boltz2_template_keys():
    """Check that upstream adds the force settings with templates and its placeholder without."""
    templated = boltz2.features(system("dimer_template"))
    plain = boltz2.features(system("dimer"))
    assert set(templated) - set(plain) == {"template_force", "template_force_threshold"}
    np.testing.assert_array_equal(plain["template_restype"][..., 0], 1)


def test_boltz2_template_must_align():
    from jfold.featurize import Chain, System, Template

    template = Template("XX", ({"CA": (0, 0, 0)}, {"CA": (3.8, 0, 0)}))
    with pytest.raises(ValueError, match="no template aligns"):
        boltz2.features(System((Chain("A", "XXXX", template=template),)))
