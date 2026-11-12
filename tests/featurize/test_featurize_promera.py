"""Test `jfold.featurize.views.promera` against ProMERA's featurization."""

import jax
import numpy as np
import pytest
from test_featurize import FIXTURES, SYSTEMS, assert_matches, system

from jfold.featurize import Chain, System
from jfold.featurize.views import promera
from jfold.models.promera import features as model_features

# The keys `jfold.models.promera.features` reads, plus `frames_mask` and `msa_paired`.
KEYS = {
    "restype", "profile", "deletion_mean", "is_std", "is_epitope", "token_pad_mask",
    "token_to_rep_atom", "token_bonds", "residue_index", "token_index", "asym_id",
    "entity_id", "sym_id", "frames_mask", "ref_pos", "ref_element", "ref_charge",
    "ref_hydrogens", "ref_atom_name_chars", "ref_space_uid", "atom_to_token",
    "atom_pad_mask", "atom_is_std", "msa", "msa_mask", "msa_paired", "has_deletion",
    "deletion_value",
}  # fmt: skip


def fixture(name):
    with np.load(FIXTURES / f"promera_{name}.npz", allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_promera(name, tmp_path):
    ours = promera.features(system(name))
    assert set(ours) == KEYS
    # Without a pose, ref_pos is upstream's canonical conformer and every other key is fixed.
    upstream = fixture(name)
    upstream["ref_pos"] = upstream["aux_ref_pos_canonical"]
    np.savez(tmp_path / "upstream.npz", **upstream)
    assert_matches(ours, tmp_path / "upstream.npz")


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_promera_pose(name):
    """Check that upstream's drawn poses on our canonical conformers give upstream's ref_pos."""
    upstream = fixture(name)
    canonical, uid = promera.reference_conformers(system(name))
    posed = promera.pose(
        canonical, uid, upstream["aux_ref_translation"], upstream["aux_ref_rotation"]
    ).astype(np.float32)
    num_atoms = len(posed)
    np.testing.assert_allclose(posed, upstream["ref_pos"][:num_atoms], rtol=1e-6, atol=1e-7)
    assert not upstream["ref_pos"][num_atoms:].any()


def test_promera_random_pose():
    sys = system("dimer")
    canonical = promera.features(sys)
    a = promera.features(sys, key=jax.random.key(0))
    b = promera.features(sys, key=jax.random.key(0))
    c = promera.features(sys, key=jax.random.key(1))
    np.testing.assert_array_equal(a["ref_pos"], b["ref_pos"])
    assert not np.allclose(a["ref_pos"], c["ref_pos"])
    for k in canonical.keys() - {"ref_pos"}:
        np.testing.assert_array_equal(a[k], canonical[k])
    # Each residue moves rigidly: (x + t) R^T, with R a proper rotation.
    t, rot = promera.random_pose(jax.random.key(0), len(canonical["restype"]))
    np.testing.assert_allclose(
        rot @ rot.transpose(0, 2, 1), np.broadcast_to(np.eye(3), rot.shape), atol=1e-6
    )
    np.testing.assert_allclose(np.linalg.det(rot), 1, atol=1e-6)
    mask = a["atom_pad_mask"].astype(bool)
    uid = a["ref_space_uid"][mask]
    expected = np.einsum("aj,aij->ai", canonical["ref_pos"][mask] + t[uid], rot[uid])
    np.testing.assert_allclose(a["ref_pos"][mask], expected, atol=1e-5)
    assert not a["ref_pos"][~mask].any()


def test_promera_rotation_convention():
    """Check that quaternions map to matrices as scipy's `Rotation.from_quat` does (scalar last)."""
    from scipy.spatial.transform import Rotation

    key = jax.random.key(3)
    _, rot = promera.random_pose(key, 16)
    q = np.array(jax.random.normal(jax.random.split(key)[1], (16, 4)), np.float64)
    np.testing.assert_allclose(rot, Rotation.from_quat(q).as_matrix(), atol=1e-12)


def test_promera_unk_and_layout():
    """Check UNK's atoms and the MSA of a chain with fewer rows than the others."""
    msa = ">query\nMKTAYIAKQRQISFVKSHFSRQ\n>a\nMKTAYIAKQRQISFVKSHFSRE\n>b\nMKTAYIAKQRQISFVKSHFSRA\n"
    sys = System(
        (
            Chain("A", "MKTAYIAKQRQISFVKSHFSRQ", msa=msa),
            Chain("B", "GXA"),
        )
    )
    f = promera.features(sys)
    unk = 22 + 1  # token of the X in chain B
    assert f["restype"][unk] == 22
    first = int(np.flatnonzero(f["atom_to_token"] == unk)[0])
    assert (f["atom_to_token"][f["atom_pad_mask"].astype(bool)] == unk).sum() == 6
    assert f["token_to_rep_atom"][unk] == first + 1
    assert f["msa"].shape == (3, 25)
    assert (f["msa"][1:, 22:] == 0).all() and f["msa_mask"].all()
    assert f["msa_paired"][0].all() and not f["msa_paired"][1:].any()


def test_promera_taxonomy_reorders():
    """Check that hits sharing a UniRef100 taxonomy move first and are flagged as paired."""
    rows = ["MKTA", "MKTE", "MKTD", "MKTC"]
    headers = ["query", "UniRef100_a", "UniRef100_b", "UniRef100_c"]
    text = "".join(f">{h}\n{r}\n" for h, r in zip(headers, rows, strict=True))
    tax = {"UniRef100_a": 1, "UniRef100_b": 2, "UniRef100_c": 2}
    f = promera.features(System((Chain("A", "MKTA", msa=text),)), taxonomy=tax.get)
    order = [promera.TOKEN.__getitem__(r[3]) for r in ("MKTA", "MKTD", "MKTC", "MKTE")]
    assert f["msa"][:, 3].tolist() == order
    assert f["msa_paired"][:, 0].tolist() == [True, True, True, False]
    with pytest.raises(ValueError, match="taxonomy"):
        promera.features(System((Chain("A", "MKTA", msa=text),)))


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_promera_model_features(name):
    f = promera.features(system(name))
    out = model_features(f)
    num_atoms, num_tokens = f["ref_pos"].shape[0], f["restype"].shape[0]
    assert out.atoms.features.shape == (num_atoms, 397)
    assert out.token_features.shape == (num_tokens, 70)
    assert out.msa_features.shape == (f["msa"].shape[0], num_tokens, 36)
