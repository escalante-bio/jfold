"""`jfold.featurize.views.esmfold2` against upstream ESMFold2's featurization on the shared
fixture systems (`tests/featurize/data/esmfold2_<system>.npz`)."""

import numpy as np
import pytest
from test_featurize import FIXTURES, SYSTEMS, assert_matches, system

from jfold.featurize.views import esmfold2 as view
from jfold.models import esmfold2


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_esmfold2(name):
    ours = view.features(system(name))
    with np.load(FIXTURES / f"esmfold2_{name}.npz") as upstream:
        prepared = {k for k in upstream.files if not k.startswith("embed_")}
    assert set(ours) == prepared
    assert_matches(ours, FIXTURES / f"esmfold2_{name}.npz")


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_esmfold2_model_inputs(name):
    """The model adapter runs on the view's output: the embedder inputs match what the
    upstream forward hands its embedder, and the ESMC layout and input match the packed
    token inputs upstream builds."""
    f = view.features(system(name))
    upstream = np.load(FIXTURES / f"esmfold2_{name}.npz")
    features = esmfold2.features(f)
    token = features.token_features
    np.testing.assert_array_equal(token[:, :33], upstream["embed_res_type"])
    np.testing.assert_allclose(token[:, 33:66], upstream["embed_profile"], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(token[:, 66], upstream["embed_deletion_mean"], rtol=1e-6)
    # Checkpoints without MSA features zero both; the model does (`msa_profile`).
    np.testing.assert_array_equal(upstream["embed_profile_no_msa_features"], 0)
    np.testing.assert_array_equal(upstream["embed_deletion_mean_no_msa_features"], 0)

    protein = (f["mol_type"] == 0) & f["token_attention_mask"]
    layout = esmfold2.lm_layout(f["asym_id"], f["residue_index"], protein)
    np.testing.assert_array_equal(layout.sequence_id, upstream["lm_sequence_id"])
    np.testing.assert_array_equal(layout.positions, upstream["lm_positions"])
    one_hot = np.eye(64, dtype=np.float32)
    for residues in (
        esmfold2.esmc_residues(one_hot[f["res_type"]][:, :33]),
        one_hot[f["input_ids"]],
    ):
        packed = esmfold2.lm_input(layout, residues)
        np.testing.assert_array_equal(packed, one_hot[upstream["lm_input_ids"]])
