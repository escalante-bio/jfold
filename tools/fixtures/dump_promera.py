"""Dump ProMERA's upstream feature dicts for the shared featurization fixtures.

Run from ~/jfold:

    uv run python tools/fixtures/dump_promera.py

For each system in `tests/featurize/data/systems.json` this featurizes exactly as ProMERA's
inference does (`promera.inference.cofolding.Cofolding.__getitem__`): tinyprot
`Structure.from_schema`, MSAs from an `msa_dir` via `load_msa_from_dir`,
`construct_paired_msa`, `AF3Featurizer(...).featurize(compute_frames=True)`,
`promera.inference.utils.finalize_feats`, then `promera.data.utils.collate` on a batch of
one. The saved dict is that batch with the batch axis removed, i.e. what
`jfold.models.promera.features` takes. Output: `tests/featurize/data/promera_<system>.npz`.

Precomputed MSAs go through ProMERA's own mechanism: a private, temporary `msa_dir` in
which each chain's a3m is stored as `<sha256(sequence + "\\n")>.a3m`. Chains without an
a3m miss the lookup and get tinyprot's single-row dummy MSA. Using a temporary directory
keeps any MSAs cached in `~/.cache/tinyprot/msa` out of the fixtures.

Determinism: the only random step is the reference conformer (`tinyprot.ccd._get_conformer`):
each residue's centred CCD ideal coordinates get a N(0, 1) translation and then a uniform
random rotation, both from numpy's global RNG. The global RNG is seeded with `SEED` before each
system. The script also records each residue's draws and stores them next to the features:

    aux_ref_pos_canonical  [A, 3]    float32  centred CCD ideal coordinates (no jitter/rotation),
                                              padded like ref_pos
    aux_ref_translation    [R, 3]    float64  per-residue translation t (indexed by ref_space_uid)
    aux_ref_rotation       [R, 3, 3] float64  per-residue rotation matrix Rm

with ref_pos = (aux_ref_pos_canonical + t) @ Rm.T for every atom of that residue (asserted
below). String-valued keys (`ref_atom_name`, `residue_name`, `asym_id_`, `msa_chars`) are kept
as numpy unicode arrays, unpadded, as tinyprot produces them; the model does not read them.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

SEED = 0
HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parents[1] / "tests" / "featurize" / "data"


def _ensure_tinyprot_data():
    # `tinyprot.msa` opens the taxonomy LMDB at import time, so check before importing it.
    from tinyprot import init

    if not all(os.path.exists(Path(p) / "data.mdb") for p in (init._ccd_path, init._tax_path)):
        init.download_pretrained()


class _ConformerRecorder:
    """Wraps `tinyprot.ccd._get_conformer` (as imported into `tinyprot.feature`) without
    changing its output or its consumption of numpy's global RNG, recording each call's
    canonical coordinates, translation and rotation."""

    def __init__(self, original):
        self.original = original
        self.canonical, self.translation, self.rotation = [], [], []

    def __call__(self, rname, leaving=False, center=True, jitter=1.0, randrot=True):
        from scipy.spatial.transform import Rotation

        assert center and jitter == 1.0 and randrot, "unexpected _get_conformer arguments"
        canonical = self.original(rname, leaving, center=True, jitter=0.0, randrot=False)
        start = np.random.get_state()
        out = self.original(rname, leaving, center, jitter, randrot)
        end = np.random.get_state()
        # Replay the same draws: translation first, then the rotation.
        np.random.set_state(start)
        t = np.random.randn(3) * jitter
        rot = Rotation.random().as_matrix()
        assert all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), end))
        assert np.array_equal((canonical + t) @ rot.T, out)
        self.canonical.append(canonical)
        self.translation.append(t)
        self.rotation.append(rot)
        return out


def featurize(system: dict, name: str) -> dict:
    import tinyprot.feature
    from promera.data.utils import collate
    from promera.inference.utils import finalize_feats
    from tinyprot.feature import AF3Featurizer
    from tinyprot.msa import construct_paired_msa, hash_sequence, load_msa_from_dir
    from tinyprot.structure import Structure

    schema = {c["id"]: {"type": "protein", "sequence": c["sequence"]} for c in system["chains"]}
    struct = Structure.from_schema(schema)

    msa_dir = tempfile.mkdtemp(prefix="promera_msa_")
    try:
        for c in system["chains"]:
            if "msa" in c:
                key = hash_sequence(struct.chains[c["id"]].seq)
                shutil.copy(FIXTURES / c["msa"], os.path.join(msa_dir, f"{key}.a3m"))
        msas = load_msa_from_dir(msa_dir, struct.chains)
    finally:
        shutil.rmtree(msa_dir)
    for c in system["chains"]:
        assert (msas[c["id"]].path is not None) == ("msa" in c), c["id"]

    pairing = construct_paired_msa(msas)
    recorder = _ConformerRecorder(tinyprot.feature._get_conformer)
    tinyprot.feature._get_conformer = recorder
    try:
        np.random.seed(SEED)
        feats = AF3Featurizer(struct, msas, pairing).featurize(compute_frames=True)
    finally:
        tinyprot.feature._get_conformer = recorder.original
    feats = finalize_feats(feats, struct, name, 0)
    batch = collate([feats])

    out = {}
    for key, value in batch.items():
        if hasattr(value, "numpy"):  # a collated tensor: drop the batch axis
            out[key] = value[0].numpy()
        elif isinstance(value[0], np.ndarray) and value[0].dtype.kind == "U":
            out[key] = value[0]
    num_atoms = len(feats["ref_pos"])
    padded = out["ref_pos"].shape[0]
    canonical = np.concatenate(recorder.canonical)
    out["aux_ref_pos_canonical"] = np.concatenate(
        [canonical, np.zeros((padded - num_atoms, 3))]
    ).astype(np.float32)
    out["aux_ref_translation"] = np.stack(recorder.translation)
    out["aux_ref_rotation"] = np.stack(recorder.rotation)

    # ref_pos is the recorded rigid motion of the canonical conformer, per residue.
    uid = out["ref_space_uid"][:num_atoms]
    t, rot = out["aux_ref_translation"][uid], out["aux_ref_rotation"][uid]
    rebuilt = np.einsum("aj,aij->ai", canonical + t, rot)
    np.testing.assert_allclose(rebuilt, out["ref_pos"][:num_atoms], rtol=0, atol=1e-5)
    return out


def main():
    _ensure_tinyprot_data()
    systems = json.loads((FIXTURES / "systems.json").read_text())
    for name, system in systems.items():
        out = featurize(system, name)
        path = FIXTURES / f"promera_{name}.npz"
        np.savez_compressed(path, **out)
        print(
            f"{path}: {out['restype'].shape[0]} tokens, "
            f"{int(out['atom_pad_mask'].sum())}/{out['ref_pos'].shape[0]} atoms, "
            f"{out['msa'].shape[0]} MSA rows, {len(out)} keys"
        )


if __name__ == "__main__":
    main()
