"""Dump Boltz-2's upstream feature dict for each fixture system in `tests/featurize/data/systems.json`.

Run from the jfold root:

    uv run python tools/fixtures/dump_boltz2.py

Writes `tests/featurize/data/boltz2_<system>.npz`, one per system, numpy arrays only, with the batch
axis removed (the layout `jfold.models.boltz2.features` takes). Every tensor in the feature
dict is saved under its upstream key. Two extra keys are added for parity tests:

- `ref_pos_raw` `[A, 3]` float32: the reference conformer coordinates before Boltz's
  per-residue centering and random rotation/translation, padded like `ref_pos`. These
  come from a second featurization in which `center_random_augmentation` is the
  identity.
- `ref_conformer_id` `[R]` int64: the conformer (of the 10 stored in each
  `~/.boltz/mols/<RES>.pkl`) used for each residue, in token order. It is replayed from
  `np.random.default_rng(42)`, the generator `PredictionDataset.__getitem__` creates. The
  script checks that the replay reproduces `ref_pos_raw`.

Determinism:

- The conformer choice uses numpy's `default_rng(42)`, which upstream seeds itself. It is
  deterministic.
- `ref_pos` is centred per residue, then given a random rotation and a N(0, 1) Å
  translation, both drawn from torch's global RNG. The script calls `torch.manual_seed(0)`
  first, so the saved `ref_pos` is reproducible with this torch version, but a numpy/JAX
  port cannot reproduce those draws. Compare `ref_pos` up to a per-residue rigid motion,
  or compare against `ref_pos_raw`.
- Nothing else in the dict is random for protein-only inference input.

MSAs come from the fixture a3m through the YAML `msa:` field (`msa: empty` for chains
without one). The MSA server is never contacted.

Templates (the systems in `template_systems.json`, PDB files from `make_templates.py`) go in
one mmCIF holding every templated chain, renamed to its query chain, each its own entity
whose full sequence is the template's sequence (its residues with coordinates, unless the
system gives the sequence); the YAML matches them by chain (`chain_id` and `template_id`).
This is how mosaic passes templates to Boltz-2.

    uv run python tools/fixtures/dump_boltz2.py [system ...]

dumps the named systems only.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parents[1] / "tests" / "featurize" / "data"
CACHE = Path.home() / ".boltz"


def _template_cif(system: dict, path: Path) -> list[str]:
    """Write the templated chains of `system` to one mmCIF at `path`, as mosaic does; their
    chain ids."""
    import gemmi

    from jfold.featurize.system import THREE_LETTER

    st, model, entities, ids = gemmi.Structure(), gemmi.Model("0"), [], []
    for chain in system["chains"]:
        spec = chain.get("template")
        if spec is None:
            continue
        template = gemmi.read_structure(str(FIXTURES / spec["pdb"]))[0][0]
        template.name = chain["id"]
        entity = gemmi.Entity(chain["id"])
        entity.entity_type = gemmi.EntityType.Polymer
        entity.polymer_type = gemmi.PolymerType.PeptideL
        entity.subchains = [chain["id"]]
        if "sequence" in spec:
            entity.full_sequence = [THREE_LETTER[c] for c in spec["sequence"]]
        else:
            entity.full_sequence = [r.name for r in template]
        entities.append(entity)
        for residue in template:
            residue.subchain = chain["id"]
        model.add_chain(template)
        ids.append(chain["id"])
    st.add_model(model)
    st.entities = gemmi.EntityList(entities)
    st.assign_subchains()
    st.setup_entities()
    st.ensure_entities()
    st.assign_label_seq_id()
    st.setup_entities()
    st.make_mmcif_document().write_file(str(path))
    return ids


def _yaml(system: dict, directory: Path) -> str:
    lines = ["version: 1", "sequences:"]
    for chain in system["chains"]:
        msa = chain.get("msa")
        msa = str(FIXTURES / msa) if msa else "empty"
        lines += [
            "  - protein:",
            f"      id: [{chain['id']}]",
            f"      sequence: {chain['sequence']}",
            f"      msa: {msa}",
        ]
    if any("template" in chain for chain in system["chains"]):
        cif = directory / "template.cif"
        ids = ", ".join(_template_cif(system, cif))
        lines += [
            "templates:",
            f"  - cif: {cif}",
            f"    chain_id: [{ids}]",
            f"    template_id: [{ids}]",
        ]
    return "\n".join(lines) + "\n"


def _featurize(system: dict, augment: bool) -> dict[str, np.ndarray]:
    """Run the upstream pipeline on `system`; the unbatched feature dict as numpy."""
    import boltz.data.feature.featurizerv2 as fv2
    import boltz.main as bm

    original = fv2.center_random_augmentation
    if not augment:
        fv2.center_random_augmentation = lambda coords, *args, **kwargs: coords
    try:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "input.yaml"
            path.write_text(_yaml(system, directory))
            bm.process_inputs(
                data=bm.check_inputs(path),
                out_dir=directory,
                ccd_path=CACHE / "ccd.pkl",
                mol_dir=CACHE / "mols",
                msa_server_url="",
                msa_pairing_strategy="greedy",
                use_msa_server=False,
                boltz2=True,
            )
            processed = directory / "processed"
            data = bm.Boltz2InferenceDataModule(
                manifest=bm.Manifest.load(processed / "manifest.json"),
                target_dir=processed / "structures",
                msa_dir=processed / "msa",
                mol_dir=CACHE / "mols",
                num_workers=0,
                constraints_dir=processed / "constraints",
                template_dir=processed / "templates",
                extra_mols_dir=processed / "mols",
                override_method=None,
            )
            torch.manual_seed(0)
            batch = next(iter(data.predict_dataloader()))
    finally:
        fv2.center_random_augmentation = original
    return {k: v[0].numpy() for k, v in batch.items() if isinstance(v, torch.Tensor)}


def _conformer_ids(n_residues: int) -> np.ndarray:
    """Replay upstream's per-residue conformer choice: `random.choice(conf_ids)` on
    `default_rng(42)`, once per residue in token order, with no earlier draws."""
    rng = np.random.default_rng(42)
    return np.array([int(rng.choice(list(range(10)))) for _ in range(n_residues)])


def _check_conformers(f: dict, conf_ids: np.ndarray, system: dict) -> None:
    """`ref_pos_raw` is exactly the chosen conformer's coordinates of each residue's atoms."""
    import pickle

    from boltz.data import const

    names = [
        const.prot_letter_to_token.get(c, "UNK") for ch in system["chains"] for c in ch["sequence"]
    ]
    expected = []
    for name, conf_id in zip(names, conf_ids, strict=True):
        with (CACHE / "mols" / f"{name}.pkl").open("rb") as fh:
            mol = pickle.load(fh)
        by_name = {a.GetProp("name"): a.GetIdx() for a in mol.GetAtoms()}
        conformer = mol.GetConformer(int(conf_id))
        expected += [list(conformer.GetAtomPosition(by_name[a])) for a in const.ref_atoms[name]]
    expected = np.array(expected, dtype=np.float32)
    n = len(expected)
    assert np.array_equal(f["ref_pos_raw"][:n], expected), "conformer replay mismatch"
    assert not f["ref_pos_raw"][n:].any()


def main() -> None:
    systems = json.loads((FIXTURES / "systems.json").read_text())
    systems |= json.loads((FIXTURES / "template_systems.json").read_text())
    if sys.argv[1:]:
        systems = {name: systems[name] for name in sys.argv[1:]}
    for name, system in systems.items():
        features = _featurize(system, augment=True)
        raw = _featurize(system, augment=False)
        for k in features:
            if k != "ref_pos":
                np.testing.assert_array_equal(features[k], raw[k], err_msg=k)
        features["ref_pos_raw"] = raw["ref_pos"]
        n_residues = sum(len(c["sequence"]) for c in system["chains"])
        features["ref_conformer_id"] = _conformer_ids(n_residues)
        _check_conformers(features, features["ref_conformer_id"], system)
        out = FIXTURES / f"boltz2_{name}.npz"
        np.savez_compressed(out, **features)
        print(f"{out}: {len(features)} arrays")
        for k in sorted(features):
            print(f"  {k:28s} {features[k].dtype!s:8s} {features[k].shape}")


if __name__ == "__main__":
    main()
