"""Upstream Protenix feature dicts for the shared fixture systems.

Runs `protenix.data.template.featurize` (the featurizer jfold's Protenix test
uses) on every system in `tests/featurize/data/systems.json` and saves
`tests/featurize/data/protenix_<system>.npz` (numpy arrays only, no pickles).

Run from `~/jfold`:

    LAYERNORM_TYPE=torch uv run python tools/fixtures/dump_protenix.py

Determinism: the only random step for protein-only inputs is the per-residue
`ref_pos` augmentation (`protenix.utils.geometry.random_transform`: a translation from
`np.random.uniform` and a rotation from `scipy.spatial.transform.Rotation.random()`,
both drawing from numpy's global RNG). The global numpy and `random` RNGs are seeded
with `SEED` right before each system is featurized, so the files are reproducible
bit-for-bit. A torch-free featurizer cannot reproduce the exact rotations without
replaying numpy's legacy RNG in the same order, so the unaugmented conformer is saved
too (`aux_ref_pos_conformer`, see below) for parity up to a per-residue rigid motion.

Keys:
- every numpy array of the upstream dict, under its own name;
- `constraint_feature__<name>` for the nested `constraint_feature` dict;
- `aux_*` extras that are not upstream features but help parity tests:
  `aux_ref_pos_conformer` (the CCD/RDKit reference conformer coordinates per atom,
  before centring and augmentation, `atom_array.ref_pos`), `aux_atom_name`,
  `aux_res_name`, `aux_element`, `aux_seed`.

Templates (the systems in `template_systems.json`, PDB files from `make_templates.py`):
`ChainInput.template` is the PDB file's chain as gemmi reads it, as mosaic passes it; with
a `sequence` in the system, a residue for each of its positions, those the file leaves out
without atoms. Upstream refuses a template with another number of residues than its chain;
those systems are reported and get no file.

    LAYERNORM_TYPE=torch uv run python tools/fixtures/dump_protenix.py [system ...]

dumps the named systems only.
"""

import json
import os
import random
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("LAYERNORM_TYPE", "torch")

import numpy as np

SEED = 0
HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parents[1] / "tests" / "featurize" / "data"


def template_chain(spec: dict):
    """The template `spec` as a `gemmi.Chain`: the PDB file's chain; with a `sequence`, a
    residue for each of its positions (residue number = position), those the file leaves
    out without atoms."""
    import gemmi

    from jfold.featurize.system import THREE_LETTER

    chain = gemmi.read_structure(str(FIXTURES / spec["pdb"]))[0][0]
    if "sequence" not in spec:
        return chain
    resolved = {residue.seqid.num: residue for residue in chain}
    full = gemmi.Chain(chain.name)
    for number, letter in enumerate(spec["sequence"], start=1):
        residue = resolved.get(number)
        if residue is None:
            residue = gemmi.Residue()
            residue.name = THREE_LETTER[letter]
            residue.seqid = gemmi.SeqId(number, " ")
        full.add_residue(residue)
    return full


def chain_inputs(system: dict, msa_root: Path):
    """`ChainInput`s for a system: chains with an `msa` get a precomputed MSA directory
    (`non_pairing.a3m` = the a3m; `pairing.a3m` = the query only, so nothing is paired);
    chains without one get Protenix's dummy single-sequence MSA (`compute_msa=False`).
    A chain's `template` is its `template_chain`."""
    from protenix.data.template import ChainInput

    chains = []
    for chain in system["chains"]:
        sequence = chain["sequence"]
        template = template_chain(chain["template"]) if "template" in chain else None
        msa_dir = None
        if "msa" in chain:
            msa_dir = msa_root / chain["id"]
            msa_dir.mkdir(parents=True)
            (msa_dir / "non_pairing.a3m").write_text((FIXTURES / chain["msa"]).read_text())
            (msa_dir / "pairing.a3m").write_text(f">query\n{sequence}\n")
            msa_dir = str(msa_dir)
        chains.append(
            ChainInput(sequence, compute_msa=False, precomputed_msa_dir=msa_dir, template=template)
        )
    return chains


def featurize_system(system: dict) -> dict[str, np.ndarray]:
    from protenix.data.template import featurize

    with tempfile.TemporaryDirectory() as tmp:
        chains = chain_inputs(system, Path(tmp))
        np.random.seed(SEED)
        random.seed(SEED)
        features, atom_array, _ = featurize(chains)

    out = {}
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            out[key] = value
        elif isinstance(value, dict):
            for sub, sub_value in value.items():
                out[f"{key}__{sub}"] = np.asarray(sub_value)
        else:
            raise TypeError(f"{key}: {type(value)}")
    out["aux_ref_pos_conformer"] = np.asarray(atom_array.ref_pos, np.float32)
    out["aux_atom_name"] = np.asarray(atom_array.atom_name, str)
    out["aux_res_name"] = np.asarray(atom_array.res_name, str)
    out["aux_element"] = np.asarray(atom_array.element, str)
    out["aux_seed"] = np.asarray(SEED)
    return out


def main():
    systems = json.loads((FIXTURES / "systems.json").read_text())
    systems |= json.loads((FIXTURES / "template_systems.json").read_text())
    if sys.argv[1:]:
        systems = {name: systems[name] for name in sys.argv[1:]}
    for name, system in systems.items():
        try:
            out = featurize_system(system)
        except ValueError as error:  # `_extract_chain_template`'s length check
            if not str(error).startswith("Template chain has"):
                raise
            print(f"{name}: upstream refuses it ({error})")
            continue
        path = FIXTURES / f"protenix_{name}.npz"
        np.savez_compressed(path, **out)
        print(
            f"{path}: {len(out)} arrays, {out['asym_id'].shape[0]} tokens, "
            f"{out['ref_pos'].shape[0]} atoms, msa {out['msa'].shape}"
        )


if __name__ == "__main__":
    main()
