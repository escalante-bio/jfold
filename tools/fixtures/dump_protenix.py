"""Write Protenix's upstream features for the test systems to `tests/featurize/data/`.

Each file also holds `aux_ref_pos_conformer`, the conformers before the random
pose, which a featurizer without torch cannot easily reproduce.

    LAYERNORM_TYPE=torch uv run python tools/fixtures/dump_protenix.py [system ...]
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
    """Read the template as a `gemmi.Chain`, adding atom-less residues for positions missing from
    the PDB file."""
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
    """Build the upstream `ChainInput`s, with a query-only pairing file so nothing is paired."""
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
        except ValueError as error:  # From `_extract_chain_template`'s length check.
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
