"""Write BioEmu2's upstream features for the test systems to `tests/featurize/data/`.

    uv run python tools/fixtures/dump_bioemu2.py
"""

import json
from pathlib import Path

import numpy as np
from bioemu2.biomoa.utils.feature_converter import convert_densebatch_to_dictionary
from bioemu2.data_interface import Protein, System
from bioemu2.msa.alignment import MSA
from bioemu2.structure.batch import batch_collate
from bioemu2.structure.preparation import prepare_structure

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "featurize" / "data"
MODE = "structure"


def upstream_features(
    chains: list[dict], mode: str = MODE, paired_queries: bool = False
) -> dict[str, np.ndarray]:
    """Return upstream's unbatched features for `chains`, passing each query as its paired A3M if
    `paired_queries`."""
    system = System(
        entities=[Protein(sequence=chain["sequence"], asym_ids=[chain["id"]]) for chain in chains]
    )
    msa = MSA(
        unpaired_a3m={
            chain["id"]: (
                (FIXTURES / chain["msa"]).read_text()
                if "msa" in chain
                else f">query\n{chain['sequence']}\n"
            )
            for chain in chains
        },
        paired_a3m={chain["id"]: f">query\n{chain['sequence']}\n" for chain in chains}
        if paired_queries
        else None,
    )
    structure = prepare_structure(system, msa=msa, mode=mode, contacts=[])
    feats = convert_densebatch_to_dictionary(batch_collate([structure]))
    return {key: value[0].numpy() for key, value in feats.items()}


def paired_query_systems(systems: dict) -> dict:
    """Return the heteromers to save with each chain's query as its paired A3M."""
    dimer = systems["dimer"]["chains"]
    return {"dimer": dimer, "dimer_msa": [dimer[0] | {"msa": "chain_A.a3m"}, dimer[1]]}


def main() -> None:
    systems = json.loads((FIXTURES / "systems.json").read_text())
    runs = [(name, spec["chains"], False) for name, spec in systems.items()]
    runs += [
        (f"{name}_paired_queries", chains, True)
        for name, chains in paired_query_systems(systems).items()
    ]
    for name, chains, paired_queries in runs:
        feats = upstream_features(chains, paired_queries=paired_queries)
        path = FIXTURES / f"bioemu2_{name}.npz"
        np.savez_compressed(path, **feats)
        print(
            f"{path}: {feats['asym_id'].shape[0]} tokens, "
            f"{int(feats['atom_pad_mask'].sum())}/{feats['atom_pad_mask'].shape[0]} atoms, "
            f"{feats['msa'].shape[0]} MSA rows, {len(feats)} keys"
        )


if __name__ == "__main__":
    main()
