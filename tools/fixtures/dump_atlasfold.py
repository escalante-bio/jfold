"""Write AtlasFold's upstream features for the test systems to `tests/featurize/data/`.

    uv run python tools/fixtures/dump_atlasfold.py
"""

import json
from pathlib import Path

import numpy as np
from atlasfold.common.protein import ProteinMultimer
from atlasfold.runner import FoldingRunner
from atlasfold.runner_multimer import MultimerFoldingRunner

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parents[1] / "tests" / "featurize" / "data"
HOMOMER = {"A": "MKTXGYIAK", "B": "GSHMW", "C": "MKTXGYIAK"}


def main():
    systems = json.loads((FIXTURES / "systems.json").read_text())
    systems["homomer"] = {"chains": [{"id": k, "sequence": v} for k, v in HOMOMER.items()]}
    for name, spec in systems.items():
        if any("msa" in c for c in spec["chains"]):
            continue
        chains = [c["sequence"] for c in spec["chains"]]
        length = sum(map(len, chains))
        runner = MultimerFoldingRunner
        batch = runner._make_batch_features(
            [ProteinMultimer.get_empty(name, chains)], runner._get_length_bucket(length)
        )
        np.savez(FIXTURES / f"atlasfold_m_{name}.npz", **{k: v[0] for k, v in batch.items()})
        print(f"atlasfold_m_{name}: {sorted(batch)}")
        if len(chains) == 1:
            batch = FoldingRunner._make_batch_features(
                chains, FoldingRunner._get_length_bucket(length)
            )
            np.savez(FIXTURES / f"atlasfold_{name}.npz", **{k: v[0] for k, v in batch.items()})
            print(f"atlasfold_{name}: {sorted(batch)}")


if __name__ == "__main__":
    main()
