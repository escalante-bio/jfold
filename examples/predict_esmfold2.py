"""Fold a protein (or a complex) with a small ESMFold2 checkpoint, without torch: load the
converted model (with its ESMC language model), featurize with `jfold.featurize`, run the
trunk, sample structures and write them as a multi-model PDB file.

    uv run python examples/predict_esmfold2.py MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
    uv run python examples/predict_esmfold2.py SEQ_A SEQ_B --samples 3 --out complex.pdb

ESMFold2 folds from single sequences (`--msa` is optional). The default checkpoint,
ESMFold2-Experimental-Fast on ESMC 300M, has no confidence head, so the samples are not
ranked. The first run downloads the converted weights (to `$JFOLD_CACHE_DIR`,
`~/.cache/jfold`).
"""

import argparse
from pathlib import Path

import equinox as eqx
import jax
import numpy as np

from jfold import pretrained
from jfold.featurize import Chain, System
from jfold.featurize.views import esmfold2 as view
from jfold.models import esmfold2

LOOPS = 20  # upstream's `fold()` default: the first trunk pass, then 20 recycles


@eqx.filter_jit
def predict(model, features, key, num_samples: int):
    """The trunk (ESMC first, inside `embed`), then `num_samples` structures from the
    diffusion sampler."""
    k_trunk, k_sample = jax.random.split(key)
    inputs, state = model.run_trunk(features, num_passes=LOOPS + 1, key=k_trunk)
    sampler = model.structure_sampler(features, inputs, state)  # upstream's sampler settings
    return jax.vmap(sampler.sample)(jax.random.split(k_sample, num_samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sequences", nargs="+", help="one protein sequence per chain")
    parser.add_argument("--msa", nargs="*", type=Path, help="an a3m file per chain")
    parser.add_argument("--model", default="esmfold2_experimental_fast_base300m")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("prediction.pdb"))
    args = parser.parse_args()
    msas = args.msa or [None] * len(args.sequences)
    if len(msas) != len(args.sequences):
        parser.error("give one --msa file per sequence")

    # 1. The system: chains with their sequences and (optional) MSAs.
    system = System(
        tuple(
            Chain(id=chr(ord("A") + i), sequence=seq, msa=msa and msa.read_text())
            for i, (seq, msa) in enumerate(zip(args.sequences, msas, strict=True))
        )
    )

    # 2. The model, with its ESMC language model.
    model = pretrained.load(args.model)

    # 3. Features: ESMFold2's own featurization, then the model's feature vectors.
    features = esmfold2.features(view.features(system))

    # 4. Predict.
    samples = predict(model, features, jax.random.key(args.seed), args.samples)
    sites = view.atom_sites(system)  # what each atom is; padding atoms follow
    args.out.write_text(sites.to_pdb(np.asarray(samples)[:, : len(sites)]))
    print(f"wrote {args.samples} samples to {args.out}")


if __name__ == "__main__":
    main()
