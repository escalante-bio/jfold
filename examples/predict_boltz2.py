"""Fold a protein (or a complex) with Boltz-2, without torch: load the converted model,
featurize with `jfold.featurize`, run the trunk, sample structures, score them with the
confidence head and write the best one as a PDB file.

    uv run python examples/predict_boltz2.py MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
    uv run python examples/predict_boltz2.py SEQ_A SEQ_B --msa a.a3m b.a3m --samples 5 --out complex.pdb

The first run downloads the converted weights (to `$JFOLD_CACHE_DIR`, `~/.cache/jfold`).
"""

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jfold import pretrained
from jfold.featurize import Chain, System
from jfold.featurize.views import boltz2 as view
from jfold.models import boltz2

RECYCLES = 3  # upstream's default: 1 + 3 trunk passes


@eqx.filter_jit
def predict(model, features, key, num_samples: int):
    """Trunk, then `num_samples` structures from the diffusion sampler and each one's pLDDT
    logits. The model is an argument so that the compiled function is reused."""
    k_trunk, k_sample, k_confidence = jax.random.split(key, 3)
    inputs, state = model.run_trunk(features, num_passes=RECYCLES + 1, key=k_trunk)
    sampler = model.structure_sampler(features, inputs, state)  # upstream's sampler settings
    coords = jax.vmap(sampler.sample)(jax.random.split(k_sample, num_samples))
    plddt_logits = jax.vmap(
        lambda x: model.confidence_logits(features, inputs, state, x, key=k_confidence).plddt
    )(coords)
    return coords, plddt_logits


def expected_plddt(logits: jax.Array) -> jax.Array:
    """pLDDT in [0, 1]: the expectation over the head's evenly spaced bins."""
    bins = logits.shape[-1]
    return jax.nn.softmax(logits, -1) @ ((jnp.arange(bins) + 0.5) / bins)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sequences", nargs="+", help="one protein sequence per chain")
    parser.add_argument("--msa", nargs="*", type=Path, help="an a3m file per chain")
    parser.add_argument("--samples", type=int, default=5)
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

    # 2. Features: Boltz-2's own featurization (`key` poses the reference conformers, as
    #    upstream does), then the model's feature vectors.
    key_features, key_predict = jax.random.split(jax.random.key(args.seed))
    f = view.features(system, key=key_features)
    features = boltz2.features(f)

    # 3. The model: the converted checkpoint, with upstream's sampler settings.
    model = pretrained.load("boltz2")

    # 4. Predict, keep the most confident sample.
    coords, plddt_logits = predict(model, features, key_predict, args.samples)
    plddt = np.asarray(expected_plddt(plddt_logits))  # [samples, tokens]
    best = int(plddt.mean(-1).argmax())
    for i, p in enumerate(plddt.mean(-1)):
        print(f"sample {i}: mean pLDDT {100 * p:.1f}{'  <- best' if i == best else ''}")
    # The view says what each atom is; padding atoms follow the real ones.
    sites = view.atom_sites(system)
    best_coords = np.asarray(coords[best])[: len(sites)]
    args.out.write_text(sites.to_pdb(best_coords, b_factor=100 * plddt[best][sites.token]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
