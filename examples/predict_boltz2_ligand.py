"""Predict a protein-small molecule complex with Boltz-2: featurize with Boltz-2's own
pipeline (torch; it builds the ligand's conformer with RDKit), convert the feature dict to
jfold's features, predict with jfold's converted model, and write the most confident
sample as mmCIF with Boltz's own writer.

`jfold.featurize` takes proteins only; anything else (ligands, nucleic acids, modified
residues, paired MSAs) goes through the upstream featurization as here. The upstream
package comes with the `reference` dependency group:

    uv sync --group reference
    uv run python examples/predict_boltz2_ligand.py MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ --smiles "CC(=O)Oc1ccccc1C(=O)O"
    uv run python examples/predict_boltz2_ligand.py SEQUENCE --ccd ATP --msa protein.a3m --out complex.cif

Boltz's CCD and conformer library are read from `~/.boltz` (downloaded there on first use);
the converted weights from `$JFOLD_CACHE_DIR` (`~/.cache/jfold`).
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch

from jfold import pretrained
from jfold.models import boltz2

RECYCLES = 3  # upstream's default: 1 + 3 trunk passes
BOLTZ_CACHE = Path.home() / ".boltz"


def input_yaml(sequence: str, smiles: str | None, ccd: str | None, msa: Path | None) -> str:
    """Boltz's input: the protein as chain A, the ligand as chain B."""
    ligand = f"smiles: '{smiles}'" if smiles else f"ccd: {ccd}"
    return "\n".join(
        [
            "version: 1",
            "sequences:",
            "  - protein:",
            "      id: [A]",
            f"      sequence: {sequence}",
            f"      msa: {msa.resolve() if msa else 'empty'}",
            "  - ligand:",
            "      id: [B]",
            f"      {ligand}",
        ]
    )


def ccd_library(cache: Path) -> None:
    """Boltz-2's CCD and conformer library (`mols`), downloaded once into `cache` (without
    Boltz's torch checkpoints, which jfold does not read)."""
    import tarfile
    import urllib.request

    from boltz.main import MOL_URL

    if (cache / "mols").exists():
        return
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "mols.tar"
    if not archive.exists():
        print(f"downloading the CCD library to {archive}")
        urllib.request.urlretrieve(MOL_URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(cache, filter="data")


def upstream_features(yaml: str, directory: Path, seed: int):
    """Boltz-2's featurization of `yaml`, as `boltz predict` runs it: the batch (torch
    tensors, batch of one) and the processed structures' directory, for the writer."""
    import boltz.main as bm

    path = directory / "input.yaml"
    path.write_text(yaml)
    ccd_library(BOLTZ_CACHE)
    bm.process_inputs(
        data=bm.check_inputs(path),
        out_dir=directory,
        ccd_path=BOLTZ_CACHE / "ccd.pkl",
        mol_dir=BOLTZ_CACHE / "mols",
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
        mol_dir=BOLTZ_CACHE / "mols",
        num_workers=0,
        constraints_dir=processed / "constraints",
        template_dir=processed / "templates",
        extra_mols_dir=processed / "mols",
        override_method=None,
    )
    torch.manual_seed(seed)  # upstream draws the conformers' random poses from torch
    return next(iter(data.predict_dataloader())), processed / "structures"


@eqx.filter_jit
def predict(model, features, key, num_samples: int):
    """Trunk, then `num_samples` structures from the diffusion sampler and each one's pLDDT
    logits. The model is an argument, not a closure, so the compiled function is reused."""
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


def write_mmcif(batch, structures: Path, coords: np.ndarray, out: Path) -> None:
    """`coords` (`[A, 3]`, padded as the batch) written by Boltz's writer, which maps them
    back onto the processed structure (chains, residues, the ligand's atoms)."""
    from boltz.data.write.writer import BoltzWriter

    with tempfile.TemporaryDirectory() as tmp:
        writer = BoltzWriter(
            data_dir=structures, output_dir=tmp, output_format="mmcif", boltz2=True
        )
        prediction = {
            "exception": False,
            "coords": torch.tensor(coords)[None],  # one sample; the writer adds the batch
            "masks": batch["atom_pad_mask"],
        }
        writer.write_on_batch_end(None, None, prediction, None, batch, None, None)
        record = batch["record"][0]
        shutil.copy(Path(tmp) / record.id / f"{record.id}_model_0.cif", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sequence", help="the protein chain")
    ligand = parser.add_mutually_exclusive_group(required=True)
    ligand.add_argument("--smiles", help="the ligand as SMILES")
    ligand.add_argument("--ccd", help="the ligand as a CCD code, e.g. ATP")
    parser.add_argument("--msa", type=Path, help="an a3m file for the protein")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("complex.cif"))
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        # 1. Boltz-2's own featurization: torch tensors, batch of one.
        yaml = input_yaml(args.sequence, args.smiles, args.ccd, args.msa)
        batch, structures = upstream_features(yaml, Path(tmp), args.seed)

        # 2. The conversion: drop the batch axis, as numpy, then jfold's features.
        f = {k: v[0].numpy() for k, v in batch.items() if isinstance(v, torch.Tensor)}
        features = boltz2.features(f)

        # 3. The model: the converted checkpoint, with upstream's sampler settings.
        model = pretrained.boltz2()

        # 4. Predict, keep the most confident sample (pLDDT per token: residues, and each
        #    of the ligand's atoms).
        coords, plddt_logits = predict(model, features, jax.random.key(args.seed), args.samples)
        token_mask = f["token_pad_mask"].astype(bool)
        plddt = np.asarray(expected_plddt(plddt_logits))[:, token_mask]  # [samples, tokens]
        ligand = f["mol_type"][token_mask] == 3  # the NONPOLYMER tokens
        best = int(plddt.mean(-1).argmax())
        for i, p in enumerate(plddt):
            print(
                f"sample {i}: mean pLDDT {100 * p.mean():.1f}, ligand {100 * p[ligand].mean():.1f}"
                + ("  <- best" if i == best else "")
            )
        write_mmcif(batch, structures, np.asarray(coords[best]), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
