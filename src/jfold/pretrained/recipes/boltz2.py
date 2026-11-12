"""Boltz-2 (`boltz2_conf.ckpt`), loaded as `boltz predict` does."""

from dataclasses import asdict
from pathlib import Path

import equinox as eqx

from jfold.pretrained import Release


def _fetch(directory: Path) -> None:
    import boltz.main

    boltz.main.download_boltz2(directory)


def _build(checkpoint: Path):
    """Load as `boltz predict` does, without kernels (the converter reads plain torch)."""
    import boltz.main as bm
    from boltz.model.models.boltz2 import Boltz2

    return Boltz2.load_from_checkpoint(
        checkpoint,
        strict=True,
        map_location="cpu",
        predict_args={"recycling_steps": 3, "sampling_steps": 200, "diffusion_samples": 1},
        diffusion_process_args=asdict(bm.Boltz2DiffusionParams()),
        ema=False,
        use_kernels=False,
        pairformer_args=asdict(bm.PairformerArgsV2()),
        msa_args=asdict(
            bm.MSAModuleArgs(subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True)
        ),
    ).eval()


def _convert(model) -> eqx.Module:
    from jfold.convert import boltz2

    return boltz2.structure_model(model)


RELEASES = {
    "boltz2": Release(
        upstream="boltz",
        source="boltz.main.download_boltz2 (boltz2_conf.ckpt)",
        filename="boltz2_conf.ckpt",
        fetch=_fetch,
        build=_build,
        convert=_convert,
    ),
}
