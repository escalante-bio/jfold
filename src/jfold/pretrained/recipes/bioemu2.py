"""BioEmu2 (`bioemu2-preview` from `microsoft/bioemu2`), loaded as `bioemu2`'s inference does.

`sampling` is upstream's structure-prediction preset. For the equilibrium preset, use
`jfold.convert.bioemu2.sampling(upstream("bioemu2"), "equilibrium")`."""

import os
from pathlib import Path

import equinox as eqx

from jfold.pretrained import Release

CHECKPOINT = "bioemu2-preview"


def _fetch(directory: Path) -> None:
    from bioemu2.checkpoints import get_checkpoint_files

    files = get_checkpoint_files(CHECKPOINT)
    os.symlink(files.checkpoint_path, directory / "checkpoint.ckpt")
    os.symlink(files.config_path, directory / "config.yaml")


def _build(checkpoint: Path):
    from bioemu2.checkpoints import resolve_checkpoint_files
    from bioemu2.model import _load_score_model

    files = resolve_checkpoint_files(checkpoint)
    model, _ = _load_score_model(
        files.checkpoint_path, files.config_path, fingerprint_required=False
    )
    return model.eval().requires_grad_(False)


def _convert(model) -> eqx.Module:
    from jfold.convert import bioemu2

    return bioemu2.bioemu2(model)


RELEASES = {
    "bioemu2": Release(
        upstream="bioemu2",
        source=f"huggingface:microsoft/bioemu2/checkpoints/{CHECKPOINT}",
        filename="checkpoint.ckpt",
        fetch=_fetch,
        build=_build,
        convert=_convert,
    ),
}
