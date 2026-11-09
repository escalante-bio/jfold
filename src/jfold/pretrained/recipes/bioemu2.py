"""BioEmu2 (`bioemu2-preview` from `microsoft/bioemu2`), as `bioemu2`'s inference loads it.

The converted model's `sampling` is upstream's structure-prediction preset;
`jfold.convert.bioemu2.sampling(upstream("bioemu2"), "equilibrium")` is the equilibrium
one (they differ only in `step_scale`)."""

import os
from pathlib import Path

import equinox as eqx

from jfold.pretrained import Release

CHECKPOINT = "bioemu2-preview"


def _fetch(directory: Path) -> None:
    """Into the Hugging Face cache (`bioemu2.checkpoints.get_checkpoint_files`, which reuses
    any cached snapshot), the checkpoint and its config linked from `directory`."""
    from bioemu2.checkpoints import get_checkpoint_files

    files = get_checkpoint_files(CHECKPOINT)
    os.symlink(files.checkpoint_path, directory / "checkpoint.ckpt")
    os.symlink(files.config_path, directory / "config.yaml")


def _build(checkpoint: Path):
    """As `bioemu2.model.load_model_for_inference` builds it: the score model from the
    `config.yaml` beside the checkpoint, with its EMA weights, on the CPU in eval mode."""
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
