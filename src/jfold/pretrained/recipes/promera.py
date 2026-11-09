"""ProMERA (`promera_2606.ckpt` from `bjing-mit/promera`), as `python -m promera` loads it."""

import os
from pathlib import Path

import equinox as eqx

from jfold.pretrained import Release

REPO, FILENAME = "bjing-mit/promera", "promera_2606.ckpt"


def _fetch(directory: Path) -> None:
    """Into the Hugging Face cache, linked from `directory`."""
    from huggingface_hub import hf_hub_download

    os.symlink(hf_hub_download(REPO, FILENAME), directory / FILENAME)


def _build(checkpoint: Path):
    """As `python -m promera` builds it: the package's model config (`model/config.yaml`),
    then the checkpoint's EMA weights (`promera.utils.load_weights`), here loaded strictly
    so that every tensor is accounted for. Building and loading run on the CPU; only the
    forward pass needs cuequivariance's CUDA kernels."""
    import torch
    from omegaconf import OmegaConf
    from promera.model import model as promera_model

    cfg = OmegaConf.load(Path(promera_model.__file__).parent / "config.yaml")
    model = promera_model.PromeraModel(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state["ema"]["params"] if "ema" in state else state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.eval()


def _convert(model) -> eqx.Module:
    from jfold.convert import promera as convert

    return convert.structure_model(model)


RELEASES = {
    "promera": Release(
        upstream="promera",
        source=f"huggingface:{REPO}/{FILENAME}",
        filename=FILENAME,
        fetch=_fetch,
        build=_build,
        convert=_convert,
    ),
}
