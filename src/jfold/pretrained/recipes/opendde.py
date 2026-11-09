"""OpenDDE (`opendde_v1`, and its antibody-antigen weights `opendde_abag.pt`), as
`opendde`'s inference runner builds it."""

from pathlib import Path

import equinox as eqx

from jfold.pretrained import Release
from jfold.pretrained.recipes.protenix import load_state

ARCHITECTURE = "opendde_v1"


def configs():
    """`opendde_v1`'s inference configs (`build_inference_config`), with the plain-PyTorch
    triangle kernels and without the inference-time fusions (the converter reads plain torch
    modules)."""
    from opendde.config.inference import build_inference_config

    configs = build_inference_config(model_name=ARCHITECTURE, fill_required_with_null=True)
    configs.triangle_multiplicative = configs.triangle_attention = "torch"
    configs.enable_diffusion_shared_vars_cache = False
    configs.enable_efficient_fusion = False
    return configs


def _fetch(filename: str):
    def fetch(directory: Path) -> None:
        from opendde.config.dependency_url import dependency_url
        from opendde.utils.download import download_from_url

        download_from_url(dependency_url(filename), str(directory / filename))

    return fetch


def _build(checkpoint: Path):
    """As the inference runner builds and loads it, on the CPU."""
    from opendde.model.opendde import OpenDDE

    cfg = configs()
    return load_state(OpenDDE(cfg), checkpoint, strict=cfg.load_strict)


def _convert(model) -> eqx.Module:
    from jfold.convert import opendde

    return opendde.structure_model(model)


def _release(filename: str) -> Release:
    return Release(
        upstream="opendde",
        source=f"opendde.config.dependency_url.dependency_url({filename!r})",
        filename=filename,
        fetch=_fetch(filename),
        build=_build,
        convert=_convert,
    )


RELEASES = {
    "opendde_v1": _release("opendde.pt"),
    "opendde_abag": _release("opendde_abag.pt"),
}
