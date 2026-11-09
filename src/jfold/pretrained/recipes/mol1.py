"""Mol-1 (`mol-1-base`), as `mol pred` builds it."""

import copy
from pathlib import Path
from unittest import mock

import equinox as eqx
import numpy as np

from jfold.pretrained import Release
from jfold.pretrained.recipes.protenix import deep_update, download, load_state

NAME = "mol-1-base"


def _fetch(directory: Path) -> None:
    from mol.data.sources import checkpoint_url

    download(checkpoint_url(NAME), directory / f"{NAME}.pt")


def _build(checkpoint: Path):
    """As `mol pred` builds and loads it (its base, data and inference defaults, then the
    model's own settings), on the CPU, with the plain-PyTorch triangle kernels (the
    converter reads plain torch modules)."""
    from mol.config.base import configs as base_configs
    from mol.config.data import data_configs
    from mol.config.inference import inference_configs
    from mol.config.model_type import model_configs
    from mol.config.parser import parse_configs
    from mol.model.mol import Mol
    from mol.model.triangular import layers

    configs = copy.deepcopy({**base_configs, "data": data_configs, **inference_configs})
    deep_update(configs, model_configs[NAME])
    configs = parse_configs(configs, arg_str=None, fill_required_with_null=True)
    configs.triangle_multiplicative = "torch"
    configs.triangle_attention = "torch"
    # The checkpoint overwrites every parameter; skip the slow truncated-normal init.
    fast = mock.Mock(std=lambda **_: 1.0, rvs=lambda size, **_: np.zeros(size))
    with mock.patch.object(layers, "truncnorm", fast):
        model = Mol(configs)
    return load_state(model, checkpoint, strict=configs.load_strict)


def _convert(model) -> eqx.Module:
    from jfold.convert import protenix

    return protenix.structure_model(model)


RELEASES = {
    "mol1_base": Release(
        upstream="molence",
        source=f"mol.data.sources.checkpoint_url({NAME!r})",
        filename=f"{NAME}.pt",
        fetch=_fetch,
        build=_build,
        convert=_convert,
    ),
}
