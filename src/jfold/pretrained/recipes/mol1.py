"""Recipes for Mol-1 (`mol-1-base`) and Mol-1-constraint (`mol-1-constraint`)."""

import copy
from pathlib import Path
from unittest import mock

import equinox as eqx
import numpy as np

from jfold.pretrained import Release
from jfold.pretrained.recipes.protenix import deep_update, download, load_state

NAME = "mol-1-base"
CONSTRAINT_NAME = "mol-1-constraint"


def _fetcher(name: str):
    def fetch(directory: Path) -> None:
        from mol.data.sources import checkpoint_url

        download(checkpoint_url(name), directory / f"{name}.pt")

    return fetch


def _configs(*settings: dict):
    """Return Mol-1's default configs updated with each of `settings` in turn.

    The triangle kernels are set to plain PyTorch because the converter reads torch
    modules."""
    from mol.config.base import configs as base_configs
    from mol.config.data import data_configs
    from mol.config.inference import inference_configs
    from mol.config.parser import parse_configs

    configs = copy.deepcopy({**base_configs, "data": data_configs, **inference_configs})
    for update in settings:
        deep_update(configs, copy.deepcopy(update))
    configs = parse_configs(configs, arg_str=None, fill_required_with_null=True)
    configs.triangle_multiplicative = "torch"
    configs.triangle_attention = "torch"
    return configs


def _instantiate(cls, configs):
    """Return `cls(configs)` without the slow truncated-normal init, since the checkpoint
    overwrites every parameter."""
    from mol.model.triangular import layers

    fast = mock.Mock(std=lambda **_: 1.0, rvs=lambda size, **_: np.zeros(size))
    with mock.patch.object(layers, "truncnorm", fast):
        return cls(configs)


def _build(checkpoint: Path):
    """Build and load Mol-1 on the CPU the way `mol pred` does."""
    from mol.config.model_type import model_configs
    from mol.model.mol import Mol

    configs = _configs(model_configs[NAME])
    return load_state(_instantiate(Mol, configs), checkpoint, strict=configs.load_strict)


def _build_constraint(checkpoint: Path):
    """Build and load Mol-1-constraint on the CPU the way `mol cpred` does.

    Upstream's loader checks the checkpoint's constraint metadata against the config and
    loads every tensor strictly."""
    import torch
    from mol.constraint import checkpoint as constraint_checkpoint
    from mol.constraint.config import DEFAULT_CONFIG, constraint_model_configs
    from mol.constraint.model import ConstraintMol

    configs = _configs(constraint_model_configs()[CONSTRAINT_NAME], DEFAULT_CONFIG)
    model = _instantiate(ConstraintMol, configs)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = constraint_checkpoint.contact_checkpoint_state(saved, configs)
    constraint_checkpoint.load_model_state(model, state)
    return model.eval().cpu()


def _convert(model) -> eqx.Module:
    from jfold.convert import protenix

    return protenix.structure_model(model)


RELEASES = {
    "mol1_base": Release(
        upstream="molence",
        source=f"mol.data.sources.checkpoint_url({NAME!r})",
        filename=f"{NAME}.pt",
        fetch=_fetcher(NAME),
        build=_build,
        convert=_convert,
    ),
    "mol1_constraint": Release(
        upstream="molence",
        source=f"mol.data.sources.checkpoint_url({CONSTRAINT_NAME!r})",
        filename=f"{CONSTRAINT_NAME}.pt",
        fetch=_fetcher(CONSTRAINT_NAME),
        build=_build_constraint,
        convert=_convert,
    ),
}
