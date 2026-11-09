"""Protenix releases (`NAMES`: mini, tiny, base v1.0.0, base 2025-06-30, v2), as `protenix
predict` builds them; and helpers shared by the Protenix-lineage recipes (Mol-1, OpenDDE)."""

import shutil
import urllib.request
from pathlib import Path

import equinox as eqx

from jfold.pretrained import Release


def download(url: str, path: Path) -> None:
    """`url` to `path`, through a partial file so an interrupted download is not mistaken for
    a checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as f:
        shutil.copyfileobj(response, f)
    partial.rename(path)


def load_state(model, checkpoint: Path, *, strict: bool = True):
    """Load a Protenix-lineage training checkpoint (`{"model": state}`, keys possibly with
    DDP's `module.` prefix) into `model` on the CPU, as the upstream runners do; returns it in
    eval mode."""
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=strict)
    return model.eval().cpu()


def deep_update(d: dict, u: dict) -> dict:
    """Merge `u` into `d` recursively (in place)."""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def configs(name: str):
    """The inference configs of `protenix predict --model_name name`: the base, data and
    inference defaults, then the model's own settings (`configs_model_type`)."""
    from ml_collections.config_dict import ConfigDict
    from protenix.config import parse_configs
    from protenix.configs.configs_base import configs as configs_base
    from protenix.configs.configs_data import data_configs
    from protenix.configs.configs_inference import inference_configs
    from protenix.configs.configs_model_type import model_configs

    configs = parse_configs(
        configs={**configs_base, "data": data_configs, **inference_configs, "model_name": name},
        fill_required_with_null=True,
    )
    configs.update(ConfigDict(model_configs[name]))
    configs.use_deepspeed_evo_attention = False
    return configs


def _fetch(name: str):
    def fetch(directory: Path) -> None:
        from protenix.web_service.dependency_url import URL

        download(URL[name], directory / f"{name}.pt")

    return fetch


def _build(name: str):
    def build(checkpoint: Path):
        """As `protenix predict`'s `InferenceRunner` builds and loads it, on the CPU."""
        from protenix.model.protenix import Protenix

        cfg = configs(name)
        return load_state(Protenix(cfg), checkpoint, strict=cfg.load_strict)

    return build


def _convert(model) -> eqx.Module:
    from jfold.convert import protenix

    return protenix.structure_model(model)


def _release(name: str) -> Release:
    return Release(
        upstream="protenix",
        source=f"protenix.web_service.dependency_url.URL[{name!r}]",
        filename=f"{name}.pt",
        fetch=_fetch(name),
        build=_build(name),
        convert=_convert,
    )


# jfold name -> upstream model name (`protenix predict --model_name`).
NAMES = {
    "protenix_mini": "protenix_mini_default_v0.5.0",
    "protenix_tiny": "protenix_tiny_default_v0.5.0",
    "protenix_base": "protenix_base_default_v1.0.0",
    "protenix_base_20250630": "protenix_base_20250630_v1.0.0",
    "protenix_v2": "protenix-v2",
}
RELEASES = {name: _release(upstream) for name, upstream in NAMES.items()}
