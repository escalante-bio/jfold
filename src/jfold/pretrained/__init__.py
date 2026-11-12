"""Load released models, converting them from the upstream checkpoints when needed.

`load(name)` downloads the converted model from `HUB_REPO`, or else builds the upstream torch
model from its checkpoint, converts it and caches it in `$JFOLD_CACHE_DIR` (default
`~/.cache/jfold`). `releases()` lists the names, and each model's recipe is a `Release` in
`jfold.pretrained.recipes`.
"""

import importlib
import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax

from jfold import io


@dataclass(frozen=True)
class Release:
    """How to fetch, build and convert one released model.

    `assemble` and `detach` add and remove the components shared with other releases, and
    `license` is saved as `LICENSE` next to the converted model."""

    upstream: str
    source: str
    filename: str
    fetch: Callable[[Path], None]
    build: Callable[[Path], Any]
    convert: Callable[[Any], eqx.Module]
    assemble: Callable[[eqx.Module, Callable[[str], eqx.Module]], eqx.Module] | None = None
    detach: Callable[[eqx.Module], eqx.Module] | None = None
    license: str | None = None


# Recipe modules must not import upstream code until a model is built.
MODULES = (
    "boltz2",
    "protenix",
    "mol1",
    "opendde",
    "openfold3",
    "promera",
    "bioemu2",
    "esmfold2",
    "atlasfold",
)
HUB_REPO = os.environ.get("JFOLD_HUB_REPO", "escalante-bio/jfold")
RELEASES: dict[str, Release] = {}  # Filled in by `releases()`.


def cache_dir() -> Path:
    return Path(os.environ.get("JFOLD_CACHE_DIR", "~/.cache/jfold")).expanduser()


def releases() -> dict[str, Release]:
    """Return every released model's recipe by name."""
    for module in MODULES:
        recipes = importlib.import_module(f"jfold.pretrained.recipes.{module}")
        for name, release in recipes.RELEASES.items():
            RELEASES.setdefault(name, release)
    return RELEASES


def _release(name: str) -> Release:
    if name not in RELEASES and name not in releases():
        raise KeyError(f"unknown model {name!r}; released: {sorted(RELEASES)}")
    return RELEASES[name]


def checkpoint(
    name: str, path: str | Path | None = None, *, cache: str | Path | None = None
) -> Path:
    """Return the upstream checkpoint path, `<cache>/<name>/<filename>` by default, downloading
    it if missing."""
    release = _release(name)
    root = Path(cache).expanduser() if cache is not None else cache_dir()
    path = Path(path).expanduser() if path is not None else root / name / release.filename
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        release.fetch(path.parent)
    return path


def upstream(
    name: str, *, checkpoint_path: str | Path | None = None, cache: str | Path | None = None
):
    """Build the upstream torch model the same way its inference code does."""
    return _release(name).build(checkpoint(name, checkpoint_path, cache=cache))


def _from_hub(name: str, converted: Path) -> bool:
    import shutil

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
    )

    staging = converted.parent / ".hub"
    try:
        snapshot_download(HUB_REPO, allow_patterns=[f"{name}/*"], local_dir=staging)
    except (RepositoryNotFoundError, GatedRepoError):
        return False
    except (HfHubHTTPError, LocalEntryNotFoundError, OSError) as error:
        warnings.warn(f"{HUB_REPO} unreachable ({error}); converting {name} locally")
        return False
    found = (staging / name / "model.json").exists()
    if found:
        shutil.rmtree(converted, ignore_errors=True)
        (staging / name).rename(converted)
    shutil.rmtree(staging, ignore_errors=True)
    return found


def load(
    name: str,
    *,
    checkpoint_path: str | Path | None = None,
    cache: str | Path | None = None,
    refresh: bool = False,
    hub: bool = True,
) -> eqx.Module:
    """Load the converted model `name`, downloading it from `HUB_REPO` (if `hub`) or converting
    it on first use, or again if `refresh`."""
    release = _release(name)
    root = Path(cache).expanduser() if cache is not None else cache_dir()
    converted = root / name / "jfold"
    missing = not (converted / "model.json").exists()
    if refresh or (missing and not (hub and _from_hub(name, converted))):
        path = checkpoint(name, checkpoint_path, cache=cache)
        # Convert on the CPU, since `io.load` moves the saved weights to the device later.
        with jax.default_device(jax.devices("cpu")[0]):
            model = release.convert(release.build(path))
        metadata = io.provenance(
            name, checkpoint=path, source=release.source, upstream=release.upstream
        )
        save(name, model, metadata=metadata, cache=cache)
        del model  # Keep only one copy of the model in memory.
    try:
        model = io.load(converted)
    except ValueError as error:
        if not (hub and _from_hub(name, converted)):
            raise
        warnings.warn(f"{converted}: {error}; replaced with {HUB_REPO}'s copy")
        model = io.load(converted)
    if release.assemble is not None:
        model = release.assemble(model, lambda other: load(other, cache=cache, hub=hub))
    return model


def save(name: str, model: eqx.Module, *, metadata: dict, cache: str | Path | None = None) -> Path:
    """Save `model` to `<cache>/<name>/jfold/` without its shared components and with its
    licence, and return the directory."""
    release = _release(name)
    root = Path(cache).expanduser() if cache is not None else cache_dir()
    converted = root / name / "jfold"
    if release.detach is not None:
        model = release.detach(model)
    io.save(model, converted, metadata=metadata)
    if release.license is not None:
        (converted / "LICENSE").write_text(release.license)
    return converted


def upload(*names: str, cache: str | Path | None = None, message: str | None = None) -> None:
    """Upload the saved models `names` to `HUB_REPO` in one commit."""
    from huggingface_hub import CommitOperationAdd, HfApi

    root = Path(cache).expanduser() if cache is not None else cache_dir()
    operations = []
    for name in names:
        release = _release(name)
        converted = root / name / "jfold"
        io.read_structure(converted)
        files = ["model.json", "weights.safetensors"]
        if release.license is not None:
            if (converted / "LICENSE").read_text() != release.license:
                raise ValueError(f"{converted}: LICENSE differs from the release's")
            files.append("LICENSE")
        operations += [
            CommitOperationAdd(path_in_repo=f"{name}/{file}", path_or_fileobj=converted / file)
            for file in files
        ]
    HfApi().create_commit(
        repo_id=HUB_REPO,
        operations=operations,
        commit_message=message or f"Upload {', '.join(names)}",
    )
