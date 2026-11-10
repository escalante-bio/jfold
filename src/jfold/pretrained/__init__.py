"""Released models: converted from the upstream checkpoint once, then loaded from a
`jfold.io` copy.

`load(name)` loads a released model by name (`"boltz2"`, `"protenix_mini"`,
`"esmfold2_experimental_fast"`, ...; `releases()` lists them). A structure model comes
with its upstream sampler settings (`model.sampling`). On first use, the converted
copy is downloaded from `HUB_REPO`; failing that, the upstream checkpoint is fetched, the
upstream torch model built as the model's own inference builds it (`upstream(name)`, also
the checkpoint tests' reference), converted, and saved with provenance metadata. Later
loads read that copy and need neither torch nor the upstream package.

Each model's recipe is a `Release` in the `RELEASES` of a `jfold.pretrained.recipes`
module (`MODULES`).

The cache is `$JFOLD_CACHE_DIR` (default `~/.cache/jfold`): `<cache>/<name>/` holds the
upstream checkpoint (unless one is passed) and `<cache>/<name>/jfold/` the converted
model.

A part several models share (ESMFold2's ESMC) is a release of its own, stored once: a
model's copy is saved without it (`Release.detach`) and completed from it on loading
(`Release.assemble`). `upload` publishes a saved copy to `HUB_REPO`.
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
    """How to get one released model."""

    upstream: str  # the upstream package
    source: str  # where its checkpoint comes from
    filename: str  # the checkpoint's file (or directory) name
    fetch: Callable[[Path], None]  # download the checkpoint into a directory
    build: Callable[[Path], Any]  # the upstream model, as its inference builds it
    convert: Callable[[Any], eqx.Module]
    # After loading: completes the model from other releases (`assemble(model, load)`,
    # `load(name)` loading one), e.g. ESMFold2's shared ESMC.
    assemble: Callable[[eqx.Module, Callable[[str], eqx.Module]], eqx.Module] | None = None
    # Before saving: removes what `assemble` attaches, so it is stored only once.
    detach: Callable[[eqx.Module], eqx.Module] | None = None
    # The licence notice the converted copy carries (`LICENSE` beside it), where the
    # upstream licence requires one with redistributed copies.
    license: str | None = None


# The `jfold.pretrained.recipes` modules whose `RELEASES` define the released models. They
# import no upstream code until a model is built.
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
# The Hugging Face repository of converted models: `<name>/model.json` and
# `<name>/weights.safetensors` for each.
HUB_REPO = os.environ.get("JFOLD_HUB_REPO", "escalante-bio/jfold")
RELEASES: dict[str, Release] = {}  # every module's, once `releases()` has imported them


def cache_dir() -> Path:
    return Path(os.environ.get("JFOLD_CACHE_DIR", "~/.cache/jfold")).expanduser()


def releases() -> dict[str, Release]:
    """Every released model's recipe, by name."""
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
    """The upstream checkpoint: `path` if given, else `<cache>/<name>/<file>`; fetched
    into its directory if missing."""
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
    """The upstream torch model, built as its own inference builds it."""
    return _release(name).build(checkpoint(name, checkpoint_path, cache=cache))


def _from_hub(name: str, converted: Path) -> bool:
    """Download `name`'s converted copy from `HUB_REPO` into `converted`; False if the
    repository doesn't have it (or can't be reached, with a warning)."""
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
    """The converted model `name`, from `<cache>/<name>/jfold/`. If that is missing, it is
    downloaded from `HUB_REPO` (with `hub`), or else converted from the upstream checkpoint
    (`checkpoint_path`, or the fetched one) and saved there; `refresh` reconverts. A cached
    copy this jfold cannot read (saved by an older one) is downloaded again."""
    release = _release(name)
    root = Path(cache).expanduser() if cache is not None else cache_dir()
    converted = root / name / "jfold"
    missing = not (converted / "model.json").exists()
    if refresh or (missing and not (hub and _from_hub(name, converted))):
        path = checkpoint(name, checkpoint_path, cache=cache)
        # Converted on the host: the copy is only saved, and `io.load` then moves the
        # weights to the device in one transfer.
        with jax.default_device(jax.devices("cpu")[0]):
            model = release.convert(release.build(path))
        metadata = io.provenance(
            name, checkpoint=path, source=release.source, upstream=release.upstream
        )
        save(name, model, metadata=metadata, cache=cache)
        del model  # before loading the saved copy, so only one is ever in memory
    try:
        model = io.load(converted)
    except ValueError as error:  # a copy from an older jfold
        if not (hub and _from_hub(name, converted)):
            raise
        warnings.warn(f"{converted}: {error}; replaced with {HUB_REPO}'s copy")
        model = io.load(converted)
    if release.assemble is not None:
        model = release.assemble(model, lambda other: load(other, cache=cache, hub=hub))
    return model


def save(name: str, model: eqx.Module, *, metadata: dict, cache: str | Path | None = None) -> Path:
    """Write `model` as `name`'s converted copy, `<cache>/<name>/jfold/`, without the parts
    it shares with other releases (`Release.detach`), with its licence notice
    (`Release.license`); returns the directory."""
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
    """Publish the saved copies of `names` (`<cache>/<name>/jfold/`, as `save` or `load`
    wrote them, licence notice included) to `HUB_REPO` as `<name>/`, in one commit. Needs
    write access."""
    from huggingface_hub import CommitOperationAdd, HfApi

    root = Path(cache).expanduser() if cache is not None else cache_dir()
    operations = []
    for name in names:
        release = _release(name)
        converted = root / name / "jfold"
        io.read_structure(converted)  # a copy this jfold reads
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
