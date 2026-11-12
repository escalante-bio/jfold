"""Save and load converted models without pickle.

A saved model is a directory with `weights.safetensors`, holding every array by its path in
the model, and `model.json`, holding the metadata and the model's classes and fields.
Loading imports classes only from jfold, equinox and any extra `namespaces` passed in.
"""

import dataclasses
import datetime
import hashlib
import importlib
import importlib.metadata
import json
import types
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import numpy as np
from safetensors.numpy import load_file, save_file

FORMAT_VERSION = 1
_STRUCTURE, _WEIGHTS = "model.json", "weights.safetensors"
_NAMESPACES = ("jfold.", "equinox.")
# Maps class names in older files to their current names.
MOVED = {
    "jfold.blocks.confidence:SinglesToPair": "jfold.nn:SinglesToPair",
    "jfold.models.boltz2:TrunkMSA": "jfold.blocks.msa:ResidualMSA",
    "jfold.models.promera:PromeraMSA": "jfold.blocks.msa:ResidualMSA",
    "jfold.blocks.pairformer:Mol1Pairformer": "jfold.models.mol1:Mol1Pairformer",
    "jfold.sampling.config:SamplingConfig": "jfold.sampling.config:UpstreamSampler",
}


def save(
    model: eqx.Module,
    path: str | Path,
    *,
    metadata: dict | None = None,
    namespaces: tuple[str, ...] = (),
) -> None:
    """Write `model` to the directory `path`, with `namespaces` naming any extra module prefixes its
    classes come from."""
    path = Path(path)
    arrays: dict[str, np.ndarray] = {}
    structure = {
        "format_version": FORMAT_VERSION,
        "jfold_version": _jfold_version(),
        "metadata": metadata or {},
        "model": _encode(model, "", arrays, _NAMESPACES + tuple(namespaces)),
    }
    path.mkdir(parents=True, exist_ok=True)
    save_file(arrays, str(path / _WEIGHTS))
    (path / _STRUCTURE).write_text(json.dumps(structure, indent=1))


def provenance(
    model: str,
    *,
    checkpoint: str | Path,
    source: str,
    upstream: str,
    **extra: Any,
) -> dict:
    """Return the standard `save` metadata: model type, checkpoint source and SHA-256, upstream
    package version and date."""
    return {
        "model": model,
        "checkpoint": {"source": source, "sha256": _sha256(Path(checkpoint).expanduser())},
        "upstream": {"package": upstream, "version": importlib.metadata.version(upstream)},
        "converted_at": datetime.datetime.now(datetime.UTC).date().isoformat(),
        **extra,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for file in files:
        if path.is_dir():
            digest.update(str(file.relative_to(path)).encode() + b"\0")
        with file.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load(path: str | Path, *, namespaces: tuple[str, ...] = ()) -> eqx.Module:
    """Load the model saved at `path` onto JAX's default device, with the `namespaces` given to
    `save`."""
    allowed = _NAMESPACES + tuple(namespaces)
    model = _decode(
        read_structure(path)["model"], load_file(str(Path(path) / _WEIGHTS)), allowed
    )
    arrays, rest = eqx.partition(model, lambda x: isinstance(x, np.ndarray))
    # One transfer: copying a large model array by array fragments the device's memory.
    return eqx.combine(jax.device_put(arrays), rest)


def read_structure(path: str | Path) -> dict:
    """Read `model.json`, raising ValueError for an unsupported format version."""
    structure = json.loads((Path(path) / _STRUCTURE).read_text())
    if structure.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{path}: format version {structure.get('format_version')}, expected {FORMAT_VERSION}"
        )
    return structure


def _jfold_version() -> str | None:
    try:
        return importlib.metadata.version("jfold")
    except importlib.metadata.PackageNotFoundError:
        return None


def _name(obj, allowed: tuple[str, ...]) -> str:
    module = obj.__module__
    if module.startswith("equinox"):
        for public in ("equinox.nn", "equinox"):
            if getattr(importlib.import_module(public), obj.__qualname__, None) is obj:
                module = public
                break
    name = f"{module}:{obj.__qualname__}"
    if not module.startswith(allowed) or "<" in obj.__qualname__:
        raise TypeError(f"cannot save {name}: not a module-level name under {allowed}")
    return name


def _resolve(name: str, allowed: tuple[str, ...]):
    name = MOVED.get(name, name)
    module, _, qualname = name.partition(":")
    if not module.startswith(allowed):
        raise ValueError(f"refusing to load {name}: outside {allowed}")
    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            raise ValueError(f"{name} no longer exists")
    return obj


def _encode(x: Any, path: str, arrays: dict[str, np.ndarray], allowed: tuple[str, ...]) -> Any:
    child = lambda key: f"{path}.{key}" if path else str(key)
    match x:
        case jax.Array() | np.ndarray():
            if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
                raise TypeError(f"{path}: cannot save a PRNG key")
            arrays[path] = np.array(x)
            return {"array": path, "shape": list(x.shape), "dtype": str(x.dtype)}
        case eqx.Module():
            fields = {
                f.name: _encode(getattr(x, f.name), child(f.name), arrays, allowed)
                for f in dataclasses.fields(x)
            }
            return {"module": _name(type(x), allowed), "fields": fields}
        case None | bool() | int() | float() | str():
            return x
        case np.generic():
            return x.item()
        case tuple():
            return {"tuple": [_encode(v, child(i), arrays, allowed) for i, v in enumerate(x)]}
        case list():
            return {"list": [_encode(v, child(i), arrays, allowed) for i, v in enumerate(x)]}
        case dict() if all(isinstance(k, str) for k in x):
            return {"dict": {k: _encode(v, child(k), arrays, allowed) for k, v in x.items()}}
        case types.FunctionType():
            return {"function": _name(x, allowed)}
    raise TypeError(f"{path}: cannot save a {type(x).__name__}")


def _has_default(field: dataclasses.Field) -> bool:
    missing = dataclasses.MISSING
    return field.default is not missing or field.default_factory is not missing


def _decode(node: Any, arrays: dict[str, np.ndarray], allowed: tuple[str, ...]) -> Any:
    match node:
        case {"array": path, "shape": shape, "dtype": dtype}:
            value = arrays[path]
            if list(value.shape) != shape or str(value.dtype) != dtype:
                raise ValueError(f"{path}: weights do not match the structure")
            return value
        case {"module": name, "fields": fields}:
            cls = _resolve(name, allowed)
            by_name = {f.name: f for f in dataclasses.fields(cls)}
            missing = {n for n, f in by_name.items() if n not in fields and not _has_default(f)}
            extra = set(fields) - set(by_name)
            if missing or extra:
                raise ValueError(
                    f"{name}: fields changed (missing {sorted(missing)}, unknown {sorted(extra)})"
                )
            # Bypass __init__, as equinox does when unflattening a module.
            module = object.__new__(cls)
            for field, f in by_name.items():
                if field in fields:
                    value = _decode(fields[field], arrays, allowed)
                elif f.default is not dataclasses.MISSING:
                    value = f.default
                else:
                    value = f.default_factory()
                object.__setattr__(module, field, value)
            return module
        case {"tuple": items}:
            return tuple(_decode(v, arrays, allowed) for v in items)
        case {"list": items}:
            return [_decode(v, arrays, allowed) for v in items]
        case {"dict": items}:
            return {k: _decode(v, arrays, allowed) for k, v in items.items()}
        case {"function": name}:
            return _resolve(name, allowed)
        case dict():
            raise ValueError(f"unrecognised node {sorted(node)}")
        case _:
            return node
