"""Saving and loading converted models without pickle.

A saved model is a directory of two files:

- `weights.safetensors`: every array, keyed by its path in the model
  (`trunk.pairformer.layers.tri_mul_out.linear_a.weight`). Readable with numpy
  alone (`safetensors.numpy.load_file`).
- `model.json`: the format version, metadata (`provenance` builds the standard
  fields: model type, source checkpoint and its hash, upstream version, date), and
  the model's structure: each module's class (`jfold.blocks.pairformer:PairformerLayer`) and
  fields, plain values (numbers, strings, tuples, `null`), module-level functions by
  name (`jfold.sampling.core:masked_centering`), and each array's shape and dtype.

Loading rebuilds the modules from the JSON and fills in the arrays by path. Nothing
on disk is a JAX or pickled Python object, so it does not depend on JAX's internal
types; it depends only on jfold's class and field names, and a mismatch is reported
by name. A class that moves or is renamed keeps loading from older files through
`MOVED`.
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
# Classes and functions a file may name: loading imports nothing else.
_NAMESPACES = ("jfold.", "equinox.")
# Names older files use -> where those classes are now. Saving always writes the current
# name; add an entry here when moving or renaming a class that saved models use.
MOVED = {
    "jfold.blocks.confidence:SinglesToPair": "jfold.nn:SinglesToPair",
    "jfold.models.boltz2:TrunkMSA": "jfold.blocks.msa:ResidualMSA",
    "jfold.models.promera:PromeraMSA": "jfold.blocks.msa:ResidualMSA",
    "jfold.blocks.pairformer:Mol1Pairformer": "jfold.models.mol1:Mol1Pairformer",
    "jfold.sampling.config:SamplingConfig": "jfold.sampling.config:UpstreamSampler",
}


def save(model: eqx.Module, path: str | Path, *, metadata: dict | None = None) -> None:
    """Write `model` to the directory `path` (created if needed). `metadata` must be
    JSON-serialisable."""
    path = Path(path)
    arrays: dict[str, np.ndarray] = {}
    structure = {
        "format_version": FORMAT_VERSION,
        "jfold_version": _jfold_version(),
        "metadata": metadata or {},
        "model": _encode(model, "", arrays),
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
    """The standard metadata for a converted model, for `save`:

    - `model`: its type, e.g. `"esmfold2-experimental"`.
    - `checkpoint`: `source`, where the weights came from (a Hugging Face repo and file,
      or a URL), and `sha256`, the hash of the local file (or of every file in a
      directory, by relative name) at `checkpoint`.
    - `upstream`: the upstream package's name and installed version.
    - `converted_at`: today's date (UTC).

    `extra` entries are added as they are."""
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


def load(path: str | Path) -> eqx.Module:
    """The model saved at `path`, its arrays on JAX's default device. The arrays are read
    on the host and moved in one `device_put` (array-by-array transfers of a very large
    model fragment the device's memory pool)."""
    model = _decode(read_structure(path)["model"], load_file(str(Path(path) / _WEIGHTS)))
    arrays, rest = eqx.partition(model, lambda x: isinstance(x, np.ndarray))
    return eqx.combine(jax.device_put(arrays), rest)  # plain values stay Python values


def read_structure(path: str | Path) -> dict:
    """The parsed `model.json` (format version, metadata, structure), checked for a
    format this jfold reads."""
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


def _name(obj) -> str:
    """`module:qualname`, preferring equinox's public modules to its private ones."""
    module = obj.__module__
    if module.startswith("equinox"):
        for public in ("equinox.nn", "equinox"):
            if getattr(importlib.import_module(public), obj.__qualname__, None) is obj:
                module = public
                break
    name = f"{module}:{obj.__qualname__}"
    if not module.startswith(_NAMESPACES) or "<" in obj.__qualname__:
        raise TypeError(f"cannot save {name}: not a module-level jfold or equinox name")
    return name


def _resolve(name: str):
    name = MOVED.get(name, name)
    module, _, qualname = name.partition(":")
    if not module.startswith(_NAMESPACES):
        raise ValueError(f"refusing to load {name}: outside {_NAMESPACES}")
    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            raise ValueError(f"{name} no longer exists")
    return obj


def _encode(x: Any, path: str, arrays: dict[str, np.ndarray]) -> Any:
    child = lambda key: f"{path}.{key}" if path else str(key)
    match x:
        case jax.Array() | np.ndarray():
            if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
                raise TypeError(f"{path}: cannot save a PRNG key")
            arrays[path] = np.array(x)
            return {"array": path, "shape": list(x.shape), "dtype": str(x.dtype)}
        case eqx.Module():
            fields = {
                f.name: _encode(getattr(x, f.name), child(f.name), arrays)
                for f in dataclasses.fields(x)
            }
            return {"module": _name(type(x)), "fields": fields}
        case None | bool() | int() | float() | str():
            return x
        case np.generic():
            return x.item()
        case tuple():
            return {"tuple": [_encode(v, child(i), arrays) for i, v in enumerate(x)]}
        case list():
            return {"list": [_encode(v, child(i), arrays) for i, v in enumerate(x)]}
        case dict() if all(isinstance(k, str) for k in x):
            return {"dict": {k: _encode(v, child(k), arrays) for k, v in x.items()}}
        case types.FunctionType():
            return {"function": _name(x)}
    raise TypeError(f"{path}: cannot save a {type(x).__name__}")


def _decode(node: Any, arrays: dict[str, np.ndarray]) -> Any:
    match node:
        case {"array": path, "shape": shape, "dtype": dtype}:
            value = arrays[path]
            if list(value.shape) != shape or str(value.dtype) != dtype:
                raise ValueError(f"{path}: weights do not match the structure")
            return value
        case {"module": name, "fields": fields}:
            cls = _resolve(name)
            names = [f.name for f in dataclasses.fields(cls)]
            missing, extra = set(names) - set(fields), set(fields) - set(names)
            if missing or extra:
                raise ValueError(
                    f"{name}: fields changed (missing {sorted(missing)}, unknown {sorted(extra)})"
                )
            # As equinox rebuilds a module when unflattening: no __init__, fields as stored.
            module = object.__new__(cls)
            for field in names:
                object.__setattr__(module, field, _decode(fields[field], arrays))
            return module
        case {"tuple": items}:
            return tuple(_decode(v, arrays) for v in items)
        case {"list": items}:
            return [_decode(v, arrays) for v in items]
        case {"dict": items}:
            return {k: _decode(v, arrays) for k, v in items.items()}
        case {"function": name}:
            return _resolve(name)
        case dict():
            raise ValueError(f"unrecognised node {sorted(node)}")
        case _:
            return node
