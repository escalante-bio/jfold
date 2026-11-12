"""Residue tables of atoms and reference conformers for each upstream pipeline, extracted by
`tools/tables/` so featurization needs no upstream code."""

import functools
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Residue:
    """One residue type's atoms in the pipeline's order, with one or more reference
    conformers."""

    atoms: tuple[str, ...]
    elements: np.ndarray  # [A] atomic numbers
    charges: np.ndarray  # [A] formal charges
    conformers: np.ndarray  # [C, A, 3] float32
    extra: dict[str, np.ndarray] = field(default_factory=dict)


class ResidueTable(Mapping[str, Residue]):
    """Map three-letter residue names to residues, with `UNK` for X."""

    def __init__(self, residues: dict[str, Residue]):
        self._residues = dict(residues)

    def __getitem__(self, name: str) -> Residue:
        return self._residues[name]

    def __iter__(self):
        return iter(self._residues)

    def __len__(self) -> int:
        return len(self._residues)

    def save(self, path: str | Path) -> None:
        arrays = {"names": np.array(list(self._residues))}
        for name, r in self._residues.items():
            arrays[f"{name}/atoms"] = np.array(r.atoms)
            arrays[f"{name}/elements"] = np.asarray(r.elements, np.int64)
            arrays[f"{name}/charges"] = np.asarray(r.charges, np.float32)
            arrays[f"{name}/conformers"] = np.asarray(r.conformers, np.float32)
            for key, value in r.extra.items():
                arrays[f"{name}/extra/{key}"] = value
        np.savez_compressed(path, **arrays)

    @classmethod
    @functools.cache
    def load(cls, name: str) -> "ResidueTable":
        """Load the table `jfold/featurize/data/<name>.npz`, caching the result."""
        with np.load(DATA / f"{name}.npz", allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        residues = {}
        for res in arrays["names"]:
            prefix = f"{res}/extra/"
            residues[str(res)] = Residue(
                atoms=tuple(str(a) for a in arrays[f"{res}/atoms"]),
                elements=arrays[f"{res}/elements"],
                charges=arrays[f"{res}/charges"],
                conformers=arrays[f"{res}/conformers"],
                extra={k[len(prefix) :]: v for k, v in arrays.items() if k.startswith(prefix)},
            )
        return cls(residues)
