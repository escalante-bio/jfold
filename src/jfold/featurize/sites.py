"""Describe each atom of a view's features by chain, residue, atom name and element, so
model outputs can be written as structures without decoding the features."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from jfold.featurize.residues import ResidueTable
from jfold.featurize.system import THREE_LETTER, Chain

# Element symbols by atomic number (index 0 unused).
SYMBOLS = (
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P",
    "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
    "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re",
    "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db",
    "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)  # fmt: skip


@dataclass(frozen=True)
class AtomSites:
    """A view's real atoms in feature order, ahead of any padding atoms, with `token`
    indexing each atom's token."""

    chain_id: tuple[str, ...]
    residue_name: tuple[str, ...]
    residue_number: np.ndarray  # [A] int
    atom_name: tuple[str, ...]
    element: np.ndarray  # [A] atomic numbers
    token: np.ndarray  # [A] int

    def __len__(self) -> int:
        return len(self.atom_name)

    def to_pdb(self, coords: np.ndarray, b_factor: np.ndarray | None = None) -> str:
        """Write `coords` (`[A, 3]`, or `[M, A, 3]` for M models) as PDB text."""
        coords = np.asarray(coords, np.float64)
        single = coords.ndim == 2
        coords = coords[None] if single else coords
        if coords.shape[1:] != (len(self), 3):
            raise ValueError(f"coords {coords.shape} for {len(self)} atoms")
        b = np.zeros(coords.shape[:2]) if b_factor is None else np.asarray(b_factor, np.float64)
        b = np.broadcast_to(b, coords.shape[:2])
        lines = []
        for m, (xyz, bm) in enumerate(zip(coords, b, strict=True), 1):
            if not single:
                lines.append(f"MODEL     {m:4d}")
            for i in range(len(self)):
                name = self.atom_name[i]
                symbol = SYMBOLS[self.element[i]]
                # Names start in column 14 unless the element symbol has two letters or the
                # name has four.
                padded = name if len(name) == 4 or len(symbol) == 2 else f" {name}"
                x, y, z = xyz[i]
                lines.append(
                    f"ATOM  {(i + 1) % 100000:5d} {padded:<4s} {self.residue_name[i]:>3s} "
                    f"{self.chain_id[i]:1s}{self.residue_number[i]:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{bm[i]:6.2f}          {symbol.upper():>2s}"
                )
            if not single:
                lines.append("ENDMDL")
        return "\n".join([*lines, "END", ""])


def atom_sites(chains: Sequence[Chain], table: ResidueTable, *, oxt: bool = False) -> AtomSites:
    """Return the atom sites of `chains`, keeping `OXT` only on each chain's last residue if
    `oxt`."""
    chain_id, residue_name, residue_number, atom_name, element, token = ([] for _ in range(6))
    t = 0
    for chain in chains:
        last = len(chain.sequence) - 1
        for i, letter in enumerate(chain.sequence):
            name = THREE_LETTER[letter]
            r = table[name]
            keep = [j for j, a in enumerate(r.atoms) if not oxt or a != "OXT" or i == last]
            chain_id += [chain.id] * len(keep)
            residue_name += [name] * len(keep)
            residue_number += [i + 1] * len(keep)
            atom_name += [r.atoms[j] for j in keep]
            element.append(np.asarray(r.elements)[keep])
            token += [t] * len(keep)
            t += 1
    return AtomSites(
        chain_id=tuple(chain_id),
        residue_name=tuple(residue_name),
        residue_number=np.array(residue_number, np.int64),
        atom_name=tuple(atom_name),
        element=np.concatenate(element).astype(np.int64),
        token=np.array(token, np.int64),
    )
