"""The input: chains with their sequences, (unpaired) MSAs and template structures."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"
THREE_LETTER = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
    "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
    "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL", "X": "UNK",
}  # fmt: skip


ONE_LETTER = {v: k for k, v in THREE_LETTER.items()}


@dataclass(frozen=True, eq=False)
class Template:
    """A template structure for one chain: its residues in order, `sequence` in one-letter
    codes (`X` for unknown), and each residue's atom positions by atom name (`atoms`, one
    mapping per residue; empty for a residue without coordinates, e.g. one listed in the
    template's sequence but unresolved). Views align it to its chain as their upstream
    does."""

    sequence: str
    atoms: tuple[Mapping[str, np.ndarray], ...] = field(repr=False)

    def __post_init__(self):
        unknown = set(self.sequence) - set(THREE_LETTER)
        if unknown:
            raise ValueError(f"template: unsupported residues {sorted(unknown)}")
        if len(self.atoms) != len(self.sequence):
            raise ValueError(
                f"template: {len(self.atoms)} residues of atoms for {len(self.sequence)} "
                "in the sequence"
            )
        atoms = tuple(
            {name: np.asarray(xyz, np.float32).reshape(3) for name, xyz in residue.items()}
            for residue in self.atoms
        )
        object.__setattr__(self, "atoms", atoms)

    @classmethod
    def from_pdb(
        cls, text: str, chain: str | None = None, *, sequence: str | None = None
    ) -> "Template":
        """The template in PDB text: the first model's `ATOM` records of `chain` (the first
        chain by default), heavy atoms, the first alternate location of each. Its residues
        are those with coordinates, in file order; with `sequence`, they are placed by their
        residue numbers (1 for its first residue), so residues the file leaves out are
        unresolved."""
        residues: dict[tuple[int, str], tuple[str, dict]] = {}
        for line in text.splitlines():
            record = line[:6].strip()
            if record == "ENDMDL":
                break
            if record != "ATOM":
                continue
            if chain is None:
                chain = line[21]
            if line[21] != chain or line[16] not in " A":
                continue
            element = line[76:78].strip() or line[12:14].strip()
            if element.upper() in ("H", "D"):
                continue
            name, number = line[12:16].strip(), int(line[22:26])
            res_name = line[17:20].strip()
            if res_name not in ONE_LETTER:
                raise ValueError(f"template: unsupported residue {res_name} {number}")
            key = (number, line[26])
            _, atoms = residues.setdefault(key, (res_name, {}))
            atoms.setdefault(name, (float(line[30:38]), float(line[38:46]), float(line[46:54])))
        if not residues:
            raise ValueError(f"template: no ATOM records for chain {chain!r}")
        if sequence is None:
            names = [name for name, _ in residues.values()]
            return cls(
                "".join(ONE_LETTER[n] for n in names), tuple(a for _, a in residues.values())
            )
        atoms: list[dict] = [{} for _ in sequence]
        for (number, icode), (res_name, residue_atoms) in residues.items():
            if icode != " " or not 1 <= number <= len(sequence):
                raise ValueError(f"template: residue {number}{icode.strip()} outside the sequence")
            if THREE_LETTER[sequence[number - 1]] != res_name:
                raise ValueError(
                    f"template: residue {number} is {res_name}, not {sequence[number - 1]}"
                )
            atoms[number - 1] = residue_atoms
        return cls(sequence, tuple(atoms))


@dataclass(frozen=True)
class Chain:
    """A protein chain: `sequence` in one-letter codes (`X` for unknown), optionally its
    MSA as a3m text, whose first record is the query, and its `template` structure (one per
    chain; views that do not take templates refuse it)."""

    id: str
    sequence: str
    msa: str | None = None
    template: Template | None = None

    def __post_init__(self):
        unknown = set(self.sequence) - set(THREE_LETTER)
        if unknown:
            raise ValueError(f"chain {self.id}: unsupported residues {sorted(unknown)}")


@dataclass(frozen=True)
class System:
    """Chains in input order. Views reorder them where their upstream does."""

    chains: tuple[Chain, ...]

    def __post_init__(self):
        ids = [c.id for c in self.chains]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate chain ids {ids}")

    def chain_order(self) -> list[tuple[int, int, int]]:
        """(chain index, entity, copy), chains grouped by entity: entities (identical
        sequences) in order of first appearance, a later chain with an earlier chain's
        sequence folded into that entity."""
        entities: dict[str, list[int]] = {}
        for i, chain in enumerate(self.chains):
            entities.setdefault(chain.sequence, []).append(i)
        return [
            (i, entity, copy)
            for entity, members in enumerate(entities.values())
            for copy, i in enumerate(members)
        ]

    def without_templates(self, view: str) -> "System":
        """This system, for a view that takes no templates: an error if a chain has one."""
        templated = [c.id for c in self.chains if c.template is not None]
        if templated:
            raise NotImplementedError(f"{view} featurization takes no templates ({templated})")
        return self
