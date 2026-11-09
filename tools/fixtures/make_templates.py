"""Write the template fixtures: `tests/featurize/data/template_systems.json` and a PDB file
per template.

Run from the jfold root:

    uv run python tools/fixtures/make_templates.py

Each template residue is its Boltz-2 reference conformer (conformer 0) moved along a line,
so the geometry is plausible but not a real structure; featurization only reads it.
Residues are numbered from 1 by their position in the template's sequence, so a template
with unresolved residues leaves those numbers out. The systems exercise:

- a full-length template, alone and on one copy of a homodimer;
- truncated ends, missing backbone and CB atoms (CB of a glycine is CA);
- an internal deletion with a mutation (Boltz-2 places every template residue with one
  alignment offset; OpenFold3 drops a template that leaves a residue unaligned);
- unresolved residues and an unknown residue;
- templates on every chain, one of them on a 4-residue chain (Mol-1 ignores those);
- templates longer than their chain (OpenFold3 realigns them; Protenix and Mol-1, which
  place template residues one for one, refuse them).
"""

import json
from pathlib import Path

import numpy as np

from jfold.featurize.system import THREE_LETTER
from jfold.featurize.views.boltz2 import residues

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "featurize" / "data"
LONG = "MKTAYIAKQRQISFVKSHFSRQ"
SHORT = "GSHMLEDPV"


def pdb(sequence: str, present: list[int], drop: dict[int, list[str]], seed: int) -> str:
    """PDB text for the residues `present` (1-based positions in `sequence`), without the
    atoms `drop` lists; coordinates to 3 decimals, as PDB writes them."""
    rng = np.random.default_rng(seed)
    table = residues()
    lines, serial = [], 1
    for number in present:
        name = THREE_LETTER[sequence[number - 1]]
        residue = table[name]
        rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        coords = residue.conformers[0] @ rotation.T + np.array([3.8 * number, 0.0, 0.0])
        for atom, element, xyz in zip(residue.atoms, residue.elements, coords, strict=True):
            if atom in drop.get(number, []):
                continue
            symbol = {6: "C", 7: "N", 8: "O", 16: "S"}[int(element)]
            padded = atom if len(atom) == 4 else f" {atom}"
            x, y, z = xyz
            lines.append(
                f"ATOM  {serial:5d} {padded:<4s} {name:>3s} A{number:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {symbol:>2s}"
            )
            serial += 1
    return "\n".join([*lines, "END", ""])


def main() -> None:
    gap = LONG[:8] + LONG[11:]  # residues 9-11 deleted
    gap = gap[:14] + "W" + gap[15:]  # and a mutation
    unknown = LONG[:4] + "X" + LONG[5:]
    tetra = SHORT[:4]
    # file: (sequence, residues with coordinates, atoms left out, seed, sequence placed by
    # residue number, so the residues without coordinates are unresolved)
    templates = {
        "template_full.pdb": (LONG, range(1, 23), {}, 0, False),
        # Residues 2-8 of SHORT; its H without CB, its M without N.
        "template_truncated.pdb": (SHORT[1:8], range(1, 8), {2: ["CB"], 3: ["N"]}, 1, False),
        "template_gap.pdb": (gap, range(1, len(gap) + 1), {}, 2, False),
        "template_unresolved.pdb": (
            unknown,
            [n for n in range(1, 23) if n not in (12, 13)],
            {7: ["C"]},
            3,
            True,
        ),
        # Glycine first; its H without CB, its M without N, its V without C.
        "template_short.pdb": (SHORT, range(1, 10), {3: ["CB"], 4: ["N"], 9: ["C"]}, 10, False),
        "template_tetra.pdb": (tetra, range(1, 5), {}, 11, False),
    }
    for file, (sequence, present, drop, seed, _) in templates.items():
        (FIXTURES / file).write_text(pdb(sequence, list(present), drop, seed))

    def chain(id: str, sequence: str, template: str | None = None, msa: str | None = None):
        spec = {"id": id, "sequence": sequence}
        if msa is not None:
            spec["msa"] = msa
        if template is not None:
            template_sequence, *_, by_number = templates[template]
            spec["template"] = {"pdb": template}
            if by_number:
                spec["template"]["sequence"] = template_sequence
        return spec

    systems = {
        "monomer_template": [chain("A", LONG, "template_full.pdb")],
        "dimer_template": [chain("A", LONG), chain("B", SHORT, "template_truncated.pdb")],
        "monomer_template_gap": [chain("A", LONG, "template_gap.pdb")],
        "monomer_template_unresolved": [
            chain("A", LONG, "template_unresolved.pdb", msa="chain_A.a3m")
        ],
        "homodimer_template": [chain("A", LONG), chain("B", LONG, "template_full.pdb")],
        "dimer_template_both": [
            chain("A", LONG, "template_full.pdb"),
            chain("B", SHORT, "template_short.pdb"),
        ],
        "dimer_template_tetra": [
            chain("A", LONG, "template_full.pdb"),
            chain("B", tetra, "template_tetra.pdb"),
        ],
        # Each template covers its chain: one longer than it, one as long, with gaps in atoms.
        "dimer_template_covering": [
            chain("A", LONG[1:19], "template_full.pdb"),
            chain("B", SHORT[1:8], "template_truncated.pdb"),
        ],
    }
    out = FIXTURES / "template_systems.json"
    out.write_text(json.dumps({k: {"chains": v} for k, v in systems.items()}, indent=1) + "\n")
    print(f"wrote {out} and {len(templates)} templates")


if __name__ == "__main__":
    main()
