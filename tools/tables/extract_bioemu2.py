"""BioEmu2's residue table (`jfold/featurize/data/bioemu2.npz`) from its packaged polymer
templates: CCD heavy atoms with ideal coordinates centred on all heavy atoms (OXT
included), then OXT dropped, as `_templates_from_protein_sequence` does. X uses the
`UNK_PROT` template (ALA's). Needs the `bioemu2` package; run once:

    uv run python tools/tables/extract_bioemu2.py
"""

from bioemu2.structure.templates import load_polymer_templates

from jfold.featurize.residues import DATA, Residue, ResidueTable
from jfold.featurize.system import THREE_LETTER

templates = load_polymer_templates()
residues = {}
for name in THREE_LETTER.values():
    t = templates["UNK_PROT" if name == "UNK" else name]
    names = ["".join(chr(c + 32) for c in row if c) for row in t.name_encoding.tolist()]
    keep = [i for i, n in enumerate(names) if n != "OXT"]
    residues[name] = Residue(
        atoms=tuple(names[i] for i in keep),
        elements=t.element.numpy()[keep],
        charges=t.charge.numpy()[keep],
        conformers=t.template_coords.numpy()[keep][None],
    )
ResidueTable(residues).save(DATA / "bioemu2.npz")
print({k: len(v.atoms) for k, v in residues.items()})
