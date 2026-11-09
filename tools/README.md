# tools

Scripts jfold itself never imports. Those in `fixtures/` and `tables/` run upstream code,
so they need the upstream package installed. Run them from the repository root, e.g.
`uv run python tools/fixtures/dump_boltz2.py`.

- `fixtures/`: the featurization parity fixtures in `tests/featurize/data/`.
  `dump_<model>.py` records upstream's feature dict for each test system;
  `make_templates.py` writes the template PDBs and `template_systems.json`.
- `tables/`: the residue tables shipped in `src/jfold/featurize/data/`
  (`extract_<model>.py`): each residue's atoms and reference conformers, as upstream
  builds them, so featurization needs no upstream code.

`model_table.py` writes `docs/models/README.md` from the released copies' `model.json`.

`docs/upstream-featurization/<model>.md` describes each pipeline and its fixtures.
