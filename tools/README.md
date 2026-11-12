# tools

Scripts that jfold does not import. The scripts in `fixtures/` and `tables/` run upstream
code, so they need the upstream package installed. Run them from the repository root,
for example `uv run python tools/fixtures/dump_boltz2.py`.

- `fixtures/` generates the featurization test fixtures in `tests/featurize/data/`.
  Each `dump_<model>.py` saves upstream's features for every test system, and
  `make_templates.py` writes the template PDBs and `template_systems.json`.
- `tables/` generates the residue tables in `src/jfold/featurize/data/`. Each
  `extract_<model>.py` saves every residue's atoms and reference conformers as upstream
  builds them, so featurization does not need upstream code.

`model_table.py` generates `docs/models/README.md` from the released models' `model.json`.

`docs/upstream-featurization/<model>.md` describes each pipeline and its fixtures.
