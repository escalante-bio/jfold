"""Converters from the upstream PyTorch implementations. Each module is named for what it
converts:

- a code lineage, its blocks and the models built only from them: `boltz` (Boltz-2's and
  ProMERA's blocks), `protenix` (Protenix and Mol-1, and OpenDDE's blocks), `openfold3`,
  `esmfold2`, `bioemu2`;
- a model built from another lineage's blocks: `boltz2`, `promera` (on `boltz`'s) and
  `opendde` (on `protenix`'s);
- `torch`: the leaf converters (linear and norm layers, attention, dropout).

Each whole-model converter is a `structure_model(upstream_model)`. Importing a converter
needs torch, but not the upstream package: converters read attributes off the torch
modules they are given."""
