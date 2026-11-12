"""Converters from the upstream PyTorch implementations to jfold modules.

Each whole-model converter is `structure_model(upstream_model)`. Converters need torch
but not the upstream package, since they only read attributes off the given torch
modules."""
