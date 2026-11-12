"""Featurize systems of protein chains for each model without torch.
`jfold.featurize.views.<model>.features(system)` returns the feature dict of that model's
upstream featurization, and `docs/upstream-featurization/` documents each view's
conventions."""

from jfold.featurize.a3m import A3M, parse_a3m
from jfold.featurize.residues import Residue, ResidueTable
from jfold.featurize.sites import AtomSites
from jfold.featurize.system import Chain, System, Template

__all__ = [
    "A3M",
    "AtomSites",
    "Chain",
    "Residue",
    "ResidueTable",
    "System",
    "Template",
    "parse_a3m",
]
