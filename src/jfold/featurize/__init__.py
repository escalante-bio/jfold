"""Torch-free featurization: a system (chains, sequences, MSAs) to each model's features.

`System` describes the input once; `jfold.featurize.views.<model>.features(system, ...)`
produces exactly the feature dict that model's upstream featurization produces, so
`jfold.models.<model>.features` takes it unchanged, and `atom_sites(system)` says what each
of those features' atoms is (`AtomSites`, which writes PDB files). Each view's conventions
(vocabulary, atoms, reference conformers, ids, MSA rules) are documented in
`docs/upstream-featurization/`.

Scope: proteins (standard residues and X), unpaired MSAs.
"""

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
