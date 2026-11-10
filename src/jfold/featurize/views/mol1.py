"""Mol-1's features, as its inference featurizer (`mol.data.inference.infer_dataloader
.InferenceDataset.process_one`) produces them for proteins (see
`docs/upstream-featurization/protenix.md`, section 2): the dict `jfold.models.mol1.features`
takes, plus upstream's `is_protein`.

As Protenix, except: Mol-1's own reference conformers; MSAs by Mol-1's rules (the query
row twice when only one distinct sequence, int8-clipped deletion counts, float32 dtypes,
no MSA for chains of 4 residues or fewer); four template slots, a chain's template given
as upstream's template JSON (`mol1_templates`)."""

import jax
import numpy as np

from jfold.featurize.sites import AtomSites
from jfold.featurize.sites import atom_sites as _atom_sites
from jfold.featurize.system import System
from jfold.featurize.views.protenix import (
    JFOLD_KEYS,
    layout,
    mol1_msa,
    mol1_templates,
    residue_features,
    residues,
)

KEYS = (*JFOLD_KEYS, "is_protein")


def features(
    system: System, *, key: jax.Array | None = None, use_msa: bool | None = None
) -> dict[str, np.ndarray]:
    """Mol-1's feature dict for `system`, chains' templates included. `key`: draw each
    residue's reference-conformer pose (translate, then rotate); None leaves the conformers
    centred and unposed. `use_msa`: the dataset's `use_msa`; by default, whether any chain
    has an MSA."""
    if use_msa is None:
        use_msa = any(c.msa is not None for c in system.chains)
    f = residue_features(system, layout(system, residues("mol1")), key, translate_first=True)
    f |= mol1_msa(system, use_msa)
    f |= mol1_templates(system)
    return {k: f[k] for k in KEYS}


def atom_sites(system: System) -> AtomSites:
    """What each real atom of `features(system)` is (chains in input order, `OXT` on each
    chain's last residue)."""
    return _atom_sites(system.chains, residues("mol1"), oxt=True)
