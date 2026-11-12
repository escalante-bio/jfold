"""Featurize proteins for Mol-1 as its inference featurizer does, which follows Protenix
with Mol-1's own conformers, MSA rules and template slots. See
`docs/upstream-featurization/protenix.md`, section 2.

Mol-1-constraint also takes `Constraints`, which add the tensors `mol cpred` adds. See
upstream's `docs/constraint_semantics.md` for what each constraint means."""

from dataclasses import dataclass

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
# A residue as (chain id, residue number), numbered from 1 within its chain.
ResidueId = tuple[str, int]


@dataclass(frozen=True)
class Site:
    """Residues that should contact a set of partner chains, or stay away from them.

    Every residue must be on a chain that is not a partner. Any token of a partner chain
    counts as the partner."""

    residues: tuple[ResidueId, ...]
    partners: tuple[str, ...]

    def __post_init__(self):
        residues = tuple(sorted({(c, int(n)) for c, n in self.residues}))
        object.__setattr__(self, "residues", residues)
        object.__setattr__(self, "partners", tuple(self.partners))
        if not self.residues:
            raise ValueError("a site needs at least one residue")
        if not self.partners or len(set(self.partners)) != len(self.partners):
            raise ValueError(f"partner chains must be unique and nonempty, got {self.partners}")
        overlap = {c for c, _ in self.residues} & set(self.partners)
        if overlap:
            raise ValueError(f"chains {sorted(overlap)} hold site residues and are partners")


@dataclass(frozen=True)
class Constraints:
    """Mol-1-constraint's conditioning.

    `site` residues should contact their partners (SITE), `exclusion` residues should stay
    more than 6 Å from theirs (EXCLUSION), and each pair in `contacts` should come within
    4.5 Å (CONTACT). Distances are between the closest heavy atoms. The model is guided
    towards these, not forced to satisfy them."""

    site: Site | None = None
    exclusion: Site | None = None
    contacts: tuple[tuple[ResidueId, ResidueId], ...] = ()

    def __post_init__(self):
        pairs = set()
        for a, b in self.contacts:
            a, b = (a[0], int(a[1])), (b[0], int(b[1]))
            if a == b:
                raise ValueError(f"contact {a} pairs a residue with itself")
            pairs.add((min(a, b), max(a, b)))
        object.__setattr__(self, "contacts", tuple(sorted(pairs)))
        site, exclusion = self.site, self.exclusion
        if (
            site is not None
            and exclusion is not None
            and set(site.residues) & set(exclusion.residues)
            and set(site.partners) & set(exclusion.partners)
        ):
            raise ValueError("SITE and EXCLUSION share residues and partner chains")
        if self.exclusion is not None:
            excluded, partners = set(self.exclusion.residues), set(self.exclusion.partners)
            for a, b in self.contacts:
                if (a in excluded and b[0] in partners) or (b in excluded and a[0] in partners):
                    raise ValueError(f"contact {a}-{b} is inside the EXCLUSION scope")


def constraint_features(system: System, constraints: Constraints) -> dict[str, np.ndarray]:
    """Return the constraint tensors `mol cpred` adds to Mol-1's feature dict."""
    tokens, chain_tokens = {}, {}
    for chain in system.chains:
        start = len(tokens)
        for number in range(1, len(chain.sequence) + 1):
            tokens[(chain.id, number)] = len(tokens)
        chain_tokens[chain.id] = np.arange(start, len(tokens))
    num_tokens = len(tokens)

    def token(residue: ResidueId) -> int:
        if residue not in tokens:
            raise ValueError(f"residue {residue} is not in the system")
        return tokens[residue]

    def site_masks(site: Site | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        selected, partners = np.zeros(num_tokens, bool), np.zeros(num_tokens, bool)
        if site is None:
            return np.array(False), selected, partners
        selected[[token(r) for r in site.residues]] = True
        for chain in site.partners:
            if chain not in chain_tokens:
                raise ValueError(f"partner chain {chain!r} is not in the system")
            partners[chain_tokens[chain]] = True
        return np.array(True), selected, partners

    site, site_residues, site_partners = site_masks(constraints.site)
    exclusion, exclusion_residues, exclusion_partners = site_masks(constraints.exclusion)
    pairs = np.array([(token(a), token(b)) for a, b in constraints.contacts], np.int64).reshape(
        -1, 2
    )
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    return {
        "protein_token_mask": np.ones(num_tokens, bool),
        "site_mask": site,
        "site_residue_mask": site_residues,
        "site_partner_token_mask": site_partners,
        "exclusion_mask": exclusion,
        "exclusion_residue_mask": exclusion_residues,
        "exclusion_partner_token_mask": exclusion_partners,
        "contact_pair_indices": pairs,
        "contact_pair_mask": np.ones(len(pairs), bool),
    }


def features(
    system: System,
    *,
    key: jax.Array | None = None,
    use_msa: bool | None = None,
    constraints: Constraints | None = None,
) -> dict[str, np.ndarray]:
    """Return Mol-1's feature dict, randomly translating then rotating each reference
    conformer if `key` is given. `use_msa` defaults to whether any chain has an MSA.

    If `constraints` is given, the dict also holds their tensors, for Mol-1-constraint.
    An empty `Constraints()` gives the tensors `mol cpred` uses when there are none."""
    if use_msa is None:
        use_msa = any(c.msa is not None for c in system.chains)
    f = residue_features(system, layout(system, residues("mol1")), key, translate_first=True)
    f |= mol1_msa(system, use_msa)
    f |= mol1_templates(system)
    f = {k: f[k] for k in KEYS}
    if constraints is not None:
        f |= constraint_features(system, constraints)
    return f


def atom_sites(system: System) -> AtomSites:
    """Describe each real atom of `features(system)`."""
    return _atom_sites(system.chains, residues("mol1"), oxt=True)
