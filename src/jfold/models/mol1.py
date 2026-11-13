"""Mol-1's architecture-specific pieces."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from jfold.blocks import inputs
from jfold.blocks.pairformer import AbstractPairformer, PairformerStack
from jfold.models import protenix
from jfold.models.common import arrays
from jfold.nn import Linear, split_key
from jfold.slots import AbstractTrunkInit, Features, TrunkInputs


class Mol1Pairformer(AbstractPairformer):
    """Mol-1's trunk stack: PairMixer layers at the resident pair width, then Pairformer
    layers at a narrower one, between two linear adapters."""

    pairmixer: PairformerStack
    adapter_in: Linear
    pairformer: PairformerStack
    adapter_out: Linear

    def __call__(
        self,
        s: Float[Array, "T Cs"],
        z: Float[Array, "T T Cz"],
        mask: Bool[Array, " T"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "T Cs"], Float[Array, "T T Cz"]]:
        k_mix, k_pair = split_key(key)
        s, z = self.pairmixer(s, z, mask, key=k_mix)
        s, z = self.pairformer(s, self.adapter_in(z), mask, key=k_pair)
        return s, self.adapter_out(z)


def constraint_features(
    site: Bool[Array, ""] | Float[Array, ""],
    site_residues: Float[Array, " T"],
    site_partners: Float[Array, " T"],
    exclusion: Bool[Array, ""] | Float[Array, ""],
    exclusion_residues: Float[Array, " T"],
    exclusion_partners: Float[Array, " T"],
    contact_pairs: Int[Array, "K 2"],
    contact_mask: Bool[Array, " K"],
) -> Float[Array, "T T 5"]:
    """Build Mol-1-constraint's pair features, shaped `[T, T, 5]`.

    The first two channels mark SITE pairs, with partner tokens in rows and site residues
    in columns, then the reverse. The next two do the same for EXCLUSION, and the last
    marks each CONTACT pair in both orientations. `site` and `exclusion` turn their
    channels on, and the residue and partner masks may be soft."""
    n = site_residues.shape[0]
    site_residues, site_partners, exclusion_residues, exclusion_partners = (
        jnp.asarray(m, jnp.float32)
        for m in (site_residues, site_partners, exclusion_residues, exclusion_partners)
    )
    site, exclusion = jnp.asarray(site, jnp.float32), jnp.asarray(exclusion, jnp.float32)
    i, j = contact_pairs[:, 0], contact_pairs[:, 1]
    weight = jnp.asarray(contact_mask, jnp.float32)
    contacts = jnp.zeros((n, n)).at[i, j].add(weight).at[j, i].add(weight)
    return jnp.stack(
        [
            site * site_partners[:, None] * site_residues[None, :],
            site * site_residues[:, None] * site_partners[None, :],
            exclusion * exclusion_partners[:, None] * exclusion_residues[None, :],
            exclusion * exclusion_residues[:, None] * exclusion_partners[None, :],
            contacts,
        ],
        axis=-1,
    )


class ConstraintTrunkInit(AbstractTrunkInit[Features]):
    """Build the initial pair like `jfold.blocks.inputs.TrunkInit`, plus the embedded
    `constraint_features` from `Features.pair_features` when there are any."""

    init: inputs.TrunkInit
    constraints: Linear

    def __call__(
        self, features: Features, s_inputs: Float[Array, "T Ci"], *, key: PRNGKeyArray | None = None
    ) -> TrunkInputs:
        pair = features.pair_features
        s_init, z_init = self.init.initial(
            s_inputs,
            features.relative_positions,
            features.bonds,
            None if pair is None else self.constraints(pair),
        )
        return TrunkInputs(s_inputs=s_inputs, s_init=s_init, z_init=z_init)


SITE_KEYS = ("site_mask", "site_residue_mask", "site_partner_token_mask")
EXCLUSION_KEYS = ("exclusion_mask", "exclusion_residue_mask", "exclusion_partner_token_mask")
CONTACT_KEYS = ("contact_pair_indices", "contact_pair_mask")
CONSTRAINT_KEYS = SITE_KEYS + EXCLUSION_KEYS + CONTACT_KEYS


def constraint_pair_features(f: dict, num_tokens: int) -> Float[Array, "T T 5"] | None:
    """Build `constraint_features` from the constraint tensors in a Mol-1 feature dict.

    Each of SITE, EXCLUSION and CONTACT may be missing but not partly present. Return None
    if the dict has no constraint tensors."""
    if not any(k in f for k in CONSTRAINT_KEYS):
        return None
    for keys in (SITE_KEYS, EXCLUSION_KEYS, CONTACT_KEYS):
        if any(k in f for k in keys) and not all(k in f for k in keys):
            raise ValueError(f"incomplete constraint features: need all of {keys}")
    a = arrays({k: v for k, v in f.items() if k in CONSTRAINT_KEYS})
    off = [False, jnp.zeros(num_tokens), jnp.zeros(num_tokens)]
    site = [a[k] for k in SITE_KEYS] if SITE_KEYS[0] in a else off
    exclusion = [a[k] for k in EXCLUSION_KEYS] if EXCLUSION_KEYS[0] in a else off
    if CONTACT_KEYS[0] in a:
        contacts = a["contact_pair_indices"].astype(jnp.int32), a["contact_pair_mask"]
    else:
        contacts = jnp.zeros((0, 2), jnp.int32), jnp.zeros(0, bool)
    return constraint_features(*site, *exclusion, *contacts)


def features(f: dict) -> Features:
    """Build `jfold.model.Features` from an unbatched Mol-1 feature dict.

    Mol-1 featurizes as Protenix does. Any constraint tensors become `pair_features`."""
    features = protenix.features(f)
    pair = constraint_pair_features(f, features.asym_id.shape[0])
    if pair is None:
        return features
    return eqx.tree_at(lambda x: x.pair_features, features, pair, is_leaf=lambda x: x is None)
