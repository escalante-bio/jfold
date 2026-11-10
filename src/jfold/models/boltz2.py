"""Boltz-2 model pieces."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from jfold.blocks import inputs
from jfold.blocks.atoms import Atoms
from jfold.blocks.confidence import AbstractPairInputs
from jfold.blocks.denoiser import FourierEmbedding
from jfold.models.common import arrays
from jfold.nn import Linear
from jfold.slots import AbstractTrunkInit, Features, TrunkInputs


def atom_features(
    ref_pos: Float[Array, "N 3"],
    ref_charge: Float[Array, " N"],
    ref_element: Float[Array, "N 128"],
    ref_atom_name_chars: Float[Array, "N 4 64"],
) -> Float[Array, "N 388"]:
    """The atom feature vector Boltz-2's `AtomEncoder` embeds: position, charge, element and
    name characters."""
    return jnp.concatenate(
        [
            ref_pos,
            ref_charge[:, None],
            ref_element,
            ref_atom_name_chars.reshape(ref_pos.shape[0], -1),
        ],
        axis=-1,
    )


def msa_features(msa, has_deletion, deletion_value, is_paired):
    """The MSA feature vector `[S, T, 36]`: residue probabilities `[S, T, 33]` (one-hot for
    a fixed alignment), has-deletion, deletion value, paired."""
    return jnp.concatenate(
        [
            msa,
            has_deletion[..., None],
            deletion_value[..., None],
            is_paired[..., None],
        ],
        axis=-1,
    )


def template_features(
    restype,
    frame_rot,
    frame_t,
    frame_mask,
    cb,
    ca,
    cb_mask,
    visibility_ids,
    min_dist=3.25,
    max_dist=50.75,
    num_bins=38,
):
    """Each template's pair feature vector `[N, T, T, 109]`, as Boltz-2 computes it from the
    template's coordinates: the Cβ distogram, Cβ pair mask, direction from each residue's
    frame to the other's Cα, frame pair mask (all zero between parts the template does not
    relate), then the row and column tokens' residue types."""
    boundaries = jnp.linspace(min_dist, max_dist, num_bins - 1)
    cb_dist = jnp.linalg.norm(cb[:, :, None] - cb[:, None, :], axis=-1)
    distogram = jax.nn.one_hot((cb_dist[..., None] > boundaries).sum(-1), num_bins)
    # Cα of token i in the frame of token j, R_j^T (ca_i - t_j). Boltz-2 takes the norm over
    # a trailing axis of size one, so its "unit vector" is the sign of each component; the
    # weights were trained on that.
    vector = jnp.einsum("njkl,nijk->nijl", frame_rot, ca[:, :, None] - frame_t[:, None])
    unit_vector = jnp.sign(vector)
    pair = lambda m: (m[:, :, None] * m[:, None, :])[..., None]
    visible = (visibility_ids[:, :, None] == visibility_ids[:, None, :])[..., None]
    geometry = jnp.concatenate([distogram, pair(cb_mask), unit_vector, pair(frame_mask)], -1)
    n = restype.shape[1]
    shape = (restype.shape[0], n, n, restype.shape[-1])
    return jnp.concatenate(
        [
            geometry * visible,
            jnp.broadcast_to(restype[:, :, None, :], shape),
            jnp.broadcast_to(restype[:, None, :, :], shape),
        ],
        axis=-1,
    )


def token_features(res_type, profile, deletion_mean, method, modified, cyclic_period, mol_type):
    """The token feature vector Boltz-2 projects into `s_inputs`: residue-type probabilities
    `[T, 33]` (one-hot for a fixed sequence), MSA profile and deletion mean, method (12),
    modified (2), cyclic, and molecule type (4)."""
    return jnp.concatenate(
        [
            res_type,
            profile,
            deletion_mean[:, None],
            jax.nn.one_hot(method, 12),
            jax.nn.one_hot(modified, 2),
            jnp.minimum(cyclic_period, 1.0)[:, None],
            jax.nn.one_hot(mol_type, 4),
        ],
        axis=-1,
    )


NUM_BOND_TYPES = 7  # no bond, then `boltz.data.const.bond_types`
NUM_CONTACT_CLASSES = 5  # `boltz.data.const.contact_conditioning_info`


def bond_features(token_bonds, type_bonds):
    """The token bond feature vector `[T, T, 8]`: bonded, then the bond type (7 classes)."""
    return jnp.concatenate([token_bonds, jax.nn.one_hot(type_bonds, NUM_BOND_TYPES)], axis=-1)


def contact_features(contact_conditioning, contact_threshold):
    """The contact feature vector `[T, T, 6]`: contact class (5, one-hot: unspecified,
    unselected, pocket-to-binder, binder-to-pocket, contact), then distance threshold."""
    return jnp.concatenate([contact_conditioning, contact_threshold[..., None]], axis=-1)


class ContactConditioning(eqx.Module):
    """Boltz-2's contact conditioning: each pair's contact class and normalized threshold
    (`contact_features`), with the threshold's Fourier embedding, projected; unspecified
    and unselected pairs get a learned vector each instead."""

    fourier: FourierEmbedding
    linear: Linear  # [classes 2-4, threshold, Fourier] -> Cz
    unspecified: Float[Array, " Cz"]
    unselected: Float[Array, " Cz"]
    cutoff_min: float = eqx.field(static=True)
    cutoff_max: float = eqx.field(static=True)

    def __call__(self, contacts: Float[Array, "T T 6"]) -> Float[Array, "T T Cz"]:
        classes, threshold = contacts[..., :NUM_CONTACT_CLASSES], contacts[..., -1:]
        t = (threshold - self.cutoff_min) / (self.cutoff_max - self.cutoff_min)
        z = self.linear(jnp.concatenate([classes[..., 2:], t, self.fourier(t)], axis=-1))
        return (
            z * (1 - classes[..., :2].sum(-1, keepdims=True))
            + classes[..., 0:1] * self.unspecified
            + classes[..., 1:2] * self.unselected
        )


class PairInputs(AbstractPairInputs):
    """Boltz-2's extra pair inputs: relative positions, bonds and contacts (from
    `Features.pair_features`), each embedded."""

    relative_positions: Linear
    bonds: Linear  # bond_features -> Cz
    contacts: ContactConditioning

    def __call__(self, features: Features) -> Float[Array, "T T Cz"]:
        return (
            self.relative_positions(features.relative_positions)
            + self.bonds(features.bonds)
            + self.contacts(features.pair_features)
        )


class TrunkInit(AbstractTrunkInit[Features]):
    """`jfold.blocks.inputs.TrunkInit` with Boltz-2's contact conditioning: the features'
    `pair_features` are the `contact_features`, embedded before they are added to the
    pair."""

    init: inputs.TrunkInit
    contacts: ContactConditioning

    def __call__(
        self, features: Features, s_inputs: Float[Array, "T Ci"], *, key: PRNGKeyArray | None = None
    ) -> TrunkInputs:
        s_init, z_init = self.init.initial(
            s_inputs,
            features.relative_positions,
            features.bonds,
            self.contacts(features.pair_features),
        )
        return TrunkInputs(s_inputs=s_inputs, s_init=s_init, z_init=z_init)


def features(f: dict) -> Features:
    """`jfold.model.Features` from a Boltz-2 feature dict (`boltz.data` featurization, as
    `Boltz2InferenceDataModule` produces it; unbatched). `msa` is upstream's residue
    classes `[S, T]` or probabilities over them `[S, T, 33]` (soft sequences); every
    step is traceable, so the dict may be built inside `jit`."""
    a = arrays(f)
    atom_to_token = a["atom_to_token"].argmax(-1)
    first_atom = jnp.argmax(a["atom_to_token"], axis=0)  # each token's first atom
    float_ = lambda k: a[k].astype(jnp.float32)
    return Features(
        atoms=Atoms(
            features=atom_features(
                a["ref_pos"],
                float_("ref_charge"),
                float_("ref_element"),
                float_("ref_atom_name_chars"),
            ),
            ref_pos=a["ref_pos"],
            ref_space_uid=a["ref_space_uid"],
            atom_to_token=atom_to_token,
            mask=a["atom_pad_mask"] > 0,
        ),
        token_features=token_features(
            float_("res_type"),
            float_("profile"),
            float_("deletion_mean"),
            a["method_feature"],
            a["modified"],
            float_("cyclic_period"),
            a["mol_type"],
        ),
        relative_positions=inputs.relative_position_features(
            a["asym_id"],
            a["residue_index"],
            a["entity_id"],
            a["token_index"],
            a["sym_id"],
            cyclic_period=float_("cyclic_period"),
        ),
        token_mask=a["token_pad_mask"] > 0,
        asym_id=a["asym_id"],
        rep_atom=a["token_to_rep_atom"].argmax(-1),
        atom_position=jnp.arange(atom_to_token.shape[0]) - first_atom[atom_to_token],
        bonds=bond_features(float_("token_bonds"), a["type_bonds"]),
        pair_features=contact_features(float_("contact_conditioning"), float_("contact_threshold")),
        msa_features=msa_features(
            inputs.probabilities(a["msa"], 33),
            float_("has_deletion"),
            float_("deletion_value"),
            float_("msa_paired"),
        ),
        msa_mask=a["msa_mask"] > 0,
        template_features=template_features(
            float_("template_restype"),
            a["template_frame_rot"],
            a["template_frame_t"],
            float_("template_mask_frame"),
            a["template_cb"],
            a["template_ca"],
            float_("template_mask_cb"),
            a["visibility_ids"],
        ),
        template_mask=a["template_mask"].any(-1),
    )
