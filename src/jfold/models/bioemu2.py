"""BioEmu2's model-specific pieces: adaLN-Zero transformer layers, method and contact
conditioning of the trunk, and a dual-stream structure module that updates tokens and atoms
side by side. BioEmu2 has no confidence head, and its conditioning inputs are stored in the
features' `extras`."""

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.nn import Identity
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from jfold.blocks.atoms import AtomPairEmbedding, Atoms, TrunkToAtoms
from jfold.blocks.denoiser import DiffusionConditioning, FourierEmbedding
from jfold.blocks.diffusion import DiffusionTransformer
from jfold.blocks.inputs import TrunkInit, probabilities, relative_position_features
from jfold.blocks.windows import CenteredWindows, LocalAttention
from jfold.model import Trunk
from jfold.models.common import arrays
from jfold.nn import (
    Attention,
    LayerNorm,
    Linear,
    ReluMLP,
    SinglesToPair,
    SwiGLU,
    Transition,
    scan_layers,
)
from jfold.slots import (
    AbstractDiffusionModule,
    AbstractTrunk,
    AbstractTrunkInit,
    Features,
    TrunkInputs,
    TrunkState,
)

NUM_TOKEN_TYPES = 33
NUM_ELEMENTS = 128
NUM_NAME_CHARS, NUM_NAME_CHAR_BINS = 4, 64
NUM_POCKET_CLASSES = 4


class ZeroAdaLN(eqx.Module):
    """DiT's adaLN-Zero, returning `norm_a(a) (1 + α) + β` and the gate `γ`."""

    norm_a: LayerNorm  # no scale or offset
    norm_s: LayerNorm
    linear: Linear  # Cs -> 2 Ca + Cout

    def __call__(
        self, a: Float[Array, "N Ca"], s: Float[Array, "N Cs"]
    ) -> tuple[Float[Array, "N Ca"], Float[Array, "N Cout"]]:
        width = a.shape[-1]
        alpha, beta, gate = jnp.split(
            self.linear(jax.nn.silu(self.norm_s(s))), [width, 2 * width], axis=-1
        )
        return self.norm_a(a) * (1 + alpha) + beta, gate


class ZeroConditionedAttention(eqx.Module):
    """Self-attention with pair bias behind adaLN-Zero."""

    adaln: ZeroAdaLN
    attention: Attention | LocalAttention

    def __call__(
        self,
        a: Float[Array, "N Ca"],
        s: Float[Array, "N Cs"],
        bias: Float[Array, "heads ..."],
        mask: Bool[Array, " N"],
    ) -> Float[Array, "N Ca"]:
        x, gate = self.adaln(a, s)
        return gate * self.attention(x, x, bias, mask)


class ZeroConditionedTransition(eqx.Module):
    """A SwiGLU MLP behind adaLN-Zero."""

    adaln: ZeroAdaLN
    mlp: SwiGLU

    def __call__(self, a: Float[Array, "N Ca"], s: Float[Array, "N Cs"]) -> Float[Array, "N Co"]:
        x, gate = self.adaln(a, s)
        return gate * self.mlp(x)


class MethodConditioning(eqx.Module):
    """Embeddings of each token's experimental method for the initial single and pair."""

    single: Linear
    pair: SinglesToPair | None

    def __call__(
        self, method: Float[Array, "T M"]
    ) -> tuple[Float[Array, "T Cs"], Float[Array, "T T Cz"] | None]:
        return self.single(method), None if self.pair is None else self.pair(method)


class ContactConditioning(eqx.Module):
    """Pair features from requested contacts, with learned encodings for unspecified and
    unselected pairs."""

    fourier: FourierEmbedding
    linear: Linear
    unspecified: Float[Array, " Cz"]
    unselected: Float[Array, " Cz"]
    cutoff_min: float = eqx.field(static=True)
    cutoff_max: float = eqx.field(static=True)

    def __call__(
        self, contacts: Float[Array, "T T 5"], threshold: Float[Array, "T T"]
    ) -> Float[Array, "T T Cz"]:
        t = (threshold - self.cutoff_min) / (self.cutoff_max - self.cutoff_min)
        x = jnp.concatenate([contacts[..., 2:], t[..., None], self.fourier(t[..., None])], -1)
        unspecified, unselected = contacts[..., :1], contacts[..., 1:2]
        return (
            self.linear(x) * (1 - unspecified - unselected)
            + self.unspecified * unspecified
            + self.unselected * unselected
        )


class ConditionedTrunkInit(AbstractTrunkInit[Features]):
    """`jfold.blocks.inputs.TrunkInit` plus method and contact conditioning."""

    init: TrunkInit
    method: MethodConditioning | None
    contacts: ContactConditioning | None

    def __call__(
        self, features: Features, s_inputs: Float[Array, "T Ci"], *, key: PRNGKeyArray | None = None
    ) -> TrunkInputs:
        conditioning = features.extras
        s_extra, z_extra = None, None
        if self.method is not None:
            s_extra, z_extra = self.method(conditioning.method)
        if self.contacts is not None:
            z_contacts = self.contacts(conditioning.contacts, conditioning.contact_threshold)
            z_extra = z_contacts if z_extra is None else z_extra + z_contacts
        s, z = self.init.initial(s_inputs, features.relative_positions, features.bonds, z_extra)
        s = s if s_extra is None else s + s_extra
        return TrunkInputs(s_inputs=s_inputs, s_init=s, z_init=z)


class Dynamics(eqx.Module):
    """A companion frame and lag time, ignored when `is_conditioned` is False."""

    coords: Float[Array, "N 3"]
    lag: Float[Array, ""]
    is_conditioned: Bool[Array, ""]


class DampedHarmonicEmbedding(eqx.Module):
    """Embed `t` as envelopes `exp(-t/τ_k)` and those envelopes times `sin` and `cos` of `2π
    q t/τ_k`."""

    taus: Float[Array, " K"]
    quality_factors: Float[Array, " Q"]

    def __call__(self, t: Float[Array, ""]) -> Float[Array, " E"]:
        scaled = t / self.taus
        envelope = jnp.exp(-scaled)
        phase = 2 * jnp.pi * scaled[:, None] * self.quality_factors[None, :]
        return jnp.concatenate(
            [
                envelope,
                (envelope[:, None] * jnp.sin(phase)).reshape(-1),
                (envelope[:, None] * jnp.cos(phase)).reshape(-1),
            ]
        )


class LagGate(eqx.Module):
    """A positive mixture of exponential decays in the lag time."""

    taus: Float[Array, " K"]
    log_weight: Float[Array, " K"]

    def __call__(self, t: Float[Array, ""]) -> Float[Array, ""]:
        return jnp.sum(jnp.exp(-t / self.taus) * jax.nn.softplus(self.log_weight))


class FrameConditioning(eqx.Module):
    """The companion frame's contribution to the atom queries, optionally gated by the lag."""

    linear: Linear  # 3 -> Ca
    gate: LagGate | None

    def __call__(self, dynamics: Dynamics) -> Float[Array, "N Ca"]:
        x = self.linear(dynamics.coords)
        if self.gate is not None:
            x = x * self.gate(dynamics.lag)
        return jnp.where(dynamics.is_conditioned, x, 0.0)


class SingleConditioning(eqx.Module):
    """`DiffusionConditioning` with a lag-time embedding added before its transitions."""

    conditioning: DiffusionConditioning
    lag_embedding: DampedHarmonicEmbedding | None = None
    linear_lag: Linear | None = None

    def pair(self, z_trunk, relative_positions):
        return self.conditioning.pair(z_trunk, relative_positions)

    def single(
        self,
        s_trunk: Float[Array, "T Cs"],
        s_inputs: Float[Array, "T Ci"],
        t: Float[Array, ""],
        dynamics: Dynamics | None,
    ) -> Float[Array, "T Ca"]:
        c = self.conditioning
        if self.lag_embedding is None:
            return c.single(s_trunk, s_inputs, t)
        s = c.linear_s(c.norm_s(jnp.concatenate([s_trunk, s_inputs], -1)))
        s = s + c.linear_noise(c.norm_noise(c.fourier(jnp.log(t / c.sigma_data) / 4)))
        lag = self.linear_lag(self.lag_embedding(dynamics.lag))
        s = s + jnp.where(dynamics.is_conditioned, lag, 0.0)
        for transition in c.transitions_s:
            s = s + transition(s)
        return s


class AtomConditioning(eqx.Module):
    """Build the structure module's atom single, windowed pair and per-step atom queries."""

    embed_features: Linear  # atom feature vector -> Ca
    pair_embedding: AtomPairEmbedding
    trunk: TrunkToAtoms
    linear_c_queries: Linear
    linear_c_keys: Linear
    pair_mlp: ReluMLP
    linear_r: Linear  # noisy positions -> Ca
    frame: FrameConditioning | None
    windows: CenteredWindows

    def condition(
        self,
        atoms: Atoms,
        s_trunk: Float[Array, "T Cs"],
        z_trunk: Float[Array, "T T Cz"],
    ) -> tuple[Float[Array, "N Ca"], Float[Array, "NW queries keys Cp"]]:
        """Return the queries' starting point and the windowed pair."""
        windows, mask = self.windows, atoms.mask
        q = self.embed_features(atoms.features)
        p = self.pair_embedding(atoms.ref_pos, atoms.ref_space_uid, mask, windows)
        c, p = self.trunk(q, p, s_trunk, z_trunk, atoms.atom_to_token, mask, windows)
        index, _ = windows.key_index(mask)
        relu_c = jax.nn.relu(c)
        p = p + self.linear_c_queries(windows.query_blocks(relu_c))[:, :, None]
        p = p + self.linear_c_keys(relu_c[index])[:, None]
        return q, p + self.pair_mlp(p)

    def queries(
        self, q: Float[Array, "N Ca"], r: Float[Array, "N 3"], dynamics: Dynamics | None
    ) -> Float[Array, "N Ca"]:
        q = q + self.linear_r(r)
        if self.frame is not None:
            q = q + self.frame(dynamics)
        return q


class DualStreamCoupling(eqx.Module):
    """Couple the streams: each atom reads its token and each token reads its centre atom.
    """

    to_atoms: ZeroConditionedTransition  # [s of the atom's token, q] -> Ca
    to_tokens: ZeroConditionedTransition  # [q of the token's center atom, s] -> Cs

    def __call__(
        self,
        s: Float[Array, "T Cs"],
        q: Float[Array, "N Ca"],
        c: Float[Array, "T Cs"],
        c_atoms: Float[Array, "N Cs"],
        atom_to_token: Int[Array, " N"],
        center_atom: Int[Array, " T"],
    ) -> tuple[Float[Array, "T Cs"], Float[Array, "N Ca"]]:
        q_update = self.to_atoms(jnp.concatenate([s[atom_to_token], q], -1), c_atoms)
        s_update = self.to_tokens(jnp.concatenate([q[center_atom], s], -1), c)
        return s + s_update, q + q_update


class DualStreamTransformer(eqx.Module):
    """Stacked dual-stream layers, each an atom layer, a token layer and the coupling."""

    atoms: DiffusionTransformer  # within atom windows
    tokens: DiffusionTransformer
    coupling: DualStreamCoupling  # stacked

    def __call__(
        self,
        s: Float[Array, "T Cs"],
        q: Float[Array, "N Ca"],
        c: Float[Array, "T Cs"],
        atom_biases: Float[Array, "L heads NW queries keys"],
        token_biases: Float[Array, "L heads T T"],
        atom_to_token: Int[Array, " N"],
        center_atom: Int[Array, " T"],
        atom_mask: Bool[Array, " N"],
        token_mask: Bool[Array, " T"],
    ) -> tuple[Float[Array, "T Cs"], Float[Array, "N Ca"]]:
        c_atoms = c[atom_to_token]

        def step(layers, carry, key):
            atom_layer, token_layer, coupling, atom_bias, token_bias = layers
            s, q = carry
            q = atom_layer(q, c_atoms, atom_bias, atom_mask)
            s = token_layer(s, c, token_bias, token_mask)
            return coupling(s, q, c, c_atoms, atom_to_token, center_atom)

        layers = (self.atoms.layers, self.tokens.layers, self.coupling, atom_biases, token_biases)
        return scan_layers(step, (s, q), layers)


class Readout(eqx.Module):
    """Compute position updates from each atom and its token."""

    linear_s: Linear
    linear_q: Linear
    norm: LayerNorm
    linear_out: Linear

    def __call__(
        self, s: Float[Array, "T Cs"], q: Float[Array, "N Ca"], atom_to_token: Int[Array, " N"]
    ) -> Float[Array, "N 3"]:
        return self.linear_out(self.norm(self.linear_q(q) + self.linear_s(s)[atom_to_token]))


class DualStreamContext(eqx.Module):
    """Everything a denoising step needs that depends only on the trunk."""

    s_trunk: Float[Array, "T Cs"]
    s_inputs: Float[Array, "T Ci"]
    a_trunk: Float[Array, "T Ca"]
    queries: Float[Array, "N Ca"]
    atom_biases: Float[Array, "L heads NW queries keys"]
    token_biases: Float[Array, "L heads T T"]
    atom_to_token: Int[Array, " N"]
    center_atom: Int[Array, " T"]
    atom_mask: Bool[Array, " N"]  # the atoms the denoiser moves, which a sampler centres on
    coordinate_mask: Bool[Array, " N"]
    token_mask: Bool[Array, " T"]
    dynamics: Dynamics | None = None


class DualStreamModule(AbstractDiffusionModule[Features]):
    """BioEmu2's structure module, with EDM preconditioning."""

    conditioning: SingleConditioning
    atoms: AtomConditioning
    norm_s: LayerNorm  # s_trunk -> token stream
    linear_s: Linear
    linear_r: Linear  # the token's center atom's noisy position -> token stream
    transformer: DualStreamTransformer
    readout: Readout
    # The relative position features as the pair conditioning reads them.
    relative_positions: Linear | Identity = eqx.field(default_factory=Identity)

    @property
    def sigma_data(self) -> float:
        return self.conditioning.conditioning.sigma_data

    def prepare(self, features: Features, inputs: TrunkInputs, state: TrunkState) -> DualStreamContext:
        conditioning, atoms = features.extras, features.atoms
        relative_positions = self.relative_positions(features.relative_positions)
        z = self.conditioning.pair(state.z, relative_positions)
        q, p = self.atoms.condition(atoms, state.s, state.z)
        return DualStreamContext(
            s_trunk=state.s,
            s_inputs=inputs.s_inputs,
            a_trunk=self.linear_s(self.norm_s(state.s)),
            queries=q,
            atom_biases=self.transformer.atoms.pair_biases(p),
            token_biases=self.transformer.tokens.pair_biases(z),
            atom_to_token=atoms.atom_to_token,
            center_atom=conditioning.center_atom,
            atom_mask=atoms.mask,
            coordinate_mask=conditioning.coordinate_mask,
            token_mask=features.token_mask,
            dynamics=conditioning.dynamics,
        )

    def __call__(
        self,
        context: DualStreamContext,
        x_noisy: Float[Array, "N 3"],
        t: Float[Array, ""],
    ) -> Float[Array, "N 3"]:
        sigma_data, dynamics = self.sigma_data, context.dynamics
        r = jnp.where(context.coordinate_mask[:, None], x_noisy, 0.0)
        r = r / jnp.sqrt(t**2 + sigma_data**2)
        c = self.conditioning.single(context.s_trunk, context.s_inputs, t, dynamics)
        q = self.atoms.queries(context.queries, r, dynamics)
        s = context.a_trunk + self.linear_r(r[context.center_atom])
        s, q = self.transformer(
            s,
            q,
            c,
            context.atom_biases,
            context.token_biases,
            context.atom_to_token,
            context.center_atom,
            context.atom_mask,
            context.token_mask,
        )
        r_update = self.readout(s, q, context.atom_to_token)
        scale = sigma_data**2 + t**2
        return sigma_data**2 / scale * x_noisy + sigma_data * t / jnp.sqrt(scale) * r_update


NUM_METHODS = 17


class Conditioning(eqx.Module):
    """BioEmu2's inputs beyond the shared ones, stored in `Features.extras`."""

    method: Float[Array, "T M"]  # one-hot experimental method
    contacts: Float[Array, "T T 5"]
    contact_threshold: Float[Array, "T T"]
    center_atom: Int[Array, " T"]
    coordinate_mask: Bool[Array, " N"]  # atoms whose noisy positions the network reads
    dynamics: Dynamics | None = None


class ConditionedTrunk(AbstractTrunk[Features]):
    """`jfold.model.Trunk`, with BioEmu2's unconditional branch (`unconditional_state`)."""

    trunk: Trunk
    unconditional: Transition | None = None

    def init_state(self, inputs: TrunkInputs, *, key: PRNGKeyArray | None = None) -> TrunkState:
        return self.trunk.init_state(inputs, key=key)

    def __call__(
        self,
        features: Features,
        inputs: TrunkInputs,
        state: TrunkState,
        *,
        key: PRNGKeyArray | None = None,
    ) -> TrunkState:
        return self.trunk(features, inputs, state, key=key)

    def unconditional_state(self, inputs: TrunkInputs) -> TrunkState:
        """Return the trunk output without the MSA module and PairMixer stack."""
        z = inputs.z_init + self.unconditional(self.trunk.recycle_z.norm(inputs.z_init))
        return TrunkState(s=inputs.s_init, z=z)


def atom_features(
    ref_pos: Float[Array, "N 3"],
    charge: Float[Array, " N"],
    element: Int[Array, " N"],
    name_chars: Int[Array, "N 4"],
) -> Float[Array, "N 388"]:
    """Build the atom feature vector in the column order `convert.bioemu2` expects."""
    return jnp.concatenate(
        [
            ref_pos,
            charge[:, None],
            jax.nn.one_hot(element, NUM_ELEMENTS),
            jax.nn.one_hot(name_chars, NUM_NAME_CHAR_BINS).reshape(ref_pos.shape[0], -1),
        ],
        axis=-1,
    )


def token_features(
    res_type: Float[Array, "T 33"],
    profile: Float[Array, "T 33"],
    deletion_mean: Float[Array, " T"],
    pocket: Int[Array, " T"],
) -> Float[Array, "T 71"]:
    return jnp.concatenate(
        [
            res_type,
            profile,
            deletion_mean[:, None],
            jax.nn.one_hot(pocket, NUM_POCKET_CLASSES),
        ],
        axis=-1,
    )


def msa_features(
    msa: Float[Array, "S T 33"],
    has_deletion: Float[Array, "S T"],
    deletion_value: Float[Array, "S T"],
    paired: Float[Array, "S T"] | None,
) -> Float[Array, "S T F"]:
    """Build the MSA feature vector, with the paired flag only if `paired` is given."""
    extra = [has_deletion, deletion_value] + ([] if paired is None else [paired])
    return jnp.concatenate([msa] + [x[..., None] for x in extra], axis=-1)


def features(f: dict, *, paired: bool = True) -> Features:
    """Build `Features` from a BioEmu2 feature dict, where `res_type` and `msa` may be soft.
    Set `paired` if the model reads the MSA's pairing flag."""
    a = arrays(f)
    f32 = jnp.float32
    num_tokens = a["asym_id"].shape[0]
    atom_mask = a["atom_pad_mask"].astype(bool)
    token_mask = a["token_pad_mask"].astype(bool)
    atom_to_token = a["token_idx_at_atom"].astype(jnp.int32)
    features = Features(
        atoms=Atoms(
            features=atom_features(
                a["template_coords"].astype(f32),
                a["charge"].astype(f32),
                a["element"],
                a["name_encoding"],
            ),
            ref_pos=a["template_coords"].astype(f32),
            ref_space_uid=a["ref_space_uid"],
            atom_to_token=atom_to_token,
            mask=atom_mask,
        ),
        token_features=token_features(
            probabilities(a["res_type"], NUM_TOKEN_TYPES),
            a["profile"].astype(f32),
            a["deletion_mean"].astype(f32),
            a["pocket_feature"],
        ),
        relative_positions=relative_position_features(
            a["asym_id"],
            a["residue_index"],
            a["entity_id"],
            jnp.arange(num_tokens),
            a["sym_id"],
        ),
        token_mask=token_mask,
        asym_id=a["asym_id"],
        rep_atom=a["token_center_atom_index"],
        atom_position=jnp.zeros_like(atom_to_token),
        bonds=a["token_bonds"].astype(f32).reshape(num_tokens, num_tokens, 1),
        msa_features=msa_features(
            a["msa"].astype(f32),
            a["has_deletion"].astype(f32),
            a["deletion_value"].astype(f32),
            a["msa_paired"].astype(f32) if paired else None,
        ),
        msa_mask=a["msa_mask"].astype(bool),
    )
    dynamics = None
    if "coords_cond" in a:
        dynamics = Dynamics(
            coords=a["coords_cond"].astype(f32),
            lag=a["dt"].astype(f32).reshape(()),
            is_conditioned=a["is_dynamics_conditioned"].astype(bool).reshape(()),
        )
    conditioning = Conditioning(
        method=jax.nn.one_hot(a["method_feature"], NUM_METHODS),
        contacts=a["contact_conditioning"].astype(f32),
        contact_threshold=a["contact_threshold"].astype(f32),
        center_atom=a["token_center_atom_index"],
        coordinate_mask=a["atom_mask"].astype(bool),
        dynamics=dynamics,
    )
    return eqx.tree_at(lambda f: f.extras, features, conditioning, is_leaf=lambda x: x is None)
