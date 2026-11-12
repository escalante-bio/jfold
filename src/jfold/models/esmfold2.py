"""ESMFold2's model-specific pieces: the sliding-window atom transformer with 3D RoPE, the
ESMC language model, the confidence head's row pooling and the Parcae recurrence around the
pair trunk. The trunk only updates the pair, so its state keeps `s_inputs` as the single."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox.nn import Identity
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from jfold.blocks.atoms import Atoms, mean_atoms_to_tokens
from jfold.blocks.confidence import (
    AtomHead,
    DistanceEmbedding,
    PairHead,
    pairwise_distances,
)
from jfold.blocks.denoiser import DiffusionConditioning
from jfold.blocks.diffusion import DiffusionTransformer
from jfold.blocks.inputs import probabilities, relative_position_features
from jfold.blocks.msa import MSAModule, outer_product_sum
from jfold.blocks.pairformer import PairformerStack
from jfold.blocks.windows import CenteredWindows
from jfold.model import Recycle
from jfold.models.common import arrays
from jfold.nn import (
    Dropout,
    LayerNorm,
    Linear,
    SinglesToPair,
    SwiGLU,
    Transition,
    scan_layers,
    split_key,
)
from jfold.slots import (
    AbstractConfidenceModule,
    AbstractDiffusionModule,
    AbstractInputEmbedder,
    AbstractTrunk,
    AbstractTrunkInit,
    ConfidenceLogits,
    Features,
    TrunkInputs,
    TrunkState,
)

_RMS_EPS = float(jnp.finfo(jnp.float32).eps)  # torch's rms_norm default for float32
NUM_CLASSES = 33  # ESMFold2's residue classes


def atom_features(
    ref_pos: Float[Array, "N 3"],
    ref_charge: Float[Array, " N"],
    mask: Bool[Array, " N"],
    ref_element: Int[Array, " N"],
    ref_atom_name_chars: Int[Array, "N 4"],
) -> Float[Array, "N 389"]:
    """Build the atom feature vector that ESMFold2's encoders embed, with one-hots zeroed
    for padding."""
    real = mask[:, None].astype(jnp.float32)
    return jnp.concatenate(
        [
            ref_pos,
            ref_charge[:, None],
            real,
            jax.nn.one_hot(ref_element, 128) * real,
            jax.nn.one_hot(ref_atom_name_chars, 64).reshape(ref_pos.shape[0], -1) * real,
        ],
        axis=-1,
    )


def _rms_norm(x: Float[Array, "... d"]) -> Float[Array, "... d"]:
    return x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + _RMS_EPS)


def _rotate(x: Float[Array, "N H D"], cos: Float[Array, "N R"], sin: Float[Array, "N R"]):
    rot = 2 * cos.shape[-1]
    cos, sin = jnp.tile(cos, 2)[:, None], jnp.tile(sin, 2)[:, None]
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    x1, x2 = jnp.split(x_rot, 2, axis=-1)
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return jnp.concatenate([x_rot * cos + rotated * sin, x_pass], axis=-1)


class Rope3D(eqx.Module):
    """Rotary angles from each atom's reference position and conformer id, rounded to
    bfloat16 as in training."""

    head_dim: int = eqx.field(static=True)
    spatial_pairs: int = eqx.field(static=True)  # per axis
    uid_pairs: int = eqx.field(static=True)
    spatial_base: float = eqx.field(static=True)
    uid_base: float = eqx.field(static=True)

    def __call__(
        self, ref_pos: Float[Array, "N 3"], ref_space_uid: Int[Array, " N"]
    ) -> tuple[Float[Array, "N R"], Float[Array, "N R"]]:
        n = self.spatial_pairs
        spatial = 1.0 / self.spatial_base ** (jnp.arange(n, dtype=jnp.float32) / n)
        uid = 1.0 / self.uid_base ** (
            jnp.arange(self.uid_pairs, dtype=jnp.float32) / self.uid_pairs
        )
        angles = jnp.concatenate(
            [
                (ref_pos[:, :, None] * spatial).reshape(ref_pos.shape[0], 3 * n),
                ref_space_uid.astype(jnp.float32)[:, None] * uid,
            ],
            axis=-1,
        )
        half = self.head_dim // 2
        if angles.shape[-1] > half:
            raise ValueError("more rotary pairs than half a head")
        angles = jnp.pad(angles, [(0, 0), (0, half - angles.shape[-1])])
        bf16 = lambda x: x.astype(jnp.bfloat16).astype(jnp.float32)
        return bf16(jnp.cos(angles)), bf16(jnp.sin(angles))


class SlidingWindowAttention(eqx.Module):
    """Gated self-attention from each atom to the real atoms within `half_window` places of
    it."""

    linear_q: Linear
    linear_k: Linear
    linear_v: Linear
    linear_g: Linear
    linear_o: Linear
    num_heads: int = eqx.field(static=True)
    half_window: int = eqx.field(static=True)
    block: int = eqx.field(static=True, default=32)  # queries per block

    def __call__(
        self,
        x: Float[Array, "N d"],
        rope: tuple[Float[Array, "N R"], Float[Array, "N R"]],
        mask: Bool[Array, " N"],
    ) -> Float[Array, "N d"]:
        heads = lambda y: y.reshape(*y.shape[:-1], self.num_heads, -1)
        cos, sin = rope
        q = _rotate(_rms_norm(heads(self.linear_q(x))), cos, sin)
        k = _rotate(_rms_norm(heads(self.linear_k(x))), cos, sin)
        v = heads(self.linear_v(x))
        # Sort real atoms first, keeping their order, so that windows count only real atoms.
        order = jnp.argsort(~mask, stable=True)
        q, k, v, m = q[order], k[order], v[order], mask[order]
        windows = CenteredWindows(queries=self.block, keys=self.block + 2 * self.half_window)
        index, valid = windows.key_index(m)
        position = windows.query_blocks(jnp.arange(x.shape[0]))
        near = jnp.abs(position[:, :, None] - index[:, None, :]) <= self.half_window
        allowed = near & valid[:, None, :]
        out = jax.vmap(lambda q, k, v, a: jax.nn.dot_product_attention(q, k, v, mask=a[None]))(
            windows.query_blocks(q), k[index], v[index], allowed
        )
        out = out.reshape(-1, x.shape[-1])[: x.shape[0]][jnp.argsort(order)]
        return self.linear_o(out * jax.nn.sigmoid(self.linear_g(x)))


class SlidingWindowLayer(eqx.Module):
    """An adaLN-Zero block around sliding-window attention and a SwiGLU."""

    modulation: Linear  # silu(c) -> 6 Ca: shift, scale, gate for attention, then the FFN
    attention: SlidingWindowAttention
    ffn: SwiGLU

    def __call__(
        self,
        x: Float[Array, "N Ca"],
        modulation: Float[Array, "N 6Ca"],
        rope: tuple[Float[Array, "N R"], Float[Array, "N R"]],
        mask: Bool[Array, " N"],
    ) -> Float[Array, "N Ca"]:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = jnp.split(modulation, 6, axis=-1)
        x = x + gate_a * self.attention(_rms_norm(x) * (1 + scale_a) + shift_a, rope, mask)
        return x + gate_f * self.ffn(_rms_norm(x) * (1 + scale_f) + shift_f)


class SlidingWindowTransformer(eqx.Module):
    """Stacked `SlidingWindowLayer`s, whose modulations and rotary angles a sampler computes
    once."""

    layers: SlidingWindowLayer  # stacked
    rope: Rope3D

    def modulations(self, c: Float[Array, "N Ca"]) -> Float[Array, "layers N 6Ca"]:
        return jax.vmap(lambda layer: layer.modulation(jax.nn.silu(c)))(self.layers)

    def __call__(
        self,
        x: Float[Array, "N Ca"],
        modulations: Float[Array, "layers N 6Ca"],
        rope: tuple[Float[Array, "N R"], Float[Array, "N R"]],
        mask: Bool[Array, " N"],
    ) -> Float[Array, "N Ca"]:
        def step(layer_and_modulation, x, key):
            layer, modulation = layer_and_modulation
            return layer(x, modulation, rope, mask)

        return scan_layers(step, x, (self.layers, modulations))


class AtomContext(eqx.Module):
    """The atom transformer's inputs that do not depend on the noisy positions."""

    single: Float[Array, "N Ca"]
    rope: tuple[Float[Array, "N R"], Float[Array, "N R"]]
    modulations: Float[Array, "layers N 6Ca"]


class AtomEncoder(eqx.Module):
    """Encode atoms and average them per token, adding the noisy positions to the queries if
    `linear_r` is set."""

    embed_features: Linear  # atom feature vector -> Ca
    norm: LayerNorm
    linear_r: Linear | None  # scaled noisy positions -> Ca
    transformer: SlidingWindowTransformer
    linear_token: Linear  # Ca -> token width

    def condition(self, atoms: Atoms) -> AtomContext:
        c = self.norm(self.embed_features(atoms.features))
        return AtomContext(
            single=c,
            rope=self.transformer.rope(atoms.ref_pos, atoms.ref_space_uid),
            modulations=self.transformer.modulations(c),
        )

    def __call__(
        self,
        context: AtomContext,
        atom_to_token: Int[Array, " N"],
        mask: Bool[Array, " N"],
        num_tokens: int,
        r: Float[Array, "N 3"] | None = None,
    ) -> tuple[Float[Array, "T Ct"], Float[Array, "N Ca"]]:
        """Return per-token aggregates and atom queries for the decoder's skip connection.
        """
        q = context.single
        if self.linear_r is not None:
            q = q + self.linear_r(r)
        q = self.transformer(q, context.modulations, context.rope, mask)
        a = jax.nn.relu(self.linear_token(q))
        return mean_atoms_to_tokens(a, atom_to_token, mask, num_tokens), q


class AtomDecoder(eqx.Module):
    """Decode token updates to a position update per atom."""

    linear_a: Linear  # token width -> Ca
    transformer: SlidingWindowTransformer
    norm_out: LayerNorm
    linear_out: Linear  # Ca -> 3

    def __call__(
        self,
        a: Float[Array, "T Ct"],
        q: Float[Array, "N Ca"],
        context: AtomContext,
        atom_to_token: Int[Array, " N"],
        mask: Bool[Array, " N"],
    ) -> Float[Array, "N 3"]:
        q = q + self.linear_a(a)[atom_to_token]
        q = self.transformer(q, context.modulations, context.rope, mask)
        return self.linear_out(self.norm_out(q))


class InputEmbedder(AbstractInputEmbedder[Features]):
    """Compute `s_inputs`, zeroing the MSA profile and deletion mean if `msa_profile` is
    false."""

    encoder: AtomEncoder  # without noisy positions
    msa_profile: bool = eqx.field(static=True, default=True)

    def __call__(self, features: Features) -> Float[Array, "T Ci"]:
        atoms, token_features = features.atoms, features.token_features
        if not self.msa_profile:  # residue type, then profile and deletion mean
            token_features = token_features.at[:, NUM_CLASSES:].set(0.0)
        context = self.encoder.condition(atoms)
        a, _ = self.encoder(context, atoms.atom_to_token, atoms.mask, token_features.shape[0])
        return jnp.concatenate([a, token_features], axis=-1)


def token_features(
    res_type: Float[Array, "T 33"],
    profile: Float[Array, "T 33"],
    deletion_mean: Float[Array, " T"],
    mask: Bool[Array, " T"],
) -> Float[Array, "T 67"]:
    """Build the token features, with the residue type zeroed for padding."""
    res_type = res_type * mask[:, None]
    return jnp.concatenate([res_type, profile, deletion_mean[:, None]], axis=-1)


class DiffusionContext(eqx.Module):
    """Everything a denoising step needs that depends only on the trunk."""

    s_inputs: Float[Array, "T Ci"]
    token_mask: Bool[Array, " T"]
    atoms: Atoms
    encoder: AtomContext
    decoder: AtomContext
    token_biases: Float[Array, "L heads T T"]

    @property
    def atom_mask(self) -> Bool[Array, " N"]:
        """The atoms the denoiser moves, which a sampler centres and aligns on."""
        return self.atoms.mask


class DiffusionModule(AbstractDiffusionModule[Features]):
    """AF3 Algorithm 20 with ESMFold2's atom encoder and decoder, conditioned on `s_inputs`
    without the trunk single."""

    conditioning: DiffusionConditioning
    encoder: AtomEncoder
    norm_s: LayerNorm
    linear_s: Linear
    transformer: DiffusionTransformer
    norm_a: LayerNorm
    decoder: AtomDecoder

    @property
    def sigma_data(self) -> float:
        return self.conditioning.sigma_data

    def prepare(self, features: Features, inputs: TrunkInputs, state: TrunkState) -> DiffusionContext:
        z = self.conditioning.pair(state.z, inputs.extras.relative_positions)
        encoder = self.encoder.condition(features.atoms)
        decoder = AtomContext(
            single=encoder.single,
            rope=encoder.rope,
            modulations=self.decoder.transformer.modulations(encoder.single),
        )
        return DiffusionContext(
            s_inputs=inputs.s_inputs,
            token_mask=features.token_mask,
            atoms=features.atoms,
            encoder=encoder,
            decoder=decoder,
            token_biases=self.transformer.pair_biases(z),
        )

    def __call__(
        self,
        context: DiffusionContext,
        x_noisy: Float[Array, "N 3"],
        t: Float[Array, ""],
    ) -> Float[Array, "N 3"]:
        atoms, sigma_data = context.atoms, self.sigma_data
        no_trunk = jnp.zeros((context.s_inputs.shape[0], 0), context.s_inputs.dtype)
        s = self.conditioning.single(no_trunk, context.s_inputs, t)
        r = x_noisy / jnp.sqrt(t**2 + sigma_data**2)
        num_tokens = context.token_mask.shape[0]
        a, q = self.encoder(context.encoder, atoms.atom_to_token, atoms.mask, num_tokens, r)
        a = a + self.linear_s(self.norm_s(s))
        a = self.transformer(a, s, context.token_biases, context.token_mask)
        r_update = self.decoder(
            self.norm_a(a), q, context.decoder, atoms.atom_to_token, atoms.mask
        )
        scale = sigma_data**2 + t**2
        return sigma_data**2 / scale * x_noisy + sigma_data * t / jnp.sqrt(scale) * r_update


class ClampedOuterProductMean(eqx.Module):
    """The outer product mean, with the projection's bias also divided by the real row count
    (at least one)."""

    norm: LayerNorm
    linear_a: Linear
    linear_b: Linear
    linear_out: Linear

    def __call__(
        self, m: Float[Array, "S T Cm"], msa_mask: Bool[Array, "S T"]
    ) -> Float[Array, "T T Cz"]:
        outer, count = outer_product_sum(self.norm, self.linear_a, self.linear_b, m, msa_mask)
        return self.linear_out(outer) / jnp.maximum(count, 1)[..., None]


def msa_features(
    msa: Float[Array, "S T 33"],
    has_deletion: Float[Array, "S T"],
    deletion_value: Float[Array, "S T"],
    msa_mask: Bool[Array, "S T"],
) -> Float[Array, "S T 35"]:
    """Build the MSA feature vector, with residues zeroed where masked."""
    residue = msa * msa_mask[..., None]
    return jnp.concatenate([residue, has_deletion[..., None], deletion_value[..., None]], -1)


class RowAttentionPooling(eqx.Module):
    """Pool the pair into a single by attention over each row's real columns."""

    linear_score: Linear  # Cz -> 1
    linear_out: Linear  # Cz -> Cs

    def __call__(self, z: Float[Array, "T T Cz"], mask: Bool[Array, " T"]) -> Float[Array, "T Cs"]:
        scores = jnp.where(mask[None, :], self.linear_score(z)[..., 0], -1e9)
        return self.linear_out(jnp.einsum("ij,ijc->ic", jax.nn.softmax(scores, axis=-1), z))


class ConfidenceModule(AbstractConfidenceModule[Features]):
    """ESMFold2's confidence head, which pools a single from the refined pair for the
    per-atom heads. The experimental model has no PDE or resolved heads."""

    norm_s_inputs: LayerNorm
    norm_z: LayerNorm
    singles_to_pair: SinglesToPair
    distances: DistanceEmbedding
    stack: PairformerStack  # pair-only
    pooling: RowAttentionPooling
    pae: PairHead
    plddt: AtomHead
    pde: PairHead | None = None
    resolved: AtomHead | None = None

    def __call__(
        self,
        features: Features,
        inputs: TrunkInputs,
        state: TrunkState,
        coords: Float[Array, "N 3"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> ConfidenceLogits:
        extras, mask = inputs.extras, features.token_mask
        z = self.norm_z(state.z) + extras.relative_positions + extras.bonds
        z = z + self.singles_to_pair(self.norm_s_inputs(inputs.s_inputs))
        z = z + self.distances(pairwise_distances(coords[features.rep_atom]))
        _, out = self.stack(None, z, mask, key=key)
        z = z + out
        s = self.pooling(z, mask)
        atom_to_token = features.atoms.atom_to_token
        # Each atom's index within its token, capped at the heads' last position.
        position = jnp.minimum(features.atom_position, self.plddt.weight.shape[0] - 1)
        return ConfidenceLogits(
            pae=self.pae(z),
            pde=None if self.pde is None else self.pde(z),
            plddt=self.plddt(s, atom_to_token, position),
            resolved=None if self.resolved is None else self.resolved(s, atom_to_token, position),
        )


class ESMCLayer(eqx.Module):
    """A pre-norm ESMC transformer block with rotary positions and scaled-down residuals."""

    norm: LayerNorm
    linear_q: Linear
    linear_k: Linear
    linear_v: Linear
    norm_q: LayerNorm
    norm_k: LayerNorm
    linear_o: Linear
    transition: Transition
    num_heads: int = eqx.field(static=True)
    residual_scale: float = eqx.field(static=True)

    def __call__(
        self,
        x: Float[Array, "L D"],
        rope: tuple[Float[Array, "L R"], Float[Array, "L R"]],
        attend: Bool[Array, "L L"],
    ) -> Float[Array, "L D"]:
        heads = lambda y: y.reshape(*y.shape[:-1], self.num_heads, -1)
        h = self.norm(x)
        q = _rotate(heads(self.norm_q(self.linear_q(h))), *rope)
        k = _rotate(heads(self.norm_k(self.linear_k(h))), *rope)
        o = jax.nn.dot_product_attention(q, k, heads(self.linear_v(h)), mask=attend[None])
        x = x + self.linear_o(o.reshape(x.shape)) / self.residual_scale
        return x + self.transition(x) / self.residual_scale


class ESMC(eqx.Module):
    """The ESMC language model, returning every layer's input and the final normalised
    output."""

    embed: Linear  # token probability vectors -> D
    layers: ESMCLayer  # stacked
    norm: LayerNorm
    rope_base: float = eqx.field(static=True, default=10000.0)

    def __call__(
        self,
        tokens: Float[Array, "L V"],
        sequence_id: Int[Array, " L"],
        mask: Bool[Array, " L"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "layers+1 L D"]:
        """Run ESMC, with positions attending only within their `sequence_id`."""
        x = self.embed(tokens)
        head_dim = x.shape[-1] // self.layers.num_heads
        inv_freq = 1.0 / self.rope_base ** (
            jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
        )
        angles = jnp.arange(tokens.shape[0], dtype=jnp.float32)[:, None] * inv_freq
        rope = (jnp.cos(angles), jnp.sin(angles))
        attend = (sequence_id[:, None] == sequence_id[None, :]) & mask[None, :]

        @jax.checkpoint
        def body(x, layer):
            return layer(x, rope, attend), x

        x, hidden = jax.lax.scan(body, x, self.layers)
        return jnp.concatenate([hidden, self.norm(x)[None]])


# ESMC's token ids, from `esm`'s `EsmSequenceTokenizer`.
ESMC_VOCAB: dict[str, int] = {
    token: i
    for i, token in enumerate(
        ["<cls>", "<pad>", "<eos>", "<unk>", *"LAGVSERTIDPKQNFYMHWCXBUZO.-|", "<mask>"]
    )
}


class MaskedLMHead(eqx.Module):
    """ESMC's masked-language-model head (`EsmcForMaskedLM.lm_head`)."""

    dense: Linear
    norm: LayerNorm
    output: Linear  # D -> vocabulary

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... V"]:
        return self.output(self.norm(jax.nn.gelu(self.dense(x), approximate=False)))


class ESMCForMaskedLM(eqx.Module):
    """ESMC with its masked-language-model head, taking probability vectors so a soft
    sequence gets gradients."""

    head: MaskedLMHead
    # `jfold.pretrained` attaches the shared ESMC when loading.
    esmc: ESMC | None = None

    def __call__(
        self,
        tokens: Float[Array, "L V"],
        sequence_id: Int[Array, " L"] | None = None,
        mask: Bool[Array, " L"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "L V"]:
        """Return logits, treating all positions as one chain and attending to all by
        default."""
        if self.esmc is None:
            raise ValueError("this model has no ESMC: load it with `jfold.pretrained`")
        length = tokens.shape[0]
        sequence_id = jnp.zeros(length, int) if sequence_id is None else sequence_id
        mask = jnp.ones(length, bool) if mask is None else mask
        return self.head(self.esmc(tokens, sequence_id, mask, key=key)[-1])


class LanguageModelPair(eqx.Module):
    """Build a pair from a softmax-weighted mix of the language model's layer states."""

    norm_hidden: LayerNorm
    linear_hidden: Linear  # D -> Cz
    layer_logits: Float[Array, " layers"]
    linear_down: Linear
    mlp_in: Linear  # [x_i * x_j, x_i - x_j] -> Cz
    mlp_out: Linear
    norm_out: LayerNorm

    def __call__(self, hidden: Float[Array, "layers T D"]) -> Float[Array, "T T Cz"]:
        x = jnp.einsum(
            "l,ltc->tc",
            jax.nn.softmax(self.layer_logits),
            self.linear_hidden(self.norm_hidden(hidden)),
        )
        x = self.linear_down(x)
        pair = jnp.concatenate([x[:, None] * x[None, :], x[:, None] - x[None, :]], axis=-1)
        pair = self.mlp_out(jax.nn.gelu(self.mlp_in(pair), approximate=False))
        return self.norm_out(pair)


class Parcae(eqx.Module):
    """The pair trunk's linear recurrence, which decays the state and adds `B norm(input)`
    each loop."""

    norm: LayerNorm
    log_a: Float[Array, " Cz"]
    log_delta: Float[Array, " Cz"]
    b: Float[Array, "Cz Cz"]
    readout: Linear

    def __call__(
        self, z: Float[Array, "T T Cz"], injected: Float[Array, "T T Cz"]
    ) -> Float[Array, "T T Cz"]:
        delta = jax.nn.softplus(self.log_delta)
        a = jnp.exp(-delta * jnp.exp(self.log_a))
        return a * z + self.norm(injected) @ (delta[:, None] * self.b).T


class LanguageModelInput(eqx.Module):
    """ESMFold2's language-model input: either precomputed ESMC states or ESMC's input for
    the trunk init to run."""

    hidden: Float[Array, "layers T D"] | None = None
    tokens: Float[Array, "L V"] | None = None
    sequence_id: Int[Array, " L"] | None = None
    positions: Int[Array, " T"] | None = None


class TrunkExtras(eqx.Module):
    """What the trunk, diffusion and confidence modules read beyond `TrunkInputs`."""

    relative_positions: Float[Array, "T T Cz"]  # embedded
    bonds: Float[Array, "T T Cz"]  # embedded
    lm_pair: Float[Array, "T T Cz"]
    msa_features: Float[Array, "S T 35"] | None  # with masked columns, as every loop reads them
    msa_mask: Bool[Array, "S T"] | None


def mask_msa_columns(
    msa_mask: Bool[Array, "S T"], rate: float, key: PRNGKeyArray
) -> Bool[Array, "S T"]:
    """Mask a random fraction `rate` of token columns in every MSA row except the query."""
    keep = jax.random.uniform(key, msa_mask.shape[1:]) >= rate
    return msa_mask & (keep[None, :] | (jnp.arange(msa_mask.shape[0]) == 0)[:, None])


def language_model_hidden(esmc: ESMC | None, features: Features) -> Float[Array, "layers T D"]:
    """Return the features' precomputed ESMC states, or else run ESMC on their sequence."""
    lm = features.extras
    if lm.hidden is not None:
        return lm.hidden
    if esmc is None:
        raise ValueError(
            "ESMFold2 needs its language model: this model has no ESMC (load it with "
            "`jfold.pretrained.load`, which attaches it) and the features carry no "
            "`lm_hidden_states`"
        )
    mask = jnp.ones(lm.sequence_id.shape, bool)
    hidden = esmc(lm.tokens, lm.sequence_id, mask)
    return jnp.swapaxes(token_hidden_states(hidden, lm.positions), 0, 1)


class TrunkInit(AbstractTrunkInit[Features]):
    """Build the initial pair, the language model's pair and the column-masked MSA. With
    `lm_dropout`, as in the experimental type, the language model's pair is added here
    instead of on every loop."""

    singles_to_pair: SinglesToPair  # s_inputs -> pair
    relative_positions: Linear
    bonds: Linear
    language_model: LanguageModelPair
    # `jfold.pretrained` attaches the shared ESMC when loading.
    esmc: ESMC | None = None
    lm_dropout: Dropout | Identity | None = None
    # The fraction of MSA columns masked outside the query, drawn once per forward pass.
    msa_column_mask_rate: float = eqx.field(static=True, default=0.0)

    def __call__(
        self, features: Features, s_inputs: Float[Array, "T Ci"], *, key: PRNGKeyArray | None = None
    ) -> TrunkInputs:
        k_columns, k_dropout = split_key(key)
        relative_positions = self.relative_positions(features.relative_positions)
        bonds = self.bonds(features.bonds)
        lm_pair = self.language_model(language_model_hidden(self.esmc, features))
        msa, msa_mask = features.msa_features, features.msa_mask
        if msa is not None and self.msa_column_mask_rate > 0:
            if k_columns is None:
                raise ValueError("ESMFold2 masks MSA columns at random: pass a key")
            msa_mask = mask_msa_columns(msa_mask, self.msa_column_mask_rate, k_columns)
            residue = msa[..., :-2] * msa_mask[..., None]  # masked residues are zero
            msa = jnp.concatenate([residue, msa[..., -2:]], axis=-1)
        z_init = self.singles_to_pair(s_inputs) + relative_positions + bonds
        if self.lm_dropout is not None:
            z_init = z_init + self.lm_dropout(lm_pair, key=k_dropout)
        return TrunkInputs(
            s_inputs=s_inputs,
            s_init=s_inputs,
            z_init=z_init,
            extras=TrunkExtras(
                relative_positions=relative_positions,
                bonds=bonds,
                lm_pair=lm_pair,
                msa_features=msa,
                msa_mask=msa_mask,
            ),
        )


class ParcaeTrunk(AbstractTrunk[Features]):
    """The release type's pair trunk, a Parcae recurrence from a random state that `finish`
    reads out through the coda."""

    lm_encoder: PairformerStack | None
    msa: MSAModule | None
    parcae: Parcae
    trunk: PairformerStack
    coda: PairformerStack
    # On by default because upstream applies this dropout at inference too.
    lm_dropout: Dropout | Identity = eqx.field(default_factory=Identity)

    def init_state(self, inputs: TrunkInputs, *, key: PRNGKeyArray | None = None) -> TrunkState:
        """Draw a random pair from a normal with std `sqrt(2 / (5 Cz))`, truncated at 3 std.
        """
        if key is None:
            raise ValueError("ESMFold2's trunk starts from a random state: pass a key")
        std = math.sqrt(2.0 / (5.0 * inputs.z_init.shape[-1]))
        z = std * jax.random.truncated_normal(key, -3.0, 3.0, inputs.z_init.shape)
        return TrunkState(s=inputs.s_init, z=z)

    def __call__(
        self,
        features: Features,
        inputs: TrunkInputs,
        state: TrunkState,
        *,
        key: PRNGKeyArray | None = None,
    ) -> TrunkState:
        k_dropout, k_lm, k_msa, k_trunk = split_key(key, 4)
        mask, extras = features.token_mask, inputs.extras
        injected = inputs.z_init
        lm_pair = self.lm_dropout(extras.lm_pair, key=k_dropout)
        if self.lm_encoder is None:
            injected = injected + lm_pair
        if self.msa is not None and extras.msa_features is not None:
            # The MSA module's output pair (its input plus updates) replaces the input.
            _, injected = self.msa(
                injected, extras.msa_features, inputs.s_inputs, extras.msa_mask, mask, key=k_msa
            )
        if self.lm_encoder is not None:
            injected = injected + self.lm_encoder(None, lm_pair, mask, key=k_lm)[1]
        z = self.parcae(state.z, injected)
        return TrunkState(s=state.s, z=self.trunk(None, z, mask, key=k_trunk)[1])

    def finish(
        self, features: Features, state: TrunkState, *, key: PRNGKeyArray | None = None
    ) -> TrunkState:
        z = self.coda(None, self.parcae.readout(state.z), features.token_mask, key=key)[1]
        return TrunkState(s=state.s, z=z)


class TrackedMSA(eqx.Module):
    """The experimental model's MSA update, zero when the MSA has no rows after the query.
    """

    module: MSAModule

    def __call__(
        self,
        z: Float[Array, "T T Cz"],
        msa_features: Float[Array, "S T F"],
        s_inputs: Float[Array, "T Ci"],
        msa_mask: Bool[Array, "S T"],
        mask: Bool[Array, " T"],
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, "T T Cz"]:
        _, z = self.module(z, msa_features, s_inputs, msa_mask, mask, key=key)
        return jnp.where(msa_mask[1:].any(), z, 0.0)


class RecyclingPairTrunk(AbstractTrunk[Features]):
    """The experimental type's pair trunk, which recycles the previous pair as in AF3."""

    recycle: Recycle
    msa: TrackedMSA | None
    trunk: PairformerStack

    def init_state(self, inputs: TrunkInputs, *, key: PRNGKeyArray | None = None) -> TrunkState:
        return TrunkState(s=inputs.s_init, z=jnp.zeros_like(inputs.z_init))

    def __call__(
        self,
        features: Features,
        inputs: TrunkInputs,
        state: TrunkState,
        *,
        key: PRNGKeyArray | None = None,
    ) -> TrunkState:
        k_msa, k_trunk = split_key(key)
        mask, extras = features.token_mask, inputs.extras
        z = self.recycle(inputs.z_init, state.z)
        if self.msa is not None and extras.msa_features is not None:
            z = z + self.msa(
                z, extras.msa_features, inputs.s_inputs, extras.msa_mask, mask, key=k_msa
            )
        return TrunkState(s=state.s, z=self.trunk(None, z, mask, key=k_trunk)[1])


def features(f: dict, *, r_max: int = 32, s_max: int = 2) -> Features:
    """Build `Features` from the keyword arguments of `EsmFold2Model.forward`, where
    `res_type` and `msa` may be soft. Without `lm_hidden_states`, the chain, residue and
    molecule ids must be concrete because they set the layout of ESMC's input."""
    layout = None if "lm_hidden_states" in f else _static_lm_layout(f)
    a = arrays(f)
    token_mask = a["token_attention_mask"].astype(bool)
    atom_mask = a["atom_attention_mask"].astype(bool)
    num_tokens, num_atoms = token_mask.shape[0], atom_mask.shape[0]
    atom_to_token = jnp.where(atom_mask, a["atom_to_token"], 0)
    res_type = probabilities(a["res_type"], 33)
    msa = msa_mask = None
    if "msa" in a:
        msa = probabilities(a["msa"], 33)
        msa_mask = a.get("msa_attention_mask", jnp.broadcast_to(token_mask, msa.shape[:2]))
        msa_mask = msa_mask.astype(bool)
        zeros = jnp.zeros(msa.shape[:2], jnp.float32)
        msa = msa_features(
            msa,
            a.get("has_deletion", zeros).astype(jnp.float32),
            a.get("deletion_value", zeros).astype(jnp.float32),
            msa_mask,
        )
        profile = msa[..., :33].sum(0) / jnp.maximum(msa_mask.sum(0), 1)[:, None]
    else:
        profile = res_type * token_mask[:, None]
    deletion_mean = a.get("deletion_mean", jnp.zeros(num_tokens)).astype(jnp.float32)
    atom_index = jnp.arange(num_atoms)
    first_atom = jax.ops.segment_min(
        jnp.where(atom_mask, atom_index, num_atoms), atom_to_token, num_tokens
    )
    asym_id = a["asym_id"]
    lm_hidden = a.get("lm_hidden_states")
    ref_pos = a["ref_pos"].astype(jnp.float32)
    return Features(
        atoms=Atoms(
            features=atom_features(
                ref_pos,
                a["ref_charge"].astype(jnp.float32),
                atom_mask,
                a["ref_element"],
                a["ref_atom_name_chars"],
            ),
            ref_pos=ref_pos,
            ref_space_uid=a["ref_space_uid"],
            atom_to_token=atom_to_token,
            mask=atom_mask,
        ),
        token_features=token_features(res_type, profile, deletion_mean, token_mask),
        relative_positions=relative_position_features(
            asym_id,
            a["residue_index"],
            a["entity_id"],
            a["token_index"],
            a["sym_id"],
            r_max=r_max,
            s_max=s_max,
            chain_offset=asym_id[:, None] != asym_id[None, :],
        ),
        bonds=a["token_bonds"].astype(jnp.float32).reshape(num_tokens, num_tokens, 1),
        token_mask=token_mask,
        asym_id=asym_id,
        rep_atom=a["distogram_atom_idx"],
        atom_position=jnp.where(atom_mask, atom_index - first_atom[atom_to_token], 0),
        msa_features=msa,
        msa_mask=msa_mask,
        extras=LanguageModelInput(
            hidden=None if lm_hidden is None else jnp.swapaxes(lm_hidden, 0, 1).astype(jnp.float32),
            tokens=None if layout is None else lm_input(layout, esmc_residues(res_type)),
            sequence_id=None if layout is None else jnp.array(layout.sequence_id),
            positions=None if layout is None else jnp.array(layout.positions),
        ),
    )


def _static_lm_layout(f: dict) -> "LMLayout":
    keys = ("asym_id", "residue_index", "mol_type", "token_attention_mask")
    missing = [k for k in keys if k not in f]
    if missing:
        raise ValueError(f"ESMC's input needs {missing} (or pass `lm_hidden_states`)")
    if any(isinstance(f[k], jax.core.Tracer) for k in keys):
        raise ValueError(
            "ESMC's input needs concrete chain, residue and molecule ids: build the features "
            "outside `jit` (only `res_type` and `msa` may be traced), or pass "
            "`lm_hidden_states`"
        )
    ids = {k: np.asarray(f[k]) for k in keys}
    protein = (ids["mol_type"] == 0) & ids["token_attention_mask"].astype(bool)
    return lm_layout(ids["asym_id"], ids["residue_index"], protein)


# ESMC's column for each of ESMFold2's 33 residue classes, with `<unk>` for non-protein.
_ESMC_COLUMN = np.array(
    [3, 3, 5, 10, 17, 13, 23, 16, 9, 6, 21, 12, 4, 15, 20, 18, 14, 8, 11, 22, 19, 7, 24] + [3] * 10
)
_BOS, _EOS = 0, 2


def esmc_residues(res_type: Float[Array, "T 33"], vocab: int = 64) -> Float[Array, "T V"]:
    """Map residue probability vectors to ESMC's amino-acid columns."""
    return res_type @ jax.nn.one_hot(_ESMC_COLUMN, vocab)


class LMLayout(eqx.Module):
    """Where a system's protein residues sit in ESMC's input, with each chain wrapped in BOS
    and EOS."""

    source: Int[Array, " L"]  # the system token in each row, -1 for BOS/EOS
    special: Int[Array, " L"]  # the BOS/EOS token id in each special row
    sequence_id: Int[Array, " L"]  # each row's chain (`ESMC`'s `sequence_id`)
    positions: Int[Array, " T"]  # each system token's row, -1 off-protein


def lm_layout(asym_id: np.ndarray, residue_index: np.ndarray, protein: np.ndarray) -> LMLayout:
    """Build an `LMLayout` from numpy chain and residue ids."""
    index = np.flatnonzero(protein)
    keys = list(zip(asym_id[index].tolist(), residue_index[index].tolist(), strict=True))
    first = {}
    for i, key in zip(index.tolist(), keys, strict=True):
        first.setdefault(key, i)
    source, special, sequence_id, position_of = [], [], [], {}
    for chain, cid in enumerate(sorted({a for a, _ in first})):
        source.append(-1)
        special.append(_BOS)
        for key, i in first.items():
            if key[0] == cid:
                position_of[key] = len(source)
                source.append(i)
                special.append(_BOS)
        source.append(-1)
        special.append(_EOS)
        sequence_id += [chain] * (len(source) - len(sequence_id))
    positions = np.full(len(asym_id), -1)
    positions[index] = [position_of[key] for key in keys]
    return LMLayout(
        source=np.array(source),
        special=np.array(special),
        sequence_id=np.array(sequence_id),
        positions=positions,
    )


def lm_input(layout: LMLayout, residues: Float[Array, "T V"]) -> Float[Array, "L V"]:
    """Build ESMC's input matrix from residue vectors and one-hot BOS and EOS rows."""
    special = jax.nn.one_hot(layout.special, residues.shape[-1])
    return jnp.where(layout.source[:, None] >= 0, residues[layout.source], special)


def token_hidden_states(
    hidden: Float[Array, "layers L D"], positions: Int[Array, " T"]
) -> Float[Array, "T layers D"]:
    """Gather ESMC's states for each system token, zero off-protein."""
    states = jnp.where(positions[None, :, None] >= 0, hidden[:, positions], 0.0)
    return jnp.swapaxes(states, 0, 1)
