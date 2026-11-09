"""ESMFold2's model-specific pieces: the sliding-window atom transformer with 3D RoPE, its
atom encoder and decoder, the outer product mean's count clamp, the confidence head's
row pooling, the ESMC language model and its pair projection, and the Parcae recurrence
around the pair trunk.

Everything else (the pair trunk, MSA modules, diffusion transformer and conditioning,
relative positions, confidence heads, the experimental type's recycling) runs on the shared
blocks. Both model types are a `jfold.model.StructureModel`: the release type's trunk is the
Parcae recurrence (`ParcaeTrunk`), the experimental type's recycles the pair
(`RecyclingPairTrunk`). The trunk init holds ESMC, which it runs on the features' sequence
(`LanguageModelInput`, the features' `extras`) unless they carry precomputed states. The
trunk is pair-only: its state carries `s_inputs` as the single, unchanged. Inputs are
unbatched, with masks that are False for padding."""

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
    ConfidenceLogits,
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
from jfold.model import Features, Recycle, TrunkInputs, TrunkState
from jfold.models.common import arrays
from jfold.nn import Dropout, LayerNorm, Linear, SinglesToPair, Transition, scan_layers, split_key
from jfold.slots import (
    AbstractConfidenceModule,
    AbstractDiffusionModule,
    AbstractInputEmbedder,
    AbstractTrunk,
    AbstractTrunkInit,
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
    """The atom feature vector ESMFold2's encoders embed: position, charge, mask, element
    (128 classes) and name characters (4 x 64 classes), one-hots zero for padding."""
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
    """Rotary embedding on the first `2R` channels of each head, halves rotated together."""
    rot = 2 * cos.shape[-1]
    cos, sin = jnp.tile(cos, 2)[:, None], jnp.tile(sin, 2)[:, None]
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    x1, x2 = jnp.split(x_rot, 2, axis=-1)
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return jnp.concatenate([x_rot * cos + rotated * sin, x_pass], axis=-1)


class Rope3D(eqx.Module):
    """Rotary angles from each atom's reference position (per axis) and conformer id,
    zero-padded to half a head; the cosines and sines are rounded to bfloat16, as the
    model was trained."""

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
    """Gated self-attention over atoms, each attending to the real atoms within
    `half_window` places of it in the sequence of real atoms; queries and keys are
    RMS-normalized and rotated by `Rope3D`'s angles."""

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
        # Real atoms first, in order: windows then count real atoms only.
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


class SwiGLU(eqx.Module):
    """`linear_out(silu(linear_a(x)) * linear_b(x))`."""

    linear_a: Linear
    linear_b: Linear
    linear_out: Linear

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "... d"]:
        return self.linear_out(jax.nn.silu(self.linear_a(x)) * self.linear_b(x))


class SlidingWindowLayer(eqx.Module):
    """adaLN-Zero block: RMS norms scaled, shifted and gated from the atom single through
    one modulation per layer, around sliding-window attention and a SwiGLU."""

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
    """Identical `SlidingWindowLayer`s, stacked and scanned. `modulations` and the rotary
    angles depend only on the atom single and reference conformers, so a sampler computes
    them once."""

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
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "N Ca"]:
        def step(layer_and_modulation, x, k):
            layer, modulation = layer_and_modulation
            return layer(x, modulation, rope, mask)

        return scan_layers(step, x, (self.layers, modulations), key=key)


class AtomContext(eqx.Module):
    """What the atom transformer reads that does not depend on noisy positions."""

    single: Float[Array, "N Ca"]
    rope: tuple[Float[Array, "N R"], Float[Array, "N R"]]
    modulations: Float[Array, "layers N 6Ca"]


class AtomEncoder(eqx.Module):
    """Atom features, normalized, through the sliding-window transformer and averaged per
    token. With `linear_r` (the diffusion module's encoder) the noisy positions are added
    to the queries."""

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
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "T Ct"], Float[Array, "N Ca"]]:
        """Per-token aggregates, and the atom queries for the decoder's skip."""
        q = context.single
        if self.linear_r is not None:
            q = q + self.linear_r(r)
        q = self.transformer(q, context.modulations, context.rope, mask, key=key)
        a = jax.nn.relu(self.linear_token(q))
        return mean_atoms_to_tokens(a, atom_to_token, mask, num_tokens), q


class AtomDecoder(eqx.Module):
    """Token updates broadcast to atoms, the sliding-window transformer, and a position
    update per atom."""

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
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "N 3"]:
        q = q + self.linear_a(a)[atom_to_token]
        q = self.transformer(q, context.modulations, context.rope, mask, key=key)
        return self.linear_out(self.norm_out(q))


class InputEmbedder(AbstractInputEmbedder):
    """`s_inputs`: the atom encoder's per-token aggregates, then the token features.
    Without `msa_profile` (checkpoints with `disable_msa_features`) the token features' MSA
    profile and deletion mean are zero."""

    encoder: AtomEncoder  # without noisy positions
    msa_profile: bool = eqx.field(static=True, default=True)

    def __call__(self, features: Features, *, key: PRNGKeyArray | None = None) -> Float[Array, "T Ci"]:
        token_features = features.token_features
        if not self.msa_profile:  # residue type, then profile and deletion mean
            token_features = token_features.at[:, NUM_CLASSES:].set(0.0)
        return self.embed(features.atoms, token_features, key=key)

    def embed(
        self,
        atoms: Atoms,
        token_features: Float[Array, "T F"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "T Ci"]:
        context = self.encoder.condition(atoms)
        a, _ = self.encoder(
            context, atoms.atom_to_token, atoms.mask, token_features.shape[0], key=key
        )
        return jnp.concatenate([a, token_features], axis=-1)


def token_features(
    res_type: Float[Array, "T 33"],
    profile: Float[Array, "T 33"],
    deletion_mean: Float[Array, " T"],
    mask: Bool[Array, " T"],
) -> Float[Array, "T 67"]:
    """The token features after the atom aggregates: residue type (a probability vector
    over ESMFold2's 33 classes, zero for padding), MSA profile and deletion mean."""
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
        """The atoms the denoiser moves (real, not padding): what a sampler centres and
        aligns on."""
        return self.atoms.mask


class DiffusionModule(AbstractDiffusionModule):
    """AF3 Algorithm 20 with EDM preconditioning, on ESMFold2's atom encoder and decoder.
    The single is conditioned on `s_inputs` alone (no trunk single), and the atom
    encoder does not see the trunk."""

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
        return self.context(
            features.atoms,
            inputs.s_inputs,
            state.z,
            inputs.extras.relative_positions,
            features.token_mask,
        )

    def context(
        self,
        atoms: Atoms,
        s_inputs: Float[Array, "T Ci"],
        z_trunk: Float[Array, "T T Cz"],
        relative_positions: Float[Array, "T T Cz"],
        token_mask: Bool[Array, " T"],
    ) -> DiffusionContext:
        """`relative_positions`: the trunk's embedded relative position encoding."""
        z = self.conditioning.pair(z_trunk, relative_positions)
        encoder = self.encoder.condition(atoms)
        decoder = AtomContext(
            single=encoder.single,
            rope=encoder.rope,
            modulations=self.decoder.transformer.modulations(encoder.single),
        )
        return DiffusionContext(
            s_inputs=s_inputs,
            token_mask=token_mask,
            atoms=atoms,
            encoder=encoder,
            decoder=decoder,
            token_biases=self.transformer.pair_biases(z),
        )

    def __call__(
        self,
        context: DiffusionContext,
        x_noisy: Float[Array, "N 3"],
        t: Float[Array, ""],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "N 3"]:
        """The denoised positions at noise level `t`."""
        k_encoder, k_transformer, k_decoder = split_key(key, 3)
        atoms, sigma_data = context.atoms, self.sigma_data
        no_trunk = jnp.zeros((context.s_inputs.shape[0], 0), context.s_inputs.dtype)
        s = self.conditioning.single(no_trunk, context.s_inputs, t)
        r = x_noisy / jnp.sqrt(t**2 + sigma_data**2)
        num_tokens = context.token_mask.shape[0]
        a, q = self.encoder(
            context.encoder, atoms.atom_to_token, atoms.mask, num_tokens, r, key=k_encoder
        )
        a = a + self.linear_s(self.norm_s(s))
        a = self.transformer(a, s, context.token_biases, context.token_mask, key=k_transformer)
        r_update = self.decoder(
            self.norm_a(a), q, context.decoder, atoms.atom_to_token, atoms.mask, key=k_decoder
        )
        scale = sigma_data**2 + t**2
        return sigma_data**2 / scale * x_noisy + sigma_data * t / jnp.sqrt(scale) * r_update


class ClampedOuterProductMean(eqx.Module):
    """The outer product mean, projected and then divided by the number of real rows
    (at least one), bias included."""

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
    """The MSA feature vector `[S, T, 35]`: residue (a probability vector over 33 classes,
    zero where masked), has-deletion, deletion value."""
    residue = msa * msa_mask[..., None]
    return jnp.concatenate([residue, has_deletion[..., None], deletion_value[..., None]], -1)


class RowAttentionPooling(eqx.Module):
    """A single from the pair: each row's attention-weighted mean over real columns."""

    linear_score: Linear  # Cz -> 1
    linear_out: Linear  # Cz -> Cs

    def __call__(self, z: Float[Array, "T T Cz"], mask: Bool[Array, " T"]) -> Float[Array, "T Cs"]:
        scores = jnp.where(mask[None, :], self.linear_score(z)[..., 0], -1e9)
        return self.linear_out(jnp.einsum("ij,ijc->ic", jax.nn.softmax(scores, axis=-1), z))


class ConfidenceModule(AbstractConfidenceModule):
    """ESMFold2's confidence head: the trunk pair plus embeddings of `s_inputs`, the
    relative positions and bonds, and the predicted distances, through a pair stack (its
    output added back to its input); a single pooled from the pair rows feeds per-atom
    pLDDT and resolved heads. The experimental model's head has neither PDE nor resolved
    logits (None in its `ConfidenceLogits`)."""

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
        extras = inputs.extras
        return self.logits(
            inputs.s_inputs,
            state.z,
            coords[features.rep_atom],
            features.atoms.atom_to_token,
            features.atom_position,
            features.token_mask,
            extras.relative_positions + extras.bonds,
            key=key,
        )

    def logits(
        self,
        s_inputs: Float[Array, "T Ci"],
        z_trunk: Float[Array, "T T Cz"],
        rep_coords: Float[Array, "T 3"],
        atom_to_token: Int[Array, " N"],
        atom_position: Int[Array, " N"],
        mask: Bool[Array, " T"],
        pair_features: Float[Array, "T T Cz"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> ConfidenceLogits:
        """`pair_features`: the trunk's embedded relative positions and bonds.
        `atom_position`: each atom's index within its token (at most 22)."""
        z = self.norm_z(z_trunk)
        if pair_features is not None:
            z = z + pair_features
        z = z + self.singles_to_pair(self.norm_s_inputs(s_inputs))
        z = z + self.distances(pairwise_distances(rep_coords))
        _, out = self.stack(None, z, mask, key=key)
        z = z + out
        s = self.pooling(z, mask)
        position = jnp.minimum(atom_position, self.plddt.weight.shape[0] - 1)
        return ConfidenceLogits(
            pae=self.pae(z),
            pde=None if self.pde is None else self.pde(z),
            plddt=self.plddt(s, atom_to_token, position),
            resolved=None if self.resolved is None else self.resolved(s, atom_to_token, position),
        )


class ESMCLayer(eqx.Module):
    """A pre-norm transformer block of ESMC: ungated attention with layer-normed queries
    and keys and rotary positions, then a SwiGLU transition, each residual scaled down."""

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
    """The ESMC protein language model, returning every layer's input and the final
    normalized output."""

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
        """`tokens`: probability vectors over ESMC's vocabulary (`lm_input`).
        `sequence_id`: which chain each position is in; positions attend within their
        chain."""
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


class LanguageModelPair(eqx.Module):
    """The pair from the language model's hidden states: each layer projected, combined
    by learned softmax weights, then an outer product and difference through an MLP."""

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
    """The pair trunk's linear recurrence: each loop decays the state by `a` and injects
    `B norm(input)` before the pair stack; `readout` maps the final state out."""

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
    """ESMFold2's language-model input (`Features.extras`): ESMC's precomputed states per
    token (`hidden`, zero off-protein), or ESMC's input for the trunk init to run it on
    (`lm_input`): residue probability vectors in ESMC's vocabulary, each row's chain, each
    token's row."""

    hidden: Float[Array, "layers T D"] | None = None
    tokens: Float[Array, "L V"] | None = None
    sequence_id: Int[Array, " L"] | None = None
    positions: Int[Array, " T"] | None = None


class TrunkExtras(eqx.Module):
    """What ESMFold2's trunk, diffusion and confidence modules read beyond `TrunkInputs`
    (its `extras`), computed once per forward pass."""

    relative_positions: Float[Array, "T T Cz"]  # embedded
    bonds: Float[Array, "T T Cz"]  # embedded
    lm_pair: Float[Array, "T T Cz"]
    msa_features: Float[Array, "S T 35"] | None  # column-masked, as every loop reads them
    msa_mask: Bool[Array, "S T"] | None


def mask_msa_columns(
    msa_mask: Bool[Array, "S T"], rate: float, key: PRNGKeyArray
) -> Bool[Array, "S T"]:
    """`msa_mask` with a random `rate` of the token columns masked in every row but the
    query (row 0)."""
    keep = jax.random.uniform(key, msa_mask.shape[1:]) >= rate
    return msa_mask & (keep[None, :] | (jnp.arange(msa_mask.shape[0]) == 0)[:, None])


def language_model_hidden(esmc: ESMC | None, features: Features) -> Float[Array, "layers T D"]:
    """ESMC's states per token: the features' precomputed ones, else ESMC run on their
    sequence (so a soft sequence's gradient reaches ESMC)."""
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


class TrunkInit(AbstractTrunkInit):
    """The initial pair (`s_inputs`, relative positions and bonds), the language model's
    pair (from ESMC, which it holds), and the MSA with `msa_column_mask_rate` of its
    columns masked outside the query, once per forward pass. The single is `s_inputs`.
    With `lm_dropout` (the experimental type), the language model's pair is added to the
    initial pair here, through that dropout; the release type's trunk injects it every
    loop instead."""

    singles_to_pair: SinglesToPair  # s_inputs -> pair
    relative_positions: Linear
    bonds: Linear
    language_model: LanguageModelPair
    # The language model. None only as stored: `jfold.pretrained` attaches the shared
    # ESMC on loading (`jfold.pretrained.recipes.esmfold2`).
    esmc: ESMC | None = None
    lm_dropout: Dropout | Identity | None = None
    # The share of MSA columns masked outside the query, once per forward pass.
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


class ParcaeTrunk(AbstractTrunk):
    """The release type's pair trunk: a Parcae recurrence from a random state, injecting
    the initial pair, the language model's pair and the MSA every loop, then the pair
    stack; `finish` reads the state out through the coda."""

    lm_encoder: PairformerStack | None
    msa: MSAModule | None
    parcae: Parcae
    trunk: PairformerStack
    coda: PairformerStack
    # Dropout on the language model's pair, drawn afresh every loop. On by default
    # (`inference=False`): upstream applies it at inference too.
    lm_dropout: Dropout | Identity = eqx.field(default_factory=Identity)

    def init_state(self, inputs: TrunkInputs, *, key: PRNGKeyArray | None = None) -> TrunkState:
        """A random pair: truncated normal, std `sqrt(2 / (5 Cz))`, cut at 3 std."""
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
    """The experimental model's MSA update: the MSA module's output pair (its input plus
    updates), zero when the MSA has no real rows past the query."""

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


class RecyclingPairTrunk(AbstractTrunk):
    """The experimental type's pair trunk: from zeros, each loop recycles the previous pair
    onto the initial one (AF3-style), adds the MSA update, then runs the pair stack."""

    recycle: Recycle
    msa: TrackedMSA | None
    trunk: PairformerStack

    def init_state(self, inputs: TrunkInputs, *, key: PRNGKeyArray | None = None) -> TrunkState:
        del key
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
    """`Features` from the upstream model's forward inputs (`EsmFold2Model.forward`'s
    keyword arguments, unbatched arrays). `res_type` (`[T]`) and `msa` (`[S, T]`) may
    instead be probability vectors over the 33 classes (`[T, 33]`, `[S, T, 33]`), e.g. a
    soft sequence. `lm_hidden_states` (`[T, layers, D]`, zero off-protein) are precomputed
    ESMC states per token. Without them the features carry ESMC's input instead, built
    from `res_type` (so a soft sequence reaches the language model), for the trunk init to
    run ESMC on (both in `extras`, a `LanguageModelInput`). The MSA profile is computed from
    the MSA, as upstream (the input embedder zeroes it where its checkpoint does:
    `InputEmbedder.msa_profile`). Traceable: the arrays
    may be traced under `jit`, except that without `lm_hidden_states` ESMC's layout (which
    token goes in which row) needs the chain, residue and molecule ids concrete."""
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
    """ESMC's layout from `f`'s chain, residue and molecule ids, which must be concrete."""
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


# ESMC's column for each of ESMFold2's 33 residue classes: the 20 amino acids, `X` (24)
# for the unknown residue (class 22), and `<unk>` (3) for the classes that are not protein.
_ESMC_COLUMN = np.array(
    [3, 3, 5, 10, 17, 13, 23, 16, 9, 6, 21, 12, 4, 15, 20, 18, 14, 8, 11, 22, 19, 7, 24] + [3] * 10
)
_BOS, _EOS = 0, 2


def esmc_residues(res_type: Float[Array, "T 33"], vocab: int = 64) -> Float[Array, "T V"]:
    """Residue probability vectors moved to ESMC's amino-acid columns."""
    return res_type @ jax.nn.one_hot(_ESMC_COLUMN, vocab)


class LMLayout(eqx.Module):
    """Where a system's protein residues sit in ESMC's input: one row per residue (a
    residue spanning several tokens keeps its first), each chain wrapped in BOS and EOS,
    chains in `asym_id` order."""

    source: Int[Array, " L"]  # the system token in each row, -1 for BOS/EOS
    special: Int[Array, " L"]  # the BOS/EOS token id in each special row
    sequence_id: Int[Array, " L"]  # each row's chain (`ESMC`'s `sequence_id`)
    positions: Int[Array, " T"]  # each system token's row, -1 off-protein


def lm_layout(asym_id: np.ndarray, residue_index: np.ndarray, protein: np.ndarray) -> LMLayout:
    """`LMLayout` from a system's chain and residue ids (numpy; the layout is static)."""
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
    """ESMC's input matrix: each system token's residue vector (`esmc_residues`, or a
    one-hot of the upstream's `input_ids`) in its row, exact one-hot BOS and EOS rows."""
    special = jax.nn.one_hot(layout.special, residues.shape[-1])
    return jnp.where(layout.source[:, None] >= 0, residues[layout.source], special)


def token_hidden_states(
    hidden: Float[Array, "layers L D"], positions: Int[Array, " T"]
) -> Float[Array, "T layers D"]:
    """ESMC's states for each system token (`LMLayout.positions`), zero off-protein, in
    the upstream's `lm_hidden_states` layout."""
    states = jnp.where(positions[None, :, None] >= 0, hidden[:, positions], 0.0)
    return jnp.swapaxes(states, 0, 1)
