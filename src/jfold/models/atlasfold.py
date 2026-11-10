"""AtlasFold and AtlasFold-M: protein structure from one sequence per chain, with no MSA,
on the AtlasLM language model.

Every trunk pass runs AtlasLM (an ESMC reproduction: `jfold.models.esmfold2.ESMCLayer`s with
rotary positions restarting in each chain) on the sequence with a fresh random share of its
residues masked, and reads its layers' states (a learned softmax mix) and attention logits
(one projection per layer) into a single and a pair. Four Pairformer layers, each first
adding a product-and-difference pair from the single, refine them; they are added to the
recycled single and pair, (AtlasFold-M) templates update the pair, and the Pairformer runs:
36 pair-only layers, then 12 with the single track.

Atoms are the 14 atom slots of each residue (atom14): `N = 14 T` positions, slot `a` of
residue `t` at `14 t + a`, with a mask for the slots a residue type lacks. The diffusion
module is AF3's, on these atoms: its atom transformers attend in windows of 4 residues to
the 12 residues around them, only within 4 residue indices on the same chain, biased by
residue offsets and, within a residue, the residue type's ideal atom offsets and distances.

The rest runs on the shared blocks: Pairformer and template layers, the diffusion
conditioning and transformers, the confidence Pairformer and heads. A converted model is a
`jfold.model.StructureModel` reading `AtlasFoldFeatures`: `ResidueEmbedder`,
`ResidueTrunkInit`, `AtlasFoldTrunk` (holding AtlasLM), `DiffusionModule`, and
`MultimerConfidence` or `MonomerConfidence`. Inputs are unbatched, with masks that are
False for padding."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox.nn import Identity
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from jfold.blocks.atoms import mean_atoms_to_tokens
from jfold.blocks.confidence import (
    ConfidenceModule,
    DistanceEmbedding,
    PairHead,
    TokenHead,
    pairwise_distances,
)
from jfold.blocks.denoiser import DiffusionConditioning
from jfold.blocks.diffusion import DiffusionTransformer, DiffusionTransformerLayer
from jfold.blocks.inputs import TrunkInit, probabilities, relative_position_features
from jfold.blocks.pairformer import AbstractPairformer, PairformerLayer, PairformerStack
from jfold.blocks.templates import TemplateEmbedder
from jfold.blocks.windows import CenteredWindows
from jfold.model import Recycle
from jfold.models.common import arrays
from jfold.models.esmfold2 import ESMCLayer, _rotate
from jfold.nn import (
    Dropout,
    LayerNorm,
    Linear,
    NormLinear,
    ReluMLP2,
    SinglesToPair,
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
    TrunkInputs,
    TrunkState,
)

NUM_CLASSES = 21  # ARNDCQEGHILKMFPSTWYV, then X
ATOMS = 14  # atom slots per residue
# AtlasLM's vocabulary (`atlaslm.alphabet.VOCAB`): <cls> <pad> <eos> <unk>, 28 residue
# letters, <mask>; the embedding has 64 rows.
LM_VOCAB = 64
_LM_RESIDUES = "LAGVSERTIDPKQNFYMHWCXBUZO.-|"
_LM_MASK = 4 + len(_LM_RESIDUES)
# AtlasLM's token for each of AtlasFold's residue classes.
_LM_COLUMN = np.array([4 + _LM_RESIDUES.index(r) for r in "ARNDCQEGHILKMFPSTWYVX"])
# Atom windows: 4 residues of queries, keys 4 residues either side.
_WINDOW = 4
_MAX_OFFSET = 4


def lm_residues(residues: Float[Array, "T 21"]) -> Float[Array, "T 64"]:
    """Residue probability vectors moved to AtlasLM's columns."""
    return residues @ jax.nn.one_hot(_LM_COLUMN, LM_VOCAB)


class LMInputs(eqx.Module):
    """AtlasLM's input: each chain in its own rows, wrapped in <cls> and <eos>."""

    tokens: Float[Array, "S 64"]  # probability vectors over the vocabulary
    positions: Int[Array, " S"]  # rotary positions, restarting at 0 in each chain
    sequence_id: Int[Array, " S"]  # positions attend to the others of the same id
    maskable: Bool[Array, " S"]  # the residue rows, which the random mask may replace
    rows: Int[Array, " T"]  # each residue's row


class AtlasFoldFeatures(eqx.Module):
    """A system's inputs in AtlasFold's vocabulary."""

    residues: Float[Array, "T 21"]  # residue probabilities (one-hot for a fixed sequence)
    relative_positions: Float[Array, "T T 73"]  # `relative_positions`
    token_mask: Bool[Array, " T"]
    asym_id: Int[Array, " T"]
    residue_index: Int[Array, " T"]
    atom_mask: Bool[Array, "T 14"]  # the atom14 slots of each residue's type
    pseudo_beta: Int[Array, " T"]  # the slot of each residue's CB (CA for glycine)
    lm: LMInputs
    template_features: Float[Array, "Nt T T 86"] | None = None  # `template_features`
    template_mask: Bool[Array, " Nt"] | None = None

    @property
    def flat_atom_mask(self) -> Bool[Array, " N"]:
        return self.atom_mask.reshape(-1)


def relative_positions(
    asym_id: Int[Array, " T"],
    residue_index: Int[Array, " T"],
    entity_id: Int[Array, " T"],
    sym_id: Int[Array, " T"],
) -> Float[Array, "T T 73"]:
    """AF3's relative position features without the token offset (every token is a
    residue): residue offset within a chain, same entity, chain offset within an entity."""
    f = relative_position_features(asym_id, residue_index, entity_id, residue_index, sym_id)
    return jnp.concatenate([f[..., :66], f[..., 132:]], axis=-1)


# AtlasLM


class AtlasLM(eqx.Module):
    """AtlasLM, ESMC's architecture: pre-norm transformer layers with rotary positions,
    attention restricted to equal `sequence_id`s. `layer` also returns the attention logits,
    which AtlasFold reads."""

    embed: Linear  # token probability vectors -> D
    layers: ESMCLayer  # stacked
    norm: LayerNorm  # after the last layer (AtlasFold reads the states before it)
    rope_base: float = eqx.field(static=True, default=10000.0)

    def rope(self, positions: Int[Array, " S"]) -> tuple[Float[Array, "S R"], Float[Array, "S R"]]:
        head_dim = self.embed.weight.shape[0] // self.layers.num_heads
        inv_freq = 1.0 / self.rope_base ** (
            jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
        )
        angles = positions.astype(jnp.float32)[:, None] * inv_freq
        return jnp.cos(angles), jnp.sin(angles)

    @staticmethod
    def layer(
        layer: ESMCLayer,
        x: Float[Array, "S D"],
        rope: tuple[Float[Array, "S R"], Float[Array, "S R"]],
        attend: Bool[Array, "S S"],
    ) -> tuple[Float[Array, "S D"], Float[Array, "heads S S"]]:
        """One layer: the new states and the attention logits (before masking)."""
        heads = lambda y: y.reshape(*y.shape[:-1], layer.num_heads, -1)
        h = layer.norm(x)
        q = _rotate(heads(layer.norm_q(layer.linear_q(h))), *rope)
        k = _rotate(heads(layer.norm_k(layer.linear_k(h))), *rope)
        v = heads(layer.linear_v(h))
        logits = jnp.einsum("qhc,khc->hqk", q, k) / jnp.sqrt(q.shape[-1])
        weights = jax.nn.softmax(jnp.where(attend, logits, -jnp.inf), axis=-1)
        o = jnp.einsum("hqk,khc->qhc", weights, v).reshape(x.shape)
        x = x + layer.linear_o(o) / layer.residual_scale
        return x + layer.transition(x) / layer.residual_scale, logits

    def __call__(
        self,
        tokens: Float[Array, "S V"],
        positions: Int[Array, " S"],
        sequence_id: Int[Array, " S"],
    ) -> Float[Array, "S D"]:
        """The final normalized states (AtlasLM's embeddings)."""
        rope = self.rope(positions)
        attend = sequence_id[:, None] == sequence_id[None, :]

        def body(x, layer):
            return self.layer(layer, x, rope, attend)[0], None

        x, _ = jax.lax.scan(jax.checkpoint(body), self.embed(tokens), self.layers)
        return self.norm(x)


# The trunk


class LMLayer(eqx.Module):
    """A Pairformer layer that first adds a pair from the single: the difference and
    product of two projections of the normalized single (`SinglesToPair`)."""

    norm: LayerNorm
    to_pair: SinglesToPair
    dropout: Dropout | Identity
    layer: PairformerLayer

    def __call__(
        self,
        s: Float[Array, "T Cs"],
        z: Float[Array, "T T Cz"],
        mask: Bool[Array, " T"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "T Cs"], Float[Array, "T T Cz"]]:
        k_dropout, k_layer = split_key(key)
        z = z + self.dropout(self.to_pair(self.norm(s)), key=k_dropout)
        return self.layer(s, z, mask, key=k_layer)


class LanguageModelEmbedder(eqx.Module):
    """AtlasLM's states and attention logits to the trunk's single and pair: the states
    (input embedding and each layer's output, normalized) mixed by softmax weights, then an
    MLP; each layer's logits, clipped to ±100 and scaled by 1/100 (zero across sequences),
    projected per layer and summed, then an MLP; then `stack`. AtlasFold-M zeroes the
    pair between chains (`intra_chain`)."""

    layer_logits: Float[Array, " layers+1"]
    norm_states: LayerNorm
    single: ReluMLP2  # D -> Cs_lm
    attention: Linear  # heads -> Cz, one per layer, stacked
    norm_pair: LayerNorm
    pair: ReluMLP2
    stack: LMLayer  # stacked
    project_s: NormLinear  # Cs_lm -> Cs
    project_z: NormLinear
    intra_chain: bool = eqx.field(static=True, default=False)

    def __call__(
        self,
        lm: AtlasLM,
        inputs: LMInputs,
        mask_rows: Bool[Array, " S"],
        asym_id: Int[Array, " T"],
        mask: Bool[Array, " T"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "T Cs"], Float[Array, "T T Cz"]]:
        """The single and pair to add to the trunk's, with AtlasLM reading `inputs` with
        `mask_rows` masked (among its residue rows)."""
        masked = mask_rows & inputs.maskable
        tokens = jnp.where(masked[:, None], jax.nn.one_hot(_LM_MASK, LM_VOCAB), inputs.tokens)
        rows = inputs.rows
        rope = lm.rope(inputs.positions)
        attend = inputs.sequence_id[:, None] == inputs.sequence_id[None, :]
        weights = jax.nn.softmax(self.layer_logits)
        x = lm.embed(tokens)

        @jax.checkpoint
        def body(carry, layer_projection_weight):
            x, states, pair = carry
            layer, projection, weight = layer_projection_weight
            x, logits = lm.layer(layer, x, rope, attend)
            logits = jnp.where(attend, jnp.clip(logits, -100.0, 100.0) / 100.0, 0.0)
            pair = pair + projection(jnp.moveaxis(logits[:, rows][:, :, rows], 0, -1))
            return (x, states + weight * self.norm_states(x[rows]), pair), None

        states = weights[0] * self.norm_states(x[rows])
        num_tokens, width = rows.shape[0], self.attention.weight.shape[1]
        pair = jnp.zeros((num_tokens, num_tokens, width), states.dtype)
        (_, states, pair), _ = jax.lax.scan(
            body, (x, states, pair), (lm.layers, self.attention, weights[1:])
        )
        s = self.single(states)
        z = self.pair(self.norm_pair(pair))
        if self.intra_chain:
            pair_mask = mask[:, None] & mask[None, :] & (asym_id[:, None] == asym_id[None, :])
            s = s * mask[:, None]
            z = z * pair_mask[..., None]

        def step(layer, carry, k):
            return layer(*carry, mask, key=k)

        s, z = scan_layers(step, (s, z), self.stack, key=key)
        return self.project_s(s), self.project_z(z)


class AtlasFoldPairformer(AbstractPairformer):
    """The trunk's Pairformer: pair-only layers, then layers with the single track."""

    pair_only: PairformerStack
    full: PairformerStack

    def __call__(
        self,
        s: Float[Array, "T Cs"],
        z: Float[Array, "T T Cz"],
        mask: Bool[Array, " T"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "T Cs"], Float[Array, "T T Cz"]]:
        k_pair, k_full = split_key(key)
        s, z = self.pair_only(s, z, mask, key=k_pair)
        return self.full(s, z, mask, key=k_full)


class AtlasFoldTrunk(AbstractTrunk[AtlasFoldFeatures]):
    """The trunk, from zeros. One pass: recycle the previous single and pair, add the
    language model's (AtlasLM reading the sequence with `mlm_rate` of its residues masked,
    drawn afresh every pass), add templates, run the Pairformer."""

    recycle_s: Recycle
    recycle_z: Recycle
    language_model: LanguageModelEmbedder
    pairformer: AtlasFoldPairformer
    templates: TemplateEmbedder | None = None
    # None only as stored: `jfold.pretrained` attaches the shared AtlasLM on loading
    # (`jfold.pretrained.recipes.atlasfold`).
    atlaslm: AtlasLM | None = None
    mlm_rate: float = eqx.field(static=True, default=0.15)

    def init_state(self, inputs: TrunkInputs, *, key: PRNGKeyArray | None = None) -> TrunkState:
        return TrunkState(s=jnp.zeros_like(inputs.s_init), z=jnp.zeros_like(inputs.z_init))

    def __call__(
        self,
        features: AtlasFoldFeatures,
        inputs: TrunkInputs,
        state: TrunkState,
        *,
        key: PRNGKeyArray,
    ) -> TrunkState:
        k_mask, k_pass = jax.random.split(key)
        mask_rows = jax.random.bernoulli(k_mask, self.mlm_rate, features.lm.maskable.shape)
        return self.masked(features, inputs, state, mask_rows, key=k_pass)

    def masked(
        self,
        features: AtlasFoldFeatures,
        inputs: TrunkInputs,
        state: TrunkState,
        mask_rows: Bool[Array, " S"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> TrunkState:
        """One pass with AtlasLM's rows `mask_rows` masked (among its residue rows)."""
        if self.atlaslm is None:
            raise ValueError(
                "AtlasFold needs its language model: this model has no AtlasLM (load it with "
                "`jfold.pretrained.load`, which attaches it)"
            )
        k_lm, k_templates, k_pairformer = split_key(key, 3)
        mask = features.token_mask
        s = self.recycle_s(inputs.s_init, state.s)
        z = self.recycle_z(inputs.z_init, state.z)
        s_lm, z_lm = self.language_model(
            self.atlaslm, features.lm, mask_rows, features.asym_id, mask, key=k_lm
        )
        s, z = s + s_lm, z + z_lm
        if self.templates is not None and features.template_features is not None:
            z = z + self.templates(
                z, features.template_features, features.template_mask, mask, key=k_templates
            )
        s, z = self.pairformer(s, z, mask, key=k_pairformer)
        return TrunkState(s=s, z=z)


# Templates


def template_features(
    residues: Float[Array, "Nt T 21"],
    pseudo_beta: Float[Array, "Nt T 3"],
    pseudo_beta_mask: Bool[Array, "Nt T"],
    backbone: Float[Array, "Nt T 3 3"],
    backbone_mask: Bool[Array, "Nt T"],
    asym_id: Int[Array, " T"],
    token_mask: Bool[Array, " T"],
) -> Float[Array, "Nt T T 86"]:
    """AtlasFold-M's template pair features, within chains only: pseudo-beta distance bins
    (39, from 3.25 Å to 50.75 Å; none below), the pseudo-beta pair mask, unit vectors to the
    other CA in each residue's backbone frame (N, CA, C), the frame pair mask, and both
    residue types."""
    same_chain = (asym_id[:, None] == asym_id[None, :]) & token_mask[:, None] & token_mask[None, :]

    def one(residues, cb, cb_mask, bb, bb_mask):
        lower = jnp.linspace(3.25, 50.75, 39) ** 2
        upper = jnp.concatenate([lower[1:], jnp.array([1e8])])
        d2 = jnp.sum((cb[None, :] - cb[:, None]) ** 2, axis=-1, keepdims=True)
        cb_pair = cb_mask[:, None] & cb_mask[None, :]
        distogram = (d2 > lower) & (d2 < upper) & (cb_pair & same_chain)[..., None]

        def normalize(x):
            return x / jnp.sqrt(jnp.maximum(jnp.sum(x**2, axis=-1, keepdims=True), 1e-12))

        n, ca, c = bb[:, 0], bb[:, 1], bb[:, 2]
        e0 = normalize(n - ca)
        e1 = c - ca
        e1 = normalize(e1 - jnp.sum(e1 * e0, axis=-1, keepdims=True) * e0)
        basis = jnp.stack([e0, e1, jnp.cross(e0, e1)], axis=-2)  # [T, 3, 3]
        to_other = ca[None, :] - ca[:, None]  # [i, j]: CA_j - CA_i
        unit = normalize(jnp.einsum("iab,ijb->ija", basis, to_other))
        bb_pair = bb_mask[:, None] & bb_mask[None, :] & same_chain
        unit = unit * bb_pair[..., None]
        t = residues.shape[0]
        return jnp.concatenate(
            [
                distogram.astype(jnp.float32),
                (cb_pair & same_chain)[..., None].astype(jnp.float32),
                unit,
                bb_pair[..., None].astype(jnp.float32),
                jnp.broadcast_to(residues[:, None], (t, t, NUM_CLASSES)),
                jnp.broadcast_to(residues[None, :], (t, t, NUM_CLASSES)),
            ],
            axis=-1,
        )

    return jax.vmap(one)(residues, pseudo_beta, pseudo_beta_mask, backbone, backbone_mask)


# Diffusion


class AtomPairEncoding(eqx.Module):
    """The atom-pair encoding of the atom windows, kept factored: the residue offset of
    each (query, key) residue pair (one-hot within ±`max_offset` on the same chain, else
    zero), and within a residue the residue type's ideal atom offsets and distances
    (`table`, mixed by the residue probabilities); also which residue pairs may attend
    (on the same chain, within `max_offset` residue indices)."""

    table: Float[Array, "21 14 14 4"]
    max_offset: int = eqx.field(static=True, default=_MAX_OFFSET)

    @property
    def windows(self) -> CenteredWindows:
        """The windows over residues; the atom windows are 14 times as long."""
        return CenteredWindows(queries=_WINDOW, keys=_WINDOW + 2 * self.max_offset)

    def __call__(
        self,
        residues: Float[Array, "T 21"],
        residue_index: Int[Array, " T"],
        asym_id: Int[Array, " T"],
        token_mask: Bool[Array, " T"],
    ) -> tuple[Float[Array, "NW q k 10"], Float[Array, "NW q k 14 14 4"], Bool[Array, "NW q k"]]:
        windows = self.windows
        index, valid = windows.key_index(token_mask)
        query = windows.query_blocks(jnp.arange(token_mask.shape[0]))
        offset = residue_index[query][:, :, None] - residue_index[index][:, None]
        same_chain = asym_id[query][:, :, None] == asym_id[index][:, None]
        r = self.max_offset
        bins = jax.nn.one_hot(jnp.clip(offset + r, 0, 2 * r + 1), 2 * r + 2)
        residue_pairs = bins * same_chain[..., None]
        same_residue = (offset == 0) & same_chain
        atoms = jnp.einsum("wqc,cabd->wqabd", residues[query], self.table)
        atom_pairs = atoms[:, :, None] * same_residue[..., None, None, None]
        allowed = (jnp.abs(offset) <= r) & same_chain & valid[:, None]
        return residue_pairs, atom_pairs, allowed


def _to_atoms(x: Float[Array, "NW q k ..."]) -> Float[Array, "NW q*14 k*14 ..."]:
    """A residue-pair array repeated over both residues' atom slots."""
    nw, q, k = x.shape[:3]
    x = jnp.broadcast_to(x[:, :, None, :, None], (nw, q, ATOMS, k, ATOMS, *x.shape[3:]))
    return x.reshape(nw, q * ATOMS, k * ATOMS, *x.shape[5:])


class AtomTransformer(eqx.Module):
    """The diffusion transformer within atom windows (`LocalAttention` in its layers),
    biased by a linear map of the atom-pair encoding, without a norm; residue pairs that
    may not attend get -1e9."""

    linear_bias: Linear  # encoding (10 residue + 4 atom channels) -> layers * heads
    layers: DiffusionTransformerLayer  # stacked

    def biases(
        self,
        encoding: tuple[
            Float[Array, "NW q k 10"], Float[Array, "NW q k 14 14 4"], Bool[Array, "NW q k"]
        ],
    ) -> Float[Array, "layers heads NW Q K"]:
        residue_pairs, atom_pairs, allowed = encoding
        weight = self.linear_bias.weight
        num_layers = jax.tree.leaves(self.layers)[0].shape[0]
        bias = _to_atoms(residue_pairs @ weight[:, :10].T)
        atom_bias = jnp.einsum("wqkabd,ed->wqakbe", atom_pairs, weight[:, 10:])
        nw, q, _, k = atom_bias.shape[:4]
        bias = bias + atom_bias.reshape(nw, q * ATOMS, k * ATOMS, -1)
        bias = bias + jnp.where(_to_atoms(allowed), 0.0, -1e9)[..., None]
        bias = jnp.moveaxis(bias, -1, 0)
        return bias.reshape(num_layers, -1, *bias.shape[1:])

    def __call__(
        self,
        a: Float[Array, "N Ca"],
        c: Float[Array, "N Ca"],
        biases: Float[Array, "layers heads NW Q K"],
        mask: Bool[Array, " N"],
    ) -> Float[Array, "N Ca"]:
        def step(layer_and_bias, a, key):
            layer, bias = layer_and_bias
            return layer(a, c, bias, mask)

        return scan_layers(step, a, (self.layers, biases))


class AtomEncoder(eqx.Module):
    """Atom queries from each residue type's per-slot embedding and the noisy positions,
    through the atom transformer; per-residue means of their projection."""

    embed: Linear  # residue probabilities -> 14 Ca, the atom single per slot
    linear_r: Linear  # noisy positions -> Ca
    transformer: AtomTransformer
    linear_token: Linear  # Ca -> Ct, through ReLU

    def atom_single(self, residues: Float[Array, "T 21"]) -> Float[Array, "N Ca"]:
        return self.embed(residues).reshape(residues.shape[0] * ATOMS, -1)

    def __call__(
        self,
        c: Float[Array, "N Ca"],
        r: Float[Array, "N 3"],
        biases: Float[Array, "layers heads NW Q K"],
        mask: Bool[Array, " N"],
    ) -> tuple[Float[Array, "T Ct"], Float[Array, "N Ca"]]:
        q = self.transformer(c + self.linear_r(r), c, biases, mask)
        a = jax.nn.relu(self.linear_token(q))
        num_tokens = mask.shape[0] // ATOMS
        atom_to_token = jnp.repeat(jnp.arange(num_tokens), ATOMS)
        return mean_atoms_to_tokens(a, atom_to_token, mask, num_tokens), q


class AtomDecoder(eqx.Module):
    """Token updates to each atom slot (a projection per slot), the atom transformer, and
    a position update per atom."""

    project_a: NormLinear  # Ct -> 14 Ca
    transformer: AtomTransformer
    norm_out: LayerNorm
    linear_out: Linear  # Ca -> 3

    def __call__(
        self,
        a: Float[Array, "T Ct"],
        q: Float[Array, "N Ca"],
        c: Float[Array, "N Ca"],
        biases: Float[Array, "layers heads NW Q K"],
        mask: Bool[Array, " N"],
    ) -> Float[Array, "N 3"]:
        q = q + self.project_a(a).reshape(q.shape)
        q = self.transformer(q, c, biases, mask)
        return self.linear_out(self.norm_out(q))


class DiffusionContext(eqx.Module):
    """Everything a denoising step needs that depends only on the trunk."""

    s_trunk: Float[Array, "T Cs"]
    residues: Float[Array, "T 21"]
    token_mask: Bool[Array, " T"]
    atom_mask: Bool[Array, " N"]  # the atoms the denoiser moves
    atom_single: Float[Array, "N Ca"]
    encoder_biases: Float[Array, "Le heads NW Q K"]
    token_biases: Float[Array, "Lt heads T T"]
    decoder_biases: Float[Array, "Ld heads NW Q K"]


class DiffusionModule(AbstractDiffusionModule[AtlasFoldFeatures]):
    """AF3 Algorithm 20 on atom14 slots, with EDM preconditioning."""

    conditioning: DiffusionConditioning
    pair_encoding: AtomPairEncoding
    encoder: AtomEncoder
    project_s: NormLinear  # conditioned single -> token width
    transformer: DiffusionTransformer
    decoder: AtomDecoder

    @property
    def sigma_data(self) -> float:
        return self.conditioning.sigma_data

    def prepare(
        self, features: AtlasFoldFeatures, inputs: TrunkInputs, state: TrunkState
    ) -> DiffusionContext:
        z = self.conditioning.pair(state.z, features.relative_positions)
        encoding = self.pair_encoding(
            features.residues, features.residue_index, features.asym_id, features.token_mask
        )
        return DiffusionContext(
            s_trunk=state.s,
            residues=inputs.s_inputs,
            token_mask=features.token_mask,
            atom_mask=features.flat_atom_mask,
            atom_single=self.encoder.atom_single(features.residues),
            encoder_biases=self.encoder.transformer.biases(encoding),
            token_biases=self.transformer.pair_biases(z),
            decoder_biases=self.decoder.transformer.biases(encoding),
        )

    def __call__(
        self,
        context: DiffusionContext,
        x_noisy: Float[Array, "N 3"],
        t: Float[Array, ""],
    ) -> Float[Array, "N 3"]:
        sigma_data, mask, c = self.sigma_data, context.atom_mask, context.atom_single
        s = self.conditioning.single(context.s_trunk, context.residues, t)
        r = x_noisy / jnp.sqrt(t**2 + sigma_data**2)
        a, q = self.encoder(c, r, context.encoder_biases, mask)
        a = a + self.project_s(s)
        a = self.transformer(a, s, context.token_biases, context.token_mask)
        r_update = self.decoder(a, q, c, context.decoder_biases, mask)
        scale = sigma_data**2 + t**2
        return sigma_data**2 / scale * x_noisy + sigma_data * t / jnp.sqrt(scale) * r_update


# Confidence


def _rep_coords(features: AtlasFoldFeatures, coords: Float[Array, "N 3"]) -> Float[Array, "T 3"]:
    """Each residue's pseudo-beta position."""
    return atom14(coords)[jnp.arange(features.pseudo_beta.shape[0]), features.pseudo_beta]


class MultimerConfidence(AbstractConfidenceModule[AtlasFoldFeatures]):
    """AtlasFold-M's confidence head: AF3's (`ConfidenceModule`, on the residue types and
    the pseudo-beta distances) with the trunk single projected first."""

    project_s: NormLinear
    module: ConfidenceModule

    def __call__(
        self,
        features: AtlasFoldFeatures,
        inputs: TrunkInputs,
        state: TrunkState,
        coords: Float[Array, "N 3"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> ConfidenceLogits:
        tokens = jnp.arange(features.token_mask.shape[0])  # the heads are per token
        return self.module.logits(
            inputs.s_inputs,
            self.project_s(state.s),
            state.z,
            _rep_coords(features, coords),
            features.asym_id,
            tokens,
            tokens * 0,
            features.token_mask,
            key=key,
        )


class MonomerConfidence(AbstractConfidenceModule[AtlasFoldFeatures]):
    """AtlasFold's confidence head: the pair conditioned on the residue types and the
    pseudo-beta distances, then one Pairformer for the single (pLDDT, resolved) and a
    pair-only one for the pair (PAE)."""

    singles_to_pair: SinglesToPair  # residue probabilities -> pair
    distances: DistanceEmbedding
    single_stack: PairformerStack
    pair_stack: PairformerStack
    pae: PairHead
    plddt: TokenHead
    resolved: TokenHead  # one logit per atom37 slot

    def __call__(
        self,
        features: AtlasFoldFeatures,
        inputs: TrunkInputs,
        state: TrunkState,
        coords: Float[Array, "N 3"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> ConfidenceLogits:
        k_single, k_pair = split_key(key)
        mask = features.token_mask
        distances = self.distances(pairwise_distances(_rep_coords(features, coords)))
        z = state.z + self.singles_to_pair(inputs.s_inputs) + distances
        s, _ = self.single_stack(state.s, z, mask, key=k_single)
        _, z = self.pair_stack(None, z, mask, key=k_pair)
        return ConfidenceLogits(
            pae=self.pae(z),
            pde=None,
            plddt=self.plddt(s, None, None),
            resolved=self.resolved(s, None, None),
        )


# Inputs


class ResidueEmbedder(AbstractInputEmbedder[AtlasFoldFeatures]):
    """`s_inputs`: the residue probabilities, as they are."""

    def __call__(self, features: AtlasFoldFeatures) -> Float[Array, "T 21"]:
        return features.residues


class ResidueTrunkInit(AbstractTrunkInit[AtlasFoldFeatures]):
    """The initial single and pair from the residue probabilities and the relative
    positions (`TrunkInit.initial`; AtlasFold has no bond features)."""

    init: TrunkInit

    def __call__(
        self,
        features: AtlasFoldFeatures,
        s_inputs: Float[Array, "T 21"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> TrunkInputs:
        s_init, z_init = self.init.initial(s_inputs, features.relative_positions)
        return TrunkInputs(s_inputs=s_inputs, s_init=s_init, z_init=z_init)


def atom14(coords: Float[Array, "N 3"]) -> Float[Array, "T 14 3"]:
    """Positions (`[14 T, 3]`, as the model samples them) by residue and atom14 slot."""
    return coords.reshape(-1, ATOMS, 3)


# Features


def features(f: dict) -> AtlasFoldFeatures:
    """`AtlasFoldFeatures` from an AtlasFold feature dict (`atlasfold.common.featurize`'s
    keys, unbatched; `template.*` optional). `aatype` (`[T, 21]`) may be any probability
    vectors, e.g. a soft sequence: they reach AtlasLM too, in its residue rows. Any of the
    arrays may be traced under `jit`."""
    a = arrays(f)
    token_mask = a["seq_mask"].astype(bool)
    residues = probabilities(a["aatype"], NUM_CLASSES)
    asym_id = a["asym_id"]
    input_ids = a["lm.input_ids"]
    rows = a["seq_tok_idx"]
    # The residue rows from `residues` (exactly `input_ids`' one-hot for a one-hot sequence).
    tokens = jax.nn.one_hot(input_ids, LM_VOCAB)
    tokens = tokens.at[jnp.where(token_mask, rows, input_ids.shape[0])].set(
        lm_residues(residues), mode="drop"
    )
    template = None
    if "template.aatype" in a:
        template = template_features(
            probabilities(a["template.aatype"], NUM_CLASSES),
            a["template.pseudo_beta"].astype(jnp.float32),
            a["template.pseudo_beta_mask"].astype(bool),
            a["template.backbone_coords"].astype(jnp.float32),
            a["template.backbone_frame_mask"].astype(bool),
            asym_id,
            token_mask,
        )
    return AtlasFoldFeatures(
        residues=residues,
        relative_positions=relative_positions(asym_id, a["res_idx"], a["entity_id"], a["sym_id"]),
        token_mask=token_mask,
        asym_id=asym_id,
        residue_index=a["res_idx"],
        atom_mask=a["atom14_mask"].astype(bool),
        pseudo_beta=a["pseudo_beta"],
        lm=LMInputs(
            tokens=tokens,
            positions=a["lm.pos_id"],
            sequence_id=a["lm.seq_id"],
            maskable=(input_ids >= 4) & (input_ids < _LM_MASK),
            rows=rows,
        ),
        template_features=template,
        template_mask=None if template is None else a["template.mask"].astype(bool),
    )
