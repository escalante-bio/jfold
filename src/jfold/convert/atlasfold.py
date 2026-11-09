"""AtlasFold's modules (`atlasfold`'s `AtlasFold` and `AtlasFold_Multimer`) and AtlasLM
(`atlaslm`'s `AtlasLM`)."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from jfold.blocks.confidence import ConfidenceModule, DistanceEmbedding, PairHead, TokenHead
from jfold.blocks.denoiser import DiffusionConditioning, FourierEmbedding
from jfold.blocks.diffusion import (
    AdaLN,
    ConditionedAttention,
    ConditionedTransition,
    DiffusionTransformer,
    DiffusionTransformerLayer,
    SeparateKeys,
)
from jfold.blocks.inputs import TrunkInit
from jfold.blocks.pairformer import (
    AttentionPairBias,
    PairformerLayer,
    PairformerStack,
    SingleUpdate,
    TriangleAttention,
    TriangleMultiplication,
)
from jfold.blocks.templates import TemplateEmbedder
from jfold.blocks.windows import CenteredWindows, LocalAttention
from jfold.convert.torch import (
    OPENFOLD_COLUMN,
    OPENFOLD_ROW,
    array,
    dropout,
    embedding,
    layer_norm,
    linear,
)
from jfold.model import Recycle, StructureModel
from jfold.models.atlasfold import (
    ATOMS,
    AtlasFoldPairformer,
    AtlasFoldTrunk,
    AtlasLM,
    AtomDecoder,
    AtomEncoder,
    AtomPairEncoding,
    AtomTransformer,
    DiffusionModule,
    LanguageModelEmbedder,
    LMLayer,
    MonomerConfidence,
    MultimerConfidence,
    NormLinear,
    ReluMLP2,
    ResidueEmbedder,
    ResidueTrunkInit,
)
from jfold.models.esmfold2 import ESMCLayer
from jfold.nn import Attention, Linear, PairBias, SinglesToPair, Transition, stack
from jfold.sampling import UpstreamSampler, masked_centering, no_alignment


def _split(m, num: int) -> list[Linear]:
    """A bias-free linear's outputs, split into `num` equal linears (torch's `chunk`)."""
    return [Linear(weight=w) for w in jnp.split(array(m.weight), num)]


def norm_linear(m) -> NormLinear:
    """A `Sequential(LayerNorm, Linear)`."""
    return NormLinear(norm=layer_norm(m[0]), linear=linear(m[1]))


# Pair blocks


def triangle_multiplication(m, *, outgoing: bool) -> TriangleMultiplication:
    return TriangleMultiplication(
        norm_in=layer_norm(m.layernorm_in),
        linear_p=linear(m.linear_in),
        linear_g=linear(m.linear_g_in),
        norm_out=layer_norm(m.layernorm_out),
        linear_out=linear(m.linear_out),
        linear_out_g=linear(m.linear_g_out),
        outgoing=outgoing,
    )


def triangle_attention(m) -> TriangleAttention:
    """Either node's: the block transposes the pair for the ending node."""
    linear_q, linear_k, linear_v = _split(m.linear_qkv, 3)
    return TriangleAttention(
        norm=layer_norm(m.layernorm),
        linear_bias=linear(m.linear_bias),
        attention=Attention(
            linear_q=linear_q,
            linear_k=linear_k,
            linear_v=linear_v,
            linear_g=linear(m.linear_g),
            linear_o=linear(m.linear_out),
            num_heads=int(m.num_heads),
        ),
    )


def transition(m) -> Transition:
    """`Transition`: a `Sequential(LayerNorm, SwiGLU, Linear)`."""
    linear_a, linear_b = _split(m[1].linear, 2)
    return Transition(
        norm=layer_norm(m[0]), linear_a=linear_a, linear_b=linear_b, linear_out=linear(m[2])
    )


def attention(m) -> Attention:
    """A `SelfAttention` or `CrossAttention`'s projections."""
    return Attention(
        linear_q=linear(m.attn.linear_q),
        linear_k=linear(m.attn.linear_k),
        linear_v=linear(m.attn.linear_v),
        linear_g=linear(m.linear_g),
        linear_o=linear(m.linear_o),
        num_heads=int(m.attn.num_heads),
    )


def pair_block(m) -> PairformerLayer:
    """A `PairBlock`'s pair and single updates (its single-to-pair step: `lm_layer`)."""
    single = None
    if m.pair_to_single:
        single = SingleUpdate(
            attention=AttentionPairBias(
                norm_s=layer_norm(m.attention.layernorm_a),
                pair_bias=PairBias(
                    norm=layer_norm(m.pair_to_single_bias[0]),
                    linear=linear(m.pair_to_single_bias[1]),
                ),
                attention=attention(m.attention),
            ),
            transition=transition(m.transition_s),
        )
    return PairformerLayer(
        tri_mul_out=triangle_multiplication(m.tri_mul_out, outgoing=True),
        tri_mul_in=triangle_multiplication(m.tri_mul_in, outgoing=False),
        tri_att_start=triangle_attention(m.tri_attn_start),
        tri_att_end=triangle_attention(m.tri_attn_end),
        transition_z=transition(m.transition_z),
        single=single,
        dropout_row=dropout(m.dropout_rowwise_z.p, OPENFOLD_ROW),
        dropout_column=dropout(m.dropout_columnwise_z.p, OPENFOLD_COLUMN),
    )


def pair_stack(blocks) -> PairformerStack:
    return PairformerStack(layers=stack([pair_block(b) for b in blocks]))


def product_difference(m) -> SinglesToPair:
    """`PairwiseProdDiff`'s pair, `linear_out([a_i - b_j, a_i * b_j])` with `a, b` the halves
    of `linear_in`, as `SinglesToPair` (on the normalized single)."""
    w_in, b_in = array(m.linear_in.weight), array(m.linear_in.bias)
    (w_a, w_b), (b_a, b_b) = jnp.split(w_in, 2), jnp.split(b_in, 2)
    w_diff, w_prod = jnp.split(array(m.linear_out.weight), 2, axis=1)
    return SinglesToPair(
        linear_i=Linear(weight=w_diff @ w_a, bias=w_diff @ b_a),
        linear_j=Linear(weight=-w_diff @ w_b, bias=-w_diff @ b_b),
        product_i=Linear(weight=w_a, bias=b_a),
        product_j=Linear(weight=w_b, bias=b_b),
        product_out=Linear(weight=w_prod),
    )


def lm_layer(m) -> LMLayer:
    """An `LMStack` block: single to pair, then the pair and single updates."""
    return LMLayer(
        norm=layer_norm(m.pairwise_prod_diff.layernorm),
        to_pair=product_difference(m.pairwise_prod_diff),
        dropout=dropout(m.dropout_z.p, ()),
        layer=pair_block(m),
    )


# The trunk


def atlaslm(m) -> AtlasLM:
    """`AtlasLM`, without its sequence head."""
    blocks = m.transformer.blocks
    layers = []
    for b in blocks:
        attn, ffn = b.attn, b.ffn
        linear_q, linear_k, linear_v = _split(attn.layernorm_qkv[1], 3)
        linear_a, linear_b = _split(ffn[1], 2)
        layers.append(
            ESMCLayer(
                norm=layer_norm(attn.layernorm_qkv[0]),
                linear_q=linear_q,
                linear_k=linear_k,
                linear_v=linear_v,
                norm_q=layer_norm(attn.q_ln),
                norm_k=layer_norm(attn.k_ln),
                linear_o=linear(attn.out_proj),
                transition=Transition(
                    norm=layer_norm(ffn[0]),
                    linear_a=linear_a,
                    linear_b=linear_b,
                    linear_out=linear(ffn[3]),
                ),
                num_heads=int(attn.n_heads),
                residual_scale=1 / float(b.scaling_factor),
            )
        )
    return AtlasLM(
        embed=embedding(m.embed),
        layers=stack(layers),
        norm=layer_norm(m.transformer.norm),
        rope_base=float(blocks[0].attn.rotary.base),
    )


def language_model_embedder(m, *, intra_chain: bool) -> LanguageModelEmbedder:
    return LanguageModelEmbedder(
        layer_logits=array(m.lm_layer_weights),
        norm_states=layer_norm(m.layernorm_lm_emb),
        single=ReluMLP2(
            linear_in=linear(m.lm_emb_to_s_lm[0]), linear_out=linear(m.lm_emb_to_s_lm[2])
        ),
        attention=stack([linear(p) for p in m.proj_lm_attn]),
        norm_pair=layer_norm(m.lm_attn_to_z_lm[0]),
        pair=ReluMLP2(
            linear_in=linear(m.lm_attn_to_z_lm[1]), linear_out=linear(m.lm_attn_to_z_lm[3])
        ),
        stack=stack([lm_layer(b) for b in m.lm_stack.blocks]),
        project_s=norm_linear(m.proj_s_lm),
        project_z=norm_linear(m.proj_z_lm),
        intra_chain=intra_chain,
    )


def pairformer(m) -> AtlasFoldPairformer:
    """The main `PairformerStack`: its pair-only blocks, then the rest."""
    blocks = list(m.blocks)
    first_full = next(i for i, b in enumerate(blocks) if b.pair_to_single)
    if not all(b.pair_to_single for b in blocks[first_full:]):
        raise ValueError("expected pair-only blocks, then blocks with the single track")
    return AtlasFoldPairformer(
        pair_only=pair_stack(blocks[:first_full]), full=pair_stack(blocks[first_full:])
    )


def template_embedder(m) -> TemplateEmbedder:
    return TemplateEmbedder(
        norm_z=layer_norm(m.proj_z[0]),
        linear_z=linear(m.proj_z[1]),
        linear_features=linear(m.linear_template),
        stack=pair_stack(m.stack.blocks),
        norm_v=layer_norm(m.layernorm_out),
        linear_u=linear(m.linear_out),
    )


def trunk_init(m) -> TrunkInit:
    linear_i, linear_j = _split(m.z_init, 2)
    return TrunkInit(
        linear_s=linear(m.s_init),
        singles_to_pair=SinglesToPair(linear_i=linear_i, linear_j=linear_j),
        relative_positions=linear(m.z_rel_pos),
    )


def trunk(m, *, multimer: bool, mlm_rate: float) -> AtlasFoldTrunk:
    templates = getattr(m, "template_module", None)
    return AtlasFoldTrunk(
        recycle_s=Recycle(norm=layer_norm(m.recycle_s[0]), linear=linear(m.recycle_s[1])),
        recycle_z=Recycle(norm=layer_norm(m.recycle_z[0]), linear=linear(m.recycle_z[1])),
        language_model=language_model_embedder(m, intra_chain=multimer),
        pairformer=pairformer(m.main_stack),
        templates=None if templates is None else template_embedder(templates),
        mlm_rate=mlm_rate,
    )


# Diffusion


def adaln(m) -> AdaLN:
    return AdaLN(
        norm_a=layer_norm(m.layernorm),
        norm_s=layer_norm(m.layernorm_cond),
        linear_scale=linear(m.linear_g),
        linear_shift=linear(m.linear_bias),
    )


def conditioned_transition(m) -> ConditionedTransition:
    linear_a, linear_b = _split(m.swiglu.linear, 2)
    return ConditionedTransition(
        adaln=adaln(m.adaln),
        linear_a=linear_a,
        linear_b=linear_b,
        linear_c=None,
        linear_out=linear(m.linear_out),
        linear_gate=linear(m.linear_g),
    )


def transformer_layer(m) -> DiffusionTransformerLayer:
    """A `DiffusionTransformerBlock` (self-attention) or `AtomTransformerBlock` (keys with
    their own adaptive norm)."""
    attn = m.attention
    cross = hasattr(attn, "adaln_a_k")
    return DiffusionTransformerLayer(
        attention=ConditionedAttention(
            adaln=adaln(attn.adaln_a_q if cross else attn.adaln_a),
            attention=attention(attn),
            linear_gate=linear(attn.linear_ada_out),
            keys=SeparateKeys(adaln(attn.adaln_a_k)) if cross else None,
        ),
        transition=conditioned_transition(m.transition),
    )


def diffusion_transformer(m, pair_conditioning) -> DiffusionTransformer:
    """The token transformer, each layer's bias from `PairConditioning`'s shared norm and
    its slice of the shared projection."""
    norm = layer_norm(pair_conditioning.layernorm_bias)
    weights = jnp.split(array(pair_conditioning.linear_bias.weight), len(m.blocks))
    return DiffusionTransformer(
        pair_bias=stack([PairBias(norm=norm, linear=Linear(weight=w)) for w in weights]),
        layers=stack([transformer_layer(b) for b in m.blocks]),
    )


def atom_transformer(m) -> AtomTransformer:
    """An `AtomAttentionStack`: its bias projection and transformer, in atom windows of
    4 residues with keys 4 residues either side."""
    windows = CenteredWindows(queries=4 * ATOMS, keys=12 * ATOMS)
    layers = eqx.tree_at(
        lambda layer: layer.attention.attention,
        stack([transformer_layer(b) for b in m.stack.blocks]),
        replace_fn=lambda a: LocalAttention(attention=a, windows=windows),
    )
    return AtomTransformer(linear_bias=linear(m.linear_pair_bais), layers=layers)


def diffusion_conditioning(head) -> DiffusionConditioning:
    single, pair = head.single_conditioning, head.pair_conditioning
    fourier = single.fourier_embed
    return DiffusionConditioning(
        norm_z=layer_norm(pair.layernorm),
        linear_z=linear(pair.linear),
        transitions_z=tuple(transition(t) for t in pair.transitions),
        norm_s=layer_norm(single.proj_single_cond[0]),
        linear_s=linear(single.proj_single_cond[1]),
        # Upstream's `cos(2π t w + b)`: the bias is outside the 2π.
        fourier=FourierEmbedding(
            weight=array(fourier.w)[0], bias=array(fourier.b)[0] / (2 * np.pi)
        ),
        norm_noise=layer_norm(single.layernorm_fourier),
        linear_noise=linear(single.linear_fourier),
        transitions_s=tuple(transition(t) for t in single.transitions),
        sigma_data=float(head.sigma_data),
    )


def diffusion_module(m) -> DiffusionModule:
    """`DiffusionHead`; `m` the model, for the atom-pair encoding's table."""
    head = m.diffusion_head
    score = head.score_model
    encoder, decoder = score.atom_encoder, score.atom_decoder
    table = array(m.atom_rel_pos_encoding.atom_rel_pos)
    return DiffusionModule(
        conditioning=diffusion_conditioning(head),
        pair_encoding=AtomPairEncoding(
            table=table.reshape(-1, ATOMS, ATOMS, 4),
            max_offset=int(m.atom_rel_pos_encoding.max_r),
        ),
        encoder=AtomEncoder(
            embed=linear(encoder.embedding_atoms),
            linear_r=linear(encoder.linear_in),
            transformer=atom_transformer(encoder.stack),
            linear_token=linear(score.proj_q_to_a[0]),
        ),
        project_s=norm_linear(score.proj_s_to_a),
        transformer=diffusion_transformer(score.diffusion_transformer, head.pair_conditioning),
        decoder=AtomDecoder(
            project_a=norm_linear(score.proj_a_to_q),
            transformer=atom_transformer(decoder.stack),
            norm_out=layer_norm(decoder.linear_out[0]),
            linear_out=linear(decoder.linear_out[1]),
        ),
    )


# Heads


def _distances(m) -> DistanceEmbedding:
    return DistanceEmbedding(
        edges=jnp.linspace(m.min_dist, m.max_dist, m.num_bins - 1),
        linear=linear(m.linear_distogram),
    )


def _token_head(m) -> TokenHead:
    return TokenHead(norm=layer_norm(m.head[0]), linear=linear(m.head[1]))


def _embed_residues(m) -> SinglesToPair:
    return SinglesToPair(linear_i=linear(m.embed_aa), linear_j=linear(m.embed_aa))


def multimer_confidence(m) -> MultimerConfidence:
    """`ConfidenceHead_Multimer`."""
    return MultimerConfidence(
        project_s=norm_linear(m.proj_s),
        module=ConfidenceModule(
            singles_to_pair=_embed_residues(m),
            distances=_distances(m),
            stack=pair_stack(m.stack.blocks),
            pae=PairHead(norm=layer_norm(m.pae_head.head[0]), linear=linear(m.pae_head.head[1])),
            pde=PairHead(
                norm=layer_norm(m.pde_head.head[0]),
                linear=linear(m.pde_head.head[1]),
                symmetric=True,
            ),
            plddt=_token_head(m.plddt_head),
            resolved=_token_head(m.experimentally_resolved_head),
        ),
    )


def monomer_confidence(m) -> MonomerConfidence:
    """`ConfidenceHead_Monomer`."""
    return MonomerConfidence(
        singles_to_pair=_embed_residues(m),
        distances=_distances(m),
        single_stack=pair_stack(m.single_stack.blocks),
        pair_stack=pair_stack(m.pair_stack.blocks),
        pae=PairHead(norm=layer_norm(m.pae_head.head[0]), linear=linear(m.pae_head.head[1])),
        plddt=_token_head(m.plddt_head),
        resolved=_token_head(m.experimentally_resolved_head),
    )


def distogram_head(m) -> PairHead:
    """`linear(z)` symmetrized after the projection (bias-free, so before is the same)."""
    return PairHead(linear=linear(m.linear), symmetric=True)


def sampling(head, config, *, num_steps: int) -> UpstreamSampler:
    """`DiffusionHead.sample`'s settings for a `SamplingConfig` (`config`): AF3's schedule
    ending at its smallest level, re-centred, randomly rotated and translated, no
    alignment. `num_steps`: the runner's (200 for AtlasFold-M; AtlasFold's grows with
    length, 20 up to 512 residues)."""
    cfg = config
    return UpstreamSampler(
        sigma_data=float(head.sigma_data),
        s_max=float(cfg.sigma_max),
        s_min=float(cfg.sigma_min),
        rho=float(cfg.rho),
        num_steps=num_steps,
        end="none",
        gamma_0=float(cfg.gamma_0),
        gamma_min=float(cfg.gamma_min),
        noise_scale=float(cfg.noise_scale),
        step_scale=float(cfg.step_scale),
        center=masked_centering,
        augment=True,
        augment_after_noise=False,
        align=no_alignment,
    )


def structure_model(m, sampling_config, *, mlm_rate: float, num_steps: int) -> StructureModel:
    """`AtlasFold` or `AtlasFold_Multimer`, without its language model (`atlaslm`
    converts it; `jfold.pretrained` attaches it as `trunk.atlaslm`). The runner's sampler
    (`sampling_config`, a `SamplingConfig`, for `num_steps` steps) and masking rate
    (`mlm_rate`)."""
    multimer = hasattr(m, "template_module")
    return StructureModel(
        input_embedder=ResidueEmbedder(),
        trunk_init=ResidueTrunkInit(init=trunk_init(m)),
        trunk=trunk(m, multimer=multimer, mlm_rate=mlm_rate),
        diffusion=diffusion_module(m),
        distogram=distogram_head(m.distogram_head),
        confidence=(multimer_confidence if multimer else monomer_confidence)(m.confidence_head),
        sampling=sampling(m.diffusion_head, sampling_config, num_steps=num_steps),
    )
