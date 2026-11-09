"""ESMFold2 (`esm.models.esmfold2`) and its ESMC language model (`esm.models.esmc`)."""

import jax.numpy as jnp
from equinox.nn import Identity

from jfold.blocks.confidence import AtomHead, DistanceEmbedding, PairHead
from jfold.blocks.denoiser import DiffusionConditioning, FourierEmbedding
from jfold.blocks.diffusion import (
    AdaLN,
    ConditionedAttention,
    ConditionedTransition,
    DiffusionTransformer,
    DiffusionTransformerLayer,
)
from jfold.blocks.msa import (
    DEFAULT_MAX_ROWS,
    MSALayer,
    MSAModule,
    MSAUpdate,
    OuterProductMean,
    PairWeightedAveraging,
)
from jfold.blocks.pairformer import PairformerStack, PairMixerLayer, TriangleMultiplication
from jfold.convert.torch import array, embedding, layer_norm, linear
from jfold.model import Recycle, StructureModel
from jfold.models.esmfold2 import (
    ESMC,
    AtomDecoder,
    AtomEncoder,
    ClampedOuterProductMean,
    ConfidenceModule,
    DiffusionModule,
    ESMCLayer,
    InputEmbedder,
    LanguageModelPair,
    Parcae,
    ParcaeTrunk,
    RecyclingPairTrunk,
    Rope3D,
    RowAttentionPooling,
    SlidingWindowAttention,
    SlidingWindowLayer,
    SlidingWindowTransformer,
    SwiGLU,
    TrackedMSA,
    TrunkInit,
)
from jfold.nn import (
    Attention,
    Dropout,
    LayerNorm,
    Linear,
    PairBias,
    SinglesToPair,
    Transition,
    stack,
)
from jfold.sampling import UpstreamSampler, masked_centering, rigid_align


def _split(weight, num: int) -> list[Linear]:
    """Bias-free linears for the `num` equal parts of a weight's outputs."""
    return [Linear(weight=w) for w in jnp.split(array(weight), num)]


def triangle_multiplication(m) -> TriangleMultiplication:
    """`TriangleMultiplicativeUpdate` (or its `TriangleMultiplicativeBlock` engine): one
    bundled projection gives the a and b projections, then their gates."""
    m = getattr(m, "_engine", m)
    linear_p, linear_g = _split(m.proj_bundle.weight, 2)
    return TriangleMultiplication(
        norm_in=layer_norm(m.norm_start),
        linear_p=linear_p,
        linear_g=linear_g,
        norm_out=layer_norm(m.norm_mix),
        linear_out=linear(m.proj_emit),
        linear_out_g=linear(m.proj_gate),
        outgoing=m.flow == "outgoing",
    )


def transition(m) -> Transition:
    """`Transition` or `PairTransition`: a layer norm and a SwiGLU with packed inputs (the
    trunk's `Transition` adds its residual itself; jfold's layers add it)."""
    linear_a, linear_b = _split(m.ffn.w12.weight, 2)
    return Transition(
        norm=layer_norm(m.norm), linear_a=linear_a, linear_b=linear_b, linear_out=linear(m.ffn.w3)
    )


def pair_update(m) -> PairMixerLayer:
    """`PairUpdateBlock`: triangle multiplications and a transition, no single track."""
    return PairMixerLayer(
        tri_mul_out=triangle_multiplication(m.tri_mul_out),
        tri_mul_in=triangle_multiplication(m.tri_mul_in),
        transition_z=transition(m.pair_transition),
        single=None,
    )


def pair_stack(m) -> PairformerStack:
    """`FoldingTrunk` (the trunk, its coda, the language model's encoder and the confidence
    head's stack)."""
    return PairformerStack(layers=stack([pair_update(block) for block in m.blocks]))


def outer_product_mean(m) -> ClampedOuterProductMean | OuterProductMean:
    linear_a, linear_b = _split(m.W.weight, 2)
    cls = OuterProductMean if m.divide_outer_before_proj else ClampedOuterProductMean
    return cls(
        norm=layer_norm(m.norm), linear_a=linear_a, linear_b=linear_b, linear_out=linear(m.Wout)
    )


def pair_weighted_averaging(m) -> PairWeightedAveraging:
    return PairWeightedAveraging(
        norm_m=layer_norm(m.norm_single),
        linear_v=linear(m.Wv),
        norm_z=layer_norm(m.compute_bias[0]),
        linear_b=linear(m.compute_bias[1]),
        linear_g=linear(m.Wgate),
        linear_o=linear(m.Wout),
        num_heads=m.n_heads,
    )


def msa_layer(m) -> MSALayer:
    """`MSAEncoderBlock`: the pair updated from the MSA first; a final block has no MSA
    update."""
    msa = None
    if not m.is_final_block:
        msa = MSAUpdate(
            pair_weighted_averaging=pair_weighted_averaging(m.msa_pair_weighted_averaging),
            transition=transition(m.msa_transition),
        )
    return MSALayer(
        outer_product_mean=outer_product_mean(m.outer_product_mean),
        msa=msa,
        pair=PairMixerLayer(
            tri_mul_out=triangle_multiplication(m.tri_mul_out),
            tri_mul_in=triangle_multiplication(m.tri_mul_in),
            transition_z=transition(m.pair_transition),
            single=None,
        ),
        outer_product_first=True,
    )


def msa_module(m, *, max_rows: int = DEFAULT_MAX_ROWS) -> MSAModule:
    """`MSAEncoder`. MSA features come from `jfold.models.esmfold2.msa_features`, laid out
    `[S, T, ...]` (the upstream's `[T, S, ...]`, transposed). `max_rows`: the rows drawn per
    loop, the query always among them (`fold`'s `msa_max_depth`, which overrides the
    config's `msa_encoder.max_depth`)."""
    blocks = list(m.blocks)
    if not blocks[-1].is_final_block:
        raise ValueError("expected the last MSA block to skip its MSA update")
    return MSAModule(
        linear_msa=linear(m.embed),
        linear_s=linear(m.project_inputs),
        layers=stack([msa_layer(b) for b in blocks[:-1]]) if len(blocks) > 1 else None,
        last=msa_layer(blocks[-1]),
        max_rows=max_rows,
    )


def experimental_msa_layer(m) -> MSALayer:
    """The experimental `MSAEncoderBlock`: the MSA updated first, then the pair from it
    (outer product divided before its projection)."""
    return MSALayer(
        outer_product_mean=outer_product_mean(m.outer_product_mean),
        msa=MSAUpdate(
            pair_weighted_averaging=pair_weighted_averaging(m.msa_pair_weighted_averaging),
            transition=transition(m.msa_transition),
        ),
        pair=PairMixerLayer(
            tri_mul_out=triangle_multiplication(m.tri_mul_out),
            tri_mul_in=triangle_multiplication(m.tri_mul_in),
            transition_z=transition(m.pair_transition),
            single=None,
        ),
        outer_product_first=False,
    )


def experimental_msa(m, *, max_rows: int = DEFAULT_MAX_ROWS) -> TrackedMSA:
    """The experimental `MSAEncoder`. MSA features and `max_rows` as for `msa_module`."""
    return TrackedMSA(
        module=MSAModule(
            linear_msa=linear(m.embed),
            linear_s=linear(m.project_inputs),
            layers=stack([experimental_msa_layer(b) for b in m.blocks]),
            max_rows=max_rows,
        )
    )


def adaln(m) -> AdaLN:
    """`AdaptiveLayerNorm`: the conditioning's norm has a scale and no offset."""
    return AdaLN(
        norm_a=LayerNorm(weight=None, bias=None, eps=float(m.eps)),
        norm_s=LayerNorm(weight=array(m.s_scale), bias=None, eps=float(m.eps)),
        linear_scale=linear(m.s_gate),
        linear_shift=linear(m.s_shift),
    )


def diffusion_transformer_layer(attn, trans) -> DiffusionTransformerLayer:
    linear_k, linear_v = _split(attn.kv_proj.weight, 2)
    linear_a, linear_b = _split(trans.lin_swish.weight, 2)
    return DiffusionTransformerLayer(
        attention=ConditionedAttention(
            adaln=adaln(attn.adaln),
            attention=Attention(
                linear_q=linear(attn.q_proj),
                linear_k=linear_k,
                linear_v=linear_v,
                linear_g=linear(attn.g_proj),
                linear_o=linear(attn.out_proj),
                num_heads=attn.num_heads,
            ),
            linear_gate=linear(attn.out_gate),
        ),
        transition=ConditionedTransition(
            adaln=adaln(trans.adaln),
            linear_a=linear_a,
            linear_b=linear_b,
            linear_c=None,
            linear_out=linear(trans.lin_out),
            linear_gate=linear(trans.output_gate),
        ),
    )


def diffusion_transformer(m) -> DiffusionTransformer:
    """The token `DiffusionTransformer` (conditioned)."""
    return DiffusionTransformer(
        pair_bias=stack(
            [
                PairBias(norm=layer_norm(a.pair_norm), linear=linear(a.pair_bias_proj))
                for a in m.attn_blocks
            ]
        ),
        layers=stack(
            [
                diffusion_transformer_layer(a, t)
                for a, t in zip(m.attn_blocks, m.transition_blocks, strict=True)
            ]
        ),
    )


def _transition_layer(m) -> Transition:
    """`TransitionLayer`: a layer norm and a SwiGLU."""
    return Transition(
        norm=layer_norm(m.norm),
        linear_a=linear(m.a_proj),
        linear_b=linear(m.b_proj),
        linear_out=linear(m.out_proj),
    )


def diffusion_conditioning(m) -> DiffusionConditioning:
    """`DiffusionConditioning`: the single is conditioned on `s_inputs` alone (pass a
    zero-width trunk single)."""
    return DiffusionConditioning(
        norm_z=layer_norm(m.z_input_norm),
        linear_z=linear(m.z_proj),
        transitions_z=tuple(_transition_layer(t) for t in m.z_transitions),
        norm_s=layer_norm(m.s_input_norm),
        linear_s=linear(m.s_proj),
        fourier=FourierEmbedding(weight=array(m.fourier.w), bias=array(m.fourier.b)),
        norm_noise=layer_norm(m.noise_norm),
        linear_noise=linear(m.noise_proj),
        transitions_s=tuple(_transition_layer(t) for t in m.s_transitions),
        sigma_data=float(m.sigma_data),
    )


def sliding_window_layer(m) -> SlidingWindowLayer:
    """`SWAAtomBlock`."""
    attn = m.attn
    linear_q, linear_k, linear_v = _split(attn.Wqkv.weight, 3)
    ffn_a, ffn_b = _split(m.ffn.w_up.weight, 2)
    return SlidingWindowLayer(
        modulation=linear(m.adaln_modulation[1]),
        attention=SlidingWindowAttention(
            linear_q=linear_q,
            linear_k=linear_k,
            linear_v=linear_v,
            linear_g=linear(attn.gate_proj),
            linear_o=linear(attn.out_proj),
            num_heads=attn.n_heads,
            half_window=attn.half_window,
        ),
        ffn=SwiGLU(linear_a=ffn_a, linear_b=ffn_b, linear_out=linear(m.ffn.w_down)),
    )


def sliding_window_transformer(m) -> SlidingWindowTransformer:
    """`SWAAtomTransformer`."""
    return SlidingWindowTransformer(
        layers=stack([sliding_window_layer(b) for b in m.blocks]),
        rope=Rope3D(
            head_dim=m.head_dim,
            spatial_pairs=m.n_spatial_rope_pairs_per_axis,
            uid_pairs=m.n_uid_rope_pairs,
            spatial_base=float(m.spatial_rope_base_frequency),
            uid_base=float(m.uid_rope_base_frequency),
        ),
    )


def atom_encoder(m) -> AtomEncoder:
    """`EsmFold2AtomEncoder`. Its coordinate projection takes the scaled noisy positions
    and a previous prediction the upstream always passes as zeros; only the first part is
    kept. Atom features come from `jfold.models.esmfold2.atom_features`."""
    linear_r = None
    if m.structure_prediction:
        linear_r = Linear(weight=array(m.coords_linear.weight[:, :3]))
    return AtomEncoder(
        embed_features=linear(m.atom_linear),
        norm=layer_norm(m.atom_norm),
        linear_r=linear_r,
        transformer=sliding_window_transformer(m.atom_transformer),
        linear_token=linear(m.atom_to_token_linear),
    )


def atom_decoder(m) -> AtomDecoder:
    """`EsmFold2AtomDecoder`."""
    return AtomDecoder(
        linear_a=linear(m.token_to_atom_linear),
        transformer=sliding_window_transformer(m.atom_transformer),
        norm_out=layer_norm(m.norm),
        linear_out=linear(m.output_linear),
    )


def diffusion_module(m) -> DiffusionModule:
    """`DiffusionModule` (the structure head's score model)."""
    return DiffusionModule(
        conditioning=diffusion_conditioning(m.conditioning),
        encoder=atom_encoder(m.atom_encoder),
        norm_s=layer_norm(m.s_step_norm),
        linear_s=linear(m.s_to_token),
        transformer=diffusion_transformer(m.token_transformer),
        norm_a=layer_norm(m.token_norm),
        decoder=atom_decoder(m.atom_decoder),
    )


def relative_positions(m) -> Linear:
    """`ResIdxAsymIdSymIdEntityIdEncoding`'s projection of
    `jfold.blocks.inputs.relative_position_features` (with chain offsets between chains:
    `chain_offset=asym_id[:, None] != asym_id[None, :]`)."""
    return linear(m.embed)


def confidence_module(m) -> ConfidenceModule:
    """`ConfidenceHead` of the release or experimental model (logits; the expected values
    and pTM are reductions of them). The experimental head has no PDE or resolved heads and
    no norm before its PAE head."""
    pae_norm = getattr(m, "pae_ln", None)
    return ConfidenceModule(
        norm_s_inputs=layer_norm(m.s_inputs_norm),
        norm_z=layer_norm(m.z_norm),
        singles_to_pair=SinglesToPair(
            linear_i=linear(m.s_to_z),
            linear_j=linear(m.s_to_z_transpose),
            product_i=linear(m.s_to_z_prod_in1),
            product_j=linear(m.s_to_z_prod_in2),
            product_out=linear(m.s_to_z_prod_out),
        ),
        distances=DistanceEmbedding(
            edges=array(m.boundaries), linear=embedding(m.dist_bin_pairwise_embed)
        ),
        stack=pair_stack(m.folding_trunk),
        pooling=RowAttentionPooling(
            linear_score=linear(m.row_attention_pooling.attn_proj),
            linear_out=linear(m.row_attention_pooling.out_proj),
        ),
        pae=PairHead(
            linear=linear(m.pae_head), norm=Identity() if pae_norm is None else layer_norm(pae_norm)
        ),
        plddt=AtomHead(weight=array(m.plddt_weight), norm=layer_norm(m.plddt_ln)),
        pde=PairHead(linear=linear(m.pde_head), norm=layer_norm(m.pde_ln))
        if hasattr(m, "pde_head")
        else None,
        resolved=AtomHead(weight=array(m.resolved_weight), norm=layer_norm(m.resolved_ln))
        if hasattr(m, "resolved_weight")
        else None,
    )


def language_model_pair(m) -> LanguageModelPair:
    """`LanguageModelShim`."""
    to_pair, norm_out = m.base_z_mlp
    return LanguageModelPair(
        norm_hidden=layer_norm(m.base_z_linear[0]),
        linear_hidden=linear(m.base_z_linear[1]),
        layer_logits=array(m.base_z_combine),
        linear_down=linear(to_pair.downproject),
        mlp_in=linear(to_pair.output_mlp[0]),
        mlp_out=linear(to_pair.output_mlp[2]),
        norm_out=layer_norm(norm_out),
    )


def _fused_norm(m) -> LayerNorm:
    """The layer norm of a Transformer Engine-style fused module."""
    return LayerNorm(
        weight=array(m.layer_norm_weight), bias=array(m.layer_norm_bias), eps=float(m.eps)
    )


def esmc_layer(m) -> ESMCLayer:
    """`EsmcUnifiedTransformerBlock` (pure-PyTorch modules, SwiGLU feed-forward)."""
    attn, ffn = m.attn, m.ffn
    linear_q, linear_k, linear_v = _split(attn.layernorm_qkv.weight, 3)
    linear_a, linear_b = _split(ffn.fc1_weight, 2)
    return ESMCLayer(
        norm=_fused_norm(attn.layernorm_qkv),
        linear_q=linear_q,
        linear_k=linear_k,
        linear_v=linear_v,
        norm_q=layer_norm(attn.q_ln),
        norm_k=layer_norm(attn.k_ln),
        linear_o=linear(attn.out_proj),
        transition=Transition(
            norm=_fused_norm(ffn),
            linear_a=linear_a,
            linear_b=linear_b,
            linear_out=Linear(weight=array(ffn.fc2_weight)),
        ),
        num_heads=attn.n_heads,
        residual_scale=float(m.scaling_factor),
    )


def esmc(m) -> ESMC:
    """`EsmcModel`, built without CUDA kernels (its pure-PyTorch modules)."""
    return ESMC(
        embed=embedding(m.embed),
        layers=stack([esmc_layer(b) for b in m.transformer.blocks]),
        norm=layer_norm(m.transformer.norm),
        rope_base=float(m.transformer.blocks[0].attn.rotary.base),
    )


def parcae(m) -> Parcae:
    """The Parcae recurrence's parameters on `EsmFold2Model`."""
    return Parcae(
        norm=layer_norm(m.parcae_input_norm),
        log_a=array(m.parcae_log_a),
        log_delta=array(m.parcae_log_delta),
        b=array(m.parcae_b_cont),
        readout=linear(m.parcae_readout),
    )


def esmfold2(m) -> StructureModel:
    """A whole `EsmFold2Model` (release type), without its ESMC backbone (`trunk_init.esmc`:
    convert it with `esmc` and attach it, as `jfold.pretrained` does). The release type
    always reads the MSA profile (it ignores `disable_msa_features`)."""
    config = m.config
    if m.msa_encoder is not None and not config.msa_encoder.overwrite:
        raise NotImplementedError("MSA encoder adding to the pair (overwrite=False)")
    lm = config.lm_encoder
    return StructureModel(
        input_embedder=InputEmbedder(
            encoder=atom_encoder(m.inputs_embedder.atom_attention_encoder)
        ),
        trunk_init=TrunkInit(
            singles_to_pair=SinglesToPair(linear_i=linear(m.z_init_1), linear_j=linear(m.z_init_2)),
            relative_positions=relative_positions(m.rel_pos),
            bonds=linear(m.token_bonds),
            language_model=language_model_pair(m.language_model),
            msa_column_mask_rate=float(config.msa_encoder.column_mask_rate),
        ),
        trunk=ParcaeTrunk(
            lm_encoder=None if m.lm_encoder is None else pair_stack(m.lm_encoder),
            msa=None if m.msa_encoder is None else msa_module(m.msa_encoder),
            parcae=parcae(m),
            trunk=pair_stack(m.folding_trunk),
            coda=pair_stack(m.parcae_coda),
            lm_dropout=lm_dropout(lm.lm_dropout if lm.per_loop_lm_dropout else 0.0),
        ),
        diffusion=diffusion_module(m.structure_head.diffusion_module),
        distogram=PairHead(linear=linear(m.distogram_head), symmetric=True),
        confidence=confidence_module(m.confidence_head),
        sampling=sampling(m.structure_head),
    )


def sampling(head, *, max_sigma: float | None = 256.0) -> UpstreamSampler:
    """`DiffusionStructureHead.sample`'s settings, with `fold`'s default cap on the
    starting noise level (`max_inference_sigma`): re-centred, randomly rotated and
    translated, the noisy sample aligned onto the denoised one (with gradients)."""
    return UpstreamSampler(
        sigma_data=float(head.sigma_data),
        s_max=float(head.inference_s_max),
        s_min=float(head.inference_s_min),
        rho=float(head.inference_p),
        num_steps=int(head.inference_num_steps),
        end="append_zero",
        max_sigma=max_sigma,
        gamma_0=float(head.gamma_0),
        gamma_min=float(head.gamma_min),
        noise_scale=float(head.noise_scale),
        step_scale=float(head.step_scale),
        center=masked_centering,
        augment=True,
        augment_after_noise=False,
        align=rigid_align,
    )


def lm_dropout(rate: float) -> Dropout | Identity:
    """ESMFold2's dropout on the language model's pair: elementwise, and on at inference
    (upstream applies it with `training=True`)."""
    return Dropout(float(rate), (), inference=False) if rate else Identity()


def esmfold2_experimental(m) -> StructureModel:
    """A whole `EsmFold2ExperimentalModel`, without its ESMC backbone (`trunk_init.esmc`:
    convert it with `esmc` and attach it, as `jfold.pretrained` does). Its token features
    carry the MSA profile unless the config disables MSA features."""
    return StructureModel(
        input_embedder=InputEmbedder(
            encoder=atom_encoder(m.inputs_embedder.atom_attention_encoder),
            msa_profile=not getattr(m.config, "disable_msa_features", False),
        ),
        trunk_init=TrunkInit(
            singles_to_pair=SinglesToPair(linear_i=linear(m.z_init_1), linear_j=linear(m.z_init_2)),
            relative_positions=relative_positions(m.rel_pos),
            bonds=linear(m.token_bonds),
            language_model=language_model_pair(m.language_model),
            lm_dropout=lm_dropout(getattr(m.config, "lm_dropout", 0.0)),
            msa_column_mask_rate=float(m.config.msa_encoder.column_mask_rate),
        ),
        trunk=RecyclingPairTrunk(
            recycle=Recycle(
                norm=layer_norm(m.pair_loop_proj[0]), linear=linear(m.pair_loop_proj[1])
            ),
            msa=None if m.msa_encoder is None else experimental_msa(m.msa_encoder),
            trunk=pair_stack(m.folding_trunk),
        ),
        diffusion=diffusion_module(m.structure_head.diffusion_module),
        distogram=PairHead(linear=linear(m.distogram_head), symmetric=True),
        confidence=None if m.confidence_head is None else confidence_module(m.confidence_head),
        sampling=sampling(m.structure_head),
    )
