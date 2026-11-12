"""Converters for OpenFold3 modules (`openfold3.core.model`)."""

import jax.numpy as jnp

from jfold.blocks.atoms import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    AtomPairEmbedding,
    TrunkToAtoms,
)
from jfold.blocks.confidence import (
    AtomHead,
    ConfidenceModule,
    DistanceEmbedding,
    PairHead,
)
from jfold.blocks.denoiser import DiffusionConditioning, DiffusionModule, FourierEmbedding
from jfold.blocks.diffusion import (
    AdaLN,
    ConditionedAttention,
    ConditionedTransition,
    DiffusionTransformer,
    DiffusionTransformerLayer,
    SeparateKeys,
)
from jfold.blocks.inputs import InputEmbedder, TrunkInit
from jfold.blocks.msa import (
    MSALayer,
    MSAModule,
    MSAUpdate,
    PairWeightedAveraging,
    ProjectedOuterProductMean,
)
from jfold.blocks.pairformer import (
    AttentionPairBias,
    PairformerLayer,
    PairformerStack,
    SingleUpdate,
    TriangleAttention,
    TriangleMultiplication,
)
from jfold.blocks.templates import TemplateEmbedder
from jfold.blocks.windows import ShiftedWindows
from jfold.convert.torch import (
    OPENFOLD_COLUMN,
    OPENFOLD_ROW,
    array,
    attention,
    below_first_bin,
    concat_inputs,
    concat_outputs,
    doubled_bias,
    dropout,
    layer_norm,
    linear,
    localize,
)
from jfold.model import Recycle, StructureModel, Trunk
from jfold.nn import PairBias, ReluMLP, SinglesToPair, Transition, stack
from jfold.sampling import UpstreamSampler, masked_centering, no_alignment


def triangle_multiplication(m, *, outgoing: bool) -> TriangleMultiplication:
    if hasattr(m, "linear_ab_p"):  # FusedTriangleMultiplicativeUpdate
        linear_p, linear_g = linear(m.linear_ab_p), linear(m.linear_ab_g)
    else:
        linear_p = concat_outputs(m.linear_a_p, m.linear_b_p)
        linear_g = concat_outputs(m.linear_a_g, m.linear_b_g)
    if bool(m._outgoing) != outgoing:
        raise ValueError(f"expected outgoing={outgoing}, got {m._outgoing}")
    return TriangleMultiplication(
        norm_in=layer_norm(m.layer_norm_in),
        linear_p=linear_p,
        linear_g=linear_g,
        norm_out=layer_norm(m.layer_norm_out),
        linear_out=linear(m.linear_z),
        linear_out_g=linear(m.linear_g),
        outgoing=outgoing,
    )


def triangle_attention(m) -> TriangleAttention:
    # Both modules are starting-node ones, and the block transposes the pair for the other.
    if not m.starting:
        raise ValueError("expected a starting-node TriangleAttention")
    return TriangleAttention(
        norm=layer_norm(m.layer_norm), linear_bias=linear(m.linear_z), attention=attention(m.mha)
    )


def transition(m) -> Transition:
    if not hasattr(m, "swiglu"):
        raise NotImplementedError("only SwiGLU transitions are supported")
    return Transition(
        norm=layer_norm(m.layer_norm),
        linear_a=linear(m.swiglu.linear_a),
        linear_b=linear(m.swiglu.linear_b),
        linear_out=linear(m.linear_out),
    )


def attention_pair_bias(m) -> AttentionPairBias:
    return AttentionPairBias(
        norm_s=layer_norm(m.layer_norm_a),
        pair_bias=PairBias(norm=layer_norm(m.layer_norm_z), linear=linear(m.linear_z)),
        attention=attention(m.mha),
    )


def pair_block(pair, single=None) -> PairformerLayer:
    """Convert a `PairBlock`, the pair half of a block, with an optional single update."""
    return PairformerLayer(
        tri_mul_out=triangle_multiplication(pair.tri_mul_out, outgoing=True),
        tri_mul_in=triangle_multiplication(pair.tri_mul_in, outgoing=False),
        tri_att_start=triangle_attention(pair.tri_att_start),
        tri_att_end=triangle_attention(pair.tri_att_end),
        transition_z=transition(pair.pair_transition),
        single=single,
        dropout_row=dropout(pair.ps_dropout_row_layer.r, OPENFOLD_ROW),
        dropout_column=dropout(pair.ps_dropout_row_layer.r, OPENFOLD_COLUMN),
    )


def pairformer_layer(m) -> PairformerLayer:
    single = SingleUpdate(
        attention=attention_pair_bias(m.attn_pair_bias), transition=transition(m.single_transition)
    )
    return pair_block(m.pair_stack, single)


def pairformer_stack(m) -> PairformerStack:
    return PairformerStack(layers=stack([pairformer_layer(b) for b in m.blocks]))


def adaln(m) -> AdaLN:
    return AdaLN(
        norm_a=layer_norm(m.layer_norm_a),
        norm_s=layer_norm(m.layer_norm_s),
        linear_scale=linear(m.linear_g),
        linear_shift=linear(m.linear_s),
    )


def conditioned_transition(m) -> ConditionedTransition:
    return ConditionedTransition(
        adaln=adaln(m.layer_norm),
        linear_a=linear(m.swiglu.linear_a),
        linear_b=linear(m.swiglu.linear_b),
        linear_c=None,
        linear_out=linear(m.linear_out),
        linear_gate=linear(m.linear_g),
    )


def diffusion_transformer_layer(m) -> DiffusionTransformerLayer:
    attn = m.attention_pair_bias
    return DiffusionTransformerLayer(
        attention=ConditionedAttention(
            adaln=adaln(attn.layer_norm_a),
            attention=attention(attn.mha),
            linear_gate=linear(attn.linear_ada_out),
        ),
        transition=conditioned_transition(m.conditioned_transition),
    )


def diffusion_transformer(m) -> DiffusionTransformer:
    """Convert a token-level `DiffusionTransformer`."""
    return DiffusionTransformer(
        pair_bias=stack(
            [
                PairBias(
                    norm=layer_norm(b.attention_pair_bias.layer_norm_z),
                    linear=linear(b.attention_pair_bias.linear_z),
                )
                for b in m.blocks
            ]
        ),
        layers=stack([diffusion_transformer_layer(b) for b in m.blocks]),
    )


def _atom_layer(m) -> DiffusionTransformerLayer:
    attn = m.attention_pair_bias
    return DiffusionTransformerLayer(
        attention=ConditionedAttention(
            adaln=adaln(attn.layer_norm_a_q),
            attention=attention(attn.mha),
            linear_gate=linear(attn.linear_ada_out),
            keys=SeparateKeys(adaln(attn.layer_norm_a_k)),
        ),
        transition=conditioned_transition(m.conditioned_transition),
    )


def atom_transformer(m) -> DiffusionTransformer:
    """Convert a windowed `DiffusionTransformer`, copying the shared pair norm to each
    layer."""
    attn = m.blocks[0].attention_pair_bias
    windows = ShiftedWindows(queries=attn.n_query, keys=attn.n_key)
    transformer = DiffusionTransformer(
        pair_bias=stack(
            [
                PairBias(
                    norm=layer_norm(m.layer_norm_z), linear=linear(b.attention_pair_bias.linear_z)
                )
                for b in m.blocks
            ]
        ),
        layers=stack([_atom_layer(b) for b in m.blocks]),
    )
    return localize(transformer, windows)


def atom_encoder(m) -> AtomAttentionEncoder:
    """Convert an `AtomAttentionEncoder`, with or without noisy positions."""
    ref = m.ref_atom_feature_embedder
    noisy = getattr(m, "noisy_position_embedder", None)
    trunk = None
    if noisy is not None:
        trunk = TrunkToAtoms(
            norm_s=layer_norm(noisy.layer_norm_s),
            linear_s=linear(noisy.linear_s),
            norm_z=layer_norm(noisy.layer_norm_z),
            linear_z=linear(noisy.linear_z),
        )
    return AtomAttentionEncoder(
        embed_features=concat_inputs(
            ref.linear_ref_pos,
            ref.linear_ref_charge,
            ref.linear_ref_mask,
            ref.linear_ref_element,
            ref.linear_ref_atom_chars,
        ),
        pair_embedding=AtomPairEmbedding(
            linear_offset=linear(ref.linear_ref_offset),
            linear_inv_sq_dist=linear(ref.linear_inv_sq_dists),
            linear_valid=linear(ref.linear_valid_mask),
        ),
        trunk=trunk,
        linear_r=None if noisy is None else linear(noisy.linear_r),
        linear_c_queries=linear(m.linear_l),
        linear_c_keys=linear(m.linear_m),
        pair_mlp=ReluMLP(layers=tuple(linear(m.pair_mlp[i]) for i in (1, 3, 5))),
        transformer=atom_transformer(m.atom_transformer),
        linear_token=linear(m.linear_q[0]),
        queries_from_conditioned=True,
    )


def atom_decoder(m) -> AtomAttentionDecoder:
    return AtomAttentionDecoder(
        linear_a=linear(m.linear_q_in),
        transformer=atom_transformer(m.atom_transformer),
        norm_out=layer_norm(m.layer_norm),
        linear_out=linear(m.linear_q_out),
    )


def diffusion_module(m) -> DiffusionModule:
    """Convert a `DiffusionModule`, which reads unprojected relative position features."""
    c = m.diffusion_conditioning
    return DiffusionModule(
        conditioning=DiffusionConditioning(
            norm_z=layer_norm(c.layer_norm_z),
            linear_z=linear(c.linear_z),
            transitions_z=tuple(transition(t) for t in c.transition_z),
            norm_s=layer_norm(c.layer_norm_s),
            linear_s=linear(c.linear_s),
            fourier=FourierEmbedding(weight=array(c.fourier_emb.w), bias=array(c.fourier_emb.b)),
            norm_noise=layer_norm(c.layer_norm_n),
            linear_noise=linear(c.linear_n),
            transitions_s=tuple(transition(t) for t in c.transition_s),
            sigma_data=float(c.sigma_data),
        ),
        encoder=atom_encoder(m.atom_attn_enc),
        norm_s=layer_norm(m.layer_norm_s),
        linear_s=linear(m.linear_s),
        transformer=diffusion_transformer(m.diffusion_transformer),
        norm_a=layer_norm(m.layer_norm_a),
        decoder=atom_decoder(m.atom_attn_dec),
    )


def pair_weighted_averaging(m) -> PairWeightedAveraging:
    return PairWeightedAveraging(
        norm_m=layer_norm(m.layer_norm_m),
        linear_v=linear(m.linear_v),
        norm_z=layer_norm(m.layer_norm_z),
        linear_b=linear(m.linear_z),
        linear_g=linear(m.linear_g),
        linear_o=linear(m.linear_o),
        num_heads=m.no_heads,
    )


def outer_product_mean(m) -> ProjectedOuterProductMean:
    return ProjectedOuterProductMean(
        norm=layer_norm(m.layer_norm),
        linear_a=linear(m.linear_1),
        linear_b=linear(m.linear_2),
        linear_out=linear(m.linear_out),
        eps=float(m.eps),
    )


def msa_layer(m) -> MSALayer:
    """Convert an `MSAModuleBlock`, leaving out the MSA update if the block skips it."""
    msa = None
    if not m.skip_msa_update:
        msa = MSAUpdate(
            pair_weighted_averaging=pair_weighted_averaging(m.msa_att_row),
            transition=transition(m.msa_transition),
            dropout=dropout(m.msa_dropout_layer.r, OPENFOLD_ROW),
        )
    return MSALayer(
        outer_product_mean=outer_product_mean(m.outer_product_mean),
        msa=msa,
        pair=pair_block(m.pair_stack),
        outer_product_first=bool(m.opm_first),
    )


def msa_module(embedder, stack_module) -> MSAModule:
    """Convert the `MSAModuleEmbedder` and `MSAModuleStack` to one `MSAModule`."""
    layers = [msa_layer(b) for b in stack_module.blocks]
    last = layers[-1] if layers[-1].msa is None else None
    regular = layers[:-1] if last is not None else layers
    return MSAModule(
        linear_msa=linear(embedder.linear_m),
        linear_s=linear(embedder.linear_s_input),
        layers=stack(regular) if regular else None,
        last=last,
        max_rows=_max_rows(embedder),
        keep_query=False,
    )


def _max_rows(embedder) -> int | None:
    if embedder.subsample_main_msa:
        raise NotImplementedError("per-chain subsampling of the main MSA")
    if not embedder.subsample_all_msa:
        return None
    if embedder.min_subsampled_all_msa != embedder.max_subsampled_all_msa:
        raise NotImplementedError("a random number of MSA rows per pass")
    return embedder.max_subsampled_all_msa


def template_embedder(m) -> TemplateEmbedder:
    """Convert a `TemplateEmbedderAllAtom`."""
    pair_embedder, pair_stack = m.template_pair_embedder, m.template_pair_stack
    if not all(b.tri_mul_first for b in pair_stack.blocks):
        raise NotImplementedError("template blocks with triangle attention first")
    return TemplateEmbedder(
        norm_z=layer_norm(pair_embedder.layer_norm_z),
        linear_z=linear(pair_embedder.linear_z),
        linear_features=concat_inputs(
            pair_embedder.dgram_linear,
            pair_embedder.pseudo_beta_mask_linear,
            pair_embedder.aatype_linear_1,
            pair_embedder.aatype_linear_2,
            pair_embedder.x_linear,
            pair_embedder.y_linear,
            pair_embedder.z_linear,
            pair_embedder.backbone_mask_linear,
        ),
        stack=PairformerStack(layers=stack([pair_block(b) for b in pair_stack.blocks])),
        norm_v=layer_norm(pair_stack.layer_norm),
        linear_u=linear(m.linear_t),
    )


def _atom_head(m) -> AtomHead:
    p, b = m.max_atoms_per_token, m.c_out
    w = array(m.linear.weight).reshape(p, b, -1).transpose(0, 2, 1)
    bias = None if m.linear.bias is None else array(m.linear.bias).reshape(p, b)
    return AtomHead(weight=w, bias=bias, norm=layer_norm(m.layer_norm))


def confidence_module(m) -> ConfidenceModule:
    """Convert `AuxiliaryHeadsAllAtom` with its `PairformerEmbedding`."""
    emb = m.pairformer_embedding
    edges = jnp.linspace(emb.min_bin, emb.max_bin, emb.no_bin)
    return ConfidenceModule(
        singles_to_pair=SinglesToPair(linear_i=linear(emb.linear_i), linear_j=linear(emb.linear_j)),
        distances=DistanceEmbedding(edges=edges, linear=below_first_bin(emb.linear_distance)),
        stack=pairformer_stack(emb.pairformer_stack),
        pae=PairHead(linear=linear(m.pae.linear), norm=layer_norm(m.pae.layer_norm)),
        # Upstream symmetrizes after the projection, so the bias counts twice.
        pde=PairHead(
            linear=doubled_bias(m.pde.linear),
            norm_terms=layer_norm(m.pde.layer_norm),
            symmetric=True,
        ),
        plddt=_atom_head(m.plddt),
        resolved=_atom_head(m.experimentally_resolved),
    )


def distogram_head(m) -> PairHead:
    """Convert a `DistogramHead`, which symmetrizes after its projection."""
    return PairHead(linear=doubled_bias(m.linear), symmetric=True)


def input_embedder(m) -> InputEmbedder:
    """Convert the part of `InputEmbedderAllAtom` that computes `s_inputs`."""
    return InputEmbedder(encoder=atom_encoder(m.atom_attn_enc))


def trunk_init(m) -> TrunkInit:
    """Convert `InputEmbedderAllAtom`'s initial single and pair."""
    return TrunkInit(
        linear_s=linear(m.linear_s),
        singles_to_pair=SinglesToPair(linear_i=linear(m.linear_z_i), linear_j=linear(m.linear_z_j)),
        relative_positions=linear(m.linear_relpos),
        bonds=linear(m.linear_token_bonds),
    )


def structure_model(m) -> StructureModel:
    """Convert a whole OpenFold3 model (`projects.of3_all_atom.model.OpenFold3`)."""
    return StructureModel(
        input_embedder=input_embedder(m.input_embedder),
        trunk_init=trunk_init(m.input_embedder),
        trunk=Trunk(
            recycle_s=Recycle(norm=layer_norm(m.layer_norm_s), linear=linear(m.linear_s)),
            recycle_z=Recycle(norm=layer_norm(m.layer_norm_z), linear=linear(m.linear_z)),
            pairformer=pairformer_stack(m.pairformer_stack),
            templates=template_embedder(m.template_embedder),
            msa=msa_module(m.msa_module_embedder, m.msa_module),
        ),
        diffusion=diffusion_module(m.diffusion_module),
        distogram=distogram_head(m.aux_heads.distogram),
        confidence=confidence_module(m.aux_heads),
        sampling=sampling(m),
    )


def sampling(m) -> UpstreamSampler:
    """Return the sampler settings of OpenFold3's `SampleDiffusion`."""
    schedule = m.config.architecture.noise_schedule
    return UpstreamSampler(
        sigma_data=float(schedule.sigma_data),
        s_max=float(schedule.s_max),
        s_min=float(schedule.s_min),
        rho=float(schedule.p),
        num_steps=int(m.shared.diffusion.no_full_rollout_steps),
        end="none",
        gamma_0=float(m.sample_diffusion.gamma_0),
        gamma_min=float(m.sample_diffusion.gamma_min),
        noise_scale=float(m.sample_diffusion.noise_scale),
        step_scale=float(m.sample_diffusion.step_scale),
        center=masked_centering,
        augment=True,
        augment_after_noise=False,
        align=no_alignment,
    )
