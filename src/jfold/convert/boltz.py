"""Convert Boltz-lineage blocks, shared by Boltz-2 and ProMERA."""

import jax.numpy as jnp
import torch
from equinox.nn import Identity

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
    TokenHead,
)
from jfold.blocks.denoiser import DiffusionConditioning, DiffusionModule, FourierEmbedding
from jfold.blocks.diffusion import (
    AdaLN,
    ConditionedAttention,
    ConditionedTransition,
    DiffusionTransformer,
    DiffusionTransformerLayer,
)
from jfold.blocks.inputs import InputEmbedder
from jfold.blocks.msa import (
    DEFAULT_MAX_ROWS,
    MSALayer,
    MSAModule,
    MSAUpdate,
    OuterProductMean,
    PairWeightedAveraging,
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
from jfold.blocks.windows import CenteredWindows
from jfold.convert.torch import (
    BOLTZ_COLUMN,
    BOLTZ_ROW,
    array,
    attention,
    dropout,
    embedding,
    layer_norm,
    linear,
    localize,
    negated_input,
)
from jfold.nn import Linear, PairBias, ReluMLP, SinglesToPair, Transition, stack


def triangle_multiplication(m, *, outgoing: bool) -> TriangleMultiplication:
    return TriangleMultiplication(
        norm_in=layer_norm(m.norm_in),
        linear_p=linear(m.p_in),
        linear_g=linear(m.g_in),
        norm_out=layer_norm(m.norm_out),
        linear_out=linear(m.p_out),
        linear_out_g=linear(m.g_out),
        outgoing=outgoing,
    )


def triangle_attention(m, *, starting: bool) -> TriangleAttention:
    # The ending-node module transposes internally, and jfold's layer transposes around it.
    if bool(m.starting) != starting:
        raise ValueError(f"expected starting={starting}, got {m.starting}")
    return TriangleAttention(
        norm=layer_norm(m.layer_norm), linear_bias=linear(m.linear), attention=attention(m.mha)
    )


def transition(m) -> Transition:
    return Transition(
        norm=layer_norm(m.norm),
        linear_a=linear(m.fc1),
        linear_b=linear(m.fc2),
        linear_out=linear(m.fc3),
    )


def _pair_bias(m) -> PairBias:
    return PairBias(norm=layer_norm(m[0]), linear=linear(m[1]))


def attention_pair_bias(m, norm_s) -> AttentionPairBias:
    return AttentionPairBias(
        norm_s=layer_norm(norm_s),
        pair_bias=_pair_bias(m.proj_z),
        attention=attention(
            m, q="proj_q", k="proj_k", v="proj_v", g="proj_g", o="proj_o", heads=m.num_heads
        ),
    )


def single_update(m) -> SingleUpdate | None:
    """Convert a layer's single update, or return None for ProMERA's pair-only layers."""
    attn = getattr(m, "attention", None)
    if attn is None:
        return None
    if hasattr(m, "pre_norm_s"):  # Boltz-2 applies the norm outside the attention
        norm_s = m.pre_norm_s
        post_norm = m.s_post_norm
    elif attn.initial_norm:
        norm_s, post_norm = attn.norm_s, None
    else:
        raise ValueError("Pairformer attention without an input norm")
    return SingleUpdate(
        attention=attention_pair_bias(attn, norm_s),
        transition=transition(m.transition_s),
        post_norm=layer_norm(post_norm)
        if isinstance(post_norm, torch.nn.LayerNorm)
        else Identity(),
    )


def pairformer_layer(m) -> PairformerLayer:
    """Convert a Boltz-2 or Boltz-1 Pairformer layer. ProMERA uses Boltz-1's."""
    m = getattr(m, "_checkpoint_wrapped_module", m)
    return PairformerLayer(
        tri_mul_out=triangle_multiplication(m.tri_mul_out, outgoing=True),
        tri_mul_in=triangle_multiplication(m.tri_mul_in, outgoing=False),
        tri_att_start=triangle_attention(m.tri_att_start, starting=True),
        tri_att_end=triangle_attention(m.tri_att_end, starting=False),
        transition_z=transition(m.transition_z),
        single=single_update(m),
        dropout_row=dropout(m.dropout, BOLTZ_ROW),
        dropout_column=dropout(m.dropout, BOLTZ_COLUMN),
    )


def pairformer_stack(m) -> PairformerStack:
    """Convert Boltz-2's or ProMERA's `PairformerModule`."""
    return PairformerStack(layers=stack([pairformer_layer(layer) for layer in m.layers]))


def adaln(m) -> AdaLN:
    return AdaLN(
        norm_a=layer_norm(m.a_norm),
        norm_s=layer_norm(m.s_norm),
        linear_scale=linear(m.s_scale),
        linear_shift=linear(m.s_bias),
    )


def conditioned_transition(m) -> ConditionedTransition:
    # `swish_gate` is one linear producing `[x | gates]`, followed by `silu(gates) * x`.
    x, gates = m.swish_gate[0].weight.chunk(2)
    return ConditionedTransition(
        adaln=adaln(m.adaln),
        linear_a=Linear(weight=array(gates)),
        linear_b=Linear(weight=array(x)),
        linear_c=linear(m.a_to_b),
        linear_out=linear(m.b_to_a),
        linear_gate=linear(m.output_projection[0]),
    )


def diffusion_transformer_layer(m) -> DiffusionTransformerLayer:
    attn = m.pair_bias_attn
    post_norm = getattr(m, "post_lnorm", None)
    return DiffusionTransformerLayer(
        attention=ConditionedAttention(
            adaln=adaln(m.adaln),
            attention=attention(
                attn,
                q="proj_q",
                k="proj_k",
                v="proj_v",
                g="proj_g",
                o="proj_o",
                heads=attn.num_heads,
            ),
            linear_gate=linear(m.output_projection_linear),
        ),
        transition=conditioned_transition(m.transition),
        post_norm=layer_norm(post_norm)
        if isinstance(post_norm, torch.nn.LayerNorm)
        else Identity(),
    )


def diffusion_transformer(m, proj_z=None, pair_bias=None) -> DiffusionTransformer:
    """Convert a `DiffusionTransformer`, taking Boltz-2's pair-bias projections from its
    conditioning as `proj_z`."""
    layers = [getattr(layer, "_checkpoint_wrapped_module", layer) for layer in m.layers]
    if pair_bias is None:
        if proj_z is None:
            proj_z = [layer.pair_bias_attn.proj_z for layer in layers]
        pair_bias = stack([_pair_bias(p) for p in proj_z])
    return DiffusionTransformer(
        pair_bias=pair_bias,
        layers=stack([diffusion_transformer_layer(layer) for layer in layers]),
    )


def atom_transformer(m, proj_z=None, pair_bias=None) -> DiffusionTransformer:
    """Convert an `AtomTransformer`, taking Boltz-2's pair-bias projections as `proj_z`."""
    windows = CenteredWindows(queries=m.attn_window_queries, keys=m.attn_window_keys)
    return localize(diffusion_transformer(m.diffusion_transformer, proj_z, pair_bias), windows)


def _relu_mlp(m) -> ReluMLP:
    return ReluMLP(layers=tuple(linear(layer) for layer in m if isinstance(layer, torch.nn.Linear)))


def atom_encoder(features, encoder=None, proj_z=None, pair_bias=None) -> AtomAttentionEncoder:
    """Convert the atom attention encoder, which Boltz-2 splits into `features`
    (`AtomEncoder`) and `encoder` (`AtomAttentionEncoder`)."""
    encoder = features if encoder is None else encoder
    trunk = None
    if features.structure_prediction:
        trunk = TrunkToAtoms(
            norm_s=layer_norm(features.s_to_c_trans[0]),
            linear_s=linear(features.s_to_c_trans[1]),
            norm_z=layer_norm(features.z_to_p_trans[0]),
            linear_z=linear(features.z_to_p_trans[1]),
        )
    return AtomAttentionEncoder(
        embed_features=linear(features.embed_atom_features),
        pair_embedding=AtomPairEmbedding(
            linear_offset=negated_input(features.embed_atompair_ref_pos),  # upstream is key - query
            linear_inv_sq_dist=linear(features.embed_atompair_ref_dist),
            linear_valid=linear(features.embed_atompair_mask),
        ),
        trunk=trunk,
        linear_r=linear(encoder.r_to_q_trans) if encoder.structure_prediction else None,
        linear_c_queries=linear(features.c_to_p_trans_q[1]),
        linear_c_keys=linear(features.c_to_p_trans_k[1]),
        pair_mlp=_relu_mlp(features.p_mlp),
        transformer=atom_transformer(encoder.atom_encoder, proj_z, pair_bias),
        linear_token=linear(encoder.atom_to_token_trans[0]),
    )


def atom_decoder(m, proj_z=None) -> AtomAttentionDecoder:
    """Convert an `AtomAttentionDecoder`, taking Boltz-2's pair-bias projections as
    `proj_z`."""
    update = m.atom_feat_to_atom_pos_update
    norm_out, linear_out = (
        (update[0], update[1]) if isinstance(update, torch.nn.Sequential) else (None, update)
    )
    return AtomAttentionDecoder(
        linear_a=linear(m.a_to_q_trans),
        transformer=atom_transformer(m.atom_decoder, proj_z),
        norm_out=Identity() if norm_out is None else layer_norm(norm_out),
        linear_out=linear(linear_out),
    )


def diffusion_module(module, conditioning=None) -> DiffusionModule:
    """Convert the score model, taking Boltz-2's separate `DiffusionConditioning` as
    `conditioning`."""
    single = module.single_conditioner
    pair = (module if conditioning is None else conditioning).pairwise_conditioner
    fourier = single.fourier_embed.proj
    if conditioning is None:
        encoder = atom_encoder(module.atom_attention_encoder)
        transformer = diffusion_transformer(module.token_transformer)
        decoder = atom_decoder(module.atom_attention_decoder)
    else:
        encoder = atom_encoder(
            conditioning.atom_encoder, module.atom_attention_encoder, conditioning.atom_enc_proj_z
        )
        transformer = diffusion_transformer(
            module.token_transformer, conditioning.token_trans_proj_z
        )
        decoder = atom_decoder(module.atom_attention_decoder, conditioning.atom_dec_proj_z)
    return DiffusionModule(
        conditioning=DiffusionConditioning(
            norm_z=layer_norm(pair.dim_pairwise_init_proj[0]),
            linear_z=linear(pair.dim_pairwise_init_proj[1]),
            transitions_z=tuple(transition(t) for t in pair.transitions),
            norm_s=layer_norm(single.norm_single),
            linear_s=linear(single.single_embed),
            fourier=FourierEmbedding(weight=array(fourier.weight[:, 0]), bias=array(fourier.bias)),
            norm_noise=layer_norm(single.norm_fourier),
            linear_noise=linear(single.fourier_to_single),
            transitions_s=tuple(transition(t) for t in single.transitions),
            sigma_data=float(single.sigma_data),
        ),
        encoder=encoder,
        norm_s=layer_norm(module.s_to_a_linear[0]),
        linear_s=linear(module.s_to_a_linear[1]),
        transformer=transformer,
        norm_a=layer_norm(module.a_norm),
        decoder=decoder,
    )


def pair_weighted_averaging(m) -> PairWeightedAveraging:
    return PairWeightedAveraging(
        norm_m=layer_norm(m.norm_m),
        linear_v=linear(m.proj_m),
        norm_z=layer_norm(m.norm_z),
        linear_b=linear(m.proj_z),
        linear_g=linear(m.proj_g),
        linear_o=linear(m.proj_o),
        num_heads=m.num_heads,
    )


def outer_product_mean(m) -> OuterProductMean:
    return OuterProductMean(
        norm=layer_norm(m.norm),
        linear_a=linear(m.proj_a),
        linear_b=linear(m.proj_b),
        linear_out=linear(m.proj_o),
    )


def msa_layer(m) -> MSALayer:
    """Convert Boltz-2's or ProMERA's `MSALayer`."""
    if hasattr(m, "pairformer_layer"):
        pair = pairformer_layer(m.pairformer_layer)
    else:
        pair = PairformerLayer(
            tri_mul_out=triangle_multiplication(m.tri_mul_out, outgoing=True),
            tri_mul_in=triangle_multiplication(m.tri_mul_in, outgoing=False),
            tri_att_start=triangle_attention(m.tri_att_start, starting=True),
            tri_att_end=triangle_attention(m.tri_att_end, starting=False),
            transition_z=transition(m.z_transition),
            single=None,
            dropout_row=dropout(m.z_dropout, BOLTZ_ROW),
            dropout_column=dropout(m.z_dropout, BOLTZ_COLUMN),
        )
    return MSALayer(
        outer_product_mean=outer_product_mean(m.outer_product_mean),
        msa=MSAUpdate(
            pair_weighted_averaging=pair_weighted_averaging(m.pair_weighted_averaging),
            transition=transition(m.msa_transition),
            dropout=dropout(m.msa_dropout, BOLTZ_ROW),
        ),
        pair=pair,
        outer_product_first=False,
    )


def msa_module(m, *, max_rows: int | None = DEFAULT_MAX_ROWS) -> MSAModule:
    """Convert Boltz-2's or ProMERA's `MSAModule`, drawing `max_rows` rows per pass."""
    layers = [getattr(layer, "_checkpoint_wrapped_module", layer) for layer in m.layers]
    return MSAModule(
        linear_msa=linear(m.msa_proj),
        linear_s=linear(m.s_proj),
        layers=stack([msa_layer(layer) for layer in layers]),
        max_rows=max_rows,
        keep_query=False,
    )


def template_embedder(m) -> TemplateEmbedder:
    """Convert Boltz-2's `TemplateV2Module`."""
    return TemplateEmbedder(
        norm_z=layer_norm(m.z_norm),
        linear_z=linear(m.z_proj),
        linear_features=linear(m.a_proj),
        stack=PairformerStack(
            layers=stack([pairformer_layer(layer) for layer in m.pairformer.layers])
        ),
        norm_v=layer_norm(m.v_norm),
        linear_u=linear(m.u_proj),
        add_stack_input=True,
    )


def _pair_head(m, name, *, norm, symmetric=False):
    if hasattr(m, f"to_{name}_intra_logits"):
        return PairHead(
            linear=linear(getattr(m, f"to_{name}_intra_logits")),
            linear_inter=linear(getattr(m, f"to_{name}_inter_logits")),
            norm=norm,
            symmetric=symmetric,
        )
    return PairHead(linear=linear(getattr(m, f"to_{name}_logits")), norm=norm, symmetric=symmetric)


def _per_token_or_atom(linear_module, *, token_level: bool, max_atoms: int, norm):
    if token_level:
        return TokenHead(linear=linear(linear_module), norm=norm)
    w = array(linear_module.weight).reshape(max_atoms, linear_module.out_features // max_atoms, -1)
    return AtomHead(weight=w.transpose(0, 2, 1), norm=norm)


def confidence_module(m) -> ConfidenceModule:
    """Convert Boltz-2's or ProMERA's `ConfidenceModule`."""
    heads = m.confidence_heads
    protenix_norms = getattr(heads, "_protenix", False)  # ProMERA option to norm heads
    norm = (
        (lambda name: layer_norm(getattr(heads, name)))
        if protenix_norms
        else (lambda name: Identity())
    )
    levels = {
        "token_level": getattr(heads, "token_level_confidence", True),
        "max_atoms": getattr(heads, "max_num_atoms_per_token", 1),
    }
    product = {}
    if getattr(m, "add_s_to_z_prod", False):
        product = {
            "product_i": linear(m.s_to_z_prod_in1),
            "product_j": linear(m.s_to_z_prod_in2),
            "product_out": linear(m.s_to_z_prod_out),
        }
    plddt = heads.to_plddt_logits
    resolved = getattr(heads, "to_resolved_logits", None)
    return ConfidenceModule(
        singles_to_pair=SinglesToPair(
            linear_i=linear(m.s_to_z), linear_j=linear(m.s_to_z_transpose), **product
        ),
        distances=DistanceEmbedding(
            edges=array(m.boundaries), linear=embedding(m.dist_bin_pairwise_embed)
        ),
        stack=pairformer_stack(m.pairformer_stack),
        pae=_pair_head(heads, "pae", norm=norm("pae_ln")),
        pde=_pair_head(heads, "pde", symmetric=True, norm=norm("pde_ln")),
        plddt=_per_token_or_atom(plddt, **levels, norm=norm("plddt_ln")),
        resolved=None
        if resolved is None
        else _per_token_or_atom(resolved, **levels, norm=Identity()),
        norm_s=Identity() if getattr(m, "no_update_s", False) else layer_norm(m.s_norm),
        norm_z=layer_norm(m.z_norm) if hasattr(m, "z_norm") else Identity(),
        norm_s_inputs=layer_norm(m.s_inputs_norm) if hasattr(m, "s_inputs_norm") else Identity(),
        linear_s_inputs_to_s=linear(m.s_input_to_s)
        if getattr(m, "add_s_input_to_s", False)
        else None,
    )


def distogram_head(m) -> PairHead:
    """Convert a `DistogramModule` with a single distogram."""
    return PairHead(linear=linear(m.distogram), symmetric=True)


def relative_positions(m) -> Linear:
    """Convert the relative position encoding's projection."""
    return linear(m.linear_layer)


def _split_pair_bias(m, depth: int) -> PairBias:
    w = array(m[1].weight)
    heads = w.shape[0] // depth
    norm = layer_norm(m[0])
    return stack(
        [
            PairBias(norm=norm, linear=Linear(weight=w[i * heads : (i + 1) * heads]))
            for i in range(depth)
        ]
    )


def input_embedder(m) -> InputEmbedder:
    """Convert Boltz-2's or ProMERA's `InputEmbedder`."""
    if not hasattr(m, "res_type_encoding"):  # ProMERA
        return InputEmbedder(encoder=atom_encoder(m.atom_attention_encoder))
    depth = len(m.atom_attention_encoder.atom_encoder.diffusion_transformer.layers)
    encoder = atom_encoder(
        m.atom_encoder,
        m.atom_attention_encoder,
        pair_bias=_split_pair_bias(m.atom_enc_proj_z, depth),
    )
    parts = [linear(m.res_type_encoding), linear(m.msa_profile_encoding)]
    for flag, name, kind in [
        ("add_method_conditioning", "method_conditioning_init", embedding),
        ("add_modified_flag", "modified_conditioning_init", embedding),
        ("add_cyclic_flag", "cyclic_conditioning_init", linear),
        ("add_mol_type_feat", "mol_type_conditioning_init", embedding),
    ]:
        if not getattr(m, flag):
            raise NotImplementedError(f"Boltz-2 input embedder without {flag}")
        parts.append(kind(getattr(m, name)))
    return InputEmbedder(
        encoder=encoder,
        linear_tokens=Linear(weight=jnp.concatenate([p.weight for p in parts], axis=1)),
    )
