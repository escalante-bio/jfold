"""Convert Protenix-lineage models: Protenix, OpenDDE, Mol-1 and Mol-1-constraint."""

import dataclasses

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
    RenormKeys,
)
from jfold.blocks.inputs import InputEmbedder, TrunkInit
from jfold.blocks.msa import (
    DEFAULT_MAX_ROWS,
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
    PairMixerLayer,
    SingleUpdate,
    TriangleAttention,
    TriangleMultiplication,
)
from jfold.blocks.templates import TemplateEmbedder
from jfold.blocks.windows import CenteredWindows
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
from jfold.models.mol1 import ConstraintTrunkInit, Mol1Pairformer
from jfold.nn import Linear, PairBias, ReluMLP, SinglesToPair, Transition, stack
from jfold.sampling import UpstreamSampler, masked_centering, no_alignment


def triangle_multiplication(m) -> TriangleMultiplication:
    return TriangleMultiplication(
        norm_in=layer_norm(m.layer_norm_in),
        linear_p=concat_outputs(m.linear_a_p, m.linear_b_p),
        linear_g=concat_outputs(m.linear_a_g, m.linear_b_g),
        norm_out=layer_norm(m.layer_norm_out),
        linear_out=linear(m.linear_z),
        linear_out_g=linear(m.linear_g),
        outgoing=bool(m._outgoing),
    )


def triangle_attention(m) -> TriangleAttention:
    # Both modules are starting-node ones, and the block transposes the pair for the other.
    if not m.starting:
        raise ValueError("expected a starting-node TriangleAttention")
    return TriangleAttention(
        norm=layer_norm(m.layer_norm), linear_bias=linear(m.linear), attention=attention(m.mha)
    )


def transition(m) -> Transition:
    return Transition(
        norm=layer_norm(m.layernorm1),
        linear_a=linear(m.linear_no_bias_a),
        linear_b=linear(m.linear_no_bias_b),
        linear_out=linear(m.linear_no_bias),
    )


def attention_pair_bias(m) -> AttentionPairBias:
    return AttentionPairBias(
        norm_s=layer_norm(m.layernorm_a),
        pair_bias=PairBias(norm=layer_norm(m.layernorm_z), linear=linear(m.linear_nobias_z)),
        attention=attention(m.attention),
    )


def single_update(m) -> SingleUpdate | None:
    """Convert a block's single update, or return None if the block has no single track."""
    attn = getattr(m, "attention_pair_bias", None)
    if attn is None:
        return None
    return SingleUpdate(
        attention=attention_pair_bias(attn), transition=transition(m.single_transition)
    )


def pairformer_layer(m) -> PairformerLayer:
    return PairformerLayer(
        tri_mul_out=triangle_multiplication(m.tri_mul_out),
        tri_mul_in=triangle_multiplication(m.tri_mul_in),
        tri_att_start=triangle_attention(m.tri_att_start),
        tri_att_end=triangle_attention(m.tri_att_end),
        transition_z=transition(m.pair_transition),
        single=single_update(m),
        dropout_row=dropout(_rate(m), OPENFOLD_ROW),
        dropout_column=dropout(_rate(m), OPENFOLD_COLUMN),
    )


def _rate(m) -> float:
    return m.dropout_row.r if hasattr(m, "dropout_row") else 0.0


def pairmixer_layer(m) -> PairMixerLayer:
    """Convert Mol-1's `PairMixerBlock`."""
    return PairMixerLayer(
        tri_mul_out=triangle_multiplication(m.tri_mul_out),
        tri_mul_in=triangle_multiplication(m.tri_mul_in),
        transition_z=transition(m.pair_transition),
        single=single_update(m),
        dropout_row=dropout(_rate(m), OPENFOLD_ROW),
    )


def pairformer_stack(m) -> PairformerStack:
    """Convert a Protenix, OpenDDE or Mol-1 `PairformerStack` without PairMixer blocks."""
    if getattr(m, "n_pairmixer_blocks", 0) or len(getattr(m, "adapters", ())):
        raise ValueError("Mol-1's trunk stack converts with mol1_pairformer")
    return PairformerStack(layers=stack([pairformer_layer(b) for b in m.blocks]))


def mol1_pairformer(m) -> Mol1Pairformer:
    """Convert Mol-1's trunk stack of PairMixer and then adapter-wrapped Pairformer
    blocks."""
    n_mix = m.n_pairmixer_blocks
    if not 0 < n_mix < len(m.blocks) or set(m.adapters) != {
        f"before_block_{n_mix}",
        "after_blocks",
    }:
        raise ValueError("expected PairMixer blocks, then Pairformer blocks between two adapters")
    return Mol1Pairformer(
        pairmixer=PairformerStack(stack([pairmixer_layer(b) for b in m.blocks[:n_mix]])),
        adapter_in=linear(m.adapters[f"before_block_{n_mix}"].pair_adapter),
        pairformer=PairformerStack(stack([pairformer_layer(b) for b in m.blocks[n_mix:]])),
        adapter_out=linear(m.adapters["after_blocks"].pair_adapter),
    )


def adaln(m) -> AdaLN:
    return AdaLN(
        norm_a=layer_norm(m.layernorm_a),
        norm_s=layer_norm(m.layernorm_s),
        linear_scale=linear(m.linear_s),
        linear_shift=linear(m.linear_nobias_s),
    )


def conditioned_transition(m) -> ConditionedTransition:
    return ConditionedTransition(
        adaln=adaln(m.adaln),
        linear_a=linear(m.linear_nobias_a1),
        linear_b=linear(m.linear_nobias_a2),
        linear_c=None,
        linear_out=linear(m.linear_nobias_b),
        linear_gate=linear(m.linear_s),
    )


def diffusion_transformer_layer(m) -> DiffusionTransformerLayer:
    attn = m.attention_pair_bias
    return DiffusionTransformerLayer(
        attention=ConditionedAttention(
            adaln=adaln(attn.layernorm_a),
            attention=attention(attn.attention),
            linear_gate=linear(attn.linear_a_last),
            keys=RenormKeys(adaln(attn.layernorm_kv)) if attn.cross_attention_mode else None,
        ),
        transition=conditioned_transition(m.conditioned_transition_block),
    )


def diffusion_transformer(m) -> DiffusionTransformer:
    """Convert a `DiffusionTransformer` of Protenix, OpenDDE or Mol-1."""
    return DiffusionTransformer(
        pair_bias=stack(
            [
                PairBias(
                    norm=layer_norm(b.attention_pair_bias.layernorm_z),
                    linear=linear(b.attention_pair_bias.linear_nobias_z),
                )
                for b in m.blocks
            ]
        ),
        layers=stack([diffusion_transformer_layer(b) for b in m.blocks]),
    )


def atom_transformer(m) -> DiffusionTransformer:
    """Convert an `AtomTransformer` of Protenix, OpenDDE or Mol-1."""
    windows = CenteredWindows(queries=m.n_queries, keys=m.n_keys)
    return localize(diffusion_transformer(m.diffusion_transformer), windows)


def atom_encoder(m) -> AtomAttentionEncoder:
    """Convert an `AtomAttentionEncoder` of Protenix, OpenDDE or Mol-1."""
    trunk = None
    if m.has_coords:
        trunk = TrunkToAtoms(
            norm_s=layer_norm(m.layernorm_s),
            linear_s=linear(m.linear_no_bias_s),
            norm_z=layer_norm(m.layernorm_z),
            linear_z=linear(m.linear_no_bias_z),
        )
    return AtomAttentionEncoder(
        embed_features=concat_inputs(
            m.linear_no_bias_ref_pos, m.linear_no_bias_ref_charge, m.linear_no_bias_f
        ),
        pair_embedding=AtomPairEmbedding(
            linear_offset=linear(m.linear_no_bias_d),
            linear_inv_sq_dist=linear(m.linear_no_bias_invd),
            linear_valid=linear(m.linear_no_bias_v),
        ),
        trunk=trunk,
        linear_r=linear(m.linear_no_bias_r) if m.has_coords else None,
        linear_c_queries=linear(m.linear_no_bias_cl),
        linear_c_keys=linear(m.linear_no_bias_cm),
        pair_mlp=ReluMLP(layers=tuple(linear(m.small_mlp[i]) for i in (1, 3, 5))),
        transformer=atom_transformer(m.atom_transformer),
        linear_token=linear(m.linear_no_bias_q),
        queries_from_conditioned=True,
    )


def atom_decoder(m) -> AtomAttentionDecoder:
    return AtomAttentionDecoder(
        linear_a=linear(m.linear_no_bias_a),
        transformer=atom_transformer(m.atom_transformer),
        norm_out=layer_norm(m.layernorm_q),
        linear_out=linear(m.linear_no_bias_out),
    )


def diffusion_conditioning(m) -> DiffusionConditioning:
    return DiffusionConditioning(
        norm_z=layer_norm(m.layernorm_z),
        linear_z=linear(m.linear_no_bias_z),
        transitions_z=(transition(m.transition_z1), transition(m.transition_z2)),
        norm_s=layer_norm(m.layernorm_s),
        linear_s=linear(m.linear_no_bias_s),
        fourier=FourierEmbedding(
            weight=array(m.fourier_embedding.w), bias=array(m.fourier_embedding.b)
        ),
        norm_noise=layer_norm(m.layernorm_n),
        linear_noise=linear(m.linear_no_bias_n),
        transitions_s=(transition(m.transition_s1), transition(m.transition_s2)),
        sigma_data=float(m.sigma_data),
        **(
            {
                "norm_z_trunk": layer_norm(m.layernorm_z_trunk),
                "linear_z_trunk": linear(m.linear_no_bias_z_trunk),
            }
            if getattr(m, "compress_pair_z", False)
            else {}
        ),
    )


def diffusion_module(m) -> DiffusionModule:
    """Convert a `DiffusionModule` of Protenix, OpenDDE or Mol-1."""
    return DiffusionModule(
        conditioning=diffusion_conditioning(m.diffusion_conditioning),
        encoder=atom_encoder(m.atom_attention_encoder),
        norm_s=layer_norm(m.layernorm_s),
        linear_s=linear(m.linear_no_bias_s),
        transformer=diffusion_transformer(m.diffusion_transformer),
        norm_a=layer_norm(m.layernorm_a),
        decoder=atom_decoder(m.atom_attention_decoder),
    )


def pair_weighted_averaging(m) -> PairWeightedAveraging:
    return PairWeightedAveraging(
        norm_m=layer_norm(m.layernorm_m),
        linear_v=linear(m.linear_no_bias_mv),
        norm_z=layer_norm(m.layernorm_z),
        linear_b=linear(m.linear_no_bias_z),
        linear_g=linear(m.linear_no_bias_mg),
        linear_o=linear(m.linear_no_bias_out),
        num_heads=m.n_heads,
    )


def outer_product_mean(m) -> ProjectedOuterProductMean:
    return ProjectedOuterProductMean(
        norm=layer_norm(m.layer_norm),
        linear_a=linear(m.linear_1),
        linear_b=linear(m.linear_2),
        linear_out=linear(m.linear_out),
        eps=float(m.eps),
    )


def msa_layer(m, *, outer_product_first: bool) -> MSALayer:
    """Convert an `MSABlock`. In Protenix and Mol-1 the last block has no MSA update."""
    msa = None
    if hasattr(m, "msa_stack"):
        msa = MSAUpdate(
            pair_weighted_averaging=pair_weighted_averaging(
                m.msa_stack.msa_pair_weighted_averaging
            ),
            transition=transition(m.msa_stack.transition_m),
            dropout=dropout(_rate(m.msa_stack), OPENFOLD_ROW),
        )
    return MSALayer(
        outer_product_mean=outer_product_mean(m.outer_product_mean_msa),
        msa=msa,
        pair=pairformer_layer(m.pair_stack),
        outer_product_first=outer_product_first,
    )


def msa_module(m, *, outer_product_first: bool = True) -> MSAModule:
    """Convert an `MSAModule` of Protenix, OpenDDE or Mol-1."""
    layers = [msa_layer(b, outer_product_first=outer_product_first) for b in m.blocks]
    last = layers[-1] if layers[-1].msa is None else None
    regular = layers[:-1] if last is not None else layers
    return MSAModule(
        linear_msa=linear(m.linear_no_bias_m),
        linear_s=linear(m.linear_no_bias_s),
        layers=stack(regular) if regular else None,
        last=last,
        keep_query=False,
        **msa_rows(m),
    )


def msa_rows(m) -> dict[str, int | None]:
    """Return `max_rows` and `min_rows` matching upstream MSA sampling at inference.

    Protenix and Mol-1 draw a random number of rows. A pass reproduces this by masking a
    draw of at most `DEFAULT_MAX_ROWS` rows, since upstream's cap of up to 16384 would
    make every pass compute on every row."""
    if hasattr(m, "msa_depth"):
        return {"max_rows": m.msa_depth, "min_rows": None}
    cfg = m.msa_configs
    if cfg["strategy"] != "random":
        raise NotImplementedError(f"MSA sampling strategy {cfg['strategy']!r}")
    cutoff = cfg["test_cutoff"]
    max_rows = DEFAULT_MAX_ROWS if cutoff <= 0 else min(cutoff, DEFAULT_MAX_ROWS)
    return {"max_rows": max_rows, "min_rows": cfg["test_lowerb"]}


def template_embedder(m) -> TemplateEmbedder:
    """Convert a `TemplateEmbedder` of Protenix, OpenDDE or Mol-1."""
    return TemplateEmbedder(
        norm_z=layer_norm(m.layernorm_z),
        linear_z=linear(m.linear_no_bias_z),
        linear_features=linear(m.linear_no_bias_a),
        stack=pairformer_stack(m.pairformer_stack),
        norm_v=layer_norm(m.layernorm_v),
        linear_u=linear(m.linear_no_bias_u),
    )


def confidence_module(m) -> ConfidenceModule:
    """Convert a `ConfidenceHead` of Protenix, OpenDDE or Mol-1."""
    return ConfidenceModule(
        singles_to_pair=SinglesToPair(
            linear_i=linear(m.linear_no_bias_s2), linear_j=linear(m.linear_no_bias_s1)
        ),
        distances=DistanceEmbedding(
            edges=array(m.lower_bins),
            linear=below_first_bin(m.linear_no_bias_d),
            linear_raw=linear(m.linear_no_bias_d_wo_onehot),
        ),
        stack=pairformer_stack(m.pairformer_stack),
        pae=PairHead(linear=linear(m.linear_no_bias_pae), norm=layer_norm(m.pae_ln)),
        pde=PairHead(
            linear=linear(m.linear_no_bias_pde), norm=layer_norm(m.pde_ln), symmetric=True
        ),
        plddt=AtomHead(weight=array(m.plddt_weight), norm=layer_norm(m.plddt_ln)),
        resolved=AtomHead(weight=array(m.resolved_weight), norm=layer_norm(m.resolved_ln)),
        norm_s=layer_norm(m.input_strunk_ln),
        clamp_s=512.0,
    )


def distogram_head(m) -> PairHead:
    """Convert a `DistogramHead`, which symmetrizes after its projection."""
    return PairHead(linear=doubled_bias(m.linear), symmetric=True)


def relative_positions(m) -> Linear:
    """Convert the relative position encoding's projection."""
    return linear(m.linear_no_bias)


def input_embedder(m) -> InputEmbedder:
    """Convert an `InputFeatureEmbedder` of Protenix, OpenDDE or Mol-1."""
    return InputEmbedder(encoder=atom_encoder(m.atom_attention_encoder))


def mol1_constraints(m) -> Linear:
    """Convert Mol-1-constraint's `constraint_conditioner` to one bias-free linear on
    `jfold.models.mol1.constraint_features`."""
    if not (hasattr(m, "exclusion") and hasattr(m, "contact")):
        raise ValueError("Mol-1-constraint converts with its EXCLUSION and CONTACT encoders")

    def projected(encoder, vectors):
        return array(encoder.output_projection.weight) @ array(vectors).T  # [Cz, n]

    return Linear(
        weight=jnp.concatenate(
            [
                projected(m, m.pair_orientation_embedding),
                projected(m.exclusion, m.exclusion.pair_orientation_embedding),
                projected(m.contact, m.contact.contact_embedding[None]),
            ],
            axis=1,
        )
    )


def structure_model(m, *, outer_product_first: bool = True) -> StructureModel:
    """Convert a whole Protenix or Mol-1 model, or OpenDDE's residue-level part with
    `outer_product_first=False`."""
    templates = m.template_embedder if m.template_embedder.n_blocks > 0 else None
    mol1 = getattr(m.pairformer_stack, "n_pairmixer_blocks", 0) > 0
    trunk_init = TrunkInit(
        linear_s=linear(m.linear_no_bias_sinit),
        singles_to_pair=SinglesToPair(
            linear_i=linear(m.linear_no_bias_zinit1), linear_j=linear(m.linear_no_bias_zinit2)
        ),
        relative_positions=relative_positions(m.relative_position_encoding),
        bonds=linear(m.linear_no_bias_token_bond),
        pair_from_s_init=True,
    )
    if hasattr(m, "constraint_conditioner"):
        trunk_init = ConstraintTrunkInit(
            init=trunk_init, constraints=mol1_constraints(m.constraint_conditioner)
        )
    return StructureModel(
        input_embedder=input_embedder(m.input_embedder),
        trunk_init=trunk_init,
        trunk=Trunk(
            recycle_s=Recycle(norm=layer_norm(m.layernorm_s), linear=linear(m.linear_no_bias_s)),
            recycle_z=Recycle(
                norm=layer_norm(m.layernorm_z_cycle), linear=linear(m.linear_no_bias_z_cycle)
            ),
            pairformer=(mol1_pairformer if mol1 else pairformer_stack)(m.pairformer_stack),
            templates=None if templates is None else template_embedder(templates),
            msa=msa_module(m.msa_module, outer_product_first=outer_product_first),
        ),
        diffusion=dataclasses.replace(
            diffusion_module(m.diffusion_module),
            relative_positions=relative_positions(m.diffusion_module.diffusion_conditioning.relpe),
        ),
        confidence=confidence_module(m.confidence_head),
        distogram=distogram_head(m.distogram_head),
        sampling=sampling(m),
    )


def sampling(m) -> UpstreamSampler:
    """Return the sampler settings of upstream `sample_diffusion`."""
    scheduler, cfg = m.inference_noise_scheduler, m.configs.sample_diffusion
    return UpstreamSampler(
        sigma_data=float(scheduler.sigma_data),
        s_max=float(scheduler.s_max),
        s_min=float(scheduler.s_min),
        rho=float(scheduler.rho),
        num_steps=int(cfg["N_step"]),
        end="zero_last",
        gamma_0=float(cfg["gamma0"]),
        gamma_min=float(cfg["gamma_min"]),
        noise_scale=float(cfg["noise_scale_lambda"]),
        step_scale=float(cfg["step_scale_eta"]),
        center=masked_centering,
        augment=True,
        augment_after_noise=False,
        align=no_alignment,
    )
