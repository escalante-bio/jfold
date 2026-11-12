"""Convert BioEmu2 (`bioemu2.biomoa`) to the shared blocks plus the parts in
`jfold.models.bioemu2`."""

import dataclasses
import inspect

import jax.numpy as jnp

from jfold.blocks.atoms import AtomAttentionEncoder, AtomPairEmbedding, TrunkToAtoms
from jfold.blocks.confidence import PairHead
from jfold.blocks.denoiser import DiffusionConditioning, FourierEmbedding
from jfold.blocks.diffusion import DiffusionTransformer, DiffusionTransformerLayer
from jfold.blocks.inputs import InputEmbedder, TrunkInit
from jfold.blocks.msa import MSALayer, MSAModule, MSAUpdate, OuterProductMean, PairWeightedAveraging
from jfold.blocks.pairformer import PairformerStack, PairMixerLayer, TriangleMultiplication
from jfold.blocks.windows import CenteredWindows
from jfold.convert.torch import (
    array,
    embedding,
    layer_norm,
    linear,
    localize,
    negated_input,
)
from jfold.model import Recycle, StructureModel, Trunk
from jfold.models.bioemu2 import (
    AtomConditioning,
    ConditionedTrunk,
    ConditionedTrunkInit,
    ContactConditioning,
    DampedHarmonicEmbedding,
    DualStreamCoupling,
    DualStreamModule,
    DualStreamTransformer,
    FrameConditioning,
    LagGate,
    MethodConditioning,
    Readout,
    SingleConditioning,
    ZeroAdaLN,
    ZeroConditionedAttention,
    ZeroConditionedTransition,
)
from jfold.nn import (
    Attention,
    LayerNorm,
    Linear,
    PairBias,
    ReluMLP,
    SinglesToPair,
    SwiGLU,
    Transition,
    stack,
)
from jfold.sampling import UpstreamSampler, masked_centering, rigid_align_stop_gradient


def triangle_multiplication(m) -> TriangleMultiplication:
    """Convert any `TriangleMultiplication*` variant, since they share weights and math."""
    return TriangleMultiplication(
        norm_in=layer_norm(m.norm_in),
        linear_p=linear(m.p_in),
        linear_g=linear(m.g_in),
        norm_out=layer_norm(m.norm_out),
        linear_out=linear(m.p_out),
        linear_out_g=linear(m.g_out),
        outgoing=m.mode == "outgoing",
    )


def transition(m) -> Transition:
    """Convert a `Transition` that has an input norm."""
    return Transition(
        norm=layer_norm(m.norm),
        linear_a=linear(m.fc1),
        linear_b=linear(m.fc2),
        linear_out=linear(m.fc3),
    )


def swiglu(m) -> SwiGLU:
    """Convert a `Transition(use_norm=False)`."""
    return SwiGLU(linear_a=linear(m.fc1), linear_b=linear(m.fc2), linear_out=linear(m.fc3))


def pairmixer_layer(m, *, transition_z="transition_z") -> PairMixerLayer:
    return PairMixerLayer(
        tri_mul_out=triangle_multiplication(m.tri_mul_out),
        tri_mul_in=triangle_multiplication(m.tri_mul_in),
        transition_z=transition(getattr(m, transition_z)),
        single=None,
    )


def pairmixer_stack(m) -> PairformerStack:
    """Convert a `PairMixerModule`, whose PairMixer layers only update the pair."""
    return PairformerStack(layers=stack([pairmixer_layer(layer) for layer in m.layers]))


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
    """Convert any `OuterProductMean*`, whose row count matches the shared module's
    because BioEmu2's MSA mask is uniform along rows."""
    w = array(m.proj_ab.weight)
    return OuterProductMean(
        norm=layer_norm(m.norm),
        linear_a=Linear(weight=w[: m.c_hidden]),
        linear_b=Linear(weight=w[m.c_hidden :]),
        linear_out=linear(m.proj_o),
    )


def msa_layer(m) -> MSALayer:
    """Convert a `PairMixerMSALayer`."""
    return MSALayer(
        outer_product_mean=outer_product_mean(m.outer_product_mean),
        msa=MSAUpdate(
            pair_weighted_averaging=pair_weighted_averaging(m.pair_weighted_averaging),
            transition=transition(m.msa_transition),
        ),
        pair=pairmixer_layer(m, transition_z="z_transition"),
        outer_product_first=False,
    )


def msa_module(m) -> MSAModule:
    """Convert a `PairMixerMSAModule`, which reads every MSA row on every pass."""
    return MSAModule(
        linear_msa=linear(m.msa_proj),
        linear_s=linear(m.s_proj),
        layers=stack([msa_layer(layer) for layer in m.layers]),
        max_rows=None,
    )


def relative_positions(m) -> Linear:
    """Convert a `FusedRelativePositionEncoder`."""
    return Linear(
        weight=jnp.concatenate(
            [
                embedding(m.embed_rel_pos).weight,
                embedding(m.embed_rel_token).weight,
                array(m.embed_same_entity.weight),
                embedding(m.embed_rel_chain).weight,
            ],
            axis=1,
        )
    )


def fourier_embedding(m) -> FourierEmbedding:
    return FourierEmbedding(weight=array(m.proj.weight)[:, 0], bias=array(m.proj.bias))


def contact_conditioning(m) -> ContactConditioning:
    return ContactConditioning(
        fourier=fourier_embedding(m.fourier_embedding),
        linear=linear(m.encoder),
        unspecified=array(m.encoding_unspecified),
        unselected=array(m.encoding_unselected),
        cutoff_min=float(m.cutoff_min),
        cutoff_max=float(m.cutoff_max),
    )


def method_conditioning(m) -> MethodConditioning:
    """Convert the method conditioning on one-hot methods."""
    pair = None
    if m.pair_embedding is not None:
        pair = SinglesToPair(
            linear_i=embedding(m.pair_embedding), linear_j=embedding(m.pair_embedding)
        )
    return MethodConditioning(single=embedding(m.embedding), pair=pair)


def trunk_init(m) -> ConditionedTrunkInit:
    """Convert a `PairMixerSequenceEmbedder`'s initial single and pair."""
    return ConditionedTrunkInit(
        init=TrunkInit(
            linear_s=linear(m.s_init),
            singles_to_pair=SinglesToPair(linear_i=linear(m.z_init_1), linear_j=linear(m.z_init_2)),
            relative_positions=relative_positions(m.rel_pos),
            bonds=linear(m.token_bonds),
        ),
        method=method_conditioning(m.method_conditioning) if m.use_method_conditioning else None,
        contacts=(
            contact_conditioning(m.contact_conditioning) if m.use_contact_conditioning else None
        ),
    )


def trunk(m) -> Trunk:
    """Convert one recycling iteration of a `PairMixerSequenceEmbedder`."""
    return Trunk(
        recycle_s=Recycle(norm=layer_norm(m.s_norm), linear=linear(m.s_recycle)),
        recycle_z=Recycle(norm=layer_norm(m.z_norm), linear=linear(m.z_recycle)),
        pairformer=pairmixer_stack(m.pairmixer_module),
        msa=msa_module(m.msa_module),
    )


def distogram(m) -> PairHead:
    """Convert a `DistogramModule`, which symmetrizes before its projection."""
    return PairHead(linear=linear(m.distogram), symmetric=True)


def zero_adaln(cond, rows: slice, width: int, lns_scale: float = 1.0) -> ZeroAdaLN:
    """Convert adaLN-Zero from the output `rows` of a
    `Sequential(LayerNorm, SiLU, Linear)`, folding upstream's `lns (norm(a) (1 + α) + β)`
    into the scale and shift."""
    w, b = array(cond[2].weight)[rows], array(cond[2].bias)[rows]
    scale = jnp.concatenate([jnp.full(width, lns_scale), jnp.full(width, lns_scale)])
    offset = jnp.concatenate([jnp.full(width, lns_scale - 1.0), jnp.zeros(width)])
    w = w.at[: 2 * width].multiply(scale[:, None])
    b = b.at[: 2 * width].set(b[: 2 * width] * scale + offset)
    return ZeroAdaLN(
        norm_a=LayerNorm(weight=None, bias=None),
        norm_s=layer_norm(cond[0]),
        linear=Linear(weight=w, bias=b),
    )


def _pair_bias_attention(m):
    m = getattr(m, "attention_layer", m)  # SlidingWindowAttentionLayer
    return m.pair_bias_attn


def pair_bias_attention(m) -> Attention:
    """Convert an `AttentionPairBias` without `initial_norm`."""
    if m.initial_norm:
        raise ValueError("expected AttentionPairBias without initial_norm")
    w, c = array(m.proj_qkv.weight), m.c_s
    return Attention(
        linear_q=Linear(weight=w[:c], bias=array(m.proj_q_bias)),
        linear_k=Linear(weight=w[c : 2 * c]),
        linear_v=Linear(weight=w[2 * c :]),
        linear_g=linear(m.proj_g),
        linear_o=linear(m.proj_o),
        num_heads=m.num_heads,
    )


def conditioned_layer(m) -> DiffusionTransformerLayer:
    """Convert a `ConditionedTransformerLayer`, unwrapping a sliding-window wrapper."""
    m = getattr(m, "transformer_layer", m)
    width = m.transition.fc1.weight.shape[1]
    lns = float(m.lns_scale)
    return DiffusionTransformerLayer(
        attention=ZeroConditionedAttention(
            adaln=zero_adaln(m.cond, slice(0, 3 * width), width, lns),
            attention=pair_bias_attention(_pair_bias_attention(m.attention)),
        ),
        transition=ZeroConditionedTransition(
            adaln=zero_adaln(m.cond, slice(3 * width, 6 * width), width, lns),
            mlp=swiglu(m.transition),
        ),
    )


def conditioned_transformer(layers, windows: CenteredWindows | None = None) -> DiffusionTransformer:
    """Convert stacked `ConditionedTransformerLayer`s, within `windows` if given."""
    layers = [getattr(layer, "transformer_layer", layer) for layer in layers]
    attns = [_pair_bias_attention(layer.attention) for layer in layers]
    transformer = DiffusionTransformer(
        pair_bias=stack(
            [PairBias(norm=layer_norm(a.norm_z), linear=linear(a.proj_z)) for a in attns]
        ),
        layers=stack([conditioned_layer(layer) for layer in layers]),
    )
    return transformer if windows is None else localize(transformer, windows)


def _windows(m) -> CenteredWindows:
    return CenteredWindows(queries=m.atoms_per_window_queries, keys=m.atoms_per_window_keys)


def _embed_features(m) -> Linear:
    return Linear(
        weight=jnp.concatenate(
            [array(m.proj_atom_continuous.weight), embedding(m.embed_element).weight]
            + [embedding(e).weight for e in m.embed_atom_name_chars],
            axis=1,
        )
    )


def _pair_embedding(m) -> AtomPairEmbedding:
    # Upstream's offset is key minus query, while jfold's is query minus key.
    return AtomPairEmbedding(
        linear_offset=negated_input(m.embed_atompair_ref_pos),
        linear_inv_sq_dist=linear(m.embed_atompair_ref_dist),
        linear_valid=linear(m.embed_atompair_mask),
    )


def _pair_mlp(m) -> ReluMLP:
    return ReluMLP(layers=tuple(linear(m.p_mlp[i]) for i in (1, 3, 5)))


def atom_encoder(m) -> AtomAttentionEncoder:
    """Convert the input embedder's `AtomAttentionEncoder`, which has no coordinates."""
    if m.structure_prediction:
        raise ValueError("use atom_conditioning for the structure module's atom encoder")
    return AtomAttentionEncoder(
        embed_features=_embed_features(m),
        pair_embedding=_pair_embedding(m),
        trunk=None,
        linear_r=None,
        linear_c_queries=linear(m.c_to_p_trans_q[1]),
        linear_c_keys=linear(m.c_to_p_trans_k[1]),
        pair_mlp=_pair_mlp(m),
        transformer=conditioned_transformer(m.atom_encoder_layers, _windows(m)),
        linear_token=linear(m.atom_to_token_trans[0]),
    )


def input_embedder(m) -> InputEmbedder:
    return InputEmbedder(encoder=atom_encoder(m.atom_attention_encoder))


def atom_conditioning(m) -> AtomConditioning:
    """Convert the structure module's `AtomAttentionEncoder`, which reads coordinates."""
    frame = None
    if m.use_dynamics_conditioning:
        gate = None
        if m.downscale_condition_contrib:
            gate = LagGate(taus=array(m.dt_gate_embed.taus), log_weight=array(m.dt_gate_log_w))
        frame = FrameConditioning(linear=linear(m.cond_to_q_trans), gate=gate)
    return AtomConditioning(
        embed_features=_embed_features(m),
        pair_embedding=_pair_embedding(m),
        trunk=TrunkToAtoms(
            norm_s=layer_norm(m.s_to_c_trans[0]),
            linear_s=linear(m.s_to_c_trans[1]),
            norm_z=layer_norm(m.z_to_p_trans[0]),
            linear_z=linear(m.z_to_p_trans[1]),
        ),
        linear_c_queries=linear(m.c_to_p_trans_q[1]),
        linear_c_keys=linear(m.c_to_p_trans_k[1]),
        pair_mlp=_pair_mlp(m),
        linear_r=linear(m.r_to_q_trans),
        frame=frame,
        windows=_windows(m),
    )


def single_conditioning(single, pairwise, sigma_data: float) -> SingleConditioning:
    """Convert `SingleConditioning` and `PairwiseConditioning`."""
    conditioning = DiffusionConditioning(
        norm_z=layer_norm(pairwise.dim_pairwise_init_proj[0]),
        linear_z=linear(pairwise.dim_pairwise_init_proj[1]),
        transitions_z=tuple(transition(t) for t in pairwise.transitions),
        norm_s=layer_norm(single.norm_single),
        linear_s=linear(single.single_embed),
        fourier=fourier_embedding(single.fourier_embed),
        norm_noise=layer_norm(single.norm_fourier),
        linear_noise=linear(single.fourier_to_single),
        transitions_s=tuple(transition(t) for t in single.transitions),
        sigma_data=float(sigma_data),
    )
    if not single.use_dt_conditioning:
        return SingleConditioning(conditioning=conditioning)
    return SingleConditioning(
        conditioning=conditioning,
        lag_embedding=DampedHarmonicEmbedding(
            taus=array(single.dt_embed.taus),
            quality_factors=array(single.dt_embed.quality_factors),
        ),
        linear_lag=linear(single.dt_embed_to_single),
    )


def coupling(m) -> DualStreamCoupling:
    lns = float(m.lns_scale)

    def mixer(cond, mlp):
        width = mlp.fc1.weight.shape[1]
        return ZeroConditionedTransition(
            adaln=zero_adaln(cond, slice(None), width, lns), mlp=swiglu(mlp)
        )

    return DualStreamCoupling(
        to_atoms=mixer(m.q_coupling_cond, m.s_to_q_mixer),
        to_tokens=mixer(m.s_coupling_cond, m.q_to_s_mixer),
    )


def readout(m) -> Readout:
    return Readout(
        linear_s=linear(m.s_to_hidden),
        linear_q=linear(m.q_to_hidden),
        norm=layer_norm(m.to_out[0]),
        linear_out=linear(m.to_out[1]),
    )


def dual_stream_transformer(m) -> DualStreamTransformer:
    """Convert the `DualStreamDiffusionLayer`s of a `DualStreamDiffusionModule`."""
    return DualStreamTransformer(
        atoms=conditioned_transformer([layer.atom_block for layer in m.layers], _windows(m)),
        tokens=conditioned_transformer([layer.token_block for layer in m.layers]),
        coupling=stack([coupling(layer.coupling) for layer in m.layers]),
    )


def dual_stream_module(m) -> DualStreamModule:
    """Convert a `DualStreamDiffusionModule`."""
    return DualStreamModule(
        conditioning=single_conditioning(
            m.single_conditioner, m.pairwise_conditioner, m.sigma_data
        ),
        atoms=atom_conditioning(m.atom_feature_embedder),
        norm_s=layer_norm(m.s_trunk_to_a[0]),
        linear_s=linear(m.s_trunk_to_a[1]),
        linear_r=linear(m.r_token_to_a),
        transformer=dual_stream_transformer(m),
        readout=readout(m.readout),
    )


def bioemu2(m) -> StructureModel:
    """Convert a whole `BioMoaCModel`, which has no confidence head."""
    embedder = m.sequence_embedder
    if m.edm_schedule.sigma_data != m.structure_module.sigma_data:
        raise ValueError("the EDM schedule and the structure module disagree on sigma_data")
    return StructureModel(
        input_embedder=input_embedder(m.input_embedder),
        trunk_init=trunk_init(embedder),
        trunk=ConditionedTrunk(
            trunk=trunk(embedder),
            unconditional=(
                transition(embedder.uncond_transition)
                if embedder.use_unconditional_branch
                else None
            ),
        ),
        # The structure module embeds the relative position features with its own linear.
        diffusion=dataclasses.replace(
            dual_stream_module(m.structure_module), relative_positions=relative_positions(m.rel_pos)
        ),
        distogram=distogram(m.distogram_module),
        sampling=sampling(m),
    )


def sampling(m, mode: str = "structure") -> UpstreamSampler:
    """Return the sampler settings of upstream's `edm` preset for `mode`, `"structure"`
    or `"equilibrium"`."""
    from bioemu2.denoising.edm import edm_sampler
    from bioemu2.denoising.samplers import EDMSampler

    defaults = {k: p.default for k, p in inspect.signature(edm_sampler).parameters.items()}
    if not defaults["alignment_reverse_diff"]:
        raise NotImplementedError("BioEmu2's sampler without alignment")
    preset = EDMSampler(step_scale={"structure": 1.5, "equilibrium": 1.0}[mode])
    schedule = m.edm_schedule
    return UpstreamSampler(
        sigma_data=float(schedule.sigma_data),
        s_max=float(schedule.sigma_max),
        s_min=float(schedule.sigma_min),
        rho=float(defaults["rho"]),
        num_steps=int(preset.num_steps),
        end="append_zero",
        gamma_0=float(defaults["gamma_0"]),
        gamma_min=float(defaults["gamma_min"]),
        noise_scale=float(preset.noise_scale),
        step_scale=float(preset.step_scale),
        center=masked_centering,
        augment=True,
        augment_after_noise=False,
        align=rigid_align_stop_gradient,
    )
