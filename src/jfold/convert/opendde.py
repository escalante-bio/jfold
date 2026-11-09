"""The whole OpenDDE model (`opendde.model.opendde.OpenDDE`). Its blocks convert with
`jfold.convert.protenix`."""

import dataclasses

import jax.numpy as jnp

from jfold.convert import protenix
from jfold.convert.torch import array, layer_norm, linear
from jfold.model import StructureModel
from jfold.models.opendde import PairEmbedding, StructuralDiffusion, StructuralTokenExpander


def structural_token_expander(m) -> StructuralTokenExpander:
    """`StructuralTokenExpander` with its full per-role-pair projection."""
    if m.pair_projection_mode != "full":
        raise ValueError(f"expected pair_projection_mode='full', got {m.pair_projection_mode!r}")
    norm, hidden, _, out = m.single_split_mlp
    n, c_z = m.n_roles, m.c_z
    flags = (
        m.same_parent_embedding,
        m.same_residue_twin_embedding,
        m.prev_bb_chain_embedding,
        m.next_bb_chain_embedding,
    )
    bias_flags = jnp.stack(
        [
            array(w)
            for w in (
                m.attn_bias_same_parent,
                m.attn_bias_same_residue_twin,
                m.attn_bias_prev_bb_chain,
                m.attn_bias_next_bb_chain,
            )
        ]
    )
    return StructuralTokenExpander(
        single_inputs_role=array(m.single_input_role_embedding.weight),
        single_role=array(m.single_role_embedding.weight),
        single_norm=layer_norm(norm),
        single_hidden=linear(hidden),
        single_out=linear(out),
        pair_projection=jnp.stack([array(p.weight) for p in m.pair_block_proj]).reshape(
            n, n, c_z, c_z
        ),
        pair_embedding=PairEmbedding(
            flags=jnp.stack([array(e.weight) for e in flags]),
            role_pair=array(m.role_pair_type_embedding.weight),
        ),
        attention_bias=PairEmbedding(
            flags=jnp.stack([jnp.zeros_like(bias_flags), bias_flags], 1)[..., None],
            role_pair=array(m.attn_bias_role_pair_type)[:, None],
        ),
    )


def structure_model(m) -> StructureModel:
    """A whole OpenDDE model: Protenix's, its diffusion run on structural tokens
    (`StructuralDiffusion`), its distogram and confidence heads reading the residue trunk
    (`pair_output_space='residue'`). Inputs come from `jfold.models.opendde.features`."""
    if not m.enable_structural_token_expansion:
        raise ValueError("expected structural-token expansion")
    if m.pair_output_space != "residue":
        raise ValueError(f"expected pair_output_space='residue', got {m.pair_output_space!r}")
    model = protenix.structure_model(m, outer_product_first=False)
    refiner = m.structural_token_refiner if m.enable_structural_token_refiner else None
    diffusion = StructuralDiffusion(
        expander=structural_token_expander(m.structural_token_expander),
        diffusion=model.diffusion,
        refiner=None if refiner is None else protenix.pairformer_stack(refiner),
    )
    return dataclasses.replace(model, diffusion=diffusion, sampling=protenix.sampling(m))
