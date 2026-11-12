"""Convert a whole Boltz-2 model using the block converters in `jfold.convert.boltz`."""

import dataclasses

import jax.numpy as jnp

from jfold.blocks import inputs
from jfold.blocks.denoiser import FourierEmbedding
from jfold.blocks.msa import ResidualMSA
from jfold.convert.boltz import (
    confidence_module,
    diffusion_module,
    distogram_head,
    input_embedder,
    msa_module,
    pairformer_stack,
    relative_positions,
    template_embedder,
)
from jfold.convert.torch import array, embedding, layer_norm, linear
from jfold.model import Recycle, StructureModel, Trunk
from jfold.models import boltz2
from jfold.nn import Linear, SinglesToPair
from jfold.sampling import (
    UpstreamSampler,
    masked_centering,
    no_alignment,
    rigid_align_stop_gradient,
)


def bonds(token_bonds, token_bonds_type) -> Linear:
    """Merge Boltz-2's bond linear and bond-type embedding into one `Linear`."""
    if token_bonds.bias is not None:
        raise ValueError("expected a bias-free token bond linear")
    return Linear(
        weight=jnp.concatenate(
            [array(token_bonds.weight), embedding(token_bonds_type).weight], axis=1
        )
    )


def contact_conditioning(m) -> boltz2.ContactConditioning:
    fourier = m.fourier_embedding.proj
    return boltz2.ContactConditioning(
        fourier=FourierEmbedding(weight=array(fourier.weight[:, 0]), bias=array(fourier.bias)),
        linear=linear(m.encoder),
        unspecified=array(m.encoding_unspecified),
        unselected=array(m.encoding_unselected),
        cutoff_min=float(m.cutoff_min),
        cutoff_max=float(m.cutoff_max),
    )


def pair_inputs(m) -> boltz2.PairInputs:
    """Convert the confidence module's relative position, bond and contact embeddings."""
    return boltz2.PairInputs(
        relative_positions=relative_positions(m.rel_pos),
        bonds=bonds(m.token_bonds, m.token_bonds_type),
        contacts=contact_conditioning(m.contact_conditioning),
    )


def structure_model(m) -> StructureModel:
    """Convert a released Boltz-2 model (`boltz.model.models.boltz2.Boltz2`)."""
    if not (m.bond_type_feature and m.use_templates):
        raise NotImplementedError("Boltz-2 without bond types or templates")
    if not m.confidence_module.add_z_input_to_z:
        raise NotImplementedError("Boltz-2 confidence without its own pair inputs")
    trunk_relative_positions = relative_positions(m.rel_pos)
    return StructureModel(
        input_embedder=input_embedder(m.input_embedder),
        trunk_init=boltz2.TrunkInit(
            init=inputs.TrunkInit(
                linear_s=linear(m.s_init),
                singles_to_pair=SinglesToPair(
                    linear_i=linear(m.z_init_1), linear_j=linear(m.z_init_2)
                ),
                relative_positions=trunk_relative_positions,
                bonds=bonds(m.token_bonds, m.token_bonds_type),
            ),
            contacts=contact_conditioning(m.contact_conditioning),
        ),
        trunk=Trunk(
            recycle_s=Recycle(norm=layer_norm(m.s_norm), linear=linear(m.s_recycle)),
            recycle_z=Recycle(norm=layer_norm(m.z_norm), linear=linear(m.z_recycle)),
            pairformer=pairformer_stack(m.pairformer_module),
            templates=template_embedder(m.template_module),
            msa=ResidualMSA(module=msa_module(m.msa_module, max_rows=msa_max_rows(m.msa_module))),
        ),
        # The diffusion module reuses the trunk's relative position embedding.
        diffusion=dataclasses.replace(
            diffusion_module(m.structure_module.score_model, conditioning=m.diffusion_conditioning),
            relative_positions=trunk_relative_positions,
        ),
        distogram=distogram_head(m.distogram_module),
        confidence=dataclasses.replace(
            confidence_module(m.confidence_module), pair_inputs=pair_inputs(m.confidence_module)
        ),
        sampling=sampling(m),
    )


def msa_max_rows(m) -> int | None:
    """Return the MSA rows Boltz-2 draws per pass, or None for every row."""
    return m.num_subsampled_msa if m.subsample_msa else None


def sampling(model) -> UpstreamSampler:
    """Return Boltz-2's `AtomDiffusion.sample` settings with `boltz predict`'s step
    count."""
    m = model.structure_module
    return UpstreamSampler(
        sigma_data=float(m.sigma_data),
        s_max=float(m.sigma_max),
        s_min=float(m.sigma_min),
        rho=float(m.rho),
        num_steps=int(model.predict_args["sampling_steps"]),
        end="append_zero",
        gamma_0=float(m.gamma_0),
        gamma_min=float(m.gamma_min),
        noise_scale=float(m.noise_scale),
        step_scale=float(m.step_scale),
        center=masked_centering,
        augment=True,
        augment_after_noise=False,
        align=rigid_align_stop_gradient if m.alignment_reverse_diff else no_alignment,
    )
