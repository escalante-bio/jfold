"""Convert a whole ProMERA model using the block converters in `jfold.convert.boltz`."""

import dataclasses

from jfold.blocks.inputs import InputEmbedder, TrunkInit
from jfold.blocks.msa import ResidualMSA
from jfold.convert import boltz
from jfold.convert.torch import layer_norm, linear
from jfold.model import Recycle, StructureModel, Trunk
from jfold.models.promera import AtomAggregate, PromeraInputEmbedder
from jfold.nn import SinglesToPair
from jfold.sampling import UpstreamSampler, masked_centering, rigid_align_stop_gradient


def input_embedder(m) -> PromeraInputEmbedder | InputEmbedder:
    """Convert ProMERA's `InputEmbedder`, whose released atom encoder has no transformer
    layers."""
    encoder = m.atom_attention_encoder
    if encoder.atom_encoder_depth > 0:
        return boltz.input_embedder(m)
    return PromeraInputEmbedder(
        encoder=AtomAggregate(
            embed_features=linear(encoder.embed_atom_features),
            linear_token=linear(encoder.atom_to_token_trans[0]),
        )
    )


def structure_model(m) -> StructureModel:
    """Convert a whole ProMERA model, using its `sm_confidence_module` for confidence."""
    cfg = m.cfg.model
    relative_positions = boltz.relative_positions(m.rel_pos)
    return StructureModel(
        input_embedder=input_embedder(m.input_embedder),
        trunk_init=TrunkInit(
            linear_s=linear(m.s_init),
            singles_to_pair=SinglesToPair(linear_i=linear(m.z_init_1), linear_j=linear(m.z_init_2)),
            relative_positions=relative_positions,
            bonds=linear(m.token_bonds),
        ),
        trunk=Trunk(
            recycle_s=Recycle(norm=layer_norm(m.s_norm), linear=linear(m.s_recycle)),
            recycle_z=Recycle(norm=layer_norm(m.z_norm), linear=linear(m.z_recycle)),
            pairformer=boltz.pairformer_stack(m.pairformer_module),
            msa=ResidualMSA(
                module=boltz.msa_module(
                    m.msa_module,
                    max_rows=cfg.msas_per_trunk_iter if cfg.subsample_msa_per_recycle else None,
                )
            ),
        ),
        # The diffusion module reuses the trunk's relative position embedding.
        diffusion=dataclasses.replace(
            boltz.diffusion_module(m.structure_module.score_model),
            relative_positions=relative_positions,
        ),
        confidence=boltz.confidence_module(m.sm_confidence_module),
        distogram=boltz.distogram_head(m.distogram_module),
        sampling=sampling(),
    )


def sampling(config: dict | None = None) -> UpstreamSampler:
    """Return ProMERA's inference sampler settings from `config`, by default the packaged
    `promera/inference/cofolding.yaml`."""
    if config is None:
        import importlib.resources

        import yaml

        path = importlib.resources.files("promera").joinpath("inference/cofolding.yaml")
        config = yaml.safe_load(path.read_text())
    diffusion = config["diffusion"]
    if not diffusion["edm_churn"]:
        raise NotImplementedError("ProMERA's sampler without EDM churn")
    return UpstreamSampler(
        sigma_data=float(diffusion["sigma_data"]),
        s_max=float(diffusion["sigma_max"]),
        s_min=float(diffusion["sigma_min"]),
        rho=float(diffusion["rho"]),
        num_steps=int(config["diffusion_steps"]),
        end="none",
        gamma_0=float(diffusion["gamma_0"]),
        gamma_min=float(diffusion["gamma_min"]),
        noise_scale=float(diffusion["noise_scale"]),
        step_scale=float(diffusion["step_scale"]),
        center=masked_centering,
        augment=True,
        augment_after_noise=True,
        align=rigid_align_stop_gradient,
    )
