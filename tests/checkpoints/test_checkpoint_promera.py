"""End-to-end parity with the released ProMERA checkpoint: the trunk, one denoising step,
confidence and distogram on a small two-chain complex featurized by tinyprot. Downloads
weights; run with `pytest -m checkpoint`. ProMERA's triangle updates exist only as
cuequivariance CUDA kernels, so the reference runs on the GPU."""

import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from reference import STEPS, assert_round_trip, assert_schedule, rel_err

pytestmark = pytest.mark.checkpoint

SEQUENCES = ("MKTAYIAKQRQISFVKSHFSRQ", "GSHMLEDPV")
NUM_RECYCLES = 2
NOISE_LEVEL = 10.0
KEY = jax.random.key(0)


def _trunk(model, features):
    inputs = model.embed(features, key=KEY)
    state = model.init_state(inputs)
    for _ in range(NUM_RECYCLES):
        state = model.step(features, inputs, state, key=KEY)
    return inputs, state


def _reference():
    """The released model, as `jfold.pretrained` builds it (from the Hugging Face cache)."""
    from huggingface_hub import hf_hub_download

    from jfold import pretrained
    from jfold.pretrained.recipes.promera import FILENAME, REPO

    return pretrained.upstream("promera", checkpoint_path=hf_hub_download(REPO, FILENAME))


def _batch():
    """tinyprot features of the complex (single-sequence MSAs), finalized and collated as
    ProMERA's inference does: a batch of one, atoms padded to a multiple of 32."""
    from promera.data.utils import collate
    from promera.inference.utils import finalize_feats
    from tinyprot import init
    from tinyprot.feature import AF3Featurizer
    from tinyprot.msa import construct_paired_msa, make_dummy_msa
    from tinyprot.structure import Structure

    if not all(os.path.exists(Path(p) / "data.mdb") for p in (init._ccd_path, init._tax_path)):
        init.download_pretrained()
    struct = Structure.from_schema(
        {chain: {"type": "protein", "sequence": s} for chain, s in zip("AB", SEQUENCES)}
    )
    msas = {chain: make_dummy_msa(c.seq) for chain, c in struct.chains.items()}
    feats = AF3Featurizer(struct, msas, construct_paired_msa(msas)).featurize()
    return collate([finalize_feats(feats, struct, "complex", 0)])


def test_promera():
    from jfold.convert import promera as convert
    from jfold.model import TrunkState
    from jfold.models import promera as promera_features

    reference = _reference()
    batch = _batch()
    num_atoms = int(batch["atom_pad_mask"].sum())
    padded_atoms = batch["ref_pos"].shape[1]
    assert padded_atoms > num_atoms  # padding is exercised
    x_noisy = 16 * np.random.default_rng(0).standard_normal((padded_atoms, 3), dtype=np.float32)

    reference.cuda()
    cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    with torch.no_grad():
        aggregate = reference.input_embedder.atom_attention_encoder(cuda)[0]
        out = reference.pairformer_forward(cuda, recycling_steps=NUM_RECYCLES)
        s_inputs, s, z = out["s_inputs"], out["s"], out["z"]
        sigma = torch.tensor([NOISE_LEVEL], device="cuda")
        denoised = reference.structure_module.preconditioned_network_forward(
            torch.from_numpy(x_noisy)[None].cuda(),
            sigma,
            network_condition_kwargs={
                "s_inputs": s_inputs,
                "s_trunk": s,
                "z_trunk": z,
                "relative_position_encoding": out["relpos"],
                "feats": cuda,
                "multiplicity": 1,
            },
            training=False,
        )["denoised_atom_coords"]
        confidence = reference.sm_confidence_module(
            cuda, out | {"sample_atom_coords": denoised}, multiplicity=1
        )
        distogram = reference.distogram_module(z)
    reference.cpu()
    ref = jax.tree.map(
        lambda x: x[0].cpu().numpy(),
        {
            "aggregate": aggregate,
            "s_inputs": s_inputs,
            "s": s,
            "z": z,
            "denoised": denoised,
            "pae": confidence["pae_logits"],
            "pde": confidence["pde_logits"],
            "plddt": confidence["plddt_logits"],
            "distogram": distogram,
        },
    )
    del cuda, out, s_inputs, s, z, denoised, confidence, distogram
    torch.cuda.empty_cache()

    model = convert.structure_model(reference)
    model = assert_round_trip(model)
    import importlib.resources
    import types

    import yaml
    from promera.diffusion.sampler import get_edm_sched_fn

    path = importlib.resources.files("promera").joinpath("inference/cofolding.yaml")
    config = yaml.safe_load(path.read_text())
    schedule = get_edm_sched_fn(types.SimpleNamespace(**config["diffusion"]))
    assert_schedule(model.sampling, [schedule(t) for t in np.linspace(0, 1, STEPS + 1)])
    features = promera_features.features(
        {k: v[0].numpy() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    )
    errors = {}
    # Standard residues drop their atom aggregate, so check it on its own.
    aggregate = model.input_embedder.encoder(features.atoms, features.token_features.shape[0])
    errors["aggregate"] = rel_err(aggregate, ref["aggregate"])
    inputs, state = eqx.filter_jit(_trunk)(model, features)
    errors["s_inputs"] = rel_err(inputs.s_inputs, ref["s_inputs"])
    errors["s"] = rel_err(state.s, ref["s"])
    errors["z"] = rel_err(state.z, ref["z"])

    # Later stages from the reference trunk, so errors do not compound.
    ref_state = TrunkState(s=jnp.array(ref["s"]), z=jnp.array(ref["z"]))
    ref_inputs = type(inputs)(
        s_inputs=jnp.array(ref["s_inputs"]), s_init=inputs.s_init, z_init=inputs.z_init
    )
    context = model.denoiser_context(features, ref_inputs, ref_state)
    ours = model.denoise(context, jnp.array(x_noisy), jnp.array(NOISE_LEVEL), key=KEY)
    errors["denoised"] = rel_err(ours[:num_atoms], ref["denoised"][:num_atoms])
    conf = model.confidence_logits(
        features, ref_inputs, ref_state, jnp.array(ref["denoised"]), key=KEY
    )
    errors["pae"] = rel_err(conf.pae, ref["pae"])
    errors["pde"] = rel_err(conf.pde, ref["pde"])
    errors["plddt"] = rel_err(conf.plddt, ref["plddt"])
    errors["distogram"] = rel_err(model.distogram_logits(ref_state), ref["distogram"])
    print(errors)

    assert conf.resolved is None
    tolerances = {"aggregate": 1e-4, "s_inputs": 1e-4, "denoised": 1e-4, "distogram": 1e-4}
    for name, err in errors.items():
        assert err < tolerances.get(name, 1e-3), (name, err)
