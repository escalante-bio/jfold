"""End-to-end parity with the released OpenFold3 checkpoint: the trunk, one denoising step,
confidence and distogram on a small real complex, from OpenFold3's own featurization.
Downloads weights; run with `pytest -m checkpoint`."""

import tempfile
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


def _reference_model():
    """The released checkpoint as `jfold.pretrained` builds it (as OpenFold3's inference
    does), from OpenFold3's default parameter directory."""
    from jopenfold3._vendor.openfold3.entry_points.parameters import get_default_checkpoint_dir

    from jfold import pretrained
    from jfold.pretrained.recipes.openfold3 import FILENAME

    path = get_default_checkpoint_dir() / FILENAME
    return pretrained.upstream("openfold3", checkpoint_path=path)


def _featurize():
    """OpenFold3's inference featurization of `SEQUENCES` (single-sequence MSAs, no
    templates)."""
    from jopenfold3._vendor.openfold3.core.data.framework.single_datasets.inference import (
        InferenceDataset,
    )
    from jopenfold3._vendor.openfold3.core.data.pipelines.preprocessing.template import (
        TemplatePreprocessorSettings,
    )
    from jopenfold3._vendor.openfold3.core.data.resources.residues import MoleculeType
    from jopenfold3._vendor.openfold3.core.data.tools.colabfold_msa_server import (
        MsaComputationSettings,
        augment_main_msa_with_query_sequence,
    )
    from jopenfold3._vendor.openfold3.projects.of3_all_atom.config.dataset_config_components import (
        MSASettings,
        TemplateSettings,
    )
    from jopenfold3._vendor.openfold3.projects.of3_all_atom.config.dataset_configs import (
        InferenceJobConfig,
    )
    from jopenfold3._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        Chain,
        InferenceQuerySet,
        Query,
    )

    chains = [
        Chain(molecule_type=MoleculeType.PROTEIN, chain_ids=[c], description=c, sequence=s)
        for c, s in zip("AB", SEQUENCES)
    ]
    queries = InferenceQuerySet(seeds=[0], queries={"q": Query(query_name="q", chains=chains)})
    with tempfile.TemporaryDirectory() as msa_dir:
        queries = augment_main_msa_with_query_sequence(
            queries, MsaComputationSettings(msa_output_directory=Path(msa_dir))
        )
        config = InferenceJobConfig(
            query_set=queries,
            seeds=[0],
            msa=MSASettings(subsample_main=False),
            template=TemplateSettings(take_top_k=True),
            template_preprocessor_settings=TemplatePreprocessorSettings(),
        )
        batch = InferenceDataset(config)[0]
    assert batch["valid_sample"].item()
    return {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}


def test_openfold3():
    from jopenfold3._vendor.openfold3.core.utils.atomize_utils import (
        get_token_representative_atoms,
    )

    from jfold.convert import openfold3
    from jfold.model import TrunkState
    from jfold.models import openfold3 as openfold3_features

    reference = _reference_model()
    batch = _featurize()
    batched = {k: v[None] for k, v in batch.items()}
    # The model's view after the trunk: a sample dimension on every feature.
    sampled = {k: v[:, None] for k, v in batched.items()}
    n_atoms = batch["atom_mask"].shape[0]
    x_noisy = 16 * np.random.default_rng(0).standard_normal((n_atoms, 3), dtype=np.float32)
    with torch.no_grad():
        s_inputs, s, z = reference.run_trunk(batched, num_cycles=NUM_RECYCLES)
        denoised = reference.diffusion_module(
            batch=sampled,
            xl_noisy=torch.from_numpy(x_noisy)[None, None],
            token_mask=sampled["token_mask"],
            atom_mask=sampled["atom_mask"],
            t=torch.tensor(NOISE_LEVEL),
            si_input=s_inputs[:, None],
            si_trunk=s[:, None],
            zij_trunk=z[:, None],
            use_conditioning=True,
        )
        aux = reference.aux_heads(
            batch=sampled,
            si_input=s_inputs[:, None],
            output={
                "si_trunk": s[:, None],
                "zij_trunk": z[:, None],
                "atom_positions_predicted": denoised,
            },
            use_zij_trunk_embedding=True,
        )
        # Each token's representative atom index, read off as a "coordinate".
        index = torch.arange(n_atoms, dtype=torch.float32)[None, :, None].expand(1, n_atoms, 3)
        rep_index, rep_mask = get_token_representative_atoms(batched, index, batched["atom_mask"])
    s_inputs, s, z, denoised = s_inputs[0], s[0], z[0], denoised[0, 0]

    model = openfold3.structure_model(reference)
    model = assert_round_trip(model)
    from jopenfold3._vendor.openfold3.core.model.structure.diffusion_module import (
        create_noise_schedule,
    )

    upstream = create_noise_schedule(
        no_rollout_steps=STEPS,
        **reference.config.architecture.noise_schedule,
        dtype=torch.float32,
        device="cpu",
    )
    assert_schedule(model.sampling, upstream)
    features = openfold3_features.features(batch)
    assert bool(rep_mask.all())
    np.testing.assert_array_equal(features.rep_atom, rep_index[0, :, 0].long().numpy())
    # Traceable: built inside `jit` from JAX arrays, the features are the same.
    traced = eqx.filter_jit(openfold3_features.features)(
        {k: jnp.array(v.numpy()) for k, v in batch.items()}
    )
    assert jax.tree.all(jax.tree.map(jnp.array_equal, traced, features))

    inputs, state = eqx.filter_jit(_trunk)(model, features)
    errors = {
        "s_inputs": rel_err(inputs.s_inputs, s_inputs),
        "s": rel_err(state.s, s),
        "z": rel_err(state.z, z),
    }

    # Later stages from the reference trunk, so errors do not compound.
    ref_state = TrunkState(s=jnp.array(s.numpy()), z=jnp.array(z.numpy()))
    ref_inputs = type(inputs)(
        s_inputs=jnp.array(s_inputs.numpy()), s_init=inputs.s_init, z_init=inputs.z_init
    )
    context = model.denoiser_context(features, ref_inputs, ref_state)
    ours = model.denoise(context, jnp.array(x_noisy), jnp.array(NOISE_LEVEL), key=KEY)
    errors["denoised"] = rel_err(ours, denoised)
    conf = model.confidence_logits(
        features, ref_inputs, ref_state, jnp.array(denoised.numpy()), key=KEY
    )
    errors["pae"] = rel_err(conf.pae, aux["pae_logits"][0, 0])
    errors["pde"] = rel_err(conf.pde, aux["pde_logits"][0, 0])
    errors["plddt"] = rel_err(conf.plddt, aux["plddt_logits"][0, 0])
    errors["resolved"] = rel_err(conf.resolved, aux["experimentally_resolved_logits"][0, 0])
    errors["distogram"] = rel_err(model.distogram_logits(ref_state), aux["distogram_logits"][0, 0])
    print(errors)

    assert errors["s_inputs"] < 1e-4
    assert errors["s"] < 1e-3
    assert errors["z"] < 1e-3
    assert errors["denoised"] < 1e-4
    for name in ("pae", "pde", "plddt", "resolved"):
        assert errors[name] < 1e-3, name
    assert errors["distogram"] < 1e-4
