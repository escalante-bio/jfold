"""Sampling a structure end to end with padded inputs: embed, trunk, denoiser context
and the sampler give the real atoms the same coordinates as without padding, given the
same random draws for the real tokens and atoms."""

import equinox as eqx
import jax
import numpy as np
import pytest
import torch
from reference import pad
from test_esmfold2 import C_Z, NUM_ATOMS, NUM_TOKENS, PADDING, _pad_system, _small_model, _system
from test_sampling import pad_noise

from jfold.convert import esmfold2 as convert
from jfold.models import esmfold2
from jfold.sampling import Noise, integrate
from jfold.slots import TrunkState

NUM_STEPS = 8


@eqx.filter_jit
def _sample(model, features, z, noise):
    key = jax.random.key(5)  # dropout is off: the trunk's key is unused
    inputs = model.embed(features, key=key)
    state = TrunkState(s=inputs.s_init, z=z)
    for _ in range(2):
        state = model.trunk(features, inputs, state, key=key)
    state = model.trunk.finish(features, state, key=key)
    sampler = model.structure_sampler(features, inputs, state, num_steps=NUM_STEPS)
    return integrate(sampler.denoise, sampler.atom_mask, sampler.settings, noise)


@pytest.mark.parametrize("experimental", [False, True], ids=["release", "experimental"])
def test_padding_leaves_the_sample_unchanged(experimental):
    """ESMFold2 samples the same real-atom coordinates with padded tokens, atoms and MSA
    rows, from the same initial state and sampler noise on the real ones."""
    m = _small_model(True, experimental=experimental)
    f = _system(torch.Generator().manual_seed(1), True)
    model = (convert.esmfold2_experimental if experimental else convert.esmfold2)(m)
    key = jax.random.key(3)
    z = 0.1 * jax.random.normal(key, (NUM_TOKENS, NUM_TOKENS, C_Z))
    steps = model.sampling.settings(NUM_STEPS).num_steps
    noise = Noise.draw(key, num_steps=steps, num_atoms=NUM_ATOMS)

    real = _sample(model, esmfold2.features(f, r_max=4), z, noise)
    z_padded = jax.numpy.array(pad(np.asarray(z), PADDING, axes=(0, 1)))
    padded = _sample(
        model, esmfold2.features(_pad_system(f), r_max=4), z_padded, pad_noise(noise, PADDING)
    )
    np.testing.assert_allclose(padded[:NUM_ATOMS], real, rtol=1e-4, atol=1e-3)
