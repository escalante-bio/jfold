import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float

from jfold.sampling import (
    Noise,
    SamplerSettings,
    euler_update,
    exponential_update,
    integrate,
    masked_centering,
    mean_centering,
    no_alignment,
    no_centering,
    random_rotation,
    rigid_align,
    rigid_align_stop_gradient,
    sample,
    schedules,
)

NUM_ATOMS = 32


def _denoise_to(target: Float[Array, "A 3"], x, sigma):
    """The exact x0-predictor for data concentrated on a single point cloud."""
    del x, sigma
    return target


def _point_denoiser(target):
    return eqx.Partial(_denoise_to, target)


MASK = jnp.ones(NUM_ATOMS)


def _target():
    return 10.0 * jax.random.normal(jax.random.key(0), (NUM_ATOMS, 3))


def _rmsd_after_alignment(x, target):
    aligned = rigid_align_stop_gradient(x, target, jnp.ones(x.shape[0]))
    return float(jnp.sqrt(((aligned - target) ** 2).sum(-1).mean()))


def _grid(num_points):
    return np.linspace(0.0, 1.0, num_points)


def _sigmas(num_steps):
    return schedules.zero_last(schedules.Karras()(_grid(num_steps + 1)))


ODE = SamplerSettings(
    sigmas=_sigmas(20),
    gamma_0=0.0,
    noise_scale=1.0,
    step_scale=1.0,
    center=no_centering,
    augment=False,
    align=no_alignment,
)


@pytest.mark.parametrize(
    "sigmas, final",
    [
        (schedules.zero_last(schedules.Karras()(_grid(11))), 0.0),
        (schedules.append_zero(schedules.Karras()(_grid(10))), 0.0),
        (schedules.Karras()(_grid(11)), 16.0 * 4e-4),
    ],
    ids=["zero_last", "append_zero", "sigma_min"],
)
def test_karras_schedule_shape_and_ends(sigmas, final):
    assert sigmas.shape == (11,)
    assert sigmas.dtype == np.float32
    assert np.isclose(sigmas[0], 16.0 * 160.0)
    assert np.all(np.diff(sigmas) < 0)
    assert np.isclose(sigmas[-1], final)


def test_karras_single_step_is_finite():
    sigmas = schedules.append_zero(schedules.Karras()(_grid(1)))

    np.testing.assert_allclose(sigmas, [16.0 * 160.0, 0.0])


def test_geometric_schedule_is_even_in_log_sigma():
    sigmas = schedules.Geometric()(_grid(11))

    np.testing.assert_allclose(sigmas[[0, -1]], [16.0 * 160.0, 16.0 * 4e-4], rtol=1e-6)
    ratios = sigmas[1:] / sigmas[:-1]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-5)


def test_geometric_is_karras_at_large_rho():
    np.testing.assert_allclose(
        schedules.Karras(rho=1e5)(_grid(11)), schedules.Geometric()(_grid(11)), rtol=1e-3
    )


def test_dilate_fixes_endpoints_and_stays_monotone():
    t = np.linspace(0.0, 1.0, 101)
    warped = schedules.dilate(t, start=0.6, end=0.8, dilation=2.667)

    np.testing.assert_allclose(warped[[0, -1]], [0.0, 1.0], atol=1e-6)
    assert np.all(np.diff(warped) > 0)


def test_cap_starts_at_cap_and_shortens_schedule():
    sigmas = schedules.cap(schedules.append_zero(schedules.Karras()(_grid(14))), 256.0)

    assert sigmas[0] == 256.0
    assert np.all(sigmas[1:] <= 256.0)
    assert len(sigmas) - 1 < 14


def test_random_rotation_is_proper_orthogonal():
    rotations = jax.vmap(random_rotation)(jax.random.split(jax.random.key(1), 8))

    eye = jnp.broadcast_to(jnp.eye(3), rotations.shape)
    np.testing.assert_allclose(rotations @ rotations.transpose(0, 2, 1), eye, atol=1e-5)
    np.testing.assert_allclose(jnp.linalg.det(rotations), 1.0, atol=1e-5)


def test_masked_centering_ignores_padding():
    x = jnp.concatenate([jnp.ones((4, 3)), 100.0 * jnp.ones((2, 3))])
    mask = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])

    np.testing.assert_allclose(masked_centering(x, mask)[:4], 0.0, atol=1e-6)
    assert not np.allclose(mean_centering(x, mask)[:4], 0.0)


@pytest.mark.parametrize("align", [rigid_align, rigid_align_stop_gradient])
def test_rigid_align_recovers_rigid_motion(align):
    target = _target()
    rotation = random_rotation(jax.random.key(2))
    moved = jnp.matmul(target, rotation, precision="highest") + jnp.array([3.0, -1.0, 2.0])

    aligned = align(moved, target, jnp.ones(NUM_ATOMS))

    np.testing.assert_allclose(aligned, target, atol=1e-3)


def test_ode_recovers_point_target():
    target = _target()
    denoise = _point_denoiser(target)

    x = sample(denoise, MASK, ODE, key=jax.random.key(3))

    np.testing.assert_allclose(x, target, atol=1e-3)


@pytest.mark.parametrize("align", [rigid_align_stop_gradient, rigid_align])
def test_stochastic_sampler_recovers_target_up_to_rigid_motion(align):
    target = _target()
    denoise = _point_denoiser(target)
    sampler = SamplerSettings(sigmas=_sigmas(20), align=align)

    xs = jax.vmap(lambda k: sample(denoise, MASK, sampler, key=k))(
        jax.random.split(jax.random.key(4), 3)
    )

    assert np.isfinite(np.asarray(xs)).all()
    for x in xs:
        assert _rmsd_after_alignment(x, target) < 0.05


@pytest.mark.parametrize("update", [euler_update, exponential_update])
def test_one_step_matches_closed_form(update):
    target, start = _target(), jax.random.normal(jax.random.key(7), (NUM_ATOMS, 3))
    sigma, sigma_next, step_scale = 40.0, 10.0, 1.5
    sampler = dataclasses.replace(
        ODE, sigmas=np.array([sigma, sigma_next], np.float32), step_scale=step_scale, update=update
    )
    noise = Noise(
        init=start,
        churn=jnp.zeros((1, NUM_ATOMS, 3)),
        rotations=jnp.eye(3)[None],
        translations=jnp.zeros((1, 3)),
    )

    x = integrate(_point_denoiser(target), MASK, sampler, noise)

    r = sigma_next / sigma
    keep = {euler_update: 1 + step_scale * (r - 1), exponential_update: r**step_scale}[update]
    np.testing.assert_allclose(x, keep * sigma * start + (1 - keep) * target, rtol=1e-5, atol=1e-4)


def test_few_step_exponential_sampler_recovers_target_up_to_rigid_motion():
    target = _target()
    sampler = SamplerSettings(
        sigmas=schedules.zero_last(schedules.Geometric()(_grid(5))), update=exponential_update
    )

    xs = jax.vmap(lambda k: sample(_point_denoiser(target), MASK, sampler, key=k))(
        jax.random.split(jax.random.key(8), 3)
    )

    for x in xs:
        assert _rmsd_after_alignment(x, target) < 0.05


def test_integrate_is_deterministic_given_noise():
    denoise = _point_denoiser(_target())
    sampler = SamplerSettings(sigmas=_sigmas(10))
    noise = Noise.draw(jax.random.key(5), num_steps=sampler.num_steps, num_atoms=NUM_ATOMS)

    np.testing.assert_array_equal(
        integrate(denoise, MASK, sampler, noise),
        integrate(denoise, MASK, sampler, noise),
    )


def test_sampler_fields_are_replaceable():
    sampler = dataclasses.replace(ODE, gamma_0=0.8, sigmas=_sigmas(5))

    assert sampler.num_steps == 5
    assert sampler.gamma_0 == 0.8
    assert sampler.align is no_alignment


@pytest.mark.parametrize("update", [euler_update, exponential_update])
@pytest.mark.parametrize("align", [no_alignment, rigid_align_stop_gradient, rigid_align])
def test_gradient_flows_through_sampler(align, update):
    denoise = _point_denoiser(_target())
    sampler = SamplerSettings(sigmas=_sigmas(10), align=align, update=update)
    noise = Noise.draw(jax.random.key(6), num_steps=sampler.num_steps, num_atoms=NUM_ATOMS)

    def loss(denoise):
        return (integrate(denoise, MASK, sampler, noise) ** 2).mean()

    (grad,) = eqx.filter_jit(eqx.filter_grad(loss))(denoise).args

    assert np.isfinite(np.asarray(grad)).all()
    assert float(jnp.abs(grad).max()) > 0.0


# Padding and per-model settings ---------------------------------------------------------

PADDING = 7


def _halfway(target, x, sigma):
    """A denoiser that reads `x`, atom by atom: real atoms never see the padding."""
    del sigma
    return 0.5 * (x + target)


@pytest.mark.parametrize(
    "policy",
    [
        {"align": no_alignment},
        {"align": rigid_align},
        {"align": rigid_align_stop_gradient},
        {"align": rigid_align_stop_gradient, "augment_after_noise": True},
    ],
    ids=["no-align", "align", "align-stop-gradient", "augment-after-noise"],
)
def test_sampling_ignores_padding(policy):
    """With masked centring, the real atoms' sample is the same with garbage padding
    atoms appended, given the same noise for the real atoms."""
    sampler = dataclasses.replace(
        SamplerSettings(sigmas=_sigmas(12), center=masked_centering), **policy
    )
    target = _target()
    garbage = 1e3 * jax.random.normal(jax.random.key(9), (PADDING, 3))
    mask = jnp.concatenate([jnp.ones(NUM_ATOMS), jnp.zeros(PADDING)])
    noise = Noise.draw(jax.random.key(4), num_steps=sampler.num_steps, num_atoms=NUM_ATOMS)
    real = integrate(eqx.Partial(_halfway, target), MASK, sampler, noise)
    padded = integrate(
        eqx.Partial(_halfway, jnp.concatenate([target, garbage])), mask, sampler, pad_noise(noise)
    )
    np.testing.assert_allclose(padded[:NUM_ATOMS], real, rtol=1e-5, atol=1e-4)


def pad_noise(noise: Noise, padding: int = PADDING) -> Noise:
    """`noise` with garbage draws for `padding` appended atoms."""
    extra = 1e2 * jax.random.normal(jax.random.key(11), (noise.churn.shape[0] + 1, padding, 3))
    return dataclasses.replace(
        noise,
        init=jnp.concatenate([noise.init, extra[0]]),
        churn=jnp.concatenate([noise.churn, extra[1:]], axis=1),
    )


def test_sampling_config_builds_the_settings():
    from jfold.sampling import UpstreamSampler

    config = UpstreamSampler(
        sigma_data=16.0,
        s_max=160.0,
        s_min=4e-4,
        rho=7.0,
        num_steps=10,
        end="append_zero",
        gamma_0=0.8,
        gamma_min=1.0,
        noise_scale=1.003,
        step_scale=1.5,
        center=masked_centering,
        augment=True,
        augment_after_noise=False,
        align=rigid_align,
    )
    sampler = config.settings()
    assert sampler.num_steps == 10 and sampler.sigmas[-1] == 0 and sampler.align is rigid_align
    np.testing.assert_allclose(sampler.sigmas[0], 16.0 * 160.0, rtol=1e-6)
    assert config.settings(4).num_steps == 4
    capped = dataclasses.replace(config, max_sigma=256.0).sigmas(50)
    assert capped[0] == 256.0 and (capped[1:] <= 256.0).all()
    zero_last = dataclasses.replace(config, end="zero_last").sigmas()
    assert zero_last.shape == (11,) and zero_last[-1] == 0
    assert dataclasses.replace(config, end="none").sigmas()[-1] > 0


def test_structure_sampler():
    """A `StructureSampler` samples as the loop does, over a batch of keys, and with other
    settings."""
    from jfold.sampling import StructureSampler

    target = _target()
    settings = SamplerSettings(sigmas=_sigmas(10))
    structure_sampler = StructureSampler(
        denoise=_point_denoiser(target), atom_mask=MASK, settings=settings
    )
    key = jax.random.key(0)
    assert jnp.array_equal(
        structure_sampler.sample(key), sample(_point_denoiser(target), MASK, settings, key=key)
    )
    noise = Noise.draw(key, num_steps=10, num_atoms=NUM_ATOMS)
    assert jnp.array_equal(structure_sampler.integrate(noise), structure_sampler.integrate(noise))
    batch = jax.vmap(structure_sampler.sample)(jax.random.split(key, 3))
    assert batch.shape == (3, NUM_ATOMS, 3)
    few = structure_sampler.with_settings(
        SamplerSettings(sigmas=_sigmas(4)), update=exponential_update
    )
    assert few.num_steps == 4 and few.settings.update is exponential_update
    assert _rmsd_after_alignment(few.sample(key), target) < 1e-3
