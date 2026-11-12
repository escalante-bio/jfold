"""A diffusion sampler (AF3 Algorithm 18) shared by all AF3-class structure modules.

It draws one sample of shape `[A, 3]` in Å. Use `jax.vmap` over keys to draw several.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

Denoise = Callable[[Float[Array, "A 3"], Float[Array, ""]], Float[Array, "A 3"]]
"""EDM x0-predictor for one sample: maps noisy coordinates `x` (Å) at noise level
`sigma` to the denoised estimate, preconditioning (c_in, c_skip, c_out, c_noise)
included. Build it as an `eqx.Partial` binding the weights and conditioning, so they stay
pytree leaves."""

# `SamplerSettings.center` and `SamplerSettings.align` are static fields, so they must be
# module-level functions. A new lambda or partial for each instance would miss the jit cache.


def no_centering(x: Float[Array, "A 3"], mask: Float[Array, " A"]) -> Float[Array, "A 3"]:
    return x


def mean_centering(x: Float[Array, "A 3"], mask: Float[Array, " A"]) -> Float[Array, "A 3"]:
    """Subtract the mean over all atoms, including padding."""
    return x - x.mean(0)


def masked_centering(x: Float[Array, "A 3"], mask: Float[Array, " A"]) -> Float[Array, "A 3"]:
    """Subtract the mean over real atoms."""
    w = mask[:, None]
    return x - (x * w).sum(0) / jnp.maximum(w.sum(), 1.0)


def no_alignment(
    x: Float[Array, "A 3"], target: Float[Array, "A 3"], mask: Float[Array, " A"]
) -> Float[Array, "A 3"]:
    return x


def euler_update(x, x0, t_hat, sigma_next, step_scale):
    """Take AF3's Euler step from `t_hat` to `sigma_next`, which is unstable with few steps."""
    return x + step_scale * (sigma_next - t_hat) * (x - x0) / t_hat


def exponential_update(x, x0, t_hat, sigma_next, step_scale):
    """Return `r^S x + (1 - r^S) x0` with `r = sigma_next / t_hat`, which is stable with few steps
    (Blitz, arXiv:2610.23703)."""
    keep = (sigma_next / t_hat) ** step_scale
    return keep * x + (1 - keep) * x0


def _kabsch(x, target, mask):
    w = mask[:, None]
    total = w.sum()
    x_c = x - (x * w).sum(0) / total
    target_mean = (target * w).sum(0) / total
    cov = jnp.einsum("ni,nj->ij", w * (target - target_mean), x_c, precision="highest")
    u, _, vh = jnp.linalg.svd(cov)
    flip = jnp.eye(3).at[2, 2].set(jnp.linalg.det(u @ vh))
    return x_c, u @ flip @ vh, target_mean


def _apply(x_c, rotation, translation):
    return jnp.einsum("ni,ji->nj", x_c, rotation, precision="highest") + translation


def rigid_align(
    x: Float[Array, "A 3"], target: Float[Array, "A 3"], mask: Float[Array, " A"]
) -> Float[Array, "A 3"]:
    """Rigidly move `x` onto `target`, with gradients through the fit."""
    return _apply(*_kabsch(x, target, mask))


def rigid_align_stop_gradient(
    x: Float[Array, "A 3"], target: Float[Array, "A 3"], mask: Float[Array, " A"]
) -> Float[Array, "A 3"]:
    """Rigidly move `x` onto `target`, with no gradients through the fitted rotation or target
    centroid."""
    x_c, rotation, target_mean = _kabsch(x, target, mask)
    return _apply(x_c, *jax.lax.stop_gradient((rotation, target_mean)))


class SamplerSettings(eqx.Module):
    """Settings for the stochastic sampler of AF3 Algorithm 18.

    `sigmas` holds `num_steps + 1` noise levels from high to low. Each step raises the noise
    from `sigma` to `t_hat = sigma * (1 + gamma)` by adding Gaussian noise scaled by
    `noise_scale`, with `gamma = gamma_0` while `sigma_next > gamma_min` and 0 after, then moves
    to `sigma_next` with `update`. `center`, `augment`, `augment_after_noise` and `align` set how
    the sample is centred, randomly rotated and aligned to the denoised estimate at each step.
    A model's `sampling.settings(...)` gives its upstream settings, which `dataclasses.replace`
    can change. For few steps, use `exponential_update` with `schedules.Geometric()`.
    """

    sigmas: Float[Array | np.ndarray, " T"]
    gamma_0: float | Float[Array, ""] = 0.8
    gamma_min: float | Float[Array, ""] = 1.0
    noise_scale: float | Float[Array, ""] = 1.003
    step_scale: float | Float[Array, ""] = 1.5
    center: Callable = eqx.field(default=masked_centering, static=True)
    augment: bool = eqx.field(default=True, static=True)
    augment_after_noise: bool = eqx.field(default=False, static=True)
    align: Callable = eqx.field(default=rigid_align_stop_gradient, static=True)
    update: Callable = eqx.field(default=euler_update, static=True)

    @property
    def num_steps(self) -> int:
        return self.sigmas.shape[0] - 1


class Noise(eqx.Module):
    """All random draws for one sampling run, with `rotations` applied as `x @ R`."""

    init: Float[Array, "A 3"]
    churn: Float[Array, "S A 3"]
    rotations: Float[Array, "S 3 3"]
    translations: Float[Array, "S 3"]

    @classmethod
    def draw(cls, key: PRNGKeyArray, *, num_steps: int, num_atoms: int) -> "Noise":
        k_init, k_churn, k_rot, k_trans = jax.random.split(key, 4)
        return cls(
            init=jax.random.normal(k_init, (num_atoms, 3)),
            churn=jax.random.normal(k_churn, (num_steps, num_atoms, 3)),
            rotations=jax.vmap(random_rotation)(jax.random.split(k_rot, num_steps)),
            translations=jax.random.normal(k_trans, (num_steps, 3)),
        )


def random_rotation(key: PRNGKeyArray) -> Float[Array, "3 3"]:
    """Return a uniformly random rotation from a Gaussian quaternion."""
    r, i, j, k = jax.random.normal(key, (4,))
    s = 2.0 / (r * r + i * i + j * j + k * k)  # Dividing by |q|^2 normalises q.
    return jnp.stack(
        [
            jnp.stack([1 - s * (j * j + k * k), s * (i * j - k * r), s * (i * k + j * r)]),
            jnp.stack([s * (i * j + k * r), 1 - s * (i * i + k * k), s * (j * k - i * r)]),
            jnp.stack([s * (i * k - j * r), s * (j * k + i * r), 1 - s * (i * i + j * j)]),
        ]
    )


@eqx.filter_jit
def integrate(
    denoise: Denoise,
    atom_mask: Float[Array, " A"],
    settings: SamplerSettings,
    noise: Noise,
) -> Float[Array, "A 3"]:
    """Run the sampler deterministically on the draws in `noise`, centring and aligning on the atoms
    in `atom_mask`."""
    s = settings
    gammas = jnp.where(s.sigmas[1:] > s.gamma_min, s.gamma_0, 0.0)

    def reframe(x, rotation, translation):
        x = s.center(x, atom_mask)
        return x @ rotation + translation if s.augment else x

    @jax.checkpoint
    def step(x, inputs):
        sigma, sigma_next, gamma, eps, rotation, translation = inputs
        if not s.augment_after_noise:
            x = reframe(x, rotation, translation)
        t_hat = sigma * (1 + gamma)
        x = x + s.noise_scale * jnp.sqrt(jnp.maximum(t_hat**2 - sigma**2, 0.0)) * eps
        if s.augment_after_noise:
            x = reframe(x, rotation, translation)
        x0 = denoise(x, t_hat)
        x = s.align(x, x0, atom_mask)
        x = s.update(x, x0, t_hat, sigma_next, s.step_scale)
        return x, None

    xs = (s.sigmas[:-1], s.sigmas[1:], gammas, noise.churn, noise.rotations, noise.translations)
    x, _ = jax.lax.scan(step, s.sigmas[0] * noise.init, xs)
    return x


def sample(
    denoise: Denoise,
    atom_mask: Float[Array, " A"],
    settings: SamplerSettings,
    *,
    key: PRNGKeyArray,
) -> Float[Array, "A 3"]:
    """Draw new `Noise` from `key` and run the sampler with it."""
    noise = Noise.draw(key, num_steps=settings.num_steps, num_atoms=atom_mask.shape[0])
    return integrate(denoise, atom_mask, settings, noise)
