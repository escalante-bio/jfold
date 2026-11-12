"""Sampler settings that match each model's upstream implementation."""

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import numpy as np

from jfold.sampling import schedules
from jfold.sampling.core import SamplerSettings


class UpstreamSampler(eqx.Module):
    """One model's upstream sampler settings, which `settings()` turns into `SamplerSettings`.

    The schedule is `schedules.Karras(rho)` from `sigma_data * s_max` to `sigma_data * s_min`.
    `end` sets how it finishes: `"append_zero"` adds a final step to 0 (EDM, Boltz),
    `"zero_last"` sets the last of `num_steps + 1` levels to 0 (AF3), and `"none"` stops at
    `sigma_data * s_min`. If set, `max_sigma` (ESMFold2) drops the levels above it. The other
    fields are as in `SamplerSettings`.
    """

    sigma_data: float = eqx.field(static=True)
    s_max: float = eqx.field(static=True)
    s_min: float = eqx.field(static=True)
    rho: float = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)
    end: Literal["append_zero", "zero_last", "none"] = eqx.field(static=True)
    gamma_0: float = eqx.field(static=True)
    gamma_min: float = eqx.field(static=True)
    noise_scale: float = eqx.field(static=True)
    step_scale: float = eqx.field(static=True)
    center: Callable = eqx.field(static=True)
    augment: bool = eqx.field(static=True)
    augment_after_noise: bool = eqx.field(static=True)
    align: Callable = eqx.field(static=True)
    max_sigma: float | None = eqx.field(static=True, default=None)

    def sigmas(
        self, num_steps: int | None = None, *, schedule: schedules.Schedule | None = None
    ) -> np.ndarray:
        """Return the noise levels from high to low, using `schedule` instead of Karras if given."""
        n = self.num_steps if num_steps is None else num_steps
        schedule = schedules.Karras(rho=self.rho) if schedule is None else schedule

        def levels(count: int) -> np.ndarray:
            grid = np.linspace(0.0, 1.0, count)
            return schedule(grid, sigma_data=self.sigma_data, s_max=self.s_max, s_min=self.s_min)

        match self.end:
            case "append_zero":
                sigmas = schedules.append_zero(levels(n))
            case "zero_last":
                sigmas = schedules.zero_last(levels(n + 1))
            case "none":
                sigmas = levels(n + 1)
            case end:
                raise ValueError(f"unknown schedule end {end!r}")
        if self.max_sigma is not None:
            sigmas = schedules.cap(sigmas, self.max_sigma)
        return sigmas

    def settings(
        self, num_steps: int | None = None, *, schedule: schedules.Schedule | None = None
    ) -> SamplerSettings:
        """Return `SamplerSettings` that reproduce the upstream sampler, with `num_steps` steps or
        fewer if `max_sigma` is set."""
        return SamplerSettings(
            sigmas=self.sigmas(num_steps, schedule=schedule),
            gamma_0=self.gamma_0,
            gamma_min=self.gamma_min,
            noise_scale=self.noise_scale,
            step_scale=self.step_scale,
            center=self.center,
            augment=self.augment,
            augment_after_noise=self.augment_after_noise,
            align=self.align,
        )
