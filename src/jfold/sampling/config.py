"""A model's own sampler settings, as its upstream defines them."""

import dataclasses
from collections.abc import Callable
from typing import Literal

import equinox as eqx
import numpy as np

from jfold.sampling import schedules
from jfold.sampling.core import SamplerSettings


class UpstreamSampler(eqx.Module):
    """Upstream's sampler for one model: its noise schedule, churn and frame policy.
    Converters attach one to each model (`model.sampling`); `settings()` builds the
    `SamplerSettings` that reproduce the upstream loop, and a model's
    `structure_sampler(...)` uses them.

    The schedule is `schedules.Karras(rho)` over `sigma_data * [s_max, s_min]` on an
    evenly spaced grid, ended by `end`:
      - `"append_zero"`: `num_steps` levels, then a final step to 0 (EDM, Boltz).
      - `"zero_last"`: `num_steps + 1` levels, the last replaced by 0 (the AF3 paper).
      - `"none"`: `num_steps + 1` levels ending at `sigma_data * s_min`.
    `max_sigma` (ESMFold2) then drops the levels above it.

    The rest are `SamplerSettings`' fields of the same names, with no defaults here: each
    converter states upstream's. (`update` is not among them: every upstream takes Euler
    steps, `SamplerSettings`' default.)

    Every converter centres on the real atoms (`masked_centering`). Upstreams that
    centre on the plain mean (Boltz-2, ProMERA, OpenDDE) count their own padding atoms
    in it; without padding the two are the same, and with it only the masked mean keeps
    padding from moving the sample.
    """

    sigma_data: float = eqx.field(static=True)
    s_max: float = eqx.field(static=True)
    s_min: float = eqx.field(static=True)
    rho: float = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)  # upstream's default
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
        """The noise levels, high to low; `schedule` replaces upstream's `Karras`."""
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
        """The upstream sampler's settings, `num_steps` steps (upstream's count by default;
        fewer survive a `max_sigma` cap)."""
        shared = {name: getattr(self, name) for name in _SHARED}
        return SamplerSettings(sigmas=self.sigmas(num_steps, schedule=schedule), **shared)


# The `SamplerSettings` fields an `UpstreamSampler` sets: all but the schedule and `update`.
_SHARED = [
    f.name for f in dataclasses.fields(SamplerSettings) if f.name not in ("sigmas", "update")
]
