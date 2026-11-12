"""Noise schedules, computed on the host as float32 NumPy arrays of sigmas.

A `Schedule` maps times `t` in [0, 1] to sigmas from `sigma_data * s_max` down to
`sigma_data * s_min`. `Karras(rho)` is the schedule AF3-class models train with, and
`Geometric()`, evenly spaced in log sigma, suits few-step sampling (Blitz, arXiv:2610.23703).
`append_zero`, `zero_last` and `cap` change how a schedule ends or starts.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Schedule(Protocol):
    def __call__(
        self, t: np.ndarray, *, sigma_data: float, s_max: float, s_min: float
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class Karras:
    rho: float = 7.0

    def __call__(
        self,
        t: np.ndarray,
        *,
        sigma_data: float = 16.0,
        s_max: float = 160.0,
        s_min: float = 4e-4,
    ) -> np.ndarray:
        inv_rho = 1.0 / self.rho
        sigmas = sigma_data * (s_max**inv_rho + t * (s_min**inv_rho - s_max**inv_rho)) ** self.rho
        return sigmas.astype(np.float32)


@dataclass(frozen=True)
class Geometric:
    def __call__(
        self,
        t: np.ndarray,
        *,
        sigma_data: float = 16.0,
        s_max: float = 160.0,
        s_min: float = 4e-4,
    ) -> np.ndarray:
        sigmas = sigma_data * np.exp(np.log(s_max) + t * (np.log(s_min) - np.log(s_max)))
        return sigmas.astype(np.float32)


def append_zero(sigmas: np.ndarray) -> np.ndarray:
    """Append a final sigma of 0 (EDM)."""
    return np.append(sigmas, 0.0).astype(np.float32)


def zero_last(sigmas: np.ndarray) -> np.ndarray:
    """Replace the last sigma with 0 (AF3 paper)."""
    return np.append(sigmas[:-1], 0.0).astype(np.float32)


def dilate(t: np.ndarray, *, start: float, end: float, dilation: float) -> np.ndarray:
    """Warp `t` piecewise-linearly so that `dilation` times more of the grid falls in
    [start, end]."""
    width, lower, upper = end - start, start, 1 - end
    assert (dilation - 1) * width <= lower + upper, "dilation too large"
    ratio = (lower + upper + (1 - dilation) * width) / (lower + upper)
    lower_prime, width_prime = lower * ratio, width * dilation
    return np.select(
        [t < lower_prime, t < lower_prime + width_prime],
        [t / ratio, (t - lower_prime) / dilation + lower],
        (t - (lower_prime + width_prime)) / ratio + lower + width,
    )


def cap(sigmas: np.ndarray, max_sigma: float) -> np.ndarray:
    """Drop the sigmas above `max_sigma` and start at `max_sigma`, which can shorten the
    schedule."""
    return np.concatenate([[max_sigma], sigmas[sigmas <= max_sigma]]).astype(np.float32)
