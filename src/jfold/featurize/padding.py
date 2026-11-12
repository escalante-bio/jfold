"""Pad the atom axis for pipelines that run atom attention in windows."""

import numpy as np


def padded_length(num_atoms: int, multiple: int, *, minimum: int = 0) -> int:
    """Round `num_atoms` up to a multiple of `multiple`, and to at least `minimum`."""
    return max(-(-num_atoms // multiple) * multiple, minimum)


def pad_to(x, length: int, value=0, dtype=None) -> np.ndarray:
    """Append rows of `value` to `x` up to `length`, casting to `dtype` if given."""
    x = np.asarray(x) if dtype is None else np.asarray(x).astype(dtype)
    return np.concatenate([x, np.full((length - len(x), *x.shape[1:]), value, x.dtype)])
