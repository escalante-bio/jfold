"""Padding the atom axis, as pipelines that run atom attention in windows do."""

import numpy as np


def padded_length(num_atoms: int, multiple: int, *, minimum: int = 0) -> int:
    """`num_atoms` rounded up to a multiple of `multiple`, and at least `minimum`."""
    return max(-(-num_atoms // multiple) * multiple, minimum)


def pad_to(x, length: int, value=0, dtype=None) -> np.ndarray:
    """`x` (as `dtype`, if given) with rows of `value` appended up to `length`."""
    x = np.asarray(x) if dtype is None else np.asarray(x).astype(dtype)
    return np.concatenate([x, np.full((length - len(x), *x.shape[1:]), value, x.dtype)])
