"""Helpers for parity tests against the upstream PyTorch implementations."""

import jax.numpy as jnp
import numpy as np
import torch

from jfold.blocks.atoms import Atoms
from jfold.slots import Features


def randomize(module: torch.nn.Module, seed: int) -> torch.nn.Module:
    """Replace every trainable parameter with random values: upstream inits zero some output
    projections, which would hide mismatches behind them. Frozen ones (bin edges, Fourier
    features) keep their fixed values."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            noise = torch.randn(p.shape, generator=gen)
            if p.ndim >= 2:
                p.copy_(noise / p.shape[-1] ** 0.5)
            else:
                p.copy_((1.0 if name.endswith("weight") else 0.0) + 0.1 * noise)
    return module.eval()


def block_features(num_tokens: int, num_atoms: int, **fields) -> Features:
    """`Features` for a test of one slot's module: `fields`, the rest empty placeholders."""
    empty = {
        "atoms": Atoms(
            features=jnp.zeros((num_atoms, 0)),
            ref_pos=jnp.zeros((num_atoms, 3)),
            ref_space_uid=jnp.zeros(num_atoms, int),
            atom_to_token=jnp.zeros(num_atoms, int),
            mask=jnp.ones(num_atoms, bool),
        ),
        "token_features": jnp.zeros((num_tokens, 0)),
        "relative_positions": jnp.zeros((num_tokens, num_tokens, 0)),
        "token_mask": jnp.ones(num_tokens, bool),
        "asym_id": jnp.zeros(num_tokens, int),
        "rep_atom": jnp.zeros(num_tokens, int),
        "atom_position": jnp.zeros(num_atoms, int),
    }
    return Features(**(empty | fields))


def rel_err(ours, ref) -> float:
    """Max absolute error relative to the reference's largest magnitude."""
    ours, ref = np.asarray(ours, np.float64), np.asarray(ref, np.float64)
    assert ours.shape == ref.shape, (ours.shape, ref.shape)
    return float(np.abs(ours - ref).max() / np.abs(ref).max())


def pad(x: np.ndarray, num: int, axes: tuple[int, ...], seed: int = 0) -> np.ndarray:
    """Append `num` positions of random garbage along each of `axes`."""
    rng = np.random.default_rng(seed)
    for axis in axes:
        shape = list(x.shape)
        shape[axis] = num
        x = np.concatenate([x, rng.standard_normal(shape).astype(x.dtype)], axis=axis)
    return x


def mask(num_real: int, num_padding: int) -> np.ndarray:
    return np.arange(num_real + num_padding) < num_real


def assert_finite(tree) -> None:
    import jax

    for leaf in jax.tree.leaves(tree):
        assert np.isfinite(np.asarray(leaf)).all()


def assert_round_trip(model, path=None):
    """Save `model` with `jfold.io` (to `path`, or a temporary directory) and load it
    back: the same structure, static fields and arrays. Returns the loaded model."""
    import tempfile

    import jax

    from jfold import io

    if path is None:
        with tempfile.TemporaryDirectory() as directory:
            return assert_round_trip(model, directory)
    io.save(model, path, metadata={"source": "test"})
    loaded = io.load(path)
    assert jax.tree.structure(loaded) == jax.tree.structure(model)
    for ours, theirs in zip(jax.tree.leaves(model), jax.tree.leaves(loaded), strict=True):
        if hasattr(ours, "dtype"):
            assert ours.dtype == theirs.dtype and np.array_equal(np.array(ours), np.array(theirs))
        else:
            assert type(ours) is type(theirs) and ours == theirs
    return loaded


STEPS = 20  # sampling steps for the schedule checks


def assert_schedule(config, upstream) -> None:
    """`config`'s noise levels for `STEPS` steps are the upstream sampler's own
    `upstream` (fewer, after a cap)."""
    upstream = np.array([float(x) for x in upstream], np.float32)
    np.testing.assert_allclose(config.sigmas(STEPS), upstream, rtol=1e-5)
