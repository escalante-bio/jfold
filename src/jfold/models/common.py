"""Feature helpers several models share."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


def as_array(v) -> jax.Array:
    """Convert to a JAX array, leaving JAX arrays and tracers unchanged."""
    return v if isinstance(v, jax.Array) else jnp.asarray(np.asarray(v))


def arrays(f: dict) -> dict[str, jax.Array]:
    """Return a feature dict's numeric entries as JAX arrays."""
    out = {}
    for k, v in f.items():
        if not isinstance(v, jax.Array) and hasattr(v, "shape"):
            v = np.asarray(v)
        if hasattr(v, "dtype") and np.dtype(v.dtype).kind in "biuf":
            out[k] = as_array(v)
    return out


def msa_features(
    msa: Float[Array, "S T C"],
    has_deletion: Float[Array, "S T"],
    deletion_value: Float[Array, "S T"],
) -> Float[Array, "S T C+2"]:
    """Concatenate the MSA, has-deletion and deletion value features."""
    return jnp.concatenate([msa, has_deletion[..., None], deletion_value[..., None]], axis=-1)


def token_features(
    restype: Float[Array, "T C"], profile: Float[Array, "T C"], deletion_mean: Float[Array, " T"]
) -> Float[Array, "T 2C+1"]:
    """Concatenate the residue type, MSA profile and deletion mean features."""
    return jnp.concatenate([restype, profile, deletion_mean[:, None]], -1)
