"""The template embedder (AF3 Algorithm 16): each template's pair features, with the trunk
pair, through a pair-only Pairformer stack, averaged over the templates.

Inputs are unbatched: template features `[N, T, T, F]` (the model's, from
`jfold.models.*.template_features`), a template mask `[N]` and a token mask `[T]`; masks are
False for padding."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from jfold.blocks.pairformer import PairformerStack
from jfold.nn import LayerNorm, Linear


class TemplateEmbedder(eqx.Module):
    """AF3 Algorithm 16."""

    norm_z: LayerNorm
    linear_z: Linear  # Cz -> Ct
    linear_features: Linear  # template feature vector -> Ct
    stack: PairformerStack  # pair-only layers
    norm_v: LayerNorm
    linear_u: Linear  # Ct -> Cz
    # Boltz-2 adds each template's input back to its stack's (already residual) output.
    add_stack_input: bool = eqx.field(static=True, default=False)

    def __call__(
        self,
        z: Float[Array, "T T Cz"],
        template_features: Float[Array, "N T T F"],
        template_mask: Bool[Array, " N"],
        mask: Bool[Array, " T"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "T T Cz"]:
        base = self.linear_z(self.norm_z(z))

        def one_template(features, k):
            v = base + self.linear_features(features)
            _, out = self.stack(None, v, mask, key=k)
            return self.norm_v(out + v if self.add_stack_input else out)

        num = template_features.shape[0]
        keys = None if key is None else jax.random.split(key, num)
        v = jax.vmap(one_template)(template_features, keys)
        weights = template_mask.astype(v.dtype)[:, None, None, None]
        u = (v * weights).sum(0) / jnp.maximum(template_mask.sum(), 1)
        # Kept out of `linear_u`'s matmul: when the mask is a constant, XLA's GPU Triton
        # GEMM fusion (JAX 0.12) folds the average into the matmul and computes it wrong.
        return self.linear_u(jax.lax.optimization_barrier(jax.nn.relu(u)))
