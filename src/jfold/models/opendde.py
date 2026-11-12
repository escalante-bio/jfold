"""OpenDDE, a Protenix-derived model whose diffusion runs on structural tokens, such as a
backbone and a side-chain token per residue. `StructuralDiffusion` expands the residue
trunk's outputs to these tokens, which are stored in the features' `extras`."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from jfold.blocks.atoms import Atoms
from jfold.blocks.denoiser import DiffusionContext, DiffusionModule
from jfold.blocks.inputs import relative_position_features
from jfold.blocks.pairformer import PairformerStack
from jfold.models import protenix
from jfold.models.common import arrays
from jfold.nn import LayerNorm, Linear
from jfold.slots import AbstractDiffusionModule, Features, TrunkInputs, TrunkState

# The class of each role in `opendde.data.tokenizer.STRUCTURAL_TOKEN_ROLES`. Class 0 is a
# backbone (protein, DNA or RNA), 1 a protein side chain, 2 a nucleotide base and 3 an atom.
ROLE_CLASS = (3, 0, 1, 0, 2, 0, 2)
# The role-pair category of each (row class, column class) pair, with 7 for any other pair.
ROLE_PAIR = ((0, 1, 4, 7), (2, 3, 7, 7), (5, 7, 6, 7), (7, 7, 7, 7))


class StructuralTokens(eqx.Module):
    """Structural tokens, each with a parent residue, a role and its parent's chain
    neighbours."""

    parent: Int[Array, " M"]
    role: Int[Array, " M"]
    prev_parent: Int[Array, " M"]
    next_parent: Int[Array, " M"]
    mask: Bool[Array, " M"]
    atoms: Atoms
    relative_positions: Float[Array, "M M R"]


class StructuralPairFeatures(eqx.Module):
    """Boolean relations between structural tokens and their role-pair category."""

    same_parent: Bool[Array, "M M"]
    twin: Bool[Array, "M M"]  # backbone with side chain or base of the same residue
    prev_backbone: Bool[Array, "M M"]  # backbone pair, column's parent precedes row's
    next_backbone: Bool[Array, "M M"]
    role_pair: Int[Array, "M M"]  # 0-6 for backbone/side chain/base pairs, 7 otherwise

    @classmethod
    def build(cls, tokens: StructuralTokens, asym_id: Int[Array, " T"]):
        parent = tokens.parent
        role_class = jnp.array(ROLE_CLASS)[tokens.role]
        backbone, other = role_class == 0, (role_class == 1) | (role_class == 2)
        chain = asym_id[parent]
        same_parent = parent[:, None] == parent[None, :]
        backbone_pair = backbone[:, None] & backbone[None, :] & (chain[:, None] == chain[None, :])
        return cls(
            same_parent=same_parent,
            twin=same_parent
            & ((backbone[:, None] & other[None, :]) | (other[:, None] & backbone[None, :])),
            prev_backbone=backbone_pair & (tokens.prev_parent[:, None] == parent[None, :]),
            next_backbone=backbone_pair & (tokens.next_parent[:, None] == parent[None, :]),
            role_pair=jnp.array(ROLE_PAIR)[role_class[:, None], role_class[None, :]],
        )

    def flags(self) -> Float[Array, "M M 4"]:
        return jnp.stack(
            [self.same_parent, self.twin, self.prev_backbone, self.next_backbone], -1
        ).astype(jnp.float32)


class PairEmbedding(eqx.Module):
    """A sum of learned vectors, one per pair relation."""

    flags: Float[Array, "4 2 C"]
    role_pair: Float[Array, "8 C"]

    def __call__(self, pair: StructuralPairFeatures) -> Float[Array, "M M C"]:
        f = pair.flags()
        off, on = self.flags[:, 0], self.flags[:, 1]
        return off.sum(0) + jnp.einsum("ijf,fc->ijc", f, on - off) + self.role_pair[pair.role_pair]


class StructuralTokenExpander(eqx.Module):
    """Expand residue trunk outputs to structural tokens, also returning their attention
    bias."""

    single_inputs_role: Float[Array, "R Ci"]
    single_role: Float[Array, "R Cs"]
    single_norm: LayerNorm
    single_hidden: Linear
    single_out: Linear
    pair_projection: Float[Array, "R R Cz Cz"]  # [row role, column role, out, in]
    pair_embedding: PairEmbedding
    attention_bias: PairEmbedding  # one channel

    def __call__(
        self,
        tokens: StructuralTokens,
        asym_id: Int[Array, " T"],
        s_inputs: Float[Array, "T Ci"],
        s: Float[Array, "T Cs"],
        z: Float[Array, "T T Cz"],
    ) -> tuple[
        Float[Array, "M Ci"], Float[Array, "M Cs"], Float[Array, "M M Cz"], Float[Array, "M M"]
    ]:
        parent, role = tokens.parent, tokens.role
        s_inputs = s_inputs[parent] + self.single_inputs_role[role]
        s = s[parent]
        s = s + self.single_out(jax.nn.silu(self.single_hidden(self.single_norm(s))))
        s = s + self.single_role[role]
        z = z[parent][:, parent]
        pair = StructuralPairFeatures.build(tokens, asym_id)
        z = z + self._project(z, role) + self.pair_embedding(pair)
        return s_inputs, s, z, self.attention_bias(pair)[..., 0]

    def _project(self, z: Float[Array, "M M Cz"], role: Int[Array, " M"]) -> Float[Array, "M M Cz"]:

        def by_row_role(delta, r):
            weights = self.pair_projection[r][role]  # by column role: [M, out, in]
            projected = jnp.einsum("ijc,joc->ijo", z, weights)
            return jnp.where((role == r)[:, None, None], projected, delta), None

        num_roles = self.pair_projection.shape[0]
        delta, _ = jax.lax.scan(by_row_role, jnp.zeros_like(z), jnp.arange(num_roles))
        return delta


class StructuralDiffusion(AbstractDiffusionModule[Features]):
    """OpenDDE's diffusion slot, which expands, refines and denoises on structural tokens.
    """

    expander: StructuralTokenExpander
    diffusion: DiffusionModule
    refiner: PairformerStack | None = None

    def structural_trunk(
        self, features: Features, inputs: TrunkInputs, state: TrunkState
    ) -> tuple[Float[Array, "M Ci"], TrunkState, Float[Array, "M M"]]:
        """Return `s_inputs`, the refined single and pair, and the attention bias on
        structural tokens."""
        tokens = features.extras
        s_inputs, s, z, bias = self.expander(
            tokens, features.asym_id, inputs.s_inputs, state.s, state.z
        )
        if self.refiner is not None:
            # The context is built without a key, so a refiner with dropout enabled raises
            # an error instead of reusing one dropout mask every time.
            s, z = self.refiner(s, z, tokens.mask, bias[None])
        return s_inputs, TrunkState(s=s, z=z), bias

    def prepare(self, features: Features, inputs: TrunkInputs, state: TrunkState) -> DiffusionContext:
        tokens = features.extras
        s_inputs, state, bias = self.structural_trunk(features, inputs, state)
        context = self.diffusion.context(
            tokens.atoms, s_inputs, state.s, state.z, tokens.relative_positions, tokens.mask
        )
        return eqx.tree_at(
            lambda c: c.token_biases, context, context.token_biases + bias[None, None]
        )

    def __call__(
        self, context: DiffusionContext, x_noisy: Float[Array, "N 3"], t: Float[Array, ""]
    ) -> Float[Array, "N 3"]:
        return self.diffusion(context, x_noisy, t)


def features(f: dict) -> Features:
    """Build Protenix's `Features` with OpenDDE's structural tokens as `extras`."""
    residues = protenix.features(f)
    a = arrays(f)
    parent = a["parent_residue_idx"]
    return eqx.tree_at(
        lambda features: features.extras,
        residues,
        StructuralTokens(
            parent=parent,
            role=a["subtoken_role_id"],
            prev_parent=a["prev_parent_residue_idx"],
            next_parent=a["next_parent_residue_idx"],
            mask=jnp.ones(parent.shape[0], bool),
            atoms=eqx.tree_at(
                lambda atoms: atoms.atom_to_token,
                residues.atoms,
                a["atom_to_structural_token_idx"],
            ),
            relative_positions=relative_position_features(
                a["asym_id"][parent],
                a["residue_index"][parent],
                a["entity_id"][parent],
                a["structural_token_index"],
                a["sym_id"][parent],
            ),
        ),
        is_leaf=lambda x: x is None,
    )
