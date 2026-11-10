"""OpenDDE: a Protenix-lineage model whose diffusion runs on structural tokens.

The trunk, distogram and confidence heads work on residues, as in Protenix. Before
diffusion, each residue's trunk outputs are copied to its structural tokens (a protein
residue's backbone and side chain; a nucleotide's backbone and base; one token per ligand
atom), conditioned on each token's role, refined by a small Pairformer, and denoised with an
extra attention bias between structural tokens: the model is a `jfold.model.StructureModel`
whose diffusion slot is `StructuralDiffusion`, with the structural tokens as the features'
`extras`."""

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

# Each structural-token role's class (`opendde.data.tokenizer.STRUCTURAL_TOKEN_ROLES`):
# 0 backbone (protein, DNA, RNA), 1 protein side chain, 2 nucleotide base, 3 none (an atom).
ROLE_CLASS = (3, 0, 1, 0, 2, 0, 2)
# The role-pair category of a (row class, column class) pair; 7 for any other pair.
ROLE_PAIR = ((0, 1, 4, 7), (2, 3, 7, 7), (5, 7, 6, 7), (7, 7, 7, 7))


class StructuralTokens(eqx.Module):
    """The structural tokens: each one's parent residue (a trunk token) and role, its
    parent's neighbours along the chain (-1 at chain ends), the atoms assigned to
    structural tokens, and the tokens' relative position features. `mask` is False for
    padding."""

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
    """A sum of learned vectors, one per pair relation: each boolean relation's table has a
    row for false and one for true; the role-pair category has eight rows."""

    flags: Float[Array, "4 2 C"]
    role_pair: Float[Array, "8 C"]

    def __call__(self, pair: StructuralPairFeatures) -> Float[Array, "M M C"]:
        f = pair.flags()
        off, on = self.flags[:, 0], self.flags[:, 1]
        return off.sum(0) + jnp.einsum("ijf,fc->ijc", f, on - off) + self.role_pair[pair.role_pair]


class StructuralTokenExpander(eqx.Module):
    """Residue trunk outputs to structural tokens: copy each token's parent residue, add
    role embeddings to the singles and an MLP to the trunk single, project each pair by its
    (row role, column role) and add relation embeddings. Also returns the scalar attention
    bias between structural tokens."""

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
        """Each pair through the projection of its (row role, column role)."""

        def by_row_role(delta, r):
            weights = self.pair_projection[r][role]  # by column role: [M, out, in]
            projected = jnp.einsum("ijc,joc->ijo", z, weights)
            return jnp.where((role == r)[:, None, None], projected, delta), None

        num_roles = self.pair_projection.shape[0]
        delta, _ = jax.lax.scan(by_row_role, jnp.zeros_like(z), jnp.arange(num_roles))
        return delta


class StructuralDiffusion(AbstractDiffusionModule[Features]):
    """OpenDDE's diffusion slot: the residue trunk's outputs expanded to structural tokens
    (`expander`, the features' `extras`), refined (`refiner`), and denoised by `diffusion`
    on those tokens with the structural attention bias added."""

    expander: StructuralTokenExpander
    diffusion: DiffusionModule
    refiner: PairformerStack | None = None

    def structural_trunk(
        self, features: Features, inputs: TrunkInputs, state: TrunkState
    ) -> tuple[Float[Array, "M Ci"], TrunkState, Float[Array, "M M"]]:
        """The trunk outputs on structural tokens: `s_inputs`, the refined single and pair,
        and the structural attention bias."""
        tokens = features.extras
        s_inputs, s, z, bias = self.expander(
            tokens, features.asym_id, inputs.s_inputs, state.s, state.z
        )
        if self.refiner is not None:
            # No key: the context is built without one, so a refiner with dropout switched
            # on raises rather than drawing the same mask every time.
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
    """`Features` from an OpenDDE feature dict (`opendde.data` featurization; numpy arrays,
    unbatched): Protenix's, with the structural tokens as `extras`."""
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
