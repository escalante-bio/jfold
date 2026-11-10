"""The diffusion module (AF3 Algorithms 20 and 21): denoise atom positions given the trunk.

`DiffusionModule.prepare` (or `context`, from arrays) computes what depends only on the
trunk and the reference conformers, once; `__call__` is one denoising step. Inputs are
unbatched: tokens `[T, ...]` and atoms `[N, ...]`, with masks that are False for padding."""

import equinox as eqx
import jax.numpy as jnp
from equinox.nn import Identity
from jaxtyping import Array, Bool, Float

from jfold.blocks.atoms import AtomAttentionDecoder, AtomAttentionEncoder, Atoms
from jfold.blocks.diffusion import DiffusionTransformer
from jfold.nn import LayerNorm, Linear, Transition
from jfold.slots import AbstractDiffusionModule, Features, TrunkInputs, TrunkState


class FourierEmbedding(eqx.Module):
    """`cos(2π (t w + b))` with fixed random `w` and `b`."""

    weight: Float[Array, " C"]
    bias: Float[Array, " C"]

    def __call__(self, t: Float[Array, ""]) -> Float[Array, " C"]:
        return jnp.cos(2 * jnp.pi * (t * self.weight + self.bias))


class DiffusionConditioning(eqx.Module):
    """AF3 Algorithm 21: the pair conditioned on relative positions (`pair`, once per
    trunk) and the single conditioned on the noise level (`single`, every step)."""

    norm_z: LayerNorm
    linear_z: Linear  # [z_trunk, relative positions] -> Cz
    transitions_z: tuple[Transition, ...]
    norm_s: LayerNorm
    linear_s: Linear  # [s_trunk, s_inputs] -> Cs
    fourier: FourierEmbedding
    norm_noise: LayerNorm
    linear_noise: Linear
    transitions_s: tuple[Transition, ...]
    sigma_data: float = eqx.field(static=True)
    # OpenDDE narrows the trunk pair to the diffusion module's width first.
    norm_z_trunk: LayerNorm | None = None
    linear_z_trunk: Linear | None = None

    def pair(
        self, z_trunk: Float[Array, "T T Cz"], relative_positions: Float[Array, "T T Cr"]
    ) -> Float[Array, "T T Cz"]:
        if self.linear_z_trunk is not None:
            z_trunk = self.linear_z_trunk(self.norm_z_trunk(z_trunk))
        z = self.linear_z(self.norm_z(jnp.concatenate([z_trunk, relative_positions], -1)))
        for transition in self.transitions_z:
            z = z + transition(z)
        return z

    def single(
        self, s_trunk: Float[Array, "T Cs"], s_inputs: Float[Array, "T Ci"], t: Float[Array, ""]
    ) -> Float[Array, "T Cs"]:
        s = self.linear_s(self.norm_s(jnp.concatenate([s_trunk, s_inputs], -1)))
        noise = self.fourier(jnp.log(t / self.sigma_data) / 4)
        s = s + self.linear_noise(self.norm_noise(noise))
        for transition in self.transitions_s:
            s = s + transition(s)
        return s


class DiffusionContext(eqx.Module):
    """Everything a denoising step needs that depends only on the trunk."""

    s_trunk: Float[Array, "T Cs"]
    s_inputs: Float[Array, "T Ci"]
    token_mask: Bool[Array, " T"]
    atoms: Atoms
    queries: Float[Array, "N Ca"]
    atom_single: Float[Array, "N Ca"]
    encoder_biases: Float[Array, "Le heads NW queries keys"]
    token_biases: Float[Array, "Lt heads T T"]
    decoder_biases: Float[Array, "Ld heads NW queries keys"]

    @property
    def atom_mask(self) -> Bool[Array, " N"]:
        """The real (not padding) atoms: what a sampler centres and aligns on."""
        return self.atoms.mask


class DiffusionModule(AbstractDiffusionModule[Features]):
    """AF3 Algorithm 20, with EDM preconditioning."""

    conditioning: DiffusionConditioning
    encoder: AtomAttentionEncoder
    norm_s: LayerNorm
    linear_s: Linear  # conditioned single -> token width
    transformer: DiffusionTransformer
    norm_a: LayerNorm
    decoder: AtomAttentionDecoder
    # The relative position features as the pair conditioning reads them: projected
    # (Protenix lineage, Boltz-2), or raw (OpenFold3).
    relative_positions: Linear | Identity = eqx.field(default_factory=Identity)

    @property
    def sigma_data(self) -> float:
        return self.conditioning.sigma_data

    def prepare(self, features: Features, inputs: TrunkInputs, state: TrunkState) -> DiffusionContext:
        return self.context(
            features.atoms,
            inputs.s_inputs,
            state.s,
            state.z,
            features.relative_positions,
            features.token_mask,
        )

    def context(
        self,
        atoms: Atoms,
        s_inputs: Float[Array, "T Ci"],
        s_trunk: Float[Array, "T Cs"],
        z_trunk: Float[Array, "T T Cz"],
        relative_position_features: Float[Array, "T T R"],
        token_mask: Bool[Array, " T"],
    ) -> DiffusionContext:
        """The context from the trunk's outputs, for tokens `[T]` and atoms `[N]` of any
        kind (OpenDDE's structural tokens)."""
        relative_positions = self.relative_positions(relative_position_features)
        z = self.conditioning.pair(z_trunk, relative_positions)
        q, c, p = self.encoder.condition(atoms, s_trunk, z)
        return DiffusionContext(
            s_trunk=s_trunk,
            s_inputs=s_inputs,
            token_mask=token_mask,
            atoms=atoms,
            queries=q,
            atom_single=c,
            encoder_biases=self.encoder.transformer.pair_biases(p),
            token_biases=self.transformer.pair_biases(z),
            decoder_biases=self.decoder.transformer.pair_biases(p),
        )

    def __call__(
        self,
        context: DiffusionContext,
        x_noisy: Float[Array, "N 3"],
        t: Float[Array, ""],
    ) -> Float[Array, "N 3"]:
        """The denoised positions at noise level `t`."""
        atoms, sigma_data = context.atoms, self.sigma_data
        s = self.conditioning.single(context.s_trunk, context.s_inputs, t)
        r = x_noisy / jnp.sqrt(t**2 + sigma_data**2)
        a, q = self.encoder(
            context.queries,
            context.atom_single,
            context.encoder_biases,
            atoms.atom_to_token,
            atoms.mask,
            context.token_mask.shape[0],
            r,
        )
        a = a + self.linear_s(self.norm_s(s))
        a = self.transformer(a, s, context.token_biases, context.token_mask)
        r_update = self.decoder(
            self.norm_a(a),
            q,
            context.atom_single,
            context.decoder_biases,
            atoms.atom_to_token,
            atoms.mask,
        )
        scale = sigma_data**2 + t**2
        return sigma_data**2 / scale * x_noisy + sigma_data * t / jnp.sqrt(scale) * r_update
