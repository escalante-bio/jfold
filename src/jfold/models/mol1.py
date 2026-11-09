"""Mol-1's architecture-specific pieces."""

from jaxtyping import Array, Bool, Float, PRNGKeyArray

from jfold.blocks.pairformer import AbstractPairformer, PairformerStack
from jfold.models import protenix
from jfold.nn import Linear, split_key


class Mol1Pairformer(AbstractPairformer):
    """Mol-1's trunk stack: PairMixer layers at the resident pair width, then Pairformer
    layers at a narrower one, between two linear adapters."""

    pairmixer: PairformerStack
    adapter_in: Linear
    pairformer: PairformerStack
    adapter_out: Linear

    def __call__(
        self,
        s: Float[Array, "T Cs"],
        z: Float[Array, "T T Cz"],
        mask: Bool[Array, " T"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "T Cs"], Float[Array, "T T Cz"]]:
        k_mix, k_pair = split_key(key)
        s, z = self.pairmixer(s, z, mask, key=k_mix)
        s, z = self.pairformer(s, self.adapter_in(z), mask, key=k_pair)
        return s, self.adapter_out(z)


def features(f: dict):
    """`jfold.model.Features` from a Mol-1 feature dict (`mol.data` featurization; numpy
    arrays, unbatched). Mol-1 featurizes as Protenix does, without constraint features."""
    return protenix.features(f)
