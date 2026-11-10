"""BioEmu2 parity against the upstream PyTorch implementation (`bioemu2.biomoa`), with random
weights: trunk blocks, transformer layers, the input embedder and the structure module."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from reference import (
    STEPS,
    assert_finite,
    assert_round_trip,
    assert_schedule,
    block_features,
    randomize,
    rel_err,
)

from jfold.blocks.atoms import Atoms
from jfold.blocks.inputs import relative_position_features
from jfold.convert import bioemu2
from jfold.models import bioemu2 as bioemu2_model
from jfold.slots import Features, TrunkInputs, TrunkState

NUM_TOKENS, NUM_ATOMS, NUM_ROWS = 7, 16, 5
W, H = 4, 8  # atoms per query window, keys per window
C_S, C_Z, C_M = 12, 16, 8  # trunk single, pair, MSA
C_ATOM, C_ATOM_PAIR = 8, 4
C_A = 16  # structure module token width
C_INPUTS = C_S + 33 + 33 + 1 + 4  # s_inputs: atom aggregate and token features
SIGMA = 7.0
KEY = jax.random.key(0)
t = torch.from_numpy


@dataclasses.dataclass
class System:
    """A random system's raw inputs (numpy, unbatched)."""

    # tokens
    asym_id: np.ndarray
    entity_id: np.ndarray
    residue_index: np.ndarray
    sym_id: np.ndarray
    method: np.ndarray
    res_type: np.ndarray
    profile: np.ndarray
    deletion_mean: np.ndarray
    pocket: np.ndarray
    center_atom: np.ndarray
    token_bonds: np.ndarray
    contacts: np.ndarray
    contact_threshold: np.ndarray
    # MSA
    msa: np.ndarray
    has_deletion: np.ndarray
    deletion_value: np.ndarray
    paired: np.ndarray
    # atoms
    atom_to_token: np.ndarray
    ref_pos: np.ndarray
    charge: np.ndarray
    element: np.ndarray
    name_chars: np.ndarray
    ref_space_uid: np.ndarray
    coordinate_mask: np.ndarray
    # trunk outputs and per-step inputs
    s_inputs: np.ndarray
    s_trunk: np.ndarray
    z_trunk: np.ndarray
    rel_pos: np.ndarray
    x_noisy: np.ndarray
    coords_cond: np.ndarray

    @property
    def num_tokens(self):
        return self.asym_id.shape[0]

    @property
    def num_atoms(self):
        return self.atom_to_token.shape[0]


def make_system(num_tokens, num_atoms, seed) -> System:
    rng = np.random.default_rng(seed)
    f32 = lambda *shape: rng.standard_normal(shape, dtype=np.float32)
    chain = (np.arange(num_tokens) >= num_tokens // 2).astype(np.int64)
    atom_to_token = np.sort(
        np.concatenate([np.arange(num_tokens), rng.integers(0, num_tokens, num_atoms - num_tokens)])
    )
    profile = rng.random((num_tokens, 33), dtype=np.float32)
    msa = rng.integers(0, 33, (NUM_ROWS, num_tokens))
    return System(
        asym_id=chain,
        entity_id=np.zeros(num_tokens, np.int64),  # one entity, two copies
        residue_index=np.arange(num_tokens) - chain * (num_tokens // 2),
        sym_id=chain,
        method=rng.integers(0, bioemu2_model.NUM_METHODS, num_tokens),
        res_type=rng.integers(0, 33, num_tokens),
        profile=profile / profile.sum(-1, keepdims=True),
        deletion_mean=rng.random(num_tokens, dtype=np.float32),
        pocket=rng.integers(0, 4, num_tokens),
        center_atom=np.searchsorted(atom_to_token, np.arange(num_tokens)),
        token_bonds=(rng.random((num_tokens, num_tokens)) < 0.2).astype(np.float32),
        contacts=np.eye(5, dtype=np.float32)[rng.integers(0, 5, (num_tokens, num_tokens))],
        contact_threshold=rng.uniform(4, 20, (num_tokens, num_tokens)).astype(np.float32),
        msa=msa,
        has_deletion=(rng.random((NUM_ROWS, num_tokens)) < 0.2).astype(np.float32),
        deletion_value=rng.random((NUM_ROWS, num_tokens), dtype=np.float32),
        paired=(rng.random((NUM_ROWS, num_tokens)) < 0.5).astype(np.float32),
        atom_to_token=atom_to_token,
        ref_pos=3 * f32(num_atoms, 3),
        charge=f32(num_atoms),
        element=rng.integers(0, 128, num_atoms),
        name_chars=rng.integers(0, 64, (num_atoms, 4)),
        ref_space_uid=atom_to_token // 2,
        coordinate_mask=rng.random(num_atoms) < 0.9,
        s_inputs=f32(num_tokens, C_INPUTS),
        s_trunk=f32(num_tokens, C_S),
        z_trunk=f32(num_tokens, num_tokens, C_Z),
        rel_pos=f32(num_tokens, num_tokens, C_Z),
        x_noisy=SIGMA * f32(num_atoms, 3),
        coords_cond=3 * f32(num_atoms, 3),
    )


def padded(x: System, tokens: int, atoms: int) -> System:
    """`x` with `tokens` garbage tokens and `atoms` garbage atoms appended (garbage atoms
    belong to random tokens; garbage tokens' center atoms are random atoms)."""
    if not (tokens or atoms):
        return x
    n, a = x.num_tokens, x.num_atoms
    g = make_system(n + tokens, a + atoms, seed=7)
    rng = np.random.default_rng(8)
    for f in dataclasses.fields(System):
        ours, garbage = getattr(x, f.name), getattr(g, f.name)
        if f.name in ("msa", "has_deletion", "deletion_value", "paired"):
            garbage[:, :n] = ours
        elif f.name in ("z_trunk", "rel_pos", "token_bonds", "contacts", "contact_threshold"):
            garbage[:n, :n] = ours
        elif garbage.shape[0] == n + tokens:
            garbage[:n] = ours
        else:
            garbage[:a] = ours
    g.atom_to_token[a:] = rng.integers(0, n + tokens, atoms)
    g.center_atom[n:] = rng.integers(0, a + atoms, tokens)
    return g


def masks(x: System):
    return jnp.arange(x.num_tokens) < NUM_TOKENS, jnp.arange(x.num_atoms) < NUM_ATOMS


def _atoms(x: System, atom_mask) -> Atoms:
    return Atoms(
        features=bioemu2_model.atom_features(
            jnp.array(x.ref_pos), jnp.array(x.charge), jnp.array(x.element), jnp.array(x.name_chars)
        ),
        ref_pos=jnp.array(x.ref_pos),
        ref_space_uid=jnp.array(x.ref_space_uid),
        atom_to_token=jnp.array(x.atom_to_token),
        mask=atom_mask,
    )


def _features(x: System) -> tuple[Features, bioemu2_model.Conditioning]:
    token_mask, atom_mask = masks(x)
    a = lambda v: jnp.array(v)
    features = Features(
        atoms=_atoms(x, atom_mask),
        token_features=bioemu2_model.token_features(
            jax.nn.one_hot(a(x.res_type), 33), a(x.profile), a(x.deletion_mean), a(x.pocket)
        ),
        relative_positions=relative_position_features(
            a(x.asym_id),
            a(x.residue_index),
            a(x.entity_id),
            jnp.arange(x.num_tokens),
            a(x.sym_id),
        ),
        token_mask=token_mask,
        asym_id=a(x.asym_id),
        rep_atom=a(x.center_atom),
        atom_position=jnp.zeros(x.num_atoms, jnp.int32),
        bonds=a(x.token_bonds)[..., None],
        msa_features=bioemu2_model.msa_features(
            jax.nn.one_hot(a(x.msa), 33), a(x.has_deletion), a(x.deletion_value), a(x.paired)
        ),
        msa_mask=(jnp.arange(NUM_ROWS) < NUM_ROWS)[:, None] & token_mask[None, :],
    )
    conditioning = bioemu2_model.Conditioning(
        method=jax.nn.one_hot(a(x.method), bioemu2_model.NUM_METHODS),
        contacts=a(x.contacts),
        contact_threshold=a(x.contact_threshold),
        center_atom=a(x.center_atom),
        coordinate_mask=a(x.coordinate_mask) & atom_mask,
    )
    return features, conditioning


def _torch_feats(x: System) -> dict:
    """The upstream's feature dict, batched."""
    n, s = x.num_tokens, NUM_ROWS
    feats = {
        "asym_id": t(x.asym_id),
        "entity_id": t(x.entity_id),
        "residue_index": t(x.residue_index),
        "sym_id": t(x.sym_id),
        "method_feature": t(x.method),
        "token_pad_mask": torch.ones(n, dtype=torch.bool),
        "token_bonds": t(x.token_bonds)[..., None],
        "contact_conditioning": t(x.contacts),
        "contact_threshold": t(x.contact_threshold),
        "msa": t(np.eye(33, dtype=np.float32)[x.msa]),
        "msa_mask": torch.ones(s, n, dtype=torch.bool),
        "has_deletion": t(x.has_deletion),
        "deletion_value": t(x.deletion_value),
        "msa_paired": t(x.paired),
        "template_coords": t(x.ref_pos),
        "charge": t(x.charge),
        "element": t(x.element),
        "name_encoding": t(x.name_chars),
        "ref_space_uid": t(x.ref_space_uid),
        "atom_pad_mask": torch.ones(x.num_atoms, dtype=torch.bool),
        "token_idx_at_atom": t(x.atom_to_token),
        "atom_mask": t(x.coordinate_mask),
        "token_center_atom_index": t(x.center_atom),
        "coords_cond": t(x.coords_cond),
    }
    return {k: v[None] for k, v in feats.items()}


# Trunk -----------------------------------------------------------------------------------------


def _sequence_embedder(**kwargs):
    from bioemu2.biomoa.modules.pairmixer import PairMixerSequenceEmbedder

    return PairMixerSequenceEmbedder(
        token_s=C_S,
        token_z=C_Z,
        s_input_dim=C_INPUTS,
        msa_s=C_M,
        msa_blocks=2,
        num_blocks=2,
        trimul_variant="full",
        **kwargs,
    )


@eqx.filter_jit
def _trunk(init, trunk, features, conditioning, s_inputs, num_iterations):
    inputs = init(dataclasses.replace(features, extras=conditioning), s_inputs)
    state = trunk.init_state(inputs)
    for _ in range(num_iterations):
        state = trunk(features, inputs, state, key=KEY)
    return inputs, state


@pytest.mark.parametrize("padding", [(0, 0), (3, 5)], ids=["unpadded", "padded"])
@pytest.mark.parametrize("method_pairs", [False, True], ids=["method-single", "method-pair"])
def test_trunk_matches_reference(method_pairs, padding):
    """`PairMixerSequenceEmbedder` (initial single and pair with method and contact
    conditioning, then two recycling iterations of the MSA module and PairMixer stack) and
    its unconditional branch, on the shared `TrunkInit` and `Trunk`."""
    module = randomize(
        _sequence_embedder(embed_method_into_pairs=method_pairs, use_unconditional_branch=True),
        seed=0,
    )
    x = make_system(NUM_TOKENS, NUM_ATOMS, seed=1)
    with torch.no_grad():
        s_ref, z_ref = module(t(x.s_inputs)[None], _torch_feats(x), num_steps_recycle=1)
        s_unc, z_unc = module(t(x.s_inputs)[None], _torch_feats(x), unconditional=True)

    init, trunk = bioemu2.trunk_init(module), bioemu2.trunk(module)
    unconditional = bioemu2.transition(module.uncond_transition)
    x_in = padded(x, *padding)
    features, conditioning = _features(x_in)
    s_inputs = jnp.array(x_in.s_inputs)
    inputs, state = _trunk(init, trunk, features, conditioning, s_inputs, 2)
    n = NUM_TOKENS
    assert rel_err(state.s[:n], s_ref[0]) < 1e-5
    assert rel_err(state.z[:n, :n], z_ref[0]) < 1e-5

    z = inputs.z_init + unconditional(trunk.recycle_z.norm(inputs.z_init))
    assert rel_err(inputs.s_init[:n], s_unc[0]) < 1e-5
    assert rel_err(z[:n, :n], z_unc[0]) < 1e-5

    def loss(s_inputs):
        _, state = _trunk(init, trunk, features, conditioning, s_inputs, 1)
        return state.z[:n, :n].sum()

    assert_finite((state, jax.grad(loss)(s_inputs)))


def test_relative_positions_match_reference():
    """`FusedRelativePositionEncoder` as a linear on the shared relative position features."""
    from bioemu2.biomoa.modules.relative_position_encoder import FusedRelativePositionEncoder

    module = randomize(FusedRelativePositionEncoder(token_z=C_Z, r_max=3, s_max=1), seed=0)
    x = make_system(NUM_TOKENS, NUM_ATOMS, seed=1)
    x.residue_index = 2 * x.residue_index  # offsets past r_max
    x.sym_id = np.array([0, 0, 0, 1, 1, 3, 3])  # chain offsets past s_max
    with torch.no_grad():
        ref = module(
            *(t(getattr(x, k))[None] for k in ("asym_id", "entity_id", "residue_index", "sym_id"))
        )
    features = relative_position_features(
        jnp.array(x.asym_id),
        jnp.array(x.residue_index),
        jnp.array(x.entity_id),
        jnp.arange(NUM_TOKENS),
        jnp.array(x.sym_id),
        r_max=3,
        s_max=1,
    )
    assert rel_err(bioemu2.relative_positions(module)(features), ref[0]) < 1e-6


def test_distogram_matches_reference():
    from bioemu2.biomoa.modules.embedders import DistogramModule

    module = randomize(DistogramModule(C_Z, 10), seed=0)
    z = make_system(NUM_TOKENS, NUM_ATOMS, seed=1).z_trunk
    with torch.no_grad():
        ref = module(t(z)[None])
    assert rel_err(bioemu2.distogram(module)(jnp.array(z)), ref[0]) < 1e-6


# Transformer layers ------------------------------------------------------------------------------


@eqx.filter_jit
def _transformer(transformer, a, s, pair, mask):
    return transformer(a, s, transformer.pair_biases(pair), mask)


@pytest.mark.parametrize("padding", [0, 3], ids=["unpadded", "padded"])
@pytest.mark.parametrize("lns_scale", [1.0, 0.5])
def test_conditioned_transformer_matches_reference(lns_scale, padding):
    """Token `ConditionedTransformerLayer`s (adaLN-Zero) as a shared `DiffusionTransformer`."""
    from bioemu2.biomoa.modules.transformer import ConditionedTransformerLayer

    layers = torch.nn.ModuleList(
        [
            ConditionedTransformerLayer(
                dim=C_A,
                dim_pairwise=C_Z,
                dim_cond=C_S,
                heads=2,
                transition_hidden=2 * C_A,
                lns_scale=lns_scale,
            )
            for _ in range(2)
        ]
    )
    randomize(layers, seed=0)
    rng = np.random.default_rng(1)
    n = NUM_TOKENS + padding
    a = rng.standard_normal((n, C_A), dtype=np.float32)
    c = rng.standard_normal((n, C_S), dtype=np.float32)
    z = rng.standard_normal((n, n, C_Z), dtype=np.float32)
    real = slice(NUM_TOKENS)
    with torch.no_grad():
        ref = t(a[real])[None]
        for layer in layers:
            ref = layer(
                ref, t(z[real, real])[None], t(c[real])[None], torch.ones(1, NUM_TOKENS).bool()
            )
    transformer = bioemu2.conditioned_transformer(layers)
    mask = jnp.arange(n) < NUM_TOKENS
    out = _transformer(transformer, jnp.array(a), jnp.array(c), jnp.array(z), mask)
    assert rel_err(out[real], ref[0]) < 1e-5
    grads = jax.grad(
        lambda a: _transformer(transformer, a, jnp.array(c), jnp.array(z), mask)[real].sum()
    )(jnp.array(a))
    assert_finite((out, grads))


@pytest.mark.parametrize("padding", [(0, 0), (3, 5)], ids=["unpadded", "padded"])
def test_input_embedder_matches_reference(padding):
    """`InputEmbedder` (its atom encoder in sliding windows) as the shared `InputEmbedder`."""
    from bioemu2.biomoa.modules.embedders import InputEmbedder

    module = randomize(
        InputEmbedder(
            atom_s=C_ATOM,
            atom_z=C_ATOM_PAIR,
            token_s=C_S,
            token_z=C_Z,
            atoms_per_window_queries=W,
            atoms_per_window_keys=H,
            atom_encoder_depth=2,
            atom_encoder_heads=2,
        ),
        seed=0,
    )
    x = make_system(NUM_TOKENS, NUM_ATOMS, seed=1)
    f = _torch_feats(x)
    with torch.no_grad():
        ref = module(
            res_type=t(x.res_type)[None],
            profile=t(x.profile)[None],
            deletion_mean=t(x.deletion_mean)[None],
            pocket_feature=t(x.pocket)[None],
            ref_pos=f["template_coords"],
            ref_charge=f["charge"],
            ref_element=f["element"],
            ref_atom_name_chars=f["name_encoding"],
            ref_space_uid=f["ref_space_uid"],
            atom_pad_mask=f["atom_pad_mask"],
            atom_to_token_ids=f["token_idx_at_atom"],
            token_pad_mask=f["token_pad_mask"],
        )
    ours = bioemu2.input_embedder(module)
    features, _ = _features(padded(x, *padding))

    @eqx.filter_jit
    def run(ours, atoms):
        return ours(dataclasses.replace(features, atoms=atoms))

    out = run(ours, features.atoms)
    assert rel_err(out[:NUM_TOKENS], ref[0]) < 1e-5
    grads = jax.grad(
        lambda pos: run(ours, eqx.tree_at(lambda a: a.ref_pos, features.atoms, pos))[
            :NUM_TOKENS
        ].sum()
    )(features.atoms.ref_pos)
    assert_finite((out, grads))


# Structure module --------------------------------------------------------------------------------


def _structure_module(dynamics, downscale):
    from bioemu2.biomoa.modules.diffusion import DualStreamDiffusionModule

    return DualStreamDiffusionModule(
        token_s=C_S,
        token_z=C_Z,
        atom_s=C_ATOM,
        atom_z=C_ATOM_PAIR,
        s_input_dim=C_INPUTS,
        a_s=C_A,
        a_z=C_Z,
        num_layers=2,
        atoms_per_window_queries=W,
        atoms_per_window_keys=H,
        dim_fourier=8,
        atom_attention_heads=2,
        token_attention_heads=2,
        use_lns=True,
        use_dynamics_conditioning=dynamics,
        downscale_condition_contrib=downscale,
        dt_embed_num_timescales=4,
    )


@eqx.filter_jit
def _denoise(module, atoms, x: dict, conditioning, token_mask, dynamics):
    features = block_features(
        token_mask.shape[0],
        atoms.mask.shape[0],
        atoms=atoms,
        token_mask=token_mask,
        relative_positions=x["rel_pos"],
        extras=dataclasses.replace(conditioning, dynamics=dynamics),
    )
    inputs = TrunkInputs(s_inputs=x["s_inputs"], s_init=x["s_trunk"], z_init=x["z_trunk"])
    context = module.prepare(features, inputs, TrunkState(s=x["s_trunk"], z=x["z_trunk"]))
    return module(context, x["x_noisy"], jnp.array(SIGMA))


CONDITIONING = {
    "equilibrium": (False, False, False),
    "dynamics-off": (True, True, False),
    "dynamics": (True, True, True),
    "dynamics-ungated": (True, False, True),
}


@pytest.mark.parametrize("padding", [(0, 0), (3, 5)], ids=["unpadded", "padded"])
@pytest.mark.parametrize("case", CONDITIONING)
def test_structure_module_matches_reference(case, padding):
    """One denoising step of `DualStreamDiffusionModule`, with LayerNorm scaling, with and
    without dynamics conditioning (on and off)."""
    use_dynamics, downscale, conditioned = CONDITIONING[case]
    module = randomize(_structure_module(use_dynamics, downscale), seed=0)
    x = make_system(NUM_TOKENS, NUM_ATOMS, seed=1)
    lag = 3.0
    f = _torch_feats(x)
    if use_dynamics:
        f["dt"] = torch.tensor([[lag]])
        f["is_dynamics_conditioned"] = torch.tensor([[conditioned]])
    sigma = torch.tensor([SIGMA])
    sigma_data = module.sigma_data
    with torch.no_grad():
        context = module.compute_static_context(
            t(x.s_trunk)[None], t(x.z_trunk)[None], t(x.rel_pos)[None], f
        )
        c_in = 1 / torch.sqrt(sigma**2 + sigma_data**2)
        r_update = module.denoise(
            context,
            t(x.s_inputs)[None],
            t(x.s_trunk)[None],
            c_in * t(x.x_noisy)[None],
            torch.log(sigma / sigma_data) / 4,
            multiplicity=1,
        )["r_update"][0]
        scale = sigma_data**2 + sigma**2
        ref = sigma_data**2 / scale * t(x.x_noisy) + sigma * sigma_data / scale.sqrt() * r_update

    ours = bioemu2.dual_stream_module(module)
    x_in = padded(x, *padding)
    token_mask, atom_mask = masks(x_in)
    _, conditioning = _features(x_in)
    dynamics = None
    if use_dynamics:
        dynamics = bioemu2_model.Dynamics(
            coords=jnp.array(x_in.coords_cond),
            lag=jnp.array(lag),
            is_conditioned=jnp.array(conditioned),
        )
    inputs = {
        k: jnp.array(getattr(x_in, k))
        for k in ("s_inputs", "s_trunk", "z_trunk", "rel_pos", "x_noisy")
    }
    atoms = _atoms(x_in, atom_mask)
    out = _denoise(ours, atoms, inputs, conditioning, token_mask, dynamics)
    assert rel_err(out[:NUM_ATOMS], ref) < 1e-5

    def loss(inputs):
        return _denoise(ours, atoms, inputs, conditioning, token_mask, dynamics)[:NUM_ATOMS].sum()

    assert_finite((out, jax.grad(loss)(inputs)))


def test_lag_embedding_matches_reference():
    from bioemu2.biomoa.modules.time import DampedHarmonicTimeEmbedding

    module = DampedHarmonicTimeEmbedding(num_timescales=5)
    ours = bioemu2_model.DampedHarmonicEmbedding(
        taus=jnp.array(module.taus.numpy()),
        quality_factors=jnp.array(module.quality_factors.numpy()),
    )
    for lag in (0.0, 2.5, 1e4):
        ref = module(torch.tensor([[lag]]))[0]
        assert rel_err(ours(jnp.array(lag)), ref) < 1e-6


# Released weights ------------------------------------------------------------------------------

SEQUENCES = ("MKTAYIAKQRQISFVKSHFSRQ", "GSHMLEDPV")
NUM_RECYCLES = 1


@pytest.mark.checkpoint
def test_bioemu2_preview():
    """The released `bioemu2-preview` on a small complex: the trunk with one recycle, the
    distogram, and one denoising step from the reference trunk. Downloads weights."""
    from bioemu2.biomoa.ops.triangle_multiplication import (
        TriangleMultiplicationCueq,
        TriangleMultiplicationFull,
    )
    from bioemu2.biomoa.utils.feature_converter import convert_densebatch_to_dictionary
    from bioemu2.checkpoints import get_checkpoint_files
    from bioemu2.data_interface import System
    from bioemu2.msa.alignment import MSA
    from bioemu2.structure.batch import batch_collate
    from bioemu2.structure.preparation import prepare_structure

    from jfold import pretrained
    from jfold.pretrained.recipes.bioemu2 import CHECKPOINT

    reference = pretrained.upstream(
        "bioemu2", checkpoint_path=get_checkpoint_files(CHECKPOINT).checkpoint_path
    )
    for m in reference.modules():  # the cuequivariance kernels need CUDA; same weights and math
        if isinstance(m, TriangleMultiplicationCueq):
            m.__class__ = TriangleMultiplicationFull

    system = System.from_description(proteins=list(SEQUENCES))
    msa = MSA(unpaired_a3m={c: f">q\n{s}\n" for c, s in zip("AB", SEQUENCES)})
    batch = batch_collate([prepare_structure(system, msa=msa, mode="structure", contacts=[])])
    feats = convert_densebatch_to_dictionary(batch)
    num_atoms = feats["atom_pad_mask"].shape[1]
    x_noisy = 16 * np.random.default_rng(0).standard_normal((num_atoms, 3), dtype=np.float32)
    sigma = 10.0
    with torch.no_grad():
        trunk = reference.compute_trunk_embeddings(batch, num_steps_recycle=NUM_RECYCLES)
        context = reference.compute_static_context(batch, trunk)
        denoised = reference.denoise(
            context, trunk, torch.from_numpy(x_noisy)[None], torch.tensor([sigma])
        ).denoised_coords[0]

    model = bioemu2.bioemu2(reference)
    model = assert_round_trip(model)
    from bioemu2.denoising.edm import edm_sample_schedule

    schedule = reference.edm_schedule
    upstream = edm_sample_schedule(
        STEPS,
        device=torch.device("cpu"),
        sigma_min=schedule.sigma_min,
        sigma_max=schedule.sigma_max,
        sigma_data=schedule.sigma_data,
    )
    assert_schedule(model.sampling, upstream)
    features = bioemu2_model.features({k: v[0].numpy() for k, v in feats.items()})

    @eqx.filter_jit
    def run_trunk(model, features):
        inputs = model.embed(features, key=KEY)
        state = model.trunk.init_state(inputs)
        for _ in range(NUM_RECYCLES + 1):
            state = model.trunk(features, inputs, state, key=KEY)
        return inputs, state

    inputs, state = run_trunk(model, features)
    assert rel_err(inputs.s_inputs, trunk.s_inputs[0]) < 1e-5
    assert rel_err(state.s, trunk.single[0]) < 1e-5
    assert rel_err(state.z, trunk.pair[0]) < 1e-5
    assert rel_err(model.distogram(state.z), trunk.distogram[0]) < 1e-5

    # The structure module from the reference trunk, so errors do not compound.
    ref_state = TrunkState(s=jnp.array(trunk.single[0].numpy()), z=jnp.array(trunk.pair[0].numpy()))
    ref_inputs = TrunkInputs(
        s_inputs=jnp.array(trunk.s_inputs[0].numpy()), s_init=inputs.s_init, z_init=inputs.z_init
    )

    @eqx.filter_jit
    def run_denoise(model, features, inputs, state, x_noisy):
        context = model.diffusion.prepare(features, inputs, state)
        return model.diffusion(context, x_noisy, jnp.array(sigma))

    ours = run_denoise(model, features, ref_inputs, ref_state, jnp.array(x_noisy))
    real = feats["atom_pad_mask"][0].numpy()  # padded atoms' outputs are unspecified
    assert rel_err(ours[real], denoised[real]) < 1e-5
