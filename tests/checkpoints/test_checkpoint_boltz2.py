"""End-to-end parity with the released Boltz-2 checkpoint: the trunk, one denoising step,
confidence and distogram on a small complex featurized by Boltz itself (an MSA and a
template on one chain, a cyclic second chain, pocket and contact constraints). Downloads
weights; run with `pytest -m checkpoint`."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from reference import STEPS, assert_round_trip, assert_schedule, pad, rel_err

pytestmark = pytest.mark.checkpoint

CACHE = Path.home() / ".boltz"
SEQUENCES = ("MKTAYIAKQRQISFVKSHFSRQ", "GSHMLEDPV")
MSA = ("MKTAYLAKQRQIS-VKSHFSRE", "MRTAYIAKqQRQISFVKSHFSKQ", "MKSAYIAKQRQVSFVKAHFSRQ")
NUM_RECYCLES = 2
NOISE_LEVEL = 10.0
KEY = jax.random.key(0)


def _helix_cif(sequence: str, path: Path) -> None:
    """An ideal α-helix (backbone and Cβ) for `sequence`, as chain A of an mmCIF."""
    import gemmi

    radius = {"N": 1.55, "CA": 2.3, "C": 1.61}
    phase = {"N": -0.49, "CA": 0.0, "C": 0.42}
    rise = {"N": -0.85, "CA": 0.0, "C": 0.95}
    chain = gemmi.Chain("A")
    for i, aa in enumerate(sequence):
        angle = np.deg2rad(100.0) * i
        pos = {
            n: np.array(
                [
                    radius[n] * np.cos(angle + phase[n]),
                    radius[n] * np.sin(angle + phase[n]),
                    1.5 * i + rise[n],
                ]
            )
            for n in radius
        }
        b, c = pos["CA"] - pos["N"], pos["C"] - pos["CA"]
        pos["CB"] = pos["CA"] - 0.58273431 * np.cross(b, c) + 0.56802827 * b - 0.54067466 * c
        residue = gemmi.Residue()
        residue.name = gemmi.expand_one_letter(aa, gemmi.ResidueKind.AA)
        residue.seqid = gemmi.SeqId(i + 1, " ")
        residue.entity_type = gemmi.EntityType.Polymer
        residue.subchain = "A"
        for name, x in pos.items():
            if name == "CB" and aa == "G":
                continue
            atom = gemmi.Atom()
            atom.name, atom.element = name, gemmi.Element(name[0])
            atom.pos, atom.occ = gemmi.Position(*x), 1.0
            residue.add_atom(atom)
        chain.add_residue(residue)
    model = gemmi.Model("1")
    model.add_chain(chain)
    structure = gemmi.Structure()
    structure.add_model(model)
    entity = gemmi.Entity("1")
    entity.entity_type = gemmi.EntityType.Polymer
    entity.polymer_type = gemmi.PolymerType.PeptideL
    entity.subchains = ["A"]
    entity.full_sequence = [r.name for r in chain]
    structure.entities = gemmi.EntityList([entity])
    structure.setup_entities()
    structure.assign_label_seq_id()
    structure.make_mmcif_document().write_file(str(path))


def _featurize(directory: Path, max_tokens: int | None = None, max_atoms: int | None = None):
    """Boltz-2's inference features (batched torch tensors), padded to `max_tokens` and
    `max_atoms` if given."""
    import boltz.main as bm

    a, b = SEQUENCES
    _helix_cif(a, directory / "template.cif")
    (directory / "a.a3m").write_text("".join(f">{i}\n{s}\n" for i, s in enumerate((a, *MSA))))
    (directory / "input.yaml").write_text(
        f"""version: 1
sequences:
  - protein:
      id: [A]
      sequence: {a}
      msa: {directory / "a.a3m"}
  - protein:
      id: [B]
      sequence: {b}
      msa: empty
      cyclic: true
constraints:
  - pocket:
      binder: B
      contacts: [[A, 5], [A, 9]]
      max_distance: 8
  - contact:
      token1: [A, 15]
      token2: [B, 4]
      max_distance: 10
templates:
  - cif: {directory / "template.cif"}
    chain_id: A
"""
    )
    bm.process_inputs(
        data=bm.check_inputs(directory / "input.yaml"),
        out_dir=directory,
        ccd_path=CACHE / "ccd.pkl",
        mol_dir=CACHE / "mols",
        msa_server_url="",
        msa_pairing_strategy="greedy",
        boltz2=True,
    )
    processed = directory / "processed"
    data = bm.Boltz2InferenceDataModule(
        manifest=bm.Manifest.load(processed / "manifest.json"),
        target_dir=processed / "structures",
        msa_dir=processed / "msa",
        mol_dir=CACHE / "mols",
        num_workers=0,
        constraints_dir=processed / "constraints",
        template_dir=processed / "templates",
    )
    loader = data.predict_dataloader()
    if max_tokens is not None:
        featurizer = loader.dataset.featurizer
        process = featurizer.process
        featurizer.process = lambda *args, **kwargs: process(
            *args, **{**kwargs, "max_tokens": max_tokens, "max_atoms": max_atoms}
        )
    batch = next(iter(loader))
    feats = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
    assert feats["template_mask"].any() and feats["msa"].shape[1] == len(MSA) + 1
    assert (feats["contact_conditioning"][..., 2:].sum() > 0) and feats["cyclic_period"].any()
    return feats


def _reference():
    """The released model as `jfold.pretrained` builds it (as `boltz predict` does)."""
    from jfold import pretrained

    return pretrained.upstream("boltz2", checkpoint_path=CACHE / "boltz2_conf.ckpt")


def _trunk(model, features):
    inputs = model.embed(features, key=KEY)
    state = model.trunk.init_state(inputs)
    for _ in range(NUM_RECYCLES):
        state = model.trunk(features, inputs, state, key=KEY)
    return inputs, state


@pytest.mark.parametrize("padding", [0, 9], ids=["unpadded", "padded"])
def test_boltz2(tmp_path, padding):
    """Parity on the real tokens and atoms; `padding` tokens (and 64 atoms) appended by
    Boltz's own featurizer for our side, with garbage in the padded trunk state."""
    from jfold.convert import boltz2 as convert
    from jfold.models import boltz2
    from jfold.slots import TrunkInputs, TrunkState

    if not (CACHE / "ccd.pkl").exists():
        import boltz.main as bm

        bm.download_boltz2(CACHE)
    feats = _featurize(tmp_path)
    num_tokens, num_atoms = feats["res_type"].shape[1], feats["ref_pos"].shape[1]
    padded = feats
    if padding:
        padded = _featurize(tmp_path, num_tokens + padding, num_atoms + 64)
        # Reference conformers are randomly placed per featurization; use the same ones.
        padded["ref_pos"][:, :num_atoms] = feats["ref_pos"]
    reference = _reference()
    atom_mask = feats["atom_pad_mask"][0].bool().numpy()
    x_noisy = 16 * np.random.default_rng(0).standard_normal((num_atoms, 3), dtype=np.float32)
    with torch.no_grad():
        s_inputs = reference.input_embedder(feats)
        # The trunk (and distogram) alone: no sampling, no confidence on the input coords.
        reference.skip_run_structure = True
        reference.confidence_prediction = False
        out = reference(feats, recycling_steps=NUM_RECYCLES - 1)
        s, z, distogram = out["s"], out["z"], out["pdistogram"][:, :, :, 0]
        q, c, to_keys, enc_bias, dec_bias, token_bias = reference.diffusion_conditioning(
            s_trunk=s, z_trunk=z, relative_position_encoding=reference.rel_pos(feats), feats=feats
        )
        denoised = reference.structure_module.preconditioned_network_forward(
            torch.from_numpy(x_noisy)[None],
            NOISE_LEVEL,
            network_condition_kwargs={
                "s_inputs": s_inputs,
                "s_trunk": s,
                "feats": feats,
                "diffusion_conditioning": {
                    "q": q,
                    "c": c,
                    "to_keys": to_keys,
                    "atom_enc_bias": enc_bias,
                    "atom_dec_bias": dec_bias,
                    "token_trans_bias": token_bias,
                },
            },
        )
        conf = reference.confidence_module(
            s_inputs=s_inputs,
            s=s,
            z=z,
            x_pred=denoised,
            feats=feats,
            pred_distogram_logits=distogram,
        )

    model = convert.structure_model(reference)
    model = assert_round_trip(model)
    assert_schedule(model.sampling, reference.structure_module.sample_schedule(STEPS))
    features = boltz2.features({k: v[0].numpy() for k, v in padded.items()})
    inputs, state = eqx.filter_jit(_trunk)(model, features)
    n = num_tokens
    errors = {
        "s_inputs": rel_err(inputs.s_inputs[:n], s_inputs[0]),
        "s": rel_err(state.s[:n], s[0]),
        "z": rel_err(state.z[:n, :n], z[0]),
    }

    # Later stages from the reference trunk, so errors do not compound.
    garbage = lambda x, axes: jnp.array(pad(x[0].numpy(), padding, axes))
    ref_state = TrunkState(s=garbage(s, (0,)), z=garbage(z, (0, 1)))
    ref_inputs = TrunkInputs(
        s_inputs=garbage(s_inputs, (0,)), s_init=inputs.s_init, z_init=inputs.z_init
    )
    context = model.diffusion.prepare(features, ref_inputs, ref_state)
    x_in = jnp.array(pad(x_noisy, 64, (0,)) if padding else x_noisy)
    ours = model.diffusion(context, x_in, jnp.array(NOISE_LEVEL))
    errors["denoised"] = rel_err(ours[:num_atoms][atom_mask], denoised[0][atom_mask])
    coords = denoised[0].numpy()
    coords = jnp.array(pad(coords, 64, (0,)) if padding else coords)
    logits = model.confidence_logits(features, ref_inputs, ref_state, coords, key=KEY)
    errors["pae"] = rel_err(logits.pae[:n, :n], conf["pae_logits"][0])
    errors["pde"] = rel_err(logits.pde[:n, :n], conf["pde_logits"][0])
    errors["plddt"] = rel_err(logits.plddt[:n], conf["plddt_logits"][0])
    errors["resolved"] = rel_err(logits.resolved[:n], conf["resolved_logits"][0])
    errors["distogram"] = rel_err(model.distogram(ref_state.z)[:n, :n], distogram[0])
    print(errors)

    assert errors["s_inputs"] < 1e-4
    assert errors["s"] < 1e-3
    assert errors["z"] < 1e-3
    assert errors["denoised"] < 1e-4
    for name in ("pae", "pde", "plddt", "resolved"):
        assert errors[name] < 1e-3
    assert errors["distogram"] < 1e-4
