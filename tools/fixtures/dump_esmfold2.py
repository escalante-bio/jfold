"""Dump ESMFold2's upstream input features for the shared fixture systems.

Run from ~/jfold:

    uv run python tools/fixtures/dump_esmfold2.py

For each system in tests/featurize/data/systems.json this writes
tests/featurize/data/esmfold2_<system>.npz with

* every tensor `ESMFold2InputBuilder.prepare_input` returns, unbatched, in upstream's dtypes;
* the ESMC token inputs upstream's `compute_lm_hidden_states` builds from them, captured by
  running that function with a recording stand-in for ESMC (no weights):
  `lm_input_ids` [Lp], `lm_sequence_id` [Lp] and `lm_positions` [T] (each token's row in
  the packed ESMC input, -1 off-protein);
* what the model forward hands its input embedder, captured from the real forward on a
  weightless (meta-device) model: `embed_res_type` [T, 33], `embed_profile` [T, 33] and
  `embed_deletion_mean` [T] for the release model (`biohub/ESMFold2`), and
  `embed_profile_no_msa_features` / `embed_deletion_mean_no_msa_features` for an
  experimental -Fast checkpoint (`disable_msa_features`, `biohub/ESMFold2-Experimental-Fast`).

`lm_hidden_states` is not included (it needs ESMC's weights).

Featurization of standard-residue proteins draws no random numbers; `prepare_input` is
called with seed=0, and the script checks that seed=None and another seed give identical
arrays. The model-time randomness (MSA column masking and row subsampling, ESMC input
masking, LM dropout) is not part of these features.

The CCD pickle is `biohub/ESMFold2`'s ccd.pkl, resolved through the Hugging Face cache
unless ESMCFOLD_CCD_PATH is set.
"""

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import torch

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "featurize" / "data"
RELEASE = "biohub/ESMFold2"
EXPERIMENTAL_FAST = "biohub/ESMFold2-Experimental-Fast"


def chains_for(system: dict):
    from esm.utils.msa import MSA
    from esm.utils.structure.input_builder import ProteinInput

    chains = []
    for chain in system["chains"]:
        msa = None
        if "msa" in chain:
            # A precomputed a3m: insertions stripped from the stored sequences, their
            # counts kept in `MSA.deletions` (identical features either way).
            msa = MSA.from_a3m(str(FIXTURES / chain["msa"]), remove_insertions=True)
        chains.append(ProteinInput(id=chain["id"], sequence=chain["sequence"], msa=msa))
    return chains


def prepare(builder, system: dict, seed: int | None):
    from esm.utils.structure.input_builder import StructurePredictionInput

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "No MSA provided ..., using single sequence mode"
        batched, _ = builder.prepare_input(
            StructurePredictionInput(sequences=chains_for(system)), seed=seed
        )
    return batched


class _RecordingESMC:
    """Stands in for ESMC in `compute_lm_hidden_states`: records its inputs and returns
    hidden states that encode each packed position (position + 1)."""

    def __call__(self, input_ids, sequence_id, output_hidden_states=True):
        self.input_ids, self.sequence_id = input_ids.clone(), sequence_id.clone()
        positions = torch.arange(input_ids.shape[1], dtype=torch.float32) + 1
        hidden = positions[None, None, :, None].expand(2, input_ids.shape[0], -1, 1)

        class Out:
            hidden_states = hidden

        return Out()


def lm_inputs(batched) -> dict:
    from esm.models.esmfold2.layers import compute_lm_hidden_states

    esmc = _RecordingESMC()
    hidden = compute_lm_hidden_states(
        esmc,
        batched["input_ids"],
        batched["asym_id"],
        batched["residue_index"],
        batched["mol_type"],
        batched["token_attention_mask"].bool(),
        lm_mask_pct=0.0,  # every released config; > 0 masks residues at random
    )
    return {
        "lm_input_ids": esmc.input_ids[0].numpy(),
        "lm_sequence_id": esmc.sequence_id[0].numpy(),
        "lm_positions": (hidden[0, :, 0, 0] - 1).long().numpy(),
    }


class _Stop(Exception):
    pass


def weightless_model(repo: str):
    """The checkpoint's model class on the meta device, with its input embedder replaced
    by a recorder that stops the forward."""
    from esm.models.esmfold2 import EsmFold2ExperimentalModel, EsmFold2Model
    from esm.models.esmfold2.config import EsmFold2Config
    from huggingface_hub import snapshot_download

    config = EsmFold2Config.from_pretrained(snapshot_download(repo, local_files_only=True))
    cls = EsmFold2ExperimentalModel if config.type == "experimental" else EsmFold2Model
    with torch.device("meta"):
        model = cls(config)
    model.seen = {}

    def record(**kwargs):
        model.seen = kwargs
        raise _Stop

    model.inputs_embedder.forward = record
    return model


def embedder_inputs(model, batched) -> dict:
    try:
        model(**batched)
    except _Stop:
        pass
    seen = model.seen
    return {k: seen[k][0].numpy() for k in ("aatype", "profile", "deletion_mean")}


def main():
    from esm.models.esmfold2 import conformers
    from esm.models.esmfold2.processor import ESMFold2InputBuilder

    builder = ESMFold2InputBuilder()
    ccd = conformers.CCD_PICKLE_PATH
    if ccd is None or not Path(ccd).exists():
        from huggingface_hub import hf_hub_download

        ccd = hf_hub_download(repo_id="biohub/ESMFold2", filename="ccd.pkl")
    print(f"CCD {ccd} md5 {hashlib.md5(Path(ccd).read_bytes()).hexdigest()}")

    release, fast = weightless_model(RELEASE), weightless_model(EXPERIMENTAL_FAST)
    systems = json.loads((FIXTURES / "systems.json").read_text())
    for name, system in systems.items():
        batched = prepare(builder, system, seed=0)
        out = {k: v[0].numpy() for k, v in batched.items() if isinstance(v, torch.Tensor)}
        for seed in (None, 1):
            other = prepare(builder, system, seed=seed)
            for k, v in out.items():
                assert np.array_equal(v, other[k][0].numpy()), f"{name}: {k} depends on seed"
        out |= lm_inputs(batched)
        embedded = embedder_inputs(release, batched)
        out |= {
            "embed_res_type": embedded["aatype"],
            "embed_profile": embedded["profile"],
            "embed_deletion_mean": embedded["deletion_mean"],
        }
        embedded = embedder_inputs(fast, batched)
        out |= {
            "embed_profile_no_msa_features": embedded["profile"],
            "embed_deletion_mean_no_msa_features": embedded["deletion_mean"],
        }
        path = FIXTURES / f"esmfold2_{name}.npz"
        np.savez_compressed(path, **out)
        tokens, atoms = out["token_attention_mask"].shape[0], out["atom_attention_mask"].shape[0]
        rows = out["msa"].shape[0]
        print(f"{path.name}: {tokens} tokens, {atoms} atoms, {rows} MSA rows, {len(out)} keys")


if __name__ == "__main__":
    main()
