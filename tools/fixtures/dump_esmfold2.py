"""Write ESMFold2's upstream features for the test systems to `tests/featurize/data/`.

Each file also holds the ESMC inputs and the input embedder's inputs, recorded
from upstream code run without weights.

    uv run python tools/fixtures/dump_esmfold2.py
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
            msa = MSA.from_a3m(str(FIXTURES / chain["msa"]), remove_insertions=True)
        chains.append(ProteinInput(id=chain["id"], sequence=chain["sequence"], msa=msa))
    return chains


def prepare(builder, system: dict, seed: int | None):
    from esm.utils.structure.input_builder import StructurePredictionInput

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Silence "No MSA provided" warnings.
        batched, _ = builder.prepare_input(
            StructurePredictionInput(sequences=chains_for(system)), seed=seed
        )
    return batched


class _RecordingESMC:
    """Stand-in for ESMC that records its inputs and returns each packed position plus one."""

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
        lm_mask_pct=0.0,  # As in every released config. Values > 0 mask at random.
    )
    return {
        "lm_input_ids": esmc.input_ids[0].numpy(),
        "lm_sequence_id": esmc.sequence_id[0].numpy(),
        "lm_positions": (hidden[0, :, 0, 0] - 1).long().numpy(),
    }


class _Stop(Exception):
    pass


def weightless_model(repo: str):
    """Build the model on the meta device, with an input embedder that records its inputs and
    stops."""
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
