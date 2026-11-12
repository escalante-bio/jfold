"""ESMFold2 releases, loaded as `esm`'s `from_pretrained` does.

Each model's ESMC language model (`esmc_6b`, `esmc_300m` or `esmc_600m`) is released once
and attached at `trunk_init.esmc` on loading. The `_confidence` releases add the confidence
heads that Synthyra trained for FastPLMs to Biohub's 300M and 600M checkpoints. To reuse a
Hugging Face cache, pass `snapshot(name)` as `checkpoint_path`.
"""

from functools import partial
from pathlib import Path

import equinox as eqx

from jfold.pretrained import Release

# Model name -> (Hugging Face repository, experimental type).
REPOS: dict[str, tuple[str, bool]] = {
    "esmfold2": ("biohub/ESMFold2", False),
    "esmfold2_fast": ("biohub/ESMFold2-Fast", False),
    "esmfold2_experimental": ("biohub/ESMFold2-Experimental", True),
    "esmfold2_experimental_fast": ("biohub/ESMFold2-Experimental-Fast", True),
    "esmfold2_experimental_cutoff2025": ("biohub/ESMFold2-Experimental-Cutoff2025", True),
    "esmfold2_experimental_fast_cutoff2025": (
        "biohub/ESMFold2-Experimental-Fast-Cutoff2025",
        True,
    ),
    # Final (step 1500k) checkpoints of the scaling runs on ESMC-300M, -600M and -6B.
    "esmfold2_experimental_fast_base300m": (
        "biohub/ESMFold2-Experimental-Fast-base300M-step1500k",
        True,
    ),
    "esmfold2_experimental_fast_base600m": (
        "biohub/ESMFold2-Experimental-Fast-base600M-step1500k",
        True,
    ),
    "esmfold2_experimental_fast_base6b": (
        "biohub/ESMFold2-Experimental-Fast-base6B-step1500k",
        True,
    ),
}
# Model name -> (base checkpoint name, confidence head repository, revision).
CONFIDENCE: dict[str, tuple[str, str, str]] = {
    "esmfold2_experimental_fast_base300m_confidence": (
        "esmfold2_experimental_fast_base300m",
        "Synthyra/ESMFold2-300",
        "b76d9e3063571de5d43825f26f07bd4da2e906f8",
    ),
    "esmfold2_experimental_fast_base600m_confidence": (
        "esmfold2_experimental_fast_base600m",
        "Synthyra/ESMFold2-600",
        "38fa0be7397a485afd4a752202447693098f3d50",
    ),
}
# The ESMC release each model uses (default `esmc_6b`), and the checkpoint each comes from.
ESMC: dict[str, str] = {
    "esmfold2_experimental_fast_base300m": "esmc_300m",
    "esmfold2_experimental_fast_base600m": "esmc_600m",
}
ESMC |= {name: ESMC[base] for name, (base, _, _) in CONFIDENCE.items()}
ESMC_SOURCES: dict[str, str] = {
    "esmc_6b": "esmfold2",
    "esmc_300m": "esmfold2_experimental_fast_base300m",
    "esmc_600m": "esmfold2_experimental_fast_base600m",
}
# ESMC with its masked-LM head: name -> (repository, shared backbone release or None). The
# 300M and 600M keep float32 backbones, since `esmc_300m` and `esmc_600m` are bfloat16.
MASKED_LM: dict[str, tuple[str, str | None]] = {
    "esmc_300m_mlm": ("Biohub/ESMC-300M", None),
    "esmc_600m_mlm": ("Biohub/ESMC-600M", None),
    "esmc_6b_mlm": ("Biohub/ESMC-6B", "esmc_6b"),
}


def snapshot(name: str, cache_dir: str | Path | None = None) -> Path:
    """Return the checkpoint of `name` in a Hugging Face cache, downloading it if missing."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    repo = REPOS[CONFIDENCE[name][0] if name in CONFIDENCE else name][0]
    try:
        return Path(snapshot_download(repo, cache_dir=cache_dir, local_files_only=True))
    except LocalEntryNotFoundError:
        return Path(snapshot_download(repo, cache_dir=cache_dir))


def _fetch(repo: str, directory: Path) -> None:
    from huggingface_hub import snapshot_download

    (directory / repo.split("/")[1]).symlink_to(snapshot_download(repo))


def _build(experimental: bool, checkpoint: Path):
    import torch

    if experimental:
        from esm.models.esmfold2.experimental import EsmFold2ExperimentalModel

        model = EsmFold2ExperimentalModel.from_pretrained(
            str(checkpoint), device="cpu", dtype=torch.float32
        )
    else:
        from esm.models.esmfold2 import EsmFold2Model

        model = EsmFold2Model.from_pretrained(
            str(checkpoint), device="cpu", dtype=torch.float32, esmc_precision="fp32"
        )
    return model.eval()


def _confidence_file(repo: str, revision: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, "model.safetensors", revision=revision)


def _fetch_with_confidence(repo: str, head_repo: str, revision: str, directory: Path) -> None:
    _fetch(repo, directory)
    _confidence_file(head_repo, revision)


def _build_with_confidence(head_repo: str, revision: str, checkpoint: Path):
    from esm.models.esmfold2.experimental import ConfidenceHead
    from safetensors import safe_open

    model = _build(True, checkpoint)
    if model.confidence_head is not None:
        raise ValueError(f"{checkpoint} has a confidence head of its own")
    model.config.confidence_head.enabled = True
    head = ConfidenceHead(model.config)
    prefix = "confidence_head."
    with safe_open(_confidence_file(head_repo, revision), "pt") as file:
        state = {
            key.removeprefix(prefix): file.get_tensor(key).float()
            for key in file.keys()  # noqa: SIM118
            if key.startswith(prefix)
        }
    head.load_state_dict(state, strict=True)
    model.confidence_head = head.eval()
    return model


def _convert(experimental: bool, m) -> eqx.Module:
    """Convert without ESMC, on the CPU so the GPU never holds two copies."""
    import jax

    from jfold.convert import esmfold2 as convert

    with jax.default_device(jax.devices("cpu")[0]):
        return (convert.esmfold2_experimental if experimental else convert.esmfold2)(m)


def _convert_esmc(m) -> eqx.Module:
    import jax

    from jfold.convert import esmfold2 as convert

    with jax.default_device(jax.devices("cpu")[0]):
        return convert.esmc(m.esmc.float())


def _build_masked_lm(checkpoint: Path):
    import torch
    from esm.models.esmc import EsmcForMaskedLM

    return EsmcForMaskedLM.from_pretrained(
        str(checkpoint), device="cpu", dtype=torch.float32
    ).eval()


def _convert_masked_lm(m) -> eqx.Module:
    import jax

    from jfold.convert import esmfold2 as convert

    with jax.default_device(jax.devices("cpu")[0]):
        return convert.esmc_for_masked_lm(m)


def _detach_masked_lm_esmc(model: eqx.Module) -> eqx.Module:
    return eqx.tree_at(lambda m: m.esmc, model, None, is_leaf=lambda x: x is None)


def _attach_masked_lm_esmc(esmc_name: str, model: eqx.Module, load) -> eqx.Module:
    return eqx.tree_at(lambda m: m.esmc, model, load(esmc_name), is_leaf=lambda x: x is None)


def _detach_esmc(model: eqx.Module) -> eqx.Module:
    return eqx.tree_at(lambda m: m.trunk_init.esmc, model, None, is_leaf=lambda x: x is None)


def _attach_esmc(experimental: bool, esmc_name: str, model: eqx.Module, load) -> eqx.Module:
    """Attach ESMC `esmc_name`, rounded to bfloat16 values for the experimental type, which
    upstream runs in bfloat16."""
    import jax
    import jax.numpy as jnp

    esmc = load(esmc_name)
    if experimental:
        esmc = jax.tree.map(
            lambda x: x.astype(jnp.bfloat16).astype(x.dtype) if eqx.is_inexact_array(x) else x,
            esmc,
        )
    return eqx.tree_at(lambda m: m.trunk_init.esmc, model, esmc, is_leaf=lambda x: x is None)


# The licence of the `esm` package and model cards, saved with converted copies.
LICENSE = """**License (MIT)**

Copyright 2026 Chan Zuckerberg Biohub, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# Appended to `LICENSE` for models with a separately trained confidence head.
CONFIDENCE_LICENSE = """
The confidence head's weights are from https://huggingface.co/{repo} (revision
{revision}), trained by Synthyra for FastPLMs (https://github.com/Synthyra/FastPLMs) and
released there under the MIT licence.
"""

RELEASES = (
    {
        name: Release(
            upstream="esm",
            source=f"https://huggingface.co/{repo}",
            filename=repo.split("/")[1],
            fetch=partial(_fetch, repo),
            build=partial(_build, experimental),
            convert=partial(_convert, experimental),
            assemble=partial(_attach_esmc, experimental, ESMC.get(name, "esmc_6b")),
            detach=_detach_esmc,
            license=LICENSE,
        )
        for name, (repo, experimental) in REPOS.items()
    }
    | {
        name: Release(
            upstream="esm",
            source=(
                f"https://huggingface.co/{REPOS[base][0]}, with the confidence head of "
                f"https://huggingface.co/{head_repo} (revision {revision})"
            ),
            filename=REPOS[base][0].split("/")[1],
            fetch=partial(_fetch_with_confidence, REPOS[base][0], head_repo, revision),
            build=partial(_build_with_confidence, head_repo, revision),
            convert=partial(_convert, True),
            assemble=partial(_attach_esmc, True, ESMC[name]),
            detach=_detach_esmc,
            license=LICENSE + CONFIDENCE_LICENSE.format(repo=head_repo, revision=revision),
        )
        for name, (base, head_repo, revision) in CONFIDENCE.items()
    }
    | {
        esmc: Release(
            upstream="esm",
            source=f"https://huggingface.co/{REPOS[fold][0]} (its ESMC)",
            filename=REPOS[fold][0].split("/")[1],
            fetch=partial(_fetch, REPOS[fold][0]),
            build=partial(_build, REPOS[fold][1]),
            convert=_convert_esmc,
            license=LICENSE,
        )
        for esmc, fold in ESMC_SOURCES.items()
    }
    | {
        name: Release(
            upstream="esm",
            source=f"https://huggingface.co/{repo}",
            filename=repo.split("/")[1],
            fetch=partial(_fetch, repo),
            build=_build_masked_lm,
            convert=_convert_masked_lm,
            assemble=None if shared is None else partial(_attach_masked_lm_esmc, shared),
            detach=None if shared is None else _detach_masked_lm_esmc,
            license=LICENSE,
        )
        for name, (repo, shared) in MASKED_LM.items()
    }
)
