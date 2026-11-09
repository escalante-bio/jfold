"""ESMFold2's Hugging Face checkpoints, as `esm`'s `from_pretrained` loads them for
inference, each converted to a `jfold.model.StructureModel` whose trunk init holds its ESMC
language model (`trunk_init.esmc`). Each ESMC (`esmc_6b`, `esmc_300m`, `esmc_600m`) is a
release of its own, stored once: a checkpoint's copy is saved without it and gets it back
on loading.

- Release type (`EsmFold2Model`): `esmfold2` (`biohub/ESMFold2`, with an MSA encoder) and
  `esmfold2_fast` (`biohub/ESMFold2-Fast`). ESMC-6B is bundled in the checkpoint; it is
  loaded in float32 (`esmc_precision="fp32"`).
- Experimental type (`EsmFold2ExperimentalModel`): `esmfold2_experimental`,
  `esmfold2_experimental_fast` and their `_cutoff2025` variants. ESMC comes from its own
  repository (`config.esmc_id`, through the Hugging Face cache); upstream always loads it in
  bfloat16, so its converted weights are bfloat16 values.

The checkpoint is the repository's snapshot directory. Pass `checkpoint_path` to reuse a
Hugging Face cache (`snapshot`); otherwise it is downloaded into jfold's cache."""

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
    # The scaling sweep's final step (step 1500k), on ESMC-300M, -600M and -6B.
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
# The shared ESMC each checkpoint reads (`esmc_6b` unless listed). Each is stored once and
# attached on loading. Its source: the checkpoint whose own ESMC it is.
ESMC: dict[str, str] = {
    "esmfold2_experimental_fast_base300m": "esmc_300m",
    "esmfold2_experimental_fast_base600m": "esmc_600m",
}
ESMC_SOURCES: dict[str, str] = {
    "esmc_6b": "esmfold2",
    "esmc_300m": "esmfold2_experimental_fast_base300m",
    "esmc_600m": "esmfold2_experimental_fast_base600m",
}
# ESMC with its masked-language-model head, from Biohub's ESMC repositories: name ->
# (repository, the ESMC release it shares, if any). ESMC-6B's backbone is exactly
# `esmc_6b`, so only its head is stored; the 300M and 600M copies keep their own float32
# backbones (`esmc_300m` and `esmc_600m` are those rounded to bfloat16).
MASKED_LM: dict[str, tuple[str, str | None]] = {
    "esmc_300m_mlm": ("Biohub/ESMC-300M", None),
    "esmc_600m_mlm": ("Biohub/ESMC-600M", None),
    "esmc_6b_mlm": ("Biohub/ESMC-6B", "esmc_6b"),
}


def snapshot(name: str, cache_dir: str | Path | None = None) -> Path:
    """The checkpoint of `name` in a Hugging Face cache (`cache_dir`, default
    `HF_HUB_CACHE`), downloaded there if missing: a `checkpoint_path` for
    `jfold.pretrained.load` and `upstream`."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    repo = REPOS[name][0]
    try:
        return Path(snapshot_download(repo, cache_dir=cache_dir, local_files_only=True))
    except LocalEntryNotFoundError:
        return Path(snapshot_download(repo, cache_dir=cache_dir))


def _fetch(repo: str, directory: Path) -> None:
    """Into the Hugging Face cache (shared with `esm`'s own loading), linked from
    `directory`."""
    from huggingface_hub import snapshot_download

    (directory / repo.split("/")[1]).symlink_to(snapshot_download(repo))


def _build(experimental: bool, checkpoint: Path):
    """As `from_pretrained` loads it for inference, on the CPU in float32 (no kernels: the
    converter reads plain torch modules)."""
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


def _convert(experimental: bool, m) -> eqx.Module:
    """The structure model, without its ESMC (converted once, as its own release, and
    attached on loading). On the host, so a GPU never holds this copy and the loaded one
    together."""
    import jax

    from jfold.convert import esmfold2 as convert

    with jax.default_device(jax.devices("cpu")[0]):
        return (convert.esmfold2_experimental if experimental else convert.esmfold2)(m)


def _convert_esmc(m) -> eqx.Module:
    """A checkpoint's ESMC, in float32 (bfloat16 values for the experimental type's)."""
    import jax

    from jfold.convert import esmfold2 as convert

    with jax.default_device(jax.devices("cpu")[0]):
        return convert.esmc(m.esmc.float())


def _build_masked_lm(checkpoint: Path):
    """`EsmcForMaskedLM`, on the CPU in float32."""
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
    """A masked-LM ESMC without its backbone, which another release stores."""
    return eqx.tree_at(lambda m: m.esmc, model, None, is_leaf=lambda x: x is None)


def _attach_masked_lm_esmc(esmc_name: str, model: eqx.Module, load) -> eqx.Module:
    return eqx.tree_at(lambda m: m.esmc, model, load(esmc_name), is_leaf=lambda x: x is None)


def _detach_esmc(model: eqx.Module) -> eqx.Module:
    """`model` without its ESMC, as its copy is stored."""
    return eqx.tree_at(lambda m: m.trunk_init.esmc, model, None, is_leaf=lambda x: x is None)


def _attach_esmc(experimental: bool, esmc_name: str, model: eqx.Module, load) -> eqx.Module:
    """`model` with its shared ESMC (`esmc_name`): as stored for the release type; rounded
    to bfloat16 values for the experimental type, which upstream always runs in bfloat16
    (its ESMC, from the same weights, is exactly that rounding; ESMC-300M's and -600M's
    step-1500k checkpoints are their final weights so rounded)."""
    import jax
    import jax.numpy as jnp

    esmc = load(esmc_name)
    if experimental:
        esmc = jax.tree.map(
            lambda x: x.astype(jnp.bfloat16).astype(x.dtype) if eqx.is_inexact_array(x) else x,
            esmc,
        )
    return eqx.tree_at(lambda m: m.trunk_init.esmc, model, esmc, is_leaf=lambda x: x is None)


# The weights' licence (the `esm` package's and the model cards'), which copies carry.
LICENSE = '''**License (MIT)**

Copyright 2026 Chan Zuckerberg Biohub, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

RELEASES = {
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
} | {
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
} | {
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
