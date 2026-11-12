"""AtlasFold releases, loaded as `atlasfold.pretrained.load_model` does.

`atlasfold` (monomers) and `atlasfold_m` (complexes, with templates) share the AtlasLM-3B
language model, released once as `atlaslm_3b` and attached at `trunk.atlaslm` on loading.
"""

from functools import partial
from pathlib import Path

import equinox as eqx

from jfold.pretrained import Release

# The licence of the `atlasfold` package and model cards, saved with converted copies.
LICENSE = '''MIT License

Copyright (c) 2026 Seonghwan Seo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

# Model name -> (Hugging Face repository, checkpoint file, multimer).
REPOS: dict[str, tuple[str, str, bool]] = {
    "atlasfold": ("SeonghwanSeo/atlasfold-260703", "weights/atlasfold-260703.pth", False),
    "atlasfold_m": ("SeonghwanSeo/atlasfold-m-260725", "weights/atlasfold-m-260725.pth", True),
}
LM_REPO = ("SeonghwanSeo/atlaslm-3b-base", "weights/atlaslm_3b_base.pth")
LM = "atlaslm_3b"
# The upstream runners' AtlasLM masking rates and sampling step counts.
MLM_RATE = {"atlasfold": 0.15, "atlasfold_m": 0.20}
NUM_STEPS = {"atlasfold": 20, "atlasfold_m": 200}


def _fetch(repo: str, filename: str, directory: Path) -> None:
    from huggingface_hub import hf_hub_download

    (directory / Path(filename).name).symlink_to(hf_hub_download(repo, filename))


def _build_lm(checkpoint: Path):
    from atlaslm.model import AtlasLM

    return AtlasLM.from_pretrained(checkpoint, device="cpu")


def _build(multimer: bool, checkpoint: Path):
    import copy

    from atlasfold.configs.atlasfold import monomer_config
    from atlasfold.configs.atlasfold_multimer import multimer_config
    from atlasfold.model import AtlasFold, AtlasFold_Multimer

    from jfold import pretrained

    lm = _build_lm(pretrained.checkpoint(LM))
    cls, config = (AtlasFold_Multimer, multimer_config) if multimer else (AtlasFold, monomer_config)
    model = cls.from_pretrained(checkpoint, config=copy.deepcopy(config), lm=lm, device="cpu")
    model.set_kernel_backend("torch")
    return model


def _convert(name: str, m) -> eqx.Module:
    from atlasfold.model.network.diffusion_head import SamplingConfig

    from jfold.convert import atlasfold as convert

    return convert.structure_model(
        m, SamplingConfig(), mlm_rate=MLM_RATE[name], num_steps=NUM_STEPS[name]
    )


def _convert_lm(m) -> eqx.Module:
    from jfold.convert import atlasfold as convert

    return convert.atlaslm(m)


def _detach_lm(model: eqx.Module) -> eqx.Module:
    return eqx.tree_at(lambda m: m.trunk.atlaslm, model, None, is_leaf=lambda x: x is None)


def _attach_lm(model: eqx.Module, load) -> eqx.Module:
    return eqx.tree_at(lambda m: m.trunk.atlaslm, model, load(LM), is_leaf=lambda x: x is None)


RELEASES = {
    name: Release(
        upstream="atlasfold",
        source=f"https://huggingface.co/{repo}/blob/main/{filename}",
        filename=Path(filename).name,
        fetch=partial(_fetch, repo, filename),
        build=partial(_build, multimer),
        convert=partial(_convert, name),
        assemble=_attach_lm,
        detach=_detach_lm,
        license=LICENSE,
    )
    for name, (repo, filename, multimer) in REPOS.items()
} | {
    LM: Release(
        upstream="atlasfold",
        source=f"https://huggingface.co/{LM_REPO[0]}/blob/main/{LM_REPO[1]}",
        filename=Path(LM_REPO[1]).name,
        fetch=partial(_fetch, *LM_REPO),
        build=_build_lm,
        convert=_convert_lm,
        license=LICENSE,
    )
}
