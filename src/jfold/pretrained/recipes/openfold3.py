"""OpenFold3 (`of3-p2-155k.pt`), loaded as `run_openfold predict` does."""

from pathlib import Path

import equinox as eqx

from jfold.pretrained import Release

# OpenFold3's default checkpoint, `DEFAULT_CHECKPOINT_NAME` in its parameter registry.
FILENAME = "of3-p2-155k.pt"


def _fetch(directory: Path) -> None:
    from jopenfold3._vendor.openfold3.entry_points.parameters import (
        DEFAULT_CHECKPOINT_NAME,
        download_model_parameters,
    )

    download_model_parameters(directory, DEFAULT_CHECKPOINT_NAME, skip_confirmation=True)


def _build(checkpoint: Path):
    """Load as `run_openfold predict` does, adding the version tensor the checkpoint lacks so
    loading can be strict."""
    from jopenfold3._vendor.openfold3.core.utils.checkpoint_loading_utils import (
        get_state_dict_from_checkpoint,
        load_checkpoint,
    )
    from jopenfold3._vendor.openfold3.projects.of3_all_atom.project_entry import (
        ModelUpdate,
        OF3ProjectEntry,
    )
    from jopenfold3._vendor.openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom

    config = OF3ProjectEntry().get_model_config_with_update(
        ModelUpdate(presets=["predict", "pae_enabled"], custom={})
    )
    config.settings.memory.eval.use_deepspeed_evo_attention = False
    runner = OpenFold3AllAtom(config)
    state_dict, _ = get_state_dict_from_checkpoint(
        load_checkpoint(checkpoint), init_from_ema_weights=True
    )
    state_dict.setdefault("model.version_tensor", runner.model.version_tensor)
    runner.load_state_dict(state_dict, strict=True)
    return runner.model.eval().cpu()


def _convert(model) -> eqx.Module:
    from jfold.convert import openfold3

    return openfold3.structure_model(model)


RELEASES = {
    "openfold3": Release(
        upstream="jopenfold3",  # Vendors openfold3.
        source="openfold3 download_model_parameters (of3-p2-155k.pt), EMA weights",
        filename=FILENAME,
        fetch=_fetch,
        build=_build,
        convert=_convert,
    ),
}
