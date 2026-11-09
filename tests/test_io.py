"""Saving and loading models with `jfold.io`: no pickle, arrays readable with numpy."""

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference import assert_round_trip, randomize
from safetensors.numpy import load_file

from jfold import io
from jfold.convert import boltz
from jfold.sampling import UpstreamSampler, masked_centering, rigid_align_stop_gradient

T, C = 6, 32


def _stack():
    from boltz.model.layers.pairformer import PairformerModule

    torch_stack = PairformerModule(
        16,
        C,
        num_blocks=2,
        num_heads=2,
        dropout=0.25,
        pairwise_head_width=4,
        pairwise_num_heads=2,
        v2=True,
    )
    return boltz.pairformer_stack(randomize(torch_stack, seed=0))


def _holder():
    """The kinds of value converted models hold, in one tree."""
    return {
        "stack": eqx.nn.inference_mode(_stack(), False),
        "sampling": UpstreamSampler(
            sigma_data=16.0,
            s_max=160.0,
            s_min=4e-4,
            rho=7.0,
            num_steps=200,
            end="append_zero",
            gamma_0=0.8,
            gamma_min=1.0,
            noise_scale=1.003,
            step_scale=1.5,
            center=masked_centering,
            augment=True,
            augment_after_noise=False,
            max_sigma=None,
            align=rigid_align_stop_gradient,
        ),
        "optional": None,
        "identity": eqx.nn.Identity(),
        "half": jnp.arange(4, dtype=jnp.bfloat16),
        "shape": (2, 3),
        "flag": True,
    }


def test_round_trip(tmp_path):
    """Structure, static fields, dropout state and arrays (bfloat16 included) survive,
    and the loaded stack computes the same thing."""
    model = _holder()
    loaded = assert_round_trip(model, tmp_path / "model")
    s, z = jnp.ones((T, 16)), jax.random.normal(jax.random.key(0), (T, T, C))
    mask, key = jnp.ones(T, bool), jax.random.key(1)
    run = eqx.filter_jit(lambda m, z: m(s, z, mask, key=key)[1])
    assert jnp.array_equal(run(model["stack"], z), run(loaded["stack"], z))
    assert loaded["sampling"].settings().align is rigid_align_stop_gradient


def test_files_are_plain(tmp_path):
    """The structure is JSON naming public classes and functions; the weights load with
    numpy alone, keyed by their path in the model."""
    io.save(_holder(), tmp_path, metadata={"source": "test"})
    structure = json.loads((tmp_path / "model.json").read_text())
    assert structure["format_version"] == io.FORMAT_VERSION
    assert structure["metadata"] == {"source": "test"}
    fields = structure["model"]["dict"]
    assert fields["identity"]["module"] == "equinox.nn:Identity"
    assert fields["sampling"]["fields"]["align"] == {
        "function": "jfold.sampling.core:rigid_align_stop_gradient"
    }
    weights = load_file(str(tmp_path / "weights.safetensors"))
    assert isinstance(weights["half"], np.ndarray)
    assert any(k.startswith("stack.layers.") for k in weights)


def _edit(path, change):
    structure = json.loads((path / "model.json").read_text())
    change(structure)
    (path / "model.json").write_text(json.dumps(structure))


def test_mismatches_are_named(tmp_path):
    io.save(_holder(), tmp_path)

    def rename(structure):
        fields = structure["model"]["dict"]["sampling"]["fields"]
        fields["sigma"] = fields.pop("sigma_data")

    _edit(tmp_path, rename)
    with pytest.raises(ValueError, match=r"missing \['sigma_data'\], unknown \['sigma'\]"):
        io.load(tmp_path)


def test_moved_classes_load_from_their_old_names(tmp_path):
    """A file naming a class where it used to live loads it from where it is now."""
    io.save(_holder(), tmp_path)
    old = "jfold.sampling.config:SamplingConfig"
    assert io.MOVED[old] == "jfold.sampling.config:UpstreamSampler"
    _edit(tmp_path, lambda s: s["model"]["dict"]["sampling"].update(module=old))
    assert isinstance(io.load(tmp_path)["sampling"], UpstreamSampler)
    for name in io.MOVED.values():
        io._resolve(name)  # every new name exists


def test_refuses_foreign_names_and_other_versions(tmp_path):
    io.save(_holder(), tmp_path)
    _edit(tmp_path, lambda s: s["model"]["dict"]["identity"].update(module="os:system"))
    with pytest.raises(ValueError, match="refusing"):
        io.load(tmp_path)
    _edit(tmp_path, lambda s: s.update(format_version=0))
    with pytest.raises(ValueError, match="format version"):
        io.load(tmp_path)


def test_rejects_unsaveable_values(tmp_path):
    with pytest.raises(TypeError, match="cannot save"):
        io.save({**_holder(), "shape": lambda x: x}, tmp_path)


def test_provenance(tmp_path):
    """The standard metadata names the checkpoint by source and hash, and the upstream
    package by its installed version; it is saved and read back without the weights."""
    import hashlib
    import importlib.metadata

    checkpoint = tmp_path / "weights.ckpt"
    checkpoint.write_bytes(b"weights")
    metadata = io.provenance(
        "boltz2", checkpoint=checkpoint, source="hf:org/repo/weights.ckpt", upstream="boltz"
    )
    assert metadata["checkpoint"] == {
        "source": "hf:org/repo/weights.ckpt",
        "sha256": hashlib.sha256(b"weights").hexdigest(),
    }
    assert metadata["upstream"] == {
        "package": "boltz",
        "version": importlib.metadata.version("boltz"),
    }
    io.save(_holder(), tmp_path / "model", metadata=metadata)
    assert io.read_structure(tmp_path / "model")["metadata"] == metadata

    # A checkpoint directory hashes its files with their names.
    (tmp_path / "shards").mkdir()
    (tmp_path / "shards" / "a").write_bytes(b"1")
    first = io.provenance("m", checkpoint=tmp_path / "shards", source="s", upstream="boltz")
    (tmp_path / "shards" / "a").rename(tmp_path / "shards" / "b")
    second = io.provenance("m", checkpoint=tmp_path / "shards", source="s", upstream="boltz")
    assert first["checkpoint"]["sha256"] != second["checkpoint"]["sha256"]
