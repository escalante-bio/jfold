"""`jfold.pretrained.load`: converts a release once, then loads the saved copy."""

import json

import jax.numpy as jnp
import pytest

from jfold import pretrained
from jfold.nn import Linear


def _toy(calls):
    def fetch(directory):
        (directory / "toy.ckpt").write_bytes(b"weights")

    def build(checkpoint):
        calls.append(checkpoint)
        return checkpoint

    return pretrained.Release(
        upstream="equinox",
        source="test",
        filename="toy.ckpt",
        fetch=fetch,
        build=build,
        convert=lambda _: Linear(weight=jnp.ones((2, 3))),
    )


def test_converts_once_then_loads_the_copy(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(pretrained.RELEASES, "toy", _toy(calls))
    model = pretrained.load("toy", cache=tmp_path, hub=False)
    assert jnp.array_equal(model.weight, jnp.ones((2, 3))) and len(calls) == 1
    pretrained.load("toy", cache=tmp_path, hub=False)
    assert len(calls) == 1  # the saved copy, not a second conversion
    pretrained.load("toy", cache=tmp_path, refresh=True, hub=False)
    assert len(calls) == 2

    structure = json.loads((tmp_path / "toy" / "jfold" / "model.json").read_text())
    assert structure["metadata"]["model"] == "toy"
    assert structure["metadata"]["checkpoint"]["source"] == "test"


def test_unknown_model():
    with pytest.raises(KeyError, match="released"):
        pretrained.load("nope")


def test_loads_from_the_hub_without_converting(tmp_path, monkeypatch):
    """A converted copy on the hub is downloaded and used as it is."""
    import huggingface_hub

    from jfold import io

    calls = []
    monkeypatch.setitem(pretrained.RELEASES, "toy", _toy(calls))
    published = tmp_path / "published"
    io.save(Linear(weight=2 * jnp.ones((2, 3))), published / "toy")

    def snapshot_download(repo, *, allow_patterns, local_dir):
        assert repo == pretrained.HUB_REPO and allow_patterns == ["toy/*"]
        import shutil

        shutil.copytree(published / "toy", local_dir / "toy")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    model = pretrained.load("toy", cache=tmp_path / "cache")
    assert jnp.array_equal(model.weight, 2 * jnp.ones((2, 3))) and not calls
    assert (tmp_path / "cache" / "toy" / "jfold" / "model.json").exists()


def test_assemble_completes_the_model(tmp_path, monkeypatch):
    """`assemble` builds on other releases, each loaded (and converted) once."""
    import dataclasses

    calls = []
    part = _toy(calls)
    whole = dataclasses.replace(_toy(calls), assemble=lambda model, load: (model, load("part")))
    monkeypatch.setitem(pretrained.RELEASES, "part", part)
    monkeypatch.setitem(pretrained.RELEASES, "whole", whole)
    model, attached = pretrained.load("whole", cache=tmp_path, hub=False)
    assert jnp.array_equal(attached.weight, model.weight) and len(calls) == 2


def test_unreadable_copy_is_fetched_again(tmp_path, monkeypatch):
    """A cached copy an older jfold saved (here: a class that no longer exists) is replaced
    by the hub's."""
    import huggingface_hub

    from jfold import io

    calls = []
    monkeypatch.setitem(pretrained.RELEASES, "toy", _toy(calls))
    cached = tmp_path / "cache" / "toy" / "jfold"
    io.save(Linear(weight=jnp.ones((2, 3))), cached)
    structure = json.loads((cached / "model.json").read_text())
    structure["model"]["module"] = "jfold.nn:Gone"
    (cached / "model.json").write_text(json.dumps(structure))
    published = tmp_path / "published"
    io.save(Linear(weight=2 * jnp.ones((2, 3))), published / "toy")

    def snapshot_download(repo, *, allow_patterns, local_dir):
        import shutil

        shutil.copytree(published / "toy", local_dir / "toy")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    with pytest.warns(UserWarning, match="no longer exists"):
        model = pretrained.load("toy", cache=tmp_path / "cache")
    assert jnp.array_equal(model.weight, 2 * jnp.ones((2, 3))) and not calls
