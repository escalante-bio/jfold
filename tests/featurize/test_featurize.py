"""`jfold.featurize` views against the upstream featurizations they reproduce, on the shared
fixture systems (`tests/featurize/data/`, written by `tools/fixtures/dump_<model>.py`)."""

import json
from pathlib import Path

import numpy as np
import pytest

from jfold.featurize import Chain, System, Template, parse_a3m
from jfold.featurize.views import bioemu2

FIXTURES = Path(__file__).parent / "data"
SYSTEMS = json.loads((FIXTURES / "systems.json").read_text())
# Systems with templates (`tools/fixtures/make_templates.py`), for the views that take them.
TEMPLATE_SYSTEMS = json.loads((FIXTURES / "template_systems.json").read_text())


def system(name: str) -> System:
    chains = []
    for c in (SYSTEMS | TEMPLATE_SYSTEMS)[name]["chains"]:
        msa = (FIXTURES / c["msa"]).read_text() if "msa" in c else None
        template = None
        if "template" in c:
            spec = c["template"]
            text = (FIXTURES / spec["pdb"]).read_text()
            template = Template.from_pdb(text, sequence=spec.get("sequence"))
        chains.append(Chain(id=c["id"], sequence=c["sequence"], msa=msa, template=template))
    return System(tuple(chains))


def assert_matches(ours: dict, fixture_path: Path) -> None:
    """Every key the view produces equals upstream's: integers and flags exactly, floats to
    float32 rounding."""
    with np.load(fixture_path, allow_pickle=False) as upstream:
        for key, value in ours.items():
            assert key in upstream.files, key
            expected = upstream[key]
            assert value.shape == expected.shape, (key, value.shape, expected.shape)
            if np.issubdtype(expected.dtype, np.floating):
                np.testing.assert_allclose(value, expected, rtol=1e-6, atol=1e-7, err_msg=key)
            else:
                np.testing.assert_array_equal(value, expected, err_msg=key)
                assert value.dtype == expected.dtype, (key, value.dtype, expected.dtype)


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_bioemu2(name):
    assert_matches(bioemu2.features(system(name)), FIXTURES / f"bioemu2_{name}.npz")


@pytest.mark.parametrize("name", ["dimer", "dimer_msa"])
def test_bioemu2_paired_queries(name):
    """Heteromers with each chain's query as its paired MSA: every chain keeps its rows."""
    dimer = system("dimer")
    if name == "dimer_msa":
        a3m = (FIXTURES / "chain_A.a3m").read_text()
        a, b = dimer.chains
        dimer = System((Chain(a.id, a.sequence, msa=a3m), b))
    fixture = FIXTURES / f"bioemu2_{name}_paired_queries.npz"
    assert_matches(bioemu2.features(dimer, paired_queries=True), fixture)


def test_parse_a3m():
    a3m = parse_a3m((FIXTURES / "chain_A.a3m").read_text())
    assert len(a3m.rows) == 6 and {len(r) for r in a3m.rows} == {22}
    # hit2 is "MKSAYIAKqqQRQLSFV--HFSRQ": two insertions before the Q at column 8.
    assert a3m.rows[2] == "MKSAYIAKQRQLSFV--HFSRQ"
    assert a3m.deletions[2].tolist() == [0] * 8 + [2] + [0] * 13
    assert a3m.headers[0] == "query"


def test_views_without_templates_refuse_them():
    with pytest.raises(NotImplementedError, match="takes no templates"):
        bioemu2.features(system("dimer_template"))
