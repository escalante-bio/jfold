"""Write OpenFold3's upstream features for the test systems to `tests/featurize/data/`.

    uv run python tools/fixtures/dump_openfold3.py [system ...]
"""

import json
import logging
import random
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parents[1] / "tests" / "featurize" / "data"
SEED = 0
SYSTEM_FILES = ("systems.json", "template_systems.json")
DROP = ("atom_array", "query_id", "seed", "repeated_sample", "valid_sample")


def featurize(system: dict, seed: int = SEED) -> dict[str, np.ndarray]:
    """Featurize one system with upstream code and return unbatched NumPy arrays."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _featurize(system, seed)


def _template_cif(system: dict, path: Path) -> None:
    import gemmi

    from jfold.featurize.system import THREE_LETTER

    st, model, entities = gemmi.Structure(), gemmi.Model("1"), []
    for chain in system["chains"]:
        spec = chain.get("template")
        if spec is None:
            continue
        template = gemmi.read_structure(str(FIXTURES / spec["pdb"]))[0][0]
        template.name = chain["id"]
        entity = gemmi.Entity(str(len(entities) + 1))
        entity.entity_type = gemmi.EntityType.Polymer
        entity.polymer_type = gemmi.PolymerType.PeptideL
        entity.subchains = [chain["id"]]
        if "sequence" in spec:
            entity.full_sequence = [THREE_LETTER[c] for c in spec["sequence"]]
        else:
            entity.full_sequence = [r.name for r in template]
        entities.append(entity)
        for residue in template:
            residue.subchain = chain["id"]
        model.add_chain(template)
    st.add_model(model)
    st.entities = gemmi.EntityList(entities)
    st.setup_entities()
    st.assign_label_seq_id()
    doc = st.make_mmcif_document()
    block = doc.sole_block()
    poly = block.get_mmcif_category("_entity_poly.")
    poly["pdbx_seq_one_letter_code_can"] = poly["pdbx_seq_one_letter_code"]
    block.set_mmcif_category("_entity_poly.", poly)
    comp = block.get_mmcif_category("_chem_comp.")
    comp["type"] = ["L-peptide linking"] * len(comp["id"])
    block.set_mmcif_category("_chem_comp.", comp)
    scheme = {"asym_id": [], "entity_id": [], "seq_id": [], "mon_id": []}
    for entity in entities:
        for number, name in enumerate(entity.full_sequence, 1):
            scheme["asym_id"].append(entity.subchains[0])
            scheme["entity_id"].append(entity.name)
            scheme["seq_id"].append(str(number))
            scheme["mon_id"].append(name)
    block.set_mmcif_category("_pdbx_poly_seq_scheme.", scheme)
    block.set_pair("_pdbx_audit_revision_history.revision_date", "2000-01-01")
    doc.write_file(str(path))


def _m8(chain: dict, path: Path) -> Path:
    n = len(chain["sequence"])
    path.parent.mkdir(parents=True)
    path.write_text(f"query\ttmpl_{chain['id']}\t1.0\t{n}\t0\t0\t1\t{n}\t1\t{n}\t0.001\t100\n")
    return path


def _featurize(system: dict, seed: int) -> dict[str, np.ndarray]:
    from jopenfold3._vendor.openfold3.core.data.framework.single_datasets import inference
    from jopenfold3._vendor.openfold3.core.data.framework.single_datasets.inference import (
        InferenceDataset,
    )
    from jopenfold3._vendor.openfold3.core.data.pipelines.preprocessing.template import (
        TemplatePreprocessor,
        TemplatePreprocessorSettings,
    )
    from jopenfold3._vendor.openfold3.core.data.resources.residues import MoleculeType
    from jopenfold3._vendor.openfold3.core.data.tools.colabfold_msa_server import (
        MsaComputationSettings,
        augment_main_msa_with_query_sequence,
    )
    from jopenfold3._vendor.openfold3.projects.of3_all_atom.config.dataset_config_components import (
        MSASettings,
        TemplateSettings,
    )
    from jopenfold3._vendor.openfold3.projects.of3_all_atom.config.dataset_configs import (
        InferenceJobConfig,
    )
    from jopenfold3._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        Chain,
        InferenceQuerySet,
        Query,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        structures = tmp / "structures"
        structures.mkdir()
        if any("template" in c for c in system["chains"]):
            _template_cif(system, structures / "tmpl.cif")
        chains = []
        for c in system["chains"]:
            msa_paths = None
            if "msa" in c:
                # OpenFold3 skips alignment files whose names it does not recognise.
                target = tmp / "msas" / c["id"] / "colabfold_main.a3m"
                target.parent.mkdir(parents=True)
                shutil.copy(FIXTURES / c["msa"], target)
                msa_paths = [target]
            chains.append(
                Chain(
                    molecule_type=MoleculeType.PROTEIN,
                    chain_ids=[c["id"]],
                    description=c["id"],
                    sequence=c["sequence"],
                    main_msa_file_paths=msa_paths,
                    template_alignment_file_path=(
                        _m8(c, tmp / "templates" / c["id"] / "hits.m8") if "template" in c else None
                    ),
                )
            )
        queries = InferenceQuerySet(
            seeds=[seed], queries={"q": Query(query_name="q", chains=chains)}
        )
        queries = augment_main_msa_with_query_sequence(
            queries, MsaComputationSettings(msa_output_directory=tmp / "dummy_msas")
        )
        templates = TemplatePreprocessorSettings(
            output_directory=tmp / "template_data",
            structure_directory=structures,
            fetch_missing_structures=False,
        )
        TemplatePreprocessor(input_set=queries, config=templates)()
        config = InferenceJobConfig(
            query_set=queries,
            seeds=[seed],
            msa=MSASettings(subsample_main=False),
            template=TemplateSettings(take_top_k=True),
            template_preprocessor_settings=templates,
        )
        process = inference.process_template_structures_of3

        def with_cache(**kwargs):
            return process(**kwargs | {"template_cache_directory": templates.cache_directory})

        inference.process_template_structures_of3 = with_cache
        try:
            dataset = InferenceDataset(config)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            batch = dataset[0]
        finally:
            inference.process_template_structures_of3 = process
    if not bool(batch["valid_sample"].item()):
        raise RuntimeError("OpenFold3 featurization failed (valid_sample=False)")
    return {k: v.numpy() for k, v in batch.items() if k not in DROP and isinstance(v, torch.Tensor)}


def main():
    logging.disable(logging.WARNING)
    systems = {}
    for file in SYSTEM_FILES:
        systems |= json.loads((FIXTURES / file).read_text())
    if sys.argv[1:]:
        systems = {name: systems[name] for name in sys.argv[1:]}
    for name, system in systems.items():
        out = featurize(system)
        again = featurize(system)
        assert out.keys() == again.keys()
        for k in out:
            np.testing.assert_array_equal(out[k], again[k], err_msg=f"{name}: {k} not reproducible")
        path = FIXTURES / f"openfold3_{name}.npz"
        np.savez_compressed(path, **out)
        n_tok, n_atom, n_msa = (
            out["token_mask"].shape[0],
            out["atom_mask"].shape[0],
            out["msa"].shape[0],
        )
        print(
            f"{path.relative_to(HERE.parents[1])}: {n_tok} tokens, {n_atom} atoms, {n_msa} MSA rows"
        )


if __name__ == "__main__":
    main()
