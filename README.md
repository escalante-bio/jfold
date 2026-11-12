# jfold

AF3-family structure prediction models in JAX
([Equinox](https://github.com/patrick-kidger/equinox)), converted from their upstream
PyTorch implementations: Boltz-2, ProMERA, Protenix, Mol-1, OpenDDE, OpenFold3, ESMFold2,
BioEmu2, AtlasFold and AtlasFold-M.

These models share most of their architecture. jfold writes each building block once, and
every model is a `StructureModel` whose parts are shared or model-specific modules
([the models, part by part](docs/models/README.md)). Converted weights load from the Hugging
Face hub, and `jfold.featurize` reproduces each model's featurization for proteins, so
prediction needs neither torch nor the upstream packages.

## Predicting

```python
import jax
from jfold import pretrained
from jfold.featurize import Chain, System
from jfold.featurize.views import boltz2 as view
from jfold.models import boltz2

k_pose, k_trunk, k_sample, k_conf = jax.random.split(jax.random.key(0), 4)
system = System((Chain("A", "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"),))
features = boltz2.features(view.features(system, key=k_pose))  # `msa=` an a3m string, per chain
model = pretrained.load("boltz2")

inputs, state = model.run_trunk(features, num_passes=4, key=k_trunk)  # 1 + upstream's 3 recycles
coords = model.structure_sampler(features, inputs, state).sample(k_sample)  # [atoms, 3]
plddt_logits = model.confidence_logits(features, inputs, state, coords, key=k_conf).plddt
```

Every model follows the same steps with its own view, `features` function and name, such as
`"protenix_mini"` or `"esmfold2_experimental_fast"` (`pretrained.releases()` lists them all). To
run the trunk one pass at a time, call `model.embed`, then `model.trunk.init_state`, then
`model.trunk(features, inputs, state, key=k)` once per pass, and finally `model.trunk.finish`
for the state the heads read. End-to-end examples:

- [`examples/predict_boltz2.py`](examples/predict_boltz2.py): Boltz-2, optionally with MSAs.
  It draws several samples and writes the most confident one as a PDB file.
- [`examples/predict_esmfold2.py`](examples/predict_esmfold2.py): an ESMFold2 checkpoint on a
  single sequence.
- [`examples/predict_boltz2_ligand.py`](examples/predict_boltz2_ligand.py): a
  protein-ligand complex featurized with Boltz-2's own torch pipeline, described below.

### Ligands, nucleic acids and other inputs

`jfold.featurize` handles proteins, with unpaired MSAs, and templates for Boltz-2, Protenix,
Mol-1 and OpenFold3. For other inputs, featurize with the upstream pipeline and pass its
unbatched feature dict, as NumPy arrays, to the model's `features` function, which accepts
the upstream format. [`examples/predict_boltz2_ligand.py`](examples/predict_boltz2_ligand.py)
does this for a protein and a small molecule given as SMILES or a CCD code:

```
uv sync --group reference   # the upstream packages
uv run python examples/predict_boltz2_ligand.py MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ --smiles "CC(=O)Oc1ccccc1C(=O)O"
```

### Sampling

```python
sampler = model.structure_sampler(features, inputs, state)  # upstream's settings
coords = jax.vmap(sampler.sample)(keys)  # several samples; the context is built once
few_step = model.structure_sampler(
    features, inputs, state, num_steps=8, schedule=schedules.Geometric(), update=exponential_update
)
```

A sampler holds the model's denoiser, the context it computes once per trunk output, and
the model's upstream sampler settings (`model.sampling`). The sampling loop underneath,
`jfold.sampling.integrate`, is the same for every model.

## Conventions

- Modules work on single examples (use `vmap` for batches) and take masks. Padding never
  changes the results for real tokens and atoms.
- Sequences are given as probability vectors over each model's residue vocabulary, one-hot
  for a fixed sequence, so a soft sequence can be optimized through the model.
- Keys are used for dropout and for each model's random choices: the MSA rows each trunk
  pass reads, drawn as upstream does (see [the models](docs/models/README.md)), ESMFold2's
  starting state and MSA column mask, and AtlasFold's masking of its language model input.
  If a key is only used for dropout, it can be omitted.
- Dropout uses each upstream's rates and is off until `eqx.nn.inference_mode(model, False)`
  turns it on, except ESMFold2's dropout on its language model's pair features, which upstream
  applies at inference. Call models through `eqx.filter_jit`.
- `jfold.blocks.pairformer.with_triangle_chunks(model, chunk_size)` makes every triangle
  attention process `chunk_size` rows at a time. This is slower, but it only holds the
  attention maps for `chunk_size` rows at once. It also limits memory under `grad`, because
  each block of rows is recomputed on the backward pass.
- Language models are stored once and attached when a model is loaded: ESMFold2's ESMC
  (`trunk_init.esmc`) and AtlasFold's AtlasLM (`trunk.atlaslm`).

## Layout

- `jfold.nn`: primitives (`Linear`, `LayerNorm`, `Attention`, `PairBias`, `Transition`,
  `Dropout`) and layer stacking.
- `jfold.blocks`: the shared building blocks (`pairformer`, `msa`, `templates`,
  `diffusion` and `atoms` for the transformers over tokens and atom windows, `denoiser` for
  the diffusion module, `confidence`, `inputs`).
- `jfold.model`: `StructureModel` (`embed`, `run_trunk`, `structure_sampler`,
  `confidence_logits`) and the AF3 recycling `Trunk`.
- `jfold.slots`: the abstract type of each `StructureModel` part, which model-specific
  modules subclass, and the values passed between parts (`Features`, `TrunkInputs`,
  `TrunkState`, `ConfidenceLogits`). Model-specific inputs go in `Features.extras`.
- `jfold.models`: each model's feature vectors (`features(dict)`) and its own modules.
- `jfold.featurize`: featurization without torch, with one view per model
  (`views.<model>.features(system)`, and `atom_sites(system)` for writing structures).
- `jfold.pretrained`: loading the released models. The recipe for each one is in
  `jfold.pretrained.recipes`.
- `jfold.convert`: converters from the torch models to jfold.
- `jfold.sampling`: the diffusion sampler, noise schedules and upstream sampler settings.
- `jfold.io`: saving and loading models without pickle (below).

## Saving models

`jfold.io.save(model, path, metadata=...)` writes `weights.safetensors`, which holds every
array keyed by its path in the model, and `model.json`, which holds the model's classes,
fields, array shapes and metadata. `jfold.io.provenance` builds the standard metadata.
`jfold.io.load(path)` rebuilds the model. Nothing is pickled, so files depend only on jfold's
class and field names. `jfold.io.MOVED` lets renamed classes keep loading, and a new field
with a default loads from files that lack it. `jfold.pretrained.save` and `upload` save and
publish a release, without the language model it shares with other releases.

## Tests

```
uv sync --group reference      # torch and the upstream packages the tests compare against
uv run pytest                  # block parity with random weights; featurization parity
uv run pytest -m checkpoint    # whole models against released weights (downloads them)
```

Plain `uv sync` installs only what prediction needs. Block tests convert each upstream
module with random weights and compare outputs, with and without padding (`tests/`).
Featurization tests compare each view with upstream's recorded features
(`tests/featurize/`). Checkpoint tests compare whole models on a small complex
(`tests/checkpoints/`). ProMERA's reference model needs a GPU.

Also in the repository: [`docs/models/`](docs/models/README.md) describes each model's parts,
`docs/upstream-featurization/` each upstream featurization pipeline the views reproduce, and
`tools/` the scripts that record the test fixtures and extract the residue tables.

## Status

Every model above converts completely and matches its released checkpoints, including the
four ESMFold2-Experimental checkpoints, using each upstream's sampler settings. Training
heads are not ported.

The 300M and 600M ESMFold2 scaling checkpoints were released without a confidence head.
jfold also provides them with the heads that [FastPLMs](https://github.com/Synthyra/FastPLMs)
trained for them: `esmfold2_experimental_fast_base300m_confidence` and
`esmfold2_experimental_fast_base600m_confidence`.

## License

The code is MIT-licensed ([LICENSE](LICENSE)). The converted weights keep their upstream
licences, listed in the [hub repository's model card](https://huggingface.co/escalante-bio/jfold).
