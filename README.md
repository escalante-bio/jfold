# jfold

AF3-family structure prediction models in JAX
([Equinox](https://github.com/patrick-kidger/equinox)), converted from their upstream
PyTorch implementations: Boltz-2, ProMERA, Protenix, Mol-1, OpenDDE, OpenFold3, ESMFold2,
BioEmu2, AtlasFold and AtlasFold-M.

These models share most of their architecture. Here each building block is written once and
every model is a `StructureModel` whose parts are filled by shared or model-specific modules
([the models, part by part](docs/models/README.md)). Converted weights load from the Hugging
Face hub, and `jfold.featurize` reproduces each model's featurization for proteins, so
predicting needs neither torch nor the upstream packages.

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
model = pretrained.boltz2()

inputs, state = model.run_trunk(features, num_passes=4, key=k_trunk)  # 1 + upstream's 3 recycles
coords = model.structure_sampler(features, inputs, state).sample(k_sample)  # [atoms, 3]
plddt_logits = model.confidence_logits(features, inputs, state, coords, key=k_conf).plddt
```

Every model follows the same steps with its own view, `features` and loader
(`pretrained.protenix_mini()`, `pretrained.esmfold2_experimental_fast()`, ...;
`pretrained.load(name)` loads one by name). End-to-end examples:

- [`examples/predict_boltz2.py`](examples/predict_boltz2.py): Boltz-2, optionally with MSAs,
  several samples, the most confident written as PDB.
- [`examples/predict_esmfold2.py`](examples/predict_esmfold2.py): an ESMFold2 checkpoint,
  single sequence.
- [`examples/predict_boltz2_ligand.py`](examples/predict_boltz2_ligand.py): a
  protein-ligand complex with Boltz-2's own (torch) featurization, below.

### Ligands, nucleic acids and other inputs

`jfold.featurize` takes proteins (with unpaired MSAs, and templates for Boltz-2, Protenix,
Mol-1 and OpenFold3). For anything else, featurize with the upstream pipeline and pass its
feature dict, unbatched and as numpy, to the model's `features`, which takes its upstream's
dict. [`examples/predict_boltz2_ligand.py`](examples/predict_boltz2_ligand.py) does this for
a protein and a small molecule, given as SMILES or a CCD code:

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

A sampler binds the model's denoiser, its once-per-trunk context and the settings of the
model's upstream sampler (`model.sampling`); the loop underneath (`jfold.sampling.integrate`)
is model-agnostic.

## Conventions

- Modules are unbatched (`vmap` for batches) and take masks; padding never changes the real
  tokens' and atoms' results.
- Sequences enter as probability vectors over each model's residue vocabulary (one-hot for a
  fixed sequence), so a soft sequence can be optimized through the model.
- Keys feed dropout and each model's random draws: the MSA rows a trunk pass reads (as its
  upstream draws them; see [the models](docs/models/README.md)), ESMFold2's starting state
  and MSA column mask, AtlasFold's masking of its language model's input. Where a key only
  feeds dropout it may be omitted.
- Dropout follows each upstream's rates and is off by default
  (`eqx.nn.inference_mode(model, False)` turns it on), except ESMFold2's on its language
  model's pair, which upstream applies at inference. Call models through `eqx.filter_jit`.
- Language models are stored once and attached on loading: ESMFold2's ESMC
  (`trunk_init.esmc`) and AtlasFold's AtlasLM (`trunk.atlaslm`).

## Layout

- `jfold.nn`: primitives (`Linear`, `LayerNorm`, `Attention`, `PairBias`, `Transition`,
  `Dropout`) and layer stacking.
- `jfold.blocks`: the shared building blocks (`pairformer`, `msa`, `templates`,
  `diffusion` and `atoms` for the transformers over tokens and atom windows, `denoiser` for
  the diffusion module, `confidence`, `inputs`).
- `jfold.model`: `StructureModel` (`run_trunk`, `structure_sampler`, `confidence_logits`,
  ...), `Features`, and the AF3 recycling `Trunk`.
- `jfold.slots`: the abstract type of each `StructureModel` part, which model-specific
  modules subclass; a model's own inputs travel in `Features.extras`.
- `jfold.models`: each model's feature vectors (`features(dict)`) and its own modules.
- `jfold.featurize`: torch-free featurization, one view per model
  (`views.<model>.features(system)`, `atom_sites(system)` for writing structures).
- `jfold.pretrained`: the released models, a loader each; their recipes in
  `jfold.pretrained.recipes`.
- `jfold.convert`: the torch → jfold converters.
- `jfold.sampling`: the diffusion sampler, noise schedules and upstream sampler settings.
- `jfold.io`: saving and loading models without pickle (below).

## Saving models

`jfold.io.save(model, path, metadata=...)` writes `weights.safetensors` (every array, by its
path in the model) and `model.json` (the model's classes, fields and array shapes, plus
metadata; `jfold.io.provenance` builds the standard fields). `jfold.io.load(path)` rebuilds
the model. Nothing is pickled, so files depend only on jfold's class and field names;
`jfold.io.MOVED` keeps renamed classes loading. `jfold.pretrained.save` and `upload` write and
publish a release's copy, without the language model it shares with other releases.

## Tests

```
uv sync --group reference      # torch and the upstream packages the tests compare against
uv run pytest                  # block parity with random weights; featurization parity
uv run pytest -m checkpoint    # whole models against released weights (downloads them)
```

Plain `uv sync` installs only what predicting needs. Block tests convert each upstream module
with random weights and compare, with and without padding (`tests/`); featurization tests
compare each view with upstream's recorded feature dicts (`tests/featurize/`); checkpoint
tests compare whole models on a small complex (`tests/checkpoints/`). ProMERA's references
run on a GPU.

Also in the repository: [`docs/models/`](docs/models/README.md) (each model's parts),
`docs/upstream-featurization/` (each upstream pipeline, which the views reproduce) and
`tools/` (scripts that record the fixtures and extract the residue tables).

## Status

Every model above converts whole and matches its released checkpoints (the four
ESMFold2-Experimental checkpoints too), with each upstream's sampler settings. Not ported:
training heads.

## License

The code is MIT-licensed ([LICENSE](LICENSE)). The converted weights keep their upstream
licences, noted beside each copy on the hub.
