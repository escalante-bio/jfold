# jfold

Shared JAX ([Equinox](https://github.com/patrick-kidger/equinox)) building blocks for
AF3-family structure prediction models, with converters from each model's upstream
PyTorch implementation.

The models (Boltz-2, ProMERA, Protenix, OpenDDE, OpenFold3, Mol-1, AtlasFold, ...) share
most of their architecture. Here each block is written once, and each model gets a converter
that maps its torch weights onto the shared modules. `jfold.featurize` reproduces each
model's featurization without torch, and `jfold.pretrained` loads the converted weights, so
predicting needs neither torch nor the upstream packages.

## Layout

- `jfold.nn`: primitives (`Linear`, `LayerNorm`, gated `Attention`, `PairBias`, SwiGLU
  `Transition`, `SinglesToPair`, `ReluMLP`, `Dropout`) and layer stacking.
- `jfold.blocks`: the building blocks the models are made of.
  - `pairformer`: triangle multiplication and attention, single attention with pair bias,
    `PairformerLayer`, `PairMixerLayer`, `PairformerStack`.
  - `msa`: pair-weighted averaging, the two outer product means, `MSAModule` (and
    `ResidualMSA`, as Boltz-2 and ProMERA apply it).
  - `templates`: `TemplateEmbedder`.
  - `diffusion`: AdaLN, conditioned attention and transition, `DiffusionTransformer` (over
    tokens, or over atom windows with `windows`).
  - `atoms`: the atom attention encoder and decoder.
  - `denoiser`: the diffusion module, split into a once-per-trunk `prepare` and a per-step
    call.
  - `confidence`: the confidence module, PAE/PDE/distogram/pLDDT heads.
  - `inputs`: relative position features, `InputEmbedder`, `TrunkInit`.
- `jfold.model`: `StructureModel`, which every model is (`embed`, `init_state`, `step`,
  `finish`, `run_trunk`, `denoiser_context`, `denoise`, `structure_sampler`,
  `distogram_logits`, `confidence_logits`); `Features`, `TrunkInputs`, `TrunkState`, and
  the AF3 recycling `Trunk`.
- `jfold.slots`: the abstract type of each `StructureModel` slot (`AbstractInputEmbedder`,
  `AbstractTrunkInit`, `AbstractTrunk`, `AbstractDiffusionModule`,
  `AbstractConfidenceModule`). Every slot reads the features, so a model's own inputs reach
  its own modules through `Features.extras` (ESMFold2's language-model input, OpenDDE's
  structural tokens, BioEmu2's conditioning) and `TrunkInputs.extras`. Within the shared
  modules, `AbstractPairformer`, `AbstractMSAModule` and `AbstractPairInputs` type the
  parts a model swaps.
- `jfold.models`: per-model feature vectors and the model-specific modules: ESMFold2's
  Parcae and recycling pair trunks and its trunk init running ESMC, OpenDDE's diffusion on
  structural tokens, BioEmu2's dual-stream structure module, AtlasFold's trunk running
  AtlasLM every pass and its diffusion on atom14 slots, and smaller pieces. A model whose
  inputs don't fit `Features` reads its own type (`AtlasFoldFeatures`).
- `jfold.sampling`: the diffusion sampler (AF3 Algorithm 18), noise schedules, and
  `UpstreamSampler`, each converted model's upstream sampler settings (`model.sampling`).
- `jfold.featurize`: torch-free featurization: a `System` of `Chain`s (sequence, optional
  a3m MSA) and one view per model (`jfold.featurize.views.<model>.features(system)`) that
  returns the feature dict the upstream pipeline would, and `atom_sites(system)`, what each
  of its atoms is (chain, residue, atom name, element; `to_pdb` writes a structure).
  Proteins, unpaired MSAs, and a template per chain (`Template`, e.g. `Template.from_pdb`)
  for Boltz-2, Protenix, Mol-1 and OpenFold3, placed as each upstream places it; the other
  views refuse templates (and AtlasFold's, a single-sequence model, MSAs). See
  `docs/upstream-featurization/`.
- `jfold.pretrained`: the released models, downloaded already converted. Each has a
  function that loads it (`boltz2()`, `protenix_mini()`, `esmfold2_experimental_fast()`,
  ...), and `load(name)` loads one by name. The recipes are in `jfold.pretrained.recipes`.
- `jfold.io`: saving and loading converted models (see below).
- `jfold.convert`: torch → jfold converters, each module named for what it converts: a
  code lineage's blocks and the models built only from them (`boltz`, `protenix`,
  `openfold3`, `esmfold2`, `bioemu2`, `atlasfold`), or a model built on another lineage's
  blocks (`boltz2`, `promera`, `opendde`). Whole models convert with `structure_model`.

Modules are unbatched (`vmap` for batches) and take masks for padded inputs; padding never
affects real tokens or atoms. Sequences enter as probability vectors over each model's
residue vocabulary (one-hot for a fixed sequence), in both the token and MSA features; each
model's `features(dict)` builds them from its own featurization.

Dropout follows each upstream's rates and mask-sharing axes. It is off by default;
`eqx.nn.inference_mode(model, False)` turns it on. The exception is ESMFold2's dropout on
its language model's pair, which upstream applies at inference too: it is on by default,
and `eqx.nn.inference_mode(model)` turns it off. Call models through `eqx.filter_jit`
(plain `jax.jit` would trace the switch). Where a key only feeds dropout (`denoise`,
`confidence_logits`, `finish`, any block) it may be omitted; dropout that is on raises
without one.

Every trunk pass, the MSA module reads the rows its upstream draws, drawn afresh
(`MSAModule.draw_rows`): Boltz-2, ProMERA and OpenFold3 1024 rows and OpenDDE 1280,
uniformly with the query like any other; ESMFold2 the query and 1023 others; Protenix and
Mol-1 a uniformly random number of rows, 1 to all of them, computed on at most 1024 (a
deeper draw is cut to 1024 rows, bounding the cost); BioEmu2 every row
(`docs/upstream-featurization/README.md`). ESMFold2 also masks a share of MSA columns outside the query
once per forward pass (`msa_column_mask_rate`, 0.1 as upstream). AtlasFold reads no MSA;
instead every pass masks a fresh share of its language model's residues (`mlm_rate`, 20%
for AtlasFold-M and 15% for AtlasFold, as upstream).

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
model = pretrained.boltz2()  # each released model has its function; `load(name)` too

inputs, state = model.run_trunk(features, num_passes=4, key=k_trunk)  # 1 + upstream's 3 recycles
coords = model.structure_sampler(features, inputs, state).sample(k_sample)  # [atoms, 3]
plddt_logits = model.confidence_logits(features, inputs, state, coords, key=k_conf).plddt
```

The other models follow the same steps with their own view and `features`. BioEmu2's
features carry its conditioning (method, contacts, dynamics) with them. An ESMFold2 model
holds its ESMC language model: `embed` runs ESMC on the sequence (soft sequences included,
so their gradient reaches it), unless the features carry precomputed `lm_hidden_states`.
Each ESMC is stored once and attached on loading (`jfold.pretrained`). An AtlasFold model's
trunk holds AtlasLM (an ESMC reproduction, `trunk.atlaslm`), stored once as `atlaslm_3b`,
and runs it inside every trunk pass; its atoms are each residue's 14 atom slots (atom14:
`jfold.models.atlasfold.atom14` reshapes the positions to `[T, 14, 3]`).

Two end-to-end examples (under `eqx.filter_jit`, several samples, written as PDB files):

```
# Boltz-2, with an MSA; keeps the most confident sample
uv run python examples/predict_boltz2.py MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG --msa ubq.a3m

# ESMFold2-Experimental-Fast on ESMC 300M, single sequence (no confidence head)
uv run python examples/predict_esmfold2.py MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
```

`jfold.featurize` takes proteins. For anything else (ligands, nucleic acids, modified
residues, paired MSAs), featurize with the upstream pipeline and pass its dict, unbatched
and as numpy, to the model's `features`: every model's `features` takes its upstream's dict.
`examples/predict_boltz2_ligand.py` does this for a protein-ligand complex with Boltz-2,
writing mmCIF with Boltz's own writer (needs `uv sync --group reference`):

```
uv run python examples/predict_boltz2_ligand.py MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ --smiles "CC(=O)Oc1ccccc1C(=O)O"
```

## Sampling

```python
structure_sampler = model.structure_sampler(features, inputs, state)  # upstream's sampler
coords = structure_sampler.sample(key)
coords = jax.vmap(structure_sampler.sample)(keys)  # several; the context is built once

few_step = model.structure_sampler(
    features,
    inputs,
    state,
    num_steps=8,
    schedule=schedules.Geometric(),
    update=exponential_update,
)
```

A `StructureSampler` binds the model's denoiser with its step-invariant context (computed
once), the atoms it moves and the sampler's settings (`SamplerSettings`, from the model's
upstream `UpstreamSampler`, `model.sampling`). The loop underneath
(`jfold.sampling.integrate`) is model-agnostic.

Padding never changes the computation: given the same random draws for the real tokens
and atoms (initial state, sampler noise), a padded system samples the same real
coordinates. The draws themselves depend on array sizes, so the same key gives a padded
and an unpadded system different samples.

## Saving models

`jfold.io.save(model, path, metadata=...)` writes a directory: `weights.safetensors`
(every array, keyed by its path in the model, readable with numpy alone) and
`model.json` (format version, metadata, and the model's structure: class names,
fields, plain values, functions by name, array shapes and dtypes). `jfold.io.load(path)`
rebuilds the model; `jfold.io.read_structure(path)["metadata"]` reads the metadata
alone. `jfold.io.provenance(model, checkpoint=..., source=..., upstream=...)` builds the
standard metadata: model type, checkpoint source and SHA-256, upstream package version,
conversion date. Nothing is pickled, so files don't depend on JAX's internal types,
only on jfold's class and field names; a mismatch is reported by name. When a class that
saved models use moves or is renamed, `jfold.io.MOVED` maps its old name to the new one, so
older files keep loading.

Released models are published to the Hugging Face hub with `jfold.pretrained`:
`save(name, model, metadata=...)` writes a release's copy without the parts it shares with
other releases (ESMFold2's ESMC, stored once as `esmc_6b`, `esmc_300m`, `esmc_600m`, and
AtlasFold's AtlasLM, as `atlaslm_3b`; attached on loading), and `upload(*names)` publishes
saved copies in one commit. A cached copy an older jfold saved is downloaded again on
loading.

## Tests

```
uv sync --group reference      # torch and the upstream packages the tests compare against
uv run pytest                  # block parity, random weights
uv run pytest -m checkpoint    # whole models against released weights (downloads them)
```

Plain `uv sync` installs only what predicting needs.

The block tests build each upstream module with random weights, run it, convert it, and
compare, with and without masked padding. The checkpoint tests featurize a small complex
with each model's own pipeline and compare the trunk, a denoising step, the confidence
heads and the distogram. ProMERA's triangle updates exist only as cuequivariance CUDA kernels, so its references run
on a GPU.

- `tests/`: the block tests and each model's own tests; `reference.py` holds their shared
  helpers.
- `tests/featurize/`: each featurization view against upstream's feature dicts, recorded in
  `tests/featurize/data/`.
- `tests/checkpoints/`: whole models against released weights.

Outside the package and its tests:

- `docs/models/`: the shared model skeleton and what fills each part in every model.
- `docs/upstream-featurization/`: how each upstream pipeline builds its features, which the
  views reproduce, quirks included.
- `tools/`: scripts that run upstream code: `tools/fixtures/` records the featurization
  fixtures, `tools/tables/` extracts the residue tables the views ship with.

## Status

Boltz-2, ProMERA, Protenix, OpenDDE, OpenFold3, Mol-1, ESMFold2, BioEmu2, AtlasFold and
AtlasFold-M convert as whole models and match their released checkpoints, as do the four
ESMFold2-Experimental checkpoints, along with each upstream's sampler settings. Not ported:
training heads.
