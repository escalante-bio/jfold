# jfold.featurize

A torch-free featurizer: one shared description of a system (`System`, `Chain`, parsed a3m
MSAs, per-model residue tables) and a thin per-model *view* that produces exactly the feature
dict each upstream pipeline produces. Each view's `atom_sites(system)` describes the atoms
of its features (chain, residue, atom name, element, token), so outputs can be written as
structures (`AtomSites.to_pdb`) without decoding the features. Scope: proteins only, unpaired
MSAs, and one template per chain (`Chain.template`, a `Template`: sequence and each residue's
atoms by name) for Boltz-2, Protenix, Mol-1 and OpenFold3, each placed as its upstream places
it (Boltz-2 by local alignment, OpenFold3 by Kalign realignment, Protenix and Mol-1 residue
by residue); OpenDDE (whose upstream only takes template search hits), ProMERA, BioEmu2,
ESMFold2 and AtlasFold refuse templates. AtlasFold, a single-sequence model, also refuses
MSAs.

The per-pipeline specs (every key jfold reads, with shape, dtype, rule and the upstream code
it comes from) are `boltz2.md`, `protenix.md` (with Mol-1 and OpenDDE), `openfold3.md`,
`promera.md`, `bioemu2.md`, `esmfold2.md` and `atlasfold.md`.

## Parity

`tests/test_featurize*.py` compare every view against upstream's own featurization of three
systems (`tests/featurize/data/systems.json`: a monomer, a heterodimer, the monomer with
`chain_A.a3m`), and the views that take templates on the templated systems of
`template_systems.json` (PDB templates written by `tools/fixtures/make_templates.py`).
`tools/fixtures/dump_<model>.py` regenerate the fixtures; they need the upstream package
installed. `tools/tables/extract_<model>.py` build the residue tables in
`src/jfold/featurize/data/` from upstream's residue data.

## Shared rules

- Chains: id, sequence, entity (identical sequences), copy index.
- Tokens: one per residue.
- Atoms: each residue's heavy atoms in CCD order (with the model's OXT and UNK rules), with
  element, formal charge (+1 on ARG NH2, HIS ND1, LYS NZ in every pipeline), atom name
  (4 characters, `ord(c) - 32`, 64 classes) and a reference conformer.
- Representative atom CB (CA for glycine); centre atom CA.
- No token bonds for standard proteins (all pipelines except ESMFold2's UNK).
- MSA: a3m match columns, with insertions counted on the next match column.

## What differs

| | Boltz-2 | Protenix | Mol-1 | OpenDDE | OpenFold3 | ProMERA | BioEmu2 | ESMFold2 |
|---|---|---|---|---|---|---|---|---|
| Vocabulary | 33 | 32 | 32 | 32 | 32 | 34 | 33 | 33 |
| Reference conformer | 1 of 10 stored RDKit conformers | own RDKit set | own RDKit set | own RDKit set | RDKit ETKDGv3 (32 stored, `random.seed` 0–31) | CCD ideal | CCD ideal, centred incl. OXT | CCD, raw |
| Random pose | per residue, skips last residue | translate→rotate | translate→rotate | rotate→translate | per residue | translate→rotate | none | none |
| OXT | none | chain end | chain end | chain end | none | none | none | none |
| Unknown residue (X) atoms | N CA C O CB | | | | N CA C O CB CG | CCD UNK (6), rep CA | ALA's 5 | N CA C O + peptide bonds |
| `residue_index` | 0-based per chain | | | | | 1-based | global counter | 0-based per chain |
| Entity / chain order | entity-grouped (reorders) | one entity per chain | | | ranked by string sort | one entity per chain | entity-grouped (reorders) | |
| `deletion_value` | (π/2)·atan(d/3), but always 0 (bug) | | capped at 127 | capped at 127 | (8/π)·atan(d/3) | (2/π)·atan(d/3) | (π/2)·atan(d/3) | (π/2)·atan(d/3) |
| `deletion_mean` | 0 (bug) | | | | | raw counts | transformed | transformed, incl. query |
| MSA rows | dedup; chains interleaved row-wise | dedup; interleaved | query twice | query twice | query twice | dedup; interleaved; UniRef100 reorder | cut to shallowest chain | interleaved; `key=` reorders |
| Profile | | float64 | | | scrambled for >1 row (bug) | | | in-model; zeroed for experimental -Fast |
| Atom padding | ×32 | | | | none | ×32 | ×32 | ×32 |

Blank cells follow the shared rule or don't apply; see the per-model spec.

## MSA rows per trunk pass (model side)

The views return every MSA row the upstream featurizer keeps (at most 16384; Boltz-2 reads
at most 8192 rows of each chain's a3m, BioEmu2 4092). Which of them a trunk pass reads is
drawn inside the model, afresh on every pass, and jfold's `MSAModule` draws them the same
way (`max_rows`, `keep_query`, `min_rows`; `MSAModule.draw_rows`):

| | Upstream, each pass | `MSAModule` |
|---|---|---|
| Boltz-2 | `torch.randperm(S)[:1024]` (`num_subsampled_msa`): 1024 rows uniformly, the query like any other | `max_rows=1024, keep_query=False` |
| ProMERA | `subsample_msa`: `min(S, 1024)` real rows uniformly (`msas_per_trunk_iter`) | `max_rows=1024, keep_query=False` |
| OpenFold3 | `_subsample_all_msa`: 1024 rows uniformly (`min_` = `max_subsampled_all_msa`), real rows first | `max_rows=1024, keep_query=False` |
| OpenDDE | `subsample_msa_feature_dict_valid_first`: 1280 rows uniformly (`msa_depth`), rows with a residue first | `max_rows=1280, keep_query=False` |
| Protenix, Mol-1 | `sample_msa_feature_dict_random_without_replacement`: `k ~ U{1..S}` rows uniformly (test `min_size`), at most 16384 (test `sample_cutoff`) | `max_rows=1024, min_rows=1, keep_query=False`: `k` rows when `k ≤ 1024`, else 1024 (each pass computes on at most 1024 rows) |
| BioEmu2 | every row | `max_rows=None` |
| ESMFold2 | `maybe_subsample_msa`: the query and 1023 other rows (`fold`'s `msa_max_depth`) | `max_rows=1024, keep_query=True` |

Every draw reads all rows of an MSA no deeper than it, except Protenix's and Mol-1's random
number of rows, which applies to any MSA of more than one row. Every model but BioEmu2 thus
subsamples a deep MSA to at most 1024 rows a pass (OpenDDE 1280); jfold bounds Protenix's and
Mol-1's draw there, where upstream's can reach 16384 rows. Row order does not change
the pair. OpenDDE's "rows with a residue" are all the rows the view writes (an all-gap row
comes only from an all-gap a3m line), so jfold draws from the real rows. `tests/test_msa.py::test_rows_drawn_as_upstream`
compares the draws with upstream's on an MSA deeper than a pass reads.

## Decisions

- **Upstream bugs are reproduced**, so each model sees the inputs it was trained on: Boltz-2's
  zeroed deletion features and unposed last residue, OpenFold3's scrambled profile and 8/π
  deletion scale.
- **Randomness comes from JAX keys.** Upstream draws poses (and OpenFold3's conformers) from
  torch's or numpy's global RNG, which can't be replayed; the views draw from the same
  distributions with a `key`, and `key=None` gives the canonical, unposed conformers. Parity
  tests compare the canonical conformers and every other feature exactly.
- **OpenFold3's conformers come from a stored table** (upstream's RDKit embedding under
  `random.seed` 0–31) instead of running RDKit per featurization.
