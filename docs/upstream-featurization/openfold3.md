# OpenFold3 featurization (proteins, unpaired MSAs)

How OpenFold3's inference data pipeline turns a protein-only query into the feature dict
that `jfold.models.openfold3.features` consumes. It is the reference for
`jfold.featurize.views.openfold3`. Scope: standard amino acids, one or more chains, one
template structure per chain (§12), no ligands, nucleic acids or modified residues; MSAs
are unpaired.

Every rule below was checked by running the vendored pipeline. The view matches every key
of the parity fixtures exactly except `ref_pos`, which is checked structurally (§14), and
`template_unit_vector`, to float32 rounding (§12).

All source paths are relative to
`jopenfold3/_vendor/openfold3/` in jfold's environment
(`.venv/lib/python3.12/site-packages/jopenfold3/_vendor/openfold3/`).

## 1. Driving the pipeline

```python
chains = [Chain(molecule_type=MoleculeType.PROTEIN, chain_ids=["A"], sequence=seq,
                main_msa_file_paths=[Path(".../A/colabfold_main.a3m")])]   # or None
queries = InferenceQuerySet(seeds=[0], queries={"q": Query(query_name="q", chains=chains)})
queries = augment_main_msa_with_query_sequence(queries, MsaComputationSettings(msa_output_directory=tmp))
config = InferenceJobConfig(query_set=queries, seeds=[0],
                            msa=MSASettings(subsample_main=False),
                            template=TemplateSettings(take_top_k=True),
                            template_preprocessor_settings=TemplatePreprocessorSettings())
batch = InferenceDataset(config)[0]          # dict of UNBATCHED torch tensors
```

This is how `tests/test_checkpoint_openfold3.py::_featurize` and
`tools/fixtures/dump_openfold3.py` drive it. Both use `subsample_main=False`, which is also the
`InferenceJobConfig` default in OpenFold3's own project configs
(`projects/of3_all_atom/config/dataset_configs.py:333,372`). The bare `MSASettings()`
default is `True`, which subsamples randomly (see §11).

**Precomputed a3m.** Pass the file in `Chain.main_msa_file_paths`; no MSA server is
involved. There are two traps in how files are keyed:
- The file **stem** names the alignment. `parse_msas_direct` (`core/data/io/sequence/msa.py:225`)
  silently skips stems that are not keys of `MSASettings.max_seq_counts`, and `create_main`
  only concatenates stems in `MSASettings.aln_order`. Name the file `colabfold_main.a3m`
  (that is what the dummy and ColabFold paths use). A file called `chain_A.a3m` is dropped,
  and the sample then fails with `valid_sample=False`.
- The **parent directory** name is the chain's "representative id"
  (`MsaSampleParserInference.create_maps`, `core/data/io/sequence/msa.py:561`). Chains whose
  files share a parent directory name share one MSA. Give each chain its own directory.

**Chains without an MSA.** `augment_main_msa_with_query_sequence`
(`core/data/tools/colabfold_msa_server.py:1066`) writes a dummy
`<out>/dummy/<seq hash>/colabfold_main.a3m` containing `">query\n" + sequence`.

## 2. Call chain

`InferenceDataset.__getitem__` → `create_all_features` (`core/data/framework/single_datasets/inference.py:323, 282`):

1. **Structure and reference molecules.** `get_structure_with_ref_mols` (inference.py:159) calls:
   - `structure_with_ref_mols_from_query` (`core/data/primitives/structure/query.py:515`): assigns entity ids and loops over chains, then over the chain ids of each chain;
   - for each residue, `structure_with_ref_mols_from_sequence` (query.py:279):
     - `atom_array_from_ccd_code` (query.py:99): biotite's CCD residue, with hydrogens removed;
     - `processed_reference_molecule_from_atom_array` → `processed_reference_molecule_from_mol` (query.py:182, 223) → `multistrategy_compute_conformer` → `compute_conformer` (`core/data/primitives/structure/conformer.py:116, 40`): an RDKit conformer;
     - OXT is removed from the atom array, and `struc.connect_via_residue_names` adds the bonds.
   - `tokenize_atom_array` and `add_token_positions` (`core/data/primitives/structure/tokenization.py:142, 39`).
2. **Structure features.** `create_structure_features` (inference.py:198) produces:
   - from `featurize_structure_of3` (`core/data/pipelines/featurization/structure.py:43`), using `create_sym_id`, `create_token_bonds` and `create_atom_to_token_index` (`core/data/primitives/featurization/structure.py:48, 91, 141`): the token features, `atom_mask`, `atom_to_token_index`, ids and bonds;
   - from `featurize_reference_conformers_of3` (`core/data/pipelines/featurization/conformer.py:35`), using `centre_random_augmentation` (`core/model/structure/diffusion_module.py:45`): the `ref_*` features.
3. **MSA features.** `create_msa_features` (inference.py:224) runs three stages:
   - **Parse:** `MsaSampleProcessorInference.__call__` (`core/data/pipelines/sample_processing/msa.py:552`) calls `MsaSampleParserInference.create_maps` and `parse_msas` (`core/data/io/sequence/msa.py:561, 649`), which call `parse_msas_direct` and `parse_a3m` (msa.py:225, 96).
   - **Assemble rows:** `create_query_seqs` (sample_processing/msa.py:57). Pairing (none here) comes next, then `create_main` (sample_processing/msa.py:236), which uses `calculate_profile` (`core/data/primitives/sequence/msa.py:1193`).
   - **Featurize:** `MsaFeaturizerOF3.__call__` (`core/data/pipelines/featurization/msa.py`) calls:
     - `create_msa_feature_precursor_of3`, `vstack_pad_msa_arrays` and `map_msas_to_tokens` (`core/data/primitives/featurization/msa.py:218, 88, 161`), using `map_str_array_to_idx_array` (`core/data/resources/residues.py:335`);
     - then `MsaFeaturizerOF3.create_features` (featurization/msa.py:80).
4. **Template features.** `create_template_features` (inference.py:241) calls `process_template_structures_of3` (`core/data/pipelines/sample_processing/template.py:36`), then `featurize_template_structures_of3` (`core/data/pipelines/featurization/template.py:49`).

## 3. Token vocabulary (32 classes)

`STANDARD_RESIDUES_WITH_GAP_3` (`core/data/resources/residues.py`), indices 0–31:

| idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3-letter | ALA | ARG | ASN | ASP | CYS | GLN | GLU | GLY | HIS | ILE | LEU | LYS | MET | PHE | PRO | SER | THR | TRP | TYR | VAL |
| 1-letter | A | R | N | D | C | Q | E | G | H | I | L | K | M | F | P | S | T | W | Y | V |

The remaining classes are:
- 20: `UNK` / `X`;
- 21–25: RNA `A G C U N`;
- 26–30: DNA `DA DG DC DT DN`;
- 31: `GAP` / `-`.

The protein block is the AlphaFold order `ARNDCQEGHILKMFPSTWYV`
(`jfold.featurize.system.AMINO_ACIDS`).

**Unknown residues in the query sequence.** Any character not in `PROTEIN_RESTYPE_1TO3`
becomes residue `UNK`, with class 20 and 6 atoms (see §5). That set is the 20 uppercase
letters plus `X`. Lowercase letters and `B`, `Z`, `U`, `O`, `J` all become `UNK`. Sequences
should be uppercase anyway: the dummy a3m writes the sequence verbatim, and a lowercase
letter there would be read as an insertion.

**Unknown residues and gaps in MSA rows.** `map_str_array_to_idx_array` (residues.py:335)
maps the 21 protein letters to 0–20 and `-` to 31. Any other character maps to 20 (`X`).
That includes `B`, `Z`, `U`, `O`, `*` and `.`.

## 4. Tokenization (proteins)

There is one token per residue. `UNK` is in `STANDARD_RESIDUES_3`, so it is a canonical
residue too and is **not** atomized. For a pure protein query, nothing is atomized:
- `is_atomized = 0` for every token;
- no residue is "modified", because every inter-residue bond is an N–C peptide bond and
  `find_modified_residue_atom_ids` excludes those.

Tokens are ordered as follows:
- **Chains:** in query order (`query.chains`), then in `chain.chain_ids` order within a chain.
- **Residues:** in sequence order within a chain.

There is no cropping and no padding at inference: `n_tokens` is the true token count.
`pad_token_dim` (`core/data/primitives/featurization/padding.py`) is therefore a no-op.

## 5. Per-residue atoms

**Source.** Atoms come from the CCD component via biotite (`biotite.structure.info.residue(res3)`,
`lru_cache`d as `get_residue_cached`). Hydrogens are removed with `remove_hydrogens`. Then
the protein leaving atom `OXT` (`MOLECULE_TYPE_TO_LEAVING_ATOMS[PROTEIN] = ["OXT"]`) is
removed from **every** residue, **including the C-terminal one**. No residue gets an OXT.

**Order.** Atoms keep the CCD order. That order equals
`token_atom_constants.AA_NAME_TO_ATOM_NAMES` (also exposed as `TOKEN_NAME_TO_ATOM_NAMES`)
for all 21 protein residues; this was checked against biotite 1.7.1's CCD:

```
ALA N CA C O CB                          LEU N CA C O CB CG CD1 CD2
ARG N CA C O CB CG CD NE CZ NH1 NH2      LYS N CA C O CB CG CD CE NZ
ASN N CA C O CB CG OD1 ND2               MET N CA C O CB CG SD CE
ASP N CA C O CB CG OD1 OD2               PHE N CA C O CB CG CD1 CD2 CE1 CE2 CZ
CYS N CA C O CB SG                       PRO N CA C O CB CG CD
GLN N CA C O CB CG CD OE1 NE2            SER N CA C O CB OG
GLU N CA C O CB CG CD OE1 OE2            THR N CA C O CB OG1 CG2
GLY N CA C O                             TRP N CA C O CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2
HIS N CA C O CB CG ND1 CD2 CE1 NE2       TYR N CA C O CB CG CD1 CD2 CE1 CE2 CZ OH
ILE N CA C O CB CG1 CG2 CD1              VAL N CA C O CB CG1 CG2
UNK N CA C O CB CG
```

`UNK` has **6** atoms (N CA C O CB CG): the CCD `UNK` is a gamma-carbon amino acid. Every
atom is real: `atom_mask` and `ref_mask` are all ones, with no padding atoms.

## 6. Reference conformer features

These come from `featurize_reference_conformers_of3`. There is one reference molecule per
residue, in token order. Inside each molecule, atoms follow the RDKit mol's order, which is
the CCD heavy-atom order. Atoms whose `in_crop_mask` is false (OXT) are skipped.

**Conformer** (`processed_reference_molecule_from_atom_array`, query.py:182). The steps are:
1. Build the CCD heavy-atom residue, **including OXT** and the CCD formal charges.
2. Convert it with `biotite.interface.rdkit.to_mol(kekulize=True)`, run `Chem.SanitizeMol`, and remove the CCD conformer.
3. Run `compute_conformer` (conformer.py:40):
   - `Chem.AddHs`;
   - `EmbedMolecule` with `AllChem.ETKDGv3()`, `clearConfs=False` and `randomSeed = random.randint(0, 10**9)`, drawn from **Python's global `random`**;
   - `Chem.RemoveAllHs`.

   If the embed fails, it is retried with `useRandomCoords=True` and a fresh `randint`. That never happens for standard residues. The timeout is 120 s.

The conformer is therefore the free amino acid, with OXT and hydrogens present during
embedding. OXT is then dropped from the features.

| key | shape | dtype | rule |
|---|---|---|---|
| `ref_pos` | `[N_atom, 3]` | float32 | Per residue: `x` is the conformer's coordinates for the kept atoms (no OXT). The result is `(x - mean(x)) @ R.T + t`; see below. |
| `ref_mask` | `[N_atom]` | int32 | `annot_used_atom_mask`: 1 unless the conformer has NaN coordinates. Always 1 here. |
| `ref_element` | `[N_atom, 119]` | int32 | One-hot of `atomic_number - 1` (C=5, N=6, O=7, S=15); element `R` → 118. |
| `ref_charge` | `[N_atom]` | float32 | RDKit formal charge, from the CCD: **+1 on LYS NZ, ARG NH2 and HIS ND1** (CCD HIS is protonated); 0 elsewhere. ASP and GLU are neutral. The arcsinh is applied by the model or adapter, not here. |
| `ref_atom_name_chars` | `[N_atom, 4, 64]` | int32 | `name.ljust(4)` (left-justified, space padded); each char gives `one_hot(ord(c) - 32, 64)`, so a space is class 0. |
| `ref_space_uid` | `[N_atom]` | int32 | Index of the reference molecule (0-based, in token order) = token index for proteins. Homomer copies get distinct uids. |

**`ref_pos` rotation and translation** (`centre_random_augmentation`, AF3 Algorithm 19).
The centering mean runs over the kept atoms only (OXT excluded). Then
`q = torch.randn(4)` and `q /= |q|`, and `R = quat_to_rot(q)` is the standard (w, x, y, z)
quaternion-to-matrix conversion:

```
[[a²+b²-c²-d², 2(bc-ad),    2(bd+ac)   ],
 [2(bc+ad),    a²-b²+c²-d², 2(cd-ab)   ],
 [2(bd-ac),    2(cd+ab),    a²-b²-c²+d²]]
```

Next comes `t = torch.randn(3)`, with a scale of 1 Å. Both draws use **torch's global
generator**, in float32, `q` before `t`, once per residue, in token order. The result is
multiplied by `ref_mask`.

## 7. Atom/token maps and masks

| key | shape | dtype | rule |
|---|---|---|---|
| `num_atoms_per_token` | `[N_tok]` | int32 | Atom count of the token's residue, per the table in §5. |
| `start_atom_index` | `[N_tok]` | int32 | Exclusive cumsum of `num_atoms_per_token`, i.e. the token's first atom. |
| `atom_to_token_index` | `[N_atom]` | int32 | `repeat(arange(N_tok), num_atoms_per_token)` (0-based, consecutive). |
| `atom_mask` | `[N_atom]` | float32 | All ones. |
| `token_mask` | `[N_tok]` | float32 | All ones. |

The adapter derives `atom_position = arange(N_atom) - start_atom_index[atom_to_token_index]`,
the index within the token, which matches the atom order in §5.

## 8. Representative atoms

These are not upstream batch keys. They come from `get_token_representative_atoms`
(`core/utils/atomize_utils.py:621`); jfold's version is
`jfold.models.openfold3.representative_atoms`. The rule is:
- **CB** for standard protein residues, which is `start_atom_index + 4` for all non-glycine residues, **including UNK**;
- **CA** for glycine (`start_atom_index + 1`);
- C4 for purines, C2 for pyrimidines, and the first atom for atomized tokens (not relevant here).

The offsets come from `atom_name_to_index_by_restype[name]["index"]`, which is the position
of the name in `TOKEN_NAME_TO_ATOM_NAMES[restype]`, dotted with the restype one-hot. This
means a soft `restype` produces nonsense indices. jfold's `features(..., rep_atom=...)`
avoids it by taking indices precomputed from a one-hot `restype`.

## 9. Chain, entity and position ids

All five keys are `[N_tok]` int32.

| key | rule |
|---|---|
| `token_index` | `token_id`: 0-based, consecutive over the whole complex. |
| `residue_index` | `res_id`: **1-based** position in the chain's sequence (`enumerate(sequence, start=1)`). |
| `asym_id` | `np.unique(chain_id_per_token, return_inverse=True)[1] + 1`: the 1-based rank of the chain-id **string** in lexicographic order, not query order. Chains `["B", "A"]` give asym 2 to the first chain. `"A10" < "A2"`. |
| `entity_id` | `entity_to_id = {e: i+1 for i, e in enumerate(sorted({chain.sequence for chain in query.chains}))}`: the 1-based rank of the **sequence string** in lexicographic order. Identical sequences share an entity, even across separate `Chain` entries. Example: in the `dimer` fixture, chain A (`MKTAY…`) gets entity **2** and chain B (`GSHM…`) gets entity **1**. |
| `sym_id` | Per entity, a 1-based counter over chains in atom-array (query) order (`create_sym_id`). |

In jfold's `relative_position_features`, `asym_id` and `entity_id` are used only through
equality, and `sym_id` only through differences. The exact numbering still matters for
array parity and for templates (an `asym_id` equality mask).

## 10. Bonds and molecule type

| key | shape | dtype | rule |
|---|---|---|---|
| `token_bonds` | `[N_tok, N_tok]` | int32 | `create_token_bonds` keeps only bonds whose **both** atoms are atomized (`filter_fully_atomized_bonds`), then sets both `[i, j]` and `[j, i]`. For proteins, the result is **all zeros**: peptide bonds are not counted. |
| `is_protein` | `[N_tok]` | int32 | 1 (from `molecule_type_id == PROTEIN`). |
| `is_rna`, `is_dna`, `is_ligand`, `is_atomized` | `[N_tok]` | int32 | 0. jfold does not read `is_ligand`. |
| `restype` | `[N_tok, 32]` | int32 | `one_hot(RESTYPE_INDEX_3.get(res_name, UNK))`. |

## 11. MSA

### Parsing an a3m

`parse_a3m` (msa.py:96), via `parse_fasta`, works as follows:
- **Records:** a record starts at `>`. Lines starting with `#` and blank lines are skipped. Wrapped sequence lines are concatenated after `strip()`.
- **Insertions:** lowercase letters (`str.islower`) are insertions. They are removed, and each one adds 1 to a running count. That count is stored on, and reset at, the next non-lowercase character.
- **Match columns:** every non-lowercase character is a column, including `-`, `.`, `*` and digits. `.` is therefore **not** treated as an insertion gap; it becomes a column that maps to `X` (20).
- **Deletion counts:** an insertion run after the last column is dropped. Columns map to residues by `res_id - 1`, so row length must be at least the chain length; extra columns are ignored.
- **No truncation:** the `max_seq_count` truncation in `parse_a3m` is a no-op, because `parsed_msa.truncate(...)` returns a new object that is discarded. All rows are kept.

### Rows

For each chain, `vstack_pad_msa_arrays` stacks, from top to bottom:
1. **The query row** (`create_query_seqs`). This is row 0 of the chain's a3m, which is the first alphabetically sorted alignment stem. Its deletions are zeroed.
2. **Paired rows.** There are none here. Pairing is attempted only for heteromers whose alignment stems are in `msas_to_pair = ["uniprot_hits", "uniprot"]`, or when `paired_msa_file_paths` is given. Neither happens with `colabfold_main.a3m`, so `num_paired_seqs == [1]`.
3. **Main rows.** These are all rows of the chain's main alignments concatenated in `aln_order`, **including the a3m's own first (query) row**. The query therefore appears **twice** (rows 0 and 1).

Row limits work as follows:
- **Main rows:** at most `max(0, max_rows - n_paired - 1) = 16383`, where `max_rows = 16384`.
- **With `subsample_main=False`:** the first rows are kept, in order.
- **With `subsample_main=True`** (the bare `MSASettings` default): `k = np.random.randint(1, min(n, lim) + 1)` and `idx = np.random.choice(n, k, replace=False)`, unsorted, from **numpy's global RNG**.

Chains are written into their token columns, which share rows. The row count is
`N_msa = 1 + n_paired + max over chains (main rows)`. A chain with fewer rows is padded
at the bottom with `-` (class 31) and deletion 0. A query-only chain contributes rows 0
and 1 (the query twice), then gaps.

### Keys

| key | shape | dtype | rule |
|---|---|---|---|
| `msa` | `[N_msa, N_tok, 32]` | int32 | One-hot of `map_str_array_to_idx_array`, as in §3. |
| `has_deletion` | `[N_msa, N_tok]` | float32 | `deletion != 0`. |
| `deletion_value` | `[N_msa, N_tok]` | float32 | `atan(d / 3) * (2.0 / acos(0) * 2)`, which is **`(8/π)·atan(d/3)`** with range [0, 4), not AF3's `(2/π)·atan(d/3)`. The operator precedence is upstream's: `2.0 / torch.acos(torch.zeros(1)) * 2`. Verified: `d = 2` gives 1.4973. |
| `msa_mask` | `[N_msa, N_tok]` | float32 | **All ones**, including padded gap rows. `map_msas_to_tokens` is called with `msa_array_vstack_mask=None`; only padding tokens would be zeroed, and there are none. |
| `profile` | `[N_tok, 32]` | float32 | See below. |
| `deletion_mean` | `[N_tok]` | float32 | `mean(deletion_matrix, axis=0)` over the chain's main MSA (all a3m rows, query included, no extra query row). The column is `res_id - 1`. |
| `num_paired_seqs` | `[1]` | int32 | `n_paired + 1` = `[1]`. jfold does not read it. |

**`profile` is not a column histogram when the MSA has more than one row.** It is computed
on the chain's main MSA before row limiting, including the a3m query row but not the
extra row 0. `calculate_profile` (`core/data/primitives/sequence/msa.py:1193`) mixes up
the index order:

```python
val_indices = msa_chunk.ravel()                                   # row-major
col_indices_local = np.repeat(np.arange(block_n_cols), n_rows)    # column-major pairing
counts = bincount(col_indices_local * 32 + val_indices).reshape(block_n_cols, 32)
return counts / n_rows
```

With `R` rows and `C ≤ 1000` columns (chunks are 1000 columns wide), the formula is
`profile[j] = bincount(idx.ravel()[j*R:(j+1)*R], minlength=32) / R`. Here `idx` is the
`[R, C]` index matrix of the chain's main MSA, and the slice is a run of `R` consecutive
entries of the row-major flattening, not column `j`. In the `monomer_msa` fixture, for
example, `profile[0]` is uniform over `M K T A Y I`, which is row 0's first six residues.

For a query-only MSA (`R = 1`), the result is the one-hot of the query, which is correct.
Rows sum to 1, with the gap included as class 31. The view reproduces the scrambled
formula, chunked per 1000 columns (`_profile`), on the assumption that the released
weights were trained with this code (not checked against the training pipeline).

## 12. Templates

### Without templates

`n_templates = 4` (`TemplateSettings`), all empty:

| key | shape | dtype | value |
|---|---|---|---|
| `template_restype` | `[4, N_tok, 32]` | int32 | One-hot of GAP (class 31) everywhere. |
| `template_pseudo_beta_mask` | `[4, N_tok]` | float32 | 0 |
| `template_backbone_frame_mask` | `[4, N_tok]` | float32 | 0 |
| `template_distogram` | `[4, N_tok, N_tok, 39]` | float32 | 0 |
| `template_unit_vector` | `[4, N_tok, N_tok, 3]` | float32 | 0 |

The jfold adapter counts all 4 templates (`template_mask = ones(4)`), as OpenFold3's
template average does, empty ones included.

### A template structure for a chain

Upstream takes templates as a per-chain **template alignment** (`Chain.template_alignment_file_path`:
`.sto`, `.a3m` or `.m8`) whose rows name template chains `<entry>_<chain>`, plus the
structures as `<structure_directory>/<entry>.cif` (`TemplatePreprocessorSettings`). An
`.m8` hit (or an a3m without start-end coordinates) carries no residue-wise alignment, so
upstream realigns the template's sequence to the chain; that is the route for a template
structure alone, and the one the view reproduces: jfold's `Chain.template` (a `Template`,
the template's sequence and each residue's atoms by name, unresolved residues with none)
is that template chain. *(verified on the systems of `template_systems.json`)*

**The installed pipeline drops every template at inference.**
`InferenceDataset.create_template_features` (`core/data/framework/single_datasets/inference.py:261`)
passes `template_cache_directory=None` to `process_template_structures_of3`, and
`sample_templates` (`core/data/primitives/structure/template.py:126`) returns no templates
without one (it returns early without a structure-array directory, and loads the cache
entry only `if k > 0 and template_cache_directory is not None`, line 210). Upstream
OpenFold3 has since passed `template_preprocessor_settings.cache_directory` there; the
fixtures are made with that one change (`dump_openfold3.py`), and the view follows it.
Everything else below is the installed code.

1. **Preprocessing** (`TemplatePreprocessor`, run by the inference data module before the
   dataset; `core/data/pipelines/preprocessing/template.py:1801`). Per template alignment,
   keyed by the chain's **sequence**: a second chain with the same sequence is skipped
   (`if input_data.query_seq_str in self.seq_hash_map`, line 1831), and afterwards every
   protein chain gets the cache entry of its sequence (`_update_inference_query_set`,
   line 1731), templated or not. So templates go by sequence: a homodimer with one copy
   templated has both copies templated, and of two chains with one sequence and different
   templates the first (input order, `n_processes = 1`) wins. The view keeps the first
   template per sequence.
2. **Realignment** (step E, line 1995; `A3mParser(realign=True)`, `core/data/io/sequence/template.py:648`).
   The template sequence is the structure's canonical sequence of that chain
   (`entity_poly.pdbx_seq_one_letter_code_can`, unresolved residues included, `X` for
   UNK). `run_kalign` (`kalign.align`, kalign-python 3.6.0, default mode) aligns three
   sequences: the query, **the query again** (the a3m's first row, also realigned), and
   the template; the query's and the template's rows give `idx_map` (`calculate_ids_hit`,
   line 380: 1-based query and template residue indices per alignment column, -1 for a
   gap, columns gapped in both dropped). The view calls `kalign.align` the same way
   (`template_alignment`); Kalign is a C library with no simple reimplementation (default
   mode: PFASUM60, consistency anchors, inline refinement), so the view needs the
   `kalign` package, except for a template with its chain's own sequence, which aligns
   to itself. No sequence-identity, coverage, length or date filters by default.
3. **A template must cover its chain.** `subset_template_index_map` (line 256) drops the
   columns whose query index is -1 but keeps those whose template index is -1, so a query
   residue facing a gap is counted among the query tokens but matches no template
   residue; `map_token_pos_to_template_residues` (line 419) then finds fewer template
   residues than query tokens and drops the template (`# Skip if still misaligned`, line
   528). A truncated template, or one with a deletion, is therefore never used. The view
   warns and drops it.
4. **Chains shorter than 5 tokens** get no templates (`min_n_tokens_per_chain = 5`,
   `core/data/pipelines/sample_processing/template.py:109`).
5. **The structure** (`parse_template_structure` → `parse_mmcif`, then
   `clean_template_atom_array`, line 374): label (PDB-assigned) residue numbering, waters
   and hydrogens removed, atoms not in the residue's CCD entry removed, and
   `add_unresolved_atoms` adds every missing CCD atom of a resolved residue and every
   unresolved residue of the sequence, with NaN coordinates. Coordinates are float32.
6. **Placement** (`create_template_feature_precursor_of3`, `core/data/primitives/featurization/template.py:58`).
   Templates are counted per chain, so a chain's first template fills slot 0 (one
   template per chain: only slot 0 is ever filled). Each query token gets its aligned
   template residue's name, its pseudo-beta atom (CB; CA for GLY) and its N, CA, C
   coordinates, NaN where missing. A template with any residue lacking a pseudo-beta
   *atom record* is skipped (line 115); after step 5 that cannot happen for standard
   residues or UNK.
7. **Features** (`featurize_template_structures_of3`, `core/data/pipelines/featurization/template.py:49`):

| key | shape, dtype | value |
|---|---|---|
| `template_restype` | `[4, N_tok, 32]` int32 | one-hot of the template residue's class (`get_with_unknown_3_to_idx`: UNK for X); GAP on tokens without one |
| `template_pseudo_beta_mask` | `[4, N_tok]` float32 | 1 where the pseudo-beta coordinate is not NaN |
| `template_backbone_frame_mask` | `[4, N_tok]` float32 | 1 where N, CA and C are all not NaN |
| `template_distogram` | `[4, N_tok, N_tok, 39]` float32 | squared pseudo-beta distance (float64) binned against `linspace(3.25, 50.75, 39)**2`, `lower < d2 < upper`, last upper 1e8; times both pseudo-beta masks and the same-chain mask |
| `template_unit_vector` | `[4, N_tok, N_tok, 3]` float32 | see below; times both frame masks and the same-chain mask |

The same-chain mask (`multichain_pair_mask`, line 106) zeroes every pair across chains,
also between copies of a homodimer that share slot 0.

**Unit vectors** (`create_template_unit_vector`, line 259; float32, missing atoms at the
origin): the frame `make_transform_from_reference(N, CA, C)` is
`Rot3Array.from_two_vectors(C - CA, N - CA)`: `e0 = normalized(C - CA)`,
`e1 = normalized((N - CA) - ((N - CA)·e0) e0)`, `e2 = e0 × e1`, origin CA, with
`normalized(v) = v / sqrt(max(v·v, 1e-12))`. The entry for (i, j) is
`normalized(R_i^T CA_j + R_i^T (-CA_i))` (the inverse frame applied to the point; not
`R_i^T (CA_j - CA_i)`). The view does the same float32 operations in the same order;
`torch.sqrt` on float32 CPU tensors is not always correctly rounded (off by one ulp
against numpy's here), so unit vectors agree to a few float32 ulps, not bit for bit.

**mosaic.** `mosaic.models.of3._inject_templates` does not use this pipeline; it writes its
own slot-0 features from a `gemmi.Chain` (residue i of the chain on query token i, no
alignment).

## 13. Batch dimension, padding and dtypes

`InferenceDataset[i]` returns **unbatched** tensors, with no padding.
`jfold.models.openfold3.features` takes them unbatched, and the view returns them
unbatched. The upstream model and the checkpoint test add the batch dimension with
`v[None]`.

Non-tensor entries are `atom_array` (biotite) and `query_id`; bookkeeping tensors are
`seed`, `repeated_sample` and `valid_sample` (bool, `[1]`). The view returns none of these.

The dtypes above match what `np.savez_compressed` stores in the fixtures. One-hot features are int32
and are cast to float by the consumers.

## 14. Randomness

| source | what | how to fix it |
|---|---|---|
| Python `random` | ETKDG seed, one `randint(0, 10**9)` per residue, drawn while the atom array is built (all chains, in token order) | `random.seed(s)` before `dataset[i]` |
| torch global RNG | `ref_pos` rotation `randn(4)` and translation `randn(3)` per residue, after all conformers | `torch.manual_seed(s)` |
| numpy global RNG | main-MSA subsampling, only if `subsample_main=True` | `subsample_main=False`, or `np.random.seed(s)` |

Model side, not featurization: every trunk pass, `MSAModuleEmbedder._subsample_all_msa`
reads 1024 rows (`min_subsampled_all_msa = max_subsampled_all_msa = 1024`) drawn
uniformly, rows with a real `msa_mask` first, the query like any other (every row when
`S <= 1024`), from torch's global RNG. jfold's `MSAModule` draws the same way
(`max_rows=1024, keep_query=False`; see `README.md`).

The `seeds` field and the `seed` tensor do **not** seed featurization. OpenFold3's runner
seeds the globals per worker (`core/utils/callbacks.py`). Everything except `ref_pos` is
deterministic. `ref_pos` is reproducible for fixed seeds, but only with the same RDKit
build (ETKDG output changes across RDKit versions) and torch build. The values here are
from RDKit 2026.09.1 and torch 2.11.0. Two calls in one process give identical output, as
do two separate processes.

Bit-matching `ref_pos` would need RDKit ETKDGv3 with the same seed stream. Conformers of
the same residue differ between seeds in side-chain torsions, not just by a rigid
transform, so no fixed conformer matches upstream up to a rigid motion.

**The view** uses a stored table instead (`src/jfold/featurize/data/openfold3.npz`, from
`tools/tables/extract_openfold3.py`): per residue, the 32 conformers upstream's
procedure produces under `random.seed(s)` for s = 0..31, stored centred (OXT excluded).
With a `key`, each residue draws one of the 32 uniformly, then a rotation from a
normalized Gaussian quaternion and an N(0, 1) Å translation. With `key=None`, every
residue takes conformer 0, centred and unposed.

The parity test (`tests/test_featurize_openfold3.py`) therefore checks `ref_pos`
structurally: the same bonded atom pairs, bond lengths within 0.05 Å of some table
conformer (ETKDG does not minimize, so bond lengths vary by up to ~0.1 Å between draws),
and each posed residue a rigid motion of a table conformer.

## 15. Dependencies of the upstream path

- **torch:** tensor construction, `one_hot`, the `ref_pos` augmentation RNG, and `atan` for `deletion_value`.
- **biotite** (1.7.1): CCD access through `biotite.structure.info` (bundled CCD; no `ccd_file_path`), the AtomArray, `connect_via_residue_names`, and `biotite.interface.rdkit.to_mol`.
- **rdkit:** sanitization, `AddHs`, ETKDGv3 embedding, `RemoveAllHs`, element symbols and formal charges.
- **func_timeout:** the conformer timeout.
- **pydantic:** configs and queries.
- **pandas:** MSA metadata.
- **kalign-python:** template realignment (§12).
- **gemmi:** not used on this path (the dump script writes the template mmCIF with it).

The view needs, for proteins, Kalign for templates, and otherwise only the residue table (§5 atoms, elements, charges, and
the stored conformers of §14), the a3m parser and the formulas above.

## 16. What jfold reads

- **`jfold.models.openfold3.features`** reads `ref_pos`, `ref_charge`, `ref_mask`, `ref_element`, `ref_atom_name_chars`, `ref_space_uid`, `atom_to_token_index`, `atom_mask`, `restype`, `profile`, `deletion_mean`, `asym_id`, `residue_index`, `entity_id`, `token_index`, `sym_id`, `token_mask`, `is_protein`, `is_dna`, `is_rna`, `is_atomized`, `start_atom_index`, `token_bonds`, `msa`, `has_deletion`, `deletion_value`, `msa_mask` and the five `template_*` keys.
- **The view** returns those plus `num_atoms_per_token`, `is_ligand` and `num_paired_seqs`: every tensor upstream returns except the bookkeeping ones (§13).

## 17. Parity fixtures

`tools/fixtures/dump_openfold3.py` (`uv run python tools/fixtures/dump_openfold3.py`)
writes `tests/featurize/data/openfold3_<system>.npz` for the systems of `systems.json` and
`template_systems.json` (names as arguments dump those only):
- **Contents:** every tensor of `InferenceDataset(config)[0]`, as numpy, **without a batch dimension**, in upstream dtypes. `atom_array`, `query_id`, `seed`, `repeated_sample` and `valid_sample` are dropped.
- **Seeding:** `random`, `numpy` and `torch` are all seeded with 0 immediately before `dataset[0]`.
- **Determinism:** each system is featurized twice, and the script checks the two results are equal.
- **MSA:** the a3m is copied to `<tmp>/<chain id>/colabfold_main.a3m`.
- **Templates:** one mmCIF, `tmpl.cif`, holds every templated chain renamed to its query
  chain, with the metadata upstream reads (canonical sequence, `pdbx_poly_seq_scheme`,
  `chem_comp` types, a release date); each templated chain's alignment is an `.m8` line
  naming `tmpl_<chain id>`. `TemplatePreprocessor` runs first, and the dataset gets the
  template cache directory (§12).

Only `ref_pos` depends on the seeds and on the RDKit and torch builds.

| system | tokens | atoms | MSA rows |
|---|---|---|---|
| monomer | 22 | 186 | 2 (query, query) |
| dimer | 31 | 253 | 2 |
| monomer_msa | 22 | 186 | 7 (query + the 6 a3m rows) |

The template systems: `monomer_template` (full-length template), `homodimer_template` (one
copy templated, both get it), `monomer_template_unresolved` (two unresolved residues, a
missing C, an X), `dimer_template_covering` (a longer template, placed at an offset, and
one missing CB and N), and two whose templates upstream drops, `dimer_template` (a
truncated template) and `monomer_template_gap` (a deletion).

## 18. Shared vs specific

**Generic AF3-style** (shared with any AF3-family view):
- one token per standard residue;
- CCD heavy-atom sets in CCD order, with hydrogens dropped;
- per-residue reference conformers with a random rotation and translation (Algorithm 19), and `ref_space_uid` per residue instance;
- `ref_element` and `ref_atom_name_chars` as one-hots (`ord(c) - 32`, 64 classes, 4 characters);
- `ref_charge` as the CCD formal charge;
- `token_index`, `asym_id`, `entity_id`, `sym_id` and `residue_index` with AF3 Table 5 semantics;
- `token_bonds` only for non-polymer (atomized) bonds;
- the 32-class residue vocabulary layout (20 aa, UNK, 5 RNA, 5 DNA, GAP) and the `msa`, `has_deletion` and `deletion_mean` definitions;
- the representative atom: CB, or CA for glycine;
- templates with 39 distogram bins and a unit vector.

**OpenFold3-specific conventions:**
1. **`OXT`:** stripped from every residue, including C-termini; it is present only while embedding the conformer.
2. **`UNK` atoms:** the 6 atoms of CCD `UNK` (`… CB CG`).
3. **ETKDG seeding:** the seed comes from Python's global `random`; the augmentation uses torch's global RNG (§14).
4. **`ref_element`:** 119 classes (`Z - 1`, plus 118 for unknown), not AF3's 128.
5. **`asym_id`:** lexicographic rank of the chain-id string.
6. **`entity_id`:** lexicographic rank of the sequence string, not order of first appearance.
7. **MSA query row:** prepended to the main MSA, so it appears twice.
8. **`deletion_value`:** `(8/π)·atan(d/3)`, not `(2/π)·atan(d/3)`.
9. **`profile`:** the row-major/column-major mixup in `calculate_profile`; exact only for single-row MSAs.
10. **`msa_mask`:** all ones, even on the gap rows that pad shorter chains. Those rows are gaps, not masked out.
11. **a3m:** `.` is a match column that maps to `X`.
12. **Alignment keys:** keyed by file stem and parent directory; unknown stems are silently dropped.
13. **Templates:** 4 slots, all counted, empty ones GAP one-hot; a chain's template (by sequence) in slot 0, realigned with Kalign, dropped unless it covers every residue; nothing across chains (§12).
14. **No padding:** no token or atom padding at inference, and an unbatched output.

## Not covered

Templates: alignments given residue-wise (an `.sto`, or an `.a3m` with the query first
and start-end coordinates, used as written), several templates per chain (slots 1-3),
template filters (sequence identity, coverage, release date), and modified residues in
templates. Paired MSAs; multiple alignment files per chain, which are sorted by stem and concatenated
in `aln_order` (profile and deletion mean are then computed over the concatenation); and
`.sto` input.
