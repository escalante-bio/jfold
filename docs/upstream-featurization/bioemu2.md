# BioEmu2 featurization (proteins only)

This spec covers how the upstream `bioemu2` package (0.1.0rc0, as installed in jfold's
environment) turns a protein-only system into the feature dict that
`jfold.models.bioemu2.features` consumes. It is the reference for
`jfold.featurize.views.bioemu2`, which reproduces every key it returns exactly.

Scope: standard amino acids plus `X`, one or more chains, unpaired MSAs, no ligands,
nucleic acids, modifications, covalent bonds, contacts or templates.

Paths are relative to `.venv/lib/python3.12/site-packages/`. The parity fixtures come from
`tools/fixtures/dump_bioemu2.py` (see [Parity fixtures](#parity-fixtures)).

## 1. Call chain

How it is driven (`tests/test_bioemu2.py::test_bioemu2_preview`,
`tools/fixtures/dump_bioemu2.py`):

```python
system = System(entities=[Protein(sequence=s, asym_ids=[c]) for c, s in chains])
msa = MSA(unpaired_a3m={c: a3m_text, ...})          # or bioemu2.load_a3m_msa / get_msa(system, path)
structure = prepare_structure(system, msa=msa, mode="structure", contacts=[])
feats = convert_densebatch_to_dictionary(batch_collate([structure]))
f = {k: v[0].numpy() for k, v in feats.items()}     # unbatched: what features() takes
```

1. `bioemu2/data_interface/system.py:System._validate_entities` (field validator).
   - Merges entities that compare equal (for proteins: same sequence) into the FIRST such
     entity, appending their `asym_ids`.
   - This reorders chains (see §8).
2. `bioemu2/structure/preparation.py:prepare_structure`.
   - Checks that the MSA chain ids equal the protein chain ids.
   - Then calls `AllAtomStructure.from_system(system, mode=mode, contacts=contacts)`.
3. `bioemu2/structure/all_atom_structure.py:_get_structure_from_system`. It runs:
   - `get_asym_id_to_chain_metadata` (line 472). `entity_id` is the entity's index in the
     merged `system.entities`. Each entity's per-chain structure comes from
     `_get_polymer_structure`.
   - `_get_polymer_structure` calls
     `bioemu2/structure/inference_inputs.py:inference_input_structure_from_protein_sequence`
     with `template_cache=bioemu2/structure/templates.py:load_polymer_templates()`. That
     loads the packaged `bioemu2/assets/polymer_templates.pt`.
   - Residue templates come from `inference_inputs.py:_templates_from_protein_sequence`,
     which drops OXT.
   - `bioemu2/structure/builder.py:StructureBuilder.add_chain` / `.build` concatenates the
     chains and offsets the indices.
   - `_add_contact_conditioning` runs only if `contacts` is non-empty (never in this scope).
   - `add_mode_conditioning(structure, mode)` adds `method_feature`.
4. Back in `prepare_structure`, the MSA:
   - `bioemu2/msa/colabfold.py:tokenize_msa`, which calls `_validated_a3m_sequences`,
     `_parse_a3m_sequences` and `_build_msa_from_colabfold_seqs`;
   - then `bioemu2/msa/arrays.py:_prepare_msa_arrays` and `_prepare_msa_arrays_inner`
     (numba);
   - then `bioemu2/msa/features.py:process_msa_features`.
5. `bioemu2/structure/batch.py:batch_collate(batch, window_size=32)`.
   - Adds the dynamics placeholders.
   - Pads atoms to a multiple of 32.
   - Builds `atom_pad_mask` and `token_pad_mask`.
6. `bioemu2/biomoa/utils/feature_converter.py:convert_densebatch_to_dictionary`.
   - Casts masks to bool.
   - Adds the contact fallback, `pocket_feature`, `ref_space_uid`, `res_type`,
     `residue_index` and `token_bonds`.
   - Transposes the MSA tensors to depth-first.

`jfold/models/bioemu2.py:features(f, paired=True)` reads the keys in §2.

MSA input without a server:
- `bioemu2.load_a3m_msa(system, {chain_id: Path})` reads the file text.
- `get_msa(system, Path)` does the same for a single unique sequence.
- Both check the query and return `MSA(unpaired_a3m={chain: text})`. They are equivalent to
  building `MSA(unpaired_a3m=...)` from the raw text.
- Chains with no alignment get the single-sequence A3M `">query\n{sequence}\n"`, as the
  test and the dump script do.

## 2. Keys

Notation: `T` = tokens (one per residue), `N` = real atoms, `Np = ceil(N/32)*32` (padded
atoms), `S` = MSA rows. All shapes are unbatched. With a single system there is no token or
MSA-row padding. The view returns exactly these keys.

| key | shape | dtype | used by |
|---|---|---|---|
| `template_coords` | (Np, 3) | float32 | jfold `ref_pos` and atom features |
| `charge` | (Np,) | float32 | atom features |
| `element` | (Np,) | int64 | atom features (one-hot 128) |
| `name_encoding` | (Np, 4) | int64 | atom features (one-hot 64 per char) |
| `ref_space_uid` | (Np,) | int64 | atom pair mask |
| `token_idx_at_atom` | (Np,) | int64 | atom to token |
| `atom_pad_mask` | (Np,) | bool | atom mask |
| `atom_mask` | (Np,) | bool | `Conditioning.coordinate_mask` |
| `token_pad_mask` | (T,) | bool | token mask |
| `token_center_atom_index` | (T,) | int64 | `rep_atom`, `Conditioning.center_atom` |
| `res_type` | (T,) | int64 | token features (one-hot 33) |
| `profile` | (T, 33) | float32 | token features |
| `deletion_mean` | (T,) | float32 | token features |
| `pocket_feature` | (T,) | int64 | token features (one-hot 4) |
| `asym_id`, `entity_id`, `sym_id`, `residue_index` | (T,) | int64 | relative positions |
| `token_bonds` | (T, T, 1) | float32 | `bonds` |
| `msa` | (S, T, 33) | int64 (one-hot) | MSA features |
| `has_deletion` | (S, T) | bool | MSA features |
| `deletion_value` | (S, T) | float32 | MSA features |
| `msa_paired` | (S, T) | int64 | MSA features (if `paired=True`) |
| `msa_mask` | (S, T) | int64 | `msa_mask` |
| `method_feature` | (T,) | int64 | `Conditioning.method` (one-hot 17) |
| `contact_conditioning` | (T, T, 5) | int64 (one-hot) | `Conditioning.contacts` |
| `contact_threshold` | (T, T) | float32 | `Conditioning.contact_threshold` |
| `coords_cond` | (Np, 3) | float32 | `Dynamics.coords` |
| `dt` | (1,) | float32 | `Dynamics.lag` |
| `is_dynamics_conditioned` | (1,) | bool | `Dynamics.is_conditioned` |
| `mol_type` | (Np,) | int64 | not read by jfold (protein = 0) |

The dict also carries keys that jfold does not read and the view does not return:
- `coords`: zeros.
- `residue_idx`: atom-level, equal to `ref_space_uid`.
- `residue_type`: atom-level `res_type`.
- `distogram_index`: global index of CB, or CA for GLY; UNK has a CB.
- `bond_index` and `template_bond_index`: (2, 0).
- `bond_type`: (0,).
- `bond_pad_mask`: (2, 0).
- `num_nodes`, `num_residues`, `num_tokens`: (1,).

The fixtures keep all of them.

## 3. Token vocabulary (33)

Source: `bioemu2/data_interface/const.py:Token`, the Boltz-1 alphabet.

| idx | token | idx | token | idx | token |
|---|---|---|---|---|---|
| 0 | PAD | 11 | ILE | 22 | UNK_PROT |
| 1 | GAP `-` | 12 | LEU | 23-26 | RNA A, G, C, U |
| 2 | ALA | 13 | LYS | 27 | UNK_RNA |
| 3 | ARG | 14 | MET | 28-31 | DNA A, G, C, T |
| 4 | ASN | 15 | PHE | 32 | UNK_DNA |
| 5 | ASP | 16 | PRO | | |
| 6 | CYS | 17 | SER | | |
| 7 | GLN | 18 | THR | | |
| 8 | GLU | 19 | TRP | | |
| 9 | GLY | 20 | TYR | | |
| 10 | HIS | 21 | VAL | | |

So amino acids 2..21 are in alphabetical order of three-letter code (one-letter order
`ARNDCQEGHILKMFPSTWYV`). One-letter mapping is `PROT_LETTER_TO_TOKEN`:
- the 20 standard letters map to their amino acids;
- `X`, `J`, `B`, `Z`, `O`, `U` map to 22 (UNK_PROT);
- `-` maps to 1 (GAP).

Allowed letters:
- **System sequence** (`Protein` validator): uppercase only, the 20 standard letters plus
  `X`. `J`, `B`, `Z`, `O`, `U` are rejected. `-` passes validation but then fails the
  template lookup.
- **A3M rows:** any key of `PROT_LETTER_TO_TOKEN` (so `B`, `Z` and the rest are accepted and
  map to 22), plus lowercase insertions. Any other character, including `.`, raises
  `KeyError`.

## 4. Tokenization

There is one token per residue. A token's atoms are exactly its residue's heavy atoms.

`res_type[t]` is the residue's token index:
`res_type = residue_type[token_center_atom_index]`. For an `X` residue it is 22.

## 5. Atoms per residue

Source: `load_polymer_templates()`, the packaged `assets/polymer_templates.pt`, keyed by
three-letter name, with `UNK_PROT` for `X`. Then `_templates_from_protein_sequence` applies
`oxt_mask = ~template.name_encoding_is_atom("OXT")`.

- **Hydrogens:** none. `template_from_cifwrapper` keeps `elements != 1`.
- **OXT:** removed from every residue, including the C-terminal one.
- **Atom order:** CCD `_chem_comp_atom` order, heavy atoms only, without OXT:

| res | n | atoms |
|---|---|---|
| ALA | 5 | N CA C O CB |
| ARG | 11 | N CA C O CB CG CD NE CZ NH1 NH2 |
| ASN | 8 | N CA C O CB CG OD1 ND2 |
| ASP | 8 | N CA C O CB CG OD1 OD2 |
| CYS | 6 | N CA C O CB SG |
| GLN | 9 | N CA C O CB CG CD OE1 NE2 |
| GLU | 9 | N CA C O CB CG CD OE1 OE2 |
| GLY | 4 | N CA C O |
| HIS | 10 | N CA C O CB CG ND1 CD2 CE1 NE2 |
| ILE | 8 | N CA C O CB CG1 CG2 CD1 |
| LEU | 8 | N CA C O CB CG CD1 CD2 |
| LYS | 9 | N CA C O CB CG CD CE NZ |
| MET | 8 | N CA C O CB CG SD CE |
| PHE | 11 | N CA C O CB CG CD1 CD2 CE1 CE2 CZ |
| PRO | 7 | N CA C O CB CG CD |
| SER | 6 | N CA C O CB OG |
| THR | 7 | N CA C O CB OG1 CG2 |
| TRP | 14 | N CA C O CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2 |
| TYR | 12 | N CA C O CB CG CD1 CD2 CE1 CE2 CZ OH |
| VAL | 7 | N CA C O CB CG1 CG2 |
| X (UNK_PROT) | 5 | N CA C O CB: **the ALA template, every field identical to ALA's**; not CCD `UNK`, which has a CG |

Atoms follow residues in token order, with no gaps between chains.

## 6. Reference conformer features

**`template_coords`** (this is `ref_pos`):
- Source: the CCD ideal coordinates (`pdbx_model_Cartn_{x,y,z}_ideal`).
- Centering:
  - Coordinates are centered by the mean over the heavy atoms INCLUDING OXT, computed in
    float64.
  - Then they are cast to float32.
  - Only then is OXT dropped.
  - So each residue's coordinates are NOT centered on its own atoms; the centroid is
    typically 0.1 to 0.4 Å off.
- No rotation and no randomness: every copy of a residue type gets identical coordinates.
  These are local per-residue frames, not placed along a chain.
- Pad atoms are 0.

Exact recipe, verified bit-exact for all 20 residues against the packaged `.pt` using
current RCSB CCD files read with gemmi:

```python
rows = [r for r in chem_comp_atom if r.type_symbol != "H"]            # CCD order
xyz = np.array([[x_ideal, y_ideal, z_ideal] for r in rows], float64)
pos = (xyz - xyz.mean(0)).astype(np.float32)
pos = pos[[i for i, r in enumerate(rows) if r.atom_id != "OXT"]]
```

**`charge`:**
- The CCD formal charge as float32 (`?` or `.` become 0).
- In this scope the only nonzero values are +1.0 on ARG `NH2`, HIS `ND1` and LYS `NZ`. The
  CCD stores the protonated forms; ASP and GLU are neutral.
- Pads are 0.

**`element`:**
- The atomic number as int64, from `mdtraj.element` by CCD `type_symbol`: N 7, C 6, O 8,
  S 16.
- Pads are 0.
- jfold one-hot encodes it to 128 classes.

**`name_encoding`:**
- `encode_atom_name`: `[ord(c) - 32 for c in name] + [0] * (4 - len(name))`, as int64
  (N, 4).
- The name is left-aligned, with 0 as padding. For example `CA` becomes `[35, 33, 0, 0]`.
- Pad atoms are `[0, 0, 0, 0]`.
- jfold one-hot encodes each character to 64 bins.

**`ref_space_uid`:**
- `= residue_idx`: a global residue counter across all chains (0..R-1), the same for every
  atom of a residue.
- For proteins it equals `token_idx_at_atom`. Pad atoms are -1.
- Upstream re-masks pads itself, so the pad value only has to be masked out.

## 7. Atom/token maps and masks

**`token_idx_at_atom`:**
- The token index of each atom (global; `StructureBuilder` adds `num_tokens` per chain).
- Pad atoms are **-1** (`batch_collate` pads `token_idx_at_atom` and `residue_idx` with
  -1).
- jfold casts it to int32 and indexes with it. A -1 wraps to the last token, which is
  harmless because the pad atoms are masked.

**`token_center_atom_index`:**
- The global atom index of each residue's `CA` (`torch.where(name == "CA")`).
- Since N, CA come first in every template, it is the residue's first atom index + 1.

**`atom_pad_mask`:** True for the N real atoms, False for the `Np - N` pad atoms.
- `Np = ceil(N / 32) * 32` (`window_size=32`). An N that is already a multiple of 32 gets
  no padding.
- Example: monomer, N=186, Np=192.

**`atom_mask`:**
- All real atoms are True (`torch.ones_like(element).bool()`); pads are False.
- So it equals `atom_pad_mask` in this scope.

**`token_pad_mask`:** all True, shape (T,).

**`mol_type`:** 0 (protein) for every atom, and 0 for pads too.

## 8. Chain, entity and position ids

Chain order:
- The order is that of the merged entities. Within an entity, chains follow `asym_ids` order.
- `System._validate_entities` folds a later entity that equals an earlier one into the
  earlier one.
- Example: entities `A:s1, B:s2, C:s1` give the order **A, C, B** (verified).
- `System.chain_ids` reports this order.

The ids:
- **`asym_id`** = the chain's position in that order (0, 1, ...). These are not the chain
  letters.
- **`entity_id`** = the index of the merged entity. Identical sequences share an entity id,
  even when they were passed as separate `Protein`s.
- **`sym_id`** = the copy number within the entity (`StructureBuilder.sym_count`): 0, 1, ...
  for the chains of a homomer, 0 otherwise.
- **`residue_index`** = `residue_idx[token_center_atom_index]`. It is a **global residue
  counter that does NOT restart at chain boundaries**: the dimer gives 0..21 for A, then
  22..30 for B.
  - Upstream's relative position encoder only takes differences within a chain
    (`b_same_chain`), so a per-chain index would give the same model output.
  - The fixtures, however, carry the global index.
  - Padded tokens (batches only) are -1.

## 9. Bonds

**`token_bonds`** is all zeros, (T, T, 1) float32, for proteins.
- `inference_input_structure_from_protein_sequence` returns
  `bond_index=torch.zeros((2, 0))`. Template bonds are dropped and no peptide bonds are
  added.
- For other inputs the converter would scatter bonded atom pairs onto token pairs,
  symmetrize, and clamp to [0, 1].

## 10. Conditioning inputs

**`pocket_feature`:** `zeros(T)` int64, always (`convert_densebatch_to_dictionary`).

**`method_feature`:** `full(T, id)` int64 (`add_mode_conditioning`).
- `"structure"` maps to `"non-md"` = **7**.
- `"equilibrium"` maps to `"md"` = **4**.
- The full table is in `bioemu2/conditioning/method.py`: 17 distinct ids, 0..16.
- With `mode=None` the key is absent and `features()` would fail; `prepare_structure`
  always sets a mode.

**`contact_conditioning`:** with `contacts=[]` it is the fallback
`make_unspecified_contact_conditioning`.
- Value: one-hot of class 0 (UNSPECIFIED) for every pair, including the diagonal:
  `cc[..., 0] = 1`, other channels 0. Shape (T, T, 5), int64.
- Classes: 0 UNSPECIFIED, 1 UNSELECTED, 2 POCKET>BINDER, 3 BINDER>POCKET, 4 CONTACT.
- Out of scope: when contacts are given, every pair is UNSELECTED, and requested pairs are
  CONTACT on both (i, j) and (j, i), with `contact_threshold = max_distance`.

**`contact_threshold`:** `zeros((T, T))` float32.

**Dynamics** (`batch_collate`, for items without `coords_cond`/`dt`):
- `coords_cond = zeros((Np, 3))` float32;
- `dt = zeros(1)` float32;
- `is_dynamics_conditioned = [False]` bool.

These keys are always present, so `features()` always builds a `Dynamics` with
`is_conditioned=False`. The mode does not change them.

## 11. MSA

### 11.1 Parsing an A3M

These rules come from `_parse_a3m_sequences` and `_validated_a3m_sequences`.

**Rows:**
- Rows are LINES, not FASTA records. Every stripped, non-empty line not starting with `>` or
  `#` is one row.
- So a wrapped (multi-line) A3M breaks, either with "aligned length" errors or wrong rows.
  `load_a3m_msa` checks only the query record, using Biopython.
- The view parses FASTA records (`jfold.featurize.a3m.parse_a3m`, shared by the views).
  For a3ms with one line per sequence, which upstream requires, the two agree.

**Query check:**
- Take row 0, drop lowercase, `-` and `.`, and uppercase the rest. It must equal the chain
  sequence.
- Row 0 itself is then tokenized as written. It is NOT replaced by the system sequence.

**Row limit:** the first `MAX_SEQS = 4092` rows are kept per chain; there is no
deduplication or filtering.

`_build_msa_from_colabfold_seqs` then walks each row, character by character:

```python
for c in row:
    if c != "-" and c.islower():      # insertion
        count += 1; continue
    residues.append(PROT_LETTER_TO_TOKEN[c])   # '-' -> 1, X/B/Z/J/O/U -> 22, '.' -> KeyError
    if count > 0: deletions.append((res_idx, count)); count = 0
    res_idx += 1
```

- Insertions count toward the NEXT aligned column.
- Insertions after the last aligned column are dropped.
- Every row must have exactly L aligned columns.
- Deletion counts are stored as int16.

### 11.2 Rows across chains, unpaired only

This is `tokenize_msa` with `paired_a3ms=None`, which is what the fixtures and the jfold
test use:

```
n_seqs = min(min(depth_c over protein chains), MAX_SEQS)   # every chain cut to the SHALLOWEST
row i, chain c: chain c's row i (residue tokens and deletion counts)
msa_paired[i, c] = 1 if homomer else (1 if i == 0 else 0)
homomer = (all protein chain sequences equal) and (more than one chain)
```

So for a heteromer whose partner has no MSA, the whole MSA collapses to the query row. This
was verified: an A with 6 rows plus a single-sequence B gives `S = 1`. For a homomer, every
row is flagged paired.

### 11.3 Features

`process_msa_features`, then `convert_densebatch_to_dictionary` transposes to rows-first.
The rows×tokens arrays are `msa_tok` (int) and `del` (int counts).

| key | shape, dtype | formula |
|---|---|---|
| `msa` | (S, T, 33) int64 | `one_hot(msa_tok, 33)`, already one-hot, NOT float |
| `msa_mask` | (S, T) int64 | `ones`: all rows, including gap-filled rows (only batch padding is 0) |
| `msa_paired` | (S, T) int64 | §11.2 |
| `has_deletion` | (S, T) bool | `del > 0` |
| `deletion_value` | (S, T) float32 | `pi/2 * arctan(del / 3)`, in float32: **π/2, not AF3's 2/π**; same as Boltz |
| `profile` | (T, 33) float32 | `msa.float().mean(rows)`; includes the query and gap rows; gap goes to channel 1 |
| `deletion_mean` | (T,) float32 | `deletion_value.mean(rows)`: the mean of the TRANSFORMED values, not of raw counts |

- Example: 2 insertions give `deletion_value = 0.9236`. For 6 rows, `deletion_mean = 0.1539`.
- Numpy reproduces the float32 values exactly in the fixtures. For very deep MSAs the
  float32 mean's summation order may differ from torch's in the last ulp.
- Single-sequence chain: `msa` is the one-hot sequence, `profile = one_hot(res_type)`,
  deletions are 0, and `msa_paired = 1`.

## 12. Padding and dtypes

`batch_collate` pads only the atom dimension, for a single system:
- **Rows:** pad atoms are appended to reach `Np`.
- **Pad value -1:** `token_idx_at_atom` and `residue_idx` (hence `ref_space_uid`).
- **Pad value 0:** every other atom field. That includes `element`, `name_encoding`,
  `charge`, `template_coords`, `mol_type`, `coords_cond` and `atom_mask`.

For batches of more than one system:
- tokens and MSA rows are padded to the batch maximum with 0;
- `res_type` is 0 on pad tokens and `residue_index` is -1.

The fixtures are single systems.

Dtypes are as in §2:
- Integers are int64. The model casts them.
- `msa`, `msa_paired`, `msa_mask` and `contact_conditioning` are int64 one-hots or flags.
- The masks and `has_deletion` are bool.
- Everything else is float32.

## 13. Randomness

**None in this pipeline.** In detail:
- There is no random rotation or recentering of reference conformers (Boltz-1 rotates
  `ref_pos`; BioEmu2 does not).
- There is no MSA subsampling or shuffling.
- There is no cropping.

Two dumps were byte-identical. The only nondeterministic MSA source is the ColabFold server
(`get_msa(system)`), which is not used here.

Randomness exists only downstream of featurization: noise and random augmentation in the
sampler. The model reads every MSA row on every trunk pass (no row subsampling; jfold's
`MSAModule` has `max_rows=None`).

## 14. Dependencies of the upstream path

- **torch:** every step (tensors, `torch.load` of `polymer_templates.pt`).
- **numba:** `_prepare_msa_arrays_inner`.
- **biopython:** A3M query checks in `alignment.py`.
- **rdkit:** an import-time dependency (`const.py`, `templates.py`); not exercised for
  proteins.
- **mdtraj:** element numbers in `template_from_cifwrapper`; also needed at import.
- **pdbecif:** CCD parsing. Not needed for proteins: their templates are prepackaged.
- **biotite:** an import-time dependency (`AtomArray` type); unused here.
- **gemmi:** not used by upstream featurization.
- **CCD:** not needed at runtime for standard proteins.

The view's residue table (`src/jfold/featurize/data/bioemu2.npz`: atom names, elements,
charges and float32 coordinates, with ALA's template for `X`) is extracted from
`polymer_templates.pt` by `tools/tables/extract_bioemu2.py`. The §6 recipe would
recompute it bit-exactly from the CCD.

## 15. Shared vs specific

**Generic AF3/Boltz-style (shared with the other pipelines):**
- One token per standard residue.
- CCD heavy-atom reference conformers, no hydrogens.
- The `name_encoding` = `ord(c) - 32`, 4 chars, convention (Boltz).
- Element as atomic number.
- CCD formal charges.
- `token_idx_at_atom`, a CA center atom, and the atom and token pad masks.
- `asym_id` / `entity_id` / `sym_id` semantics (entity = identical sequence, sym =
  copy number).
- The Boltz-1 33-token alphabet (PAD, GAP, 20 aa, UNK, RNA, DNA).
- A3M parsing: lowercase = insertion counted on the next column, `-` = GAP.
- One-hot MSA, `has_deletion`, Boltz's `π/2·arctan(d/3)` deletion value, profile = mean
  one-hot over rows, `deletion_mean` = mean of transformed values (as in Boltz-1).
- `msa_paired` with the query row = 1.

**BioEmu2-specific conventions:**
- **OXT** is dropped from every residue, including C-termini.
- **`ref_pos`:**
  - CCD *ideal* coordinates, centered over heavy atoms *including* OXT before OXT is
    dropped;
  - no random rotation;
  - key name `template_coords`.
- **`X`** is tokenized as UNK (22) but given ALA's atoms and conformer (5 atoms).
- **`ref_space_uid`** = global residue index; pad -1.
- **`token_idx_at_atom`** pad -1.
- **`residue_index`** is global across chains (no per-chain restart).
- **Entity merging** reorders chains (the first occurrence's entity collects later identical
  chains).
- **`token_bonds`** are all zero for proteins (no peptide bonds).
- **Atom padding** to a multiple of 32; no token padding for one system.
- **MSA row rules:**
  - unpaired-only heteromers are cut to the shallowest chain's depth;
  - homomers flag every row paired;
  - `msa_mask` is all ones;
  - MAX_SEQS = 4092;
  - A3M rows are lines.
- **Conditioning:**
  - `method_feature`: 7 for structure, 4 for equilibrium; 17 classes;
  - `pocket_feature` = 0 (4 classes);
  - `contact_conditioning` one-hot UNSPECIFIED (5 classes) and `contact_threshold` = 0;
  - dynamics placeholders: `coords_cond = 0`, `dt = 0`, `is_dynamics_conditioned = False`.
- **Dtypes:** int64 one-hot `msa` and `contact_conditioning`; int64 `msa_mask` and
  `msa_paired`.

## Parity fixtures

`uv run python tools/fixtures/dump_bioemu2.py` (from `~/jfold`) writes
`tests/featurize/data/bioemu2_{monomer,dimer,monomer_msa}.npz`.
- **Contents:** the unbatched upstream dict, 42 keys, in "structure" mode.
- **MSA:** unpaired A3Ms only. `monomer_msa` uses `chain_A.a3m`; the other chains get
  single-sequence A3Ms.
- **Determinism:** fully deterministic; nothing remains random, and no network is needed.

| system | T | N / Np | S |
|---|---|---|---|
| monomer | 22 | 186 / 192 | 1 |
| dimer | 31 | 253 / 256 | 1 |
| monomer_msa | 22 | 186 / 192 | 6 |

`tests/test_featurize.py::test_bioemu2` checks that the view reproduces every key
exactly (shape, dtype, values) on all three fixtures. The rules the fixtures do not
exercise (a heteromer with an MSA, A/B/C chain merging, a homomer with an MSA, `X` with
B/Z/insertions in the A3M, N = 32 exactly, equilibrium mode) were checked against upstream
directly.
