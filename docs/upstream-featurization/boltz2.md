# Boltz-2 featurization (protein-only): specification

This document specifies how upstream Boltz (`boltz` 2.2.1, the version installed in jfold's
environment) builds the Boltz-2 inference feature dict for protein-only inputs, and is the
reference for `jfold.featurize.views.boltz2`. It covers every key that
`jfold.models.boltz2.features` consumes. Scope: standard amino acids, one or more chains,
unpaired MSAs, one template per chain (§11a), no ligands, nucleic acids, modified residues
or constraints.

Paths below are relative to `site-packages/boltz/`. Every claim marked *(verified)* was
checked against the fixture dumps in `tests/featurize/data/boltz2_<system>.npz`, which
`tools/fixtures/dump_boltz2.py` produces.

Notation: `T` = tokens, which for proteins is the number of residues summed over chains.
`A` = real atoms. `A_pad` = atoms after padding. `S` = MSA rows.

---

## 1. Call chain

| Step | Function | Output |
|---|---|---|
| 1 | `main.py:check_inputs` | `[Path(input.yaml)]` |
| 2 | `main.py:process_inputs`. For Boltz-2 this loads `mol.py:load_canonicals(mol_dir)`, the 21 canonical residue `Mol`s from `~/.boltz/mols/<RES>.pkl`; `ccd.pkl` is **not** read. It then calls `main.py:process_input`. | |
| 2a | `parse/yaml.py:parse_yaml` → `parse/schema.py:parse_boltz_schema(..., boltz_2=True)` | `Target(record, structure: StructureV2, ...)` |
| 2b | `parse/schema.py:parse_polymer` (per entity) | `ParsedChain` with one `ParsedResidue` per residue |
| 2c | `parse/a3m.py:parse_a3m` (or `parse/csv.py:parse_csv`), `max_seqs=8192` | `MSA` → `processed/msa/<id>_<k>.npz` |
| 2d | | dumps `processed/{structures,records,constraints,templates,mols}` and `manifest.json` |
| 3 | `module/inferencev2.py:Boltz2InferenceDataModule.predict_dataloader` → `PredictionDataset.__getitem__` | |
| 3a | `module/inferencev2.py:load_input` | `Input(structure, msas={asym_id: MSA}, record, ...)` |
| 3b | `tokenize/boltz2.py:Boltz2Tokenizer.tokenize` → `tokenize_structure` | `Tokenized(tokens: TokenV2[], bonds: TokenBondV2[])` |
| 3c | `random = np.random.default_rng(42)`; `feature/featurizerv2.py:Boltz2Featurizer.process(training=False, max_atoms=None, max_tokens=None, max_seqs=const.max_msa_seqs (16384), pad_to_max_seqs=False, compute_frames=True, compute_constraint_features=True, override_method=None)` | |
| 3c-i | `process_ensemble_features` | `ensemble_ref_idxs=[0]`; no RNG draws |
| 3c-ii | `process_token_features` | token ids, `res_type`, bonds, contacts, method, modified, cyclic |
| 3c-iii | `process_atom_features` | `ref_*`, `atom_to_token`, rep/center atoms, frames, atom padding |
| 3c-iv | `process_msa_features` → `construct_paired_msa` → `prepare_msa_arrays` → `_prepare_msa_arrays_inner` (numba) | `msa`, deletions, `profile`, `msa_paired`, `msa_mask` |
| 3c-v | `process_template_features` with templates (§11a); `load_dummy_templates_features(tdim=1)` without | `template_*` |
| 3c-vi | `process_residue_constraint_features`, `process_chain_feature_constraints`, `process_contact_feature_constraints` | all empty for proteins |
| 4 | `module/inferencev2.py:collate` (`batch_size=1`) | `torch.stack` adds a leading batch axis of 1; `record` is kept as a list |

`jfold.models.boltz2.features` takes the **unbatched** dict, which is what the view returns
and also the layout of the fixtures.

---

## 2. Token vocabulary (`data/const.py:tokens`, 33 entries)

```
0 <pad>   1 -   2 ALA  3 ARG  4 ASN  5 ASP  6 CYS  7 GLN  8 GLU  9 GLY 10 HIS 11 ILE
12 LEU 13 LYS 14 MET 15 PHE 16 PRO 17 SER 18 THR 19 TRP 20 TYR 21 VAL 22 UNK
23 A 24 G 25 C 26 U 27 N 28 DA 29 DG 30 DC 31 DT 32 DN
```

- **Amino acid order.** Slots 2–21 are the 20 amino acids in alphabetical order of their
  three-letter codes. This is the same order as AF2/AF3's `restypes`
  (`ARNDCQEGHILKMFPSTWYV`), shifted by +2 for the leading `<pad>` and `-`. So
  `token_id = af_restype_index + 2`; GLN (7) comes before GLU (8).
- **Difference from AF3's 32-class layout.** AF3 puts the 20 amino acids at 0..19,
  UNK at 20 and gap last. Boltz puts `<pad>` at 0, gap at 1 and UNK at 22, then
  RNA (A G C U N) at 23–27 and DNA (DA DG DC DT DN) at 28–32.
- **Gap and unknown.** Gap `-` = 1. Unknown protein residue `UNK` = 22.
- **Sequence letters** (`const.prot_letter_to_token`): the 20 standard letters, then
  `X, J, B, Z, O, U → UNK` and `- → -`.
  - In the **YAML sequence**, any other character, including lowercase letters, also
    becomes UNK: `token_map.get(c, unk_token)`, at `schema.py:1163` *(verified: `a` → 22)*.
  - In an **a3m/csv row**, a character outside the map that is not lowercase (e.g. `.`)
    raises `KeyError`.
- **`res_type`**: `[T, 33]` int64, `one_hot(token_id, 33)`.

## 3. Tokenization rules for proteins

`tokenize/boltz2.py:tokenize_structure`:
- **One token per standard residue.** `res["is_standard"]` is True for every residue whose
  name is in `const.tokens`, and UNK counts too.
- **Token fields:**
  - `atom_idx`, `atom_num`: the residue's atom slice.
  - `res_idx`: the residue's 0-based index in its chain.
  - `asym_id`, `entity_id`, `sym_id`, `mol_type`: taken from the chain.
  - `center_idx` = CA. `disto_idx` = CB (CA for GLY).
  - `resolved_mask = True`. Inference atoms all have `is_present=True` and coordinates 0.
  - `modified = False`.
  - `frame_rot`/`frame_t` are computed from all-zero coordinates; they are used for
    templates only.
- **Residue name.** The YAML letter gives `res_name` and `res_type`; `MSE` would be mapped
  to `MET`.
- **Chain order (asym_id) follows entity grouping, not YAML order.**
  - `parse_boltz_schema` groups YAML items by `(entity_type, sequence)` in order of first
    appearance (`schema.py:1037`), and assigns `entity_id` = group index.
  - It then emits chains group by group (`schema.py:1345`).
  - Example: YAML chains `A(seq1), B(seq2), C(seq1)` give asym order `A, C, B`, with
    asym_id 0, 1, 2, entity_id 0, 0, 1 and sym_id 0, 1, 0 *(verified)*.
  - The view (`chain_order`) rejects chains that share a sequence but not an MSA.

## 4. Per-residue atom set and order

- **Atom set and order.** These come from `const.ref_atoms[res_name]`, a constant table.
  Atoms appear in exactly this order, chain by chain and residue by residue:

```
ALA N CA C O CB                     ARG N CA C O CB CG CD NE CZ NH1 NH2
ASN N CA C O CB CG OD1 ND2          ASP N CA C O CB CG OD1 OD2
CYS N CA C O CB SG                  GLN N CA C O CB CG CD OE1 NE2
GLU N CA C O CB CG CD OE1 OE2       GLY N CA C O
HIS N CA C O CB CG ND1 CD2 CE1 NE2  ILE N CA C O CB CG1 CG2 CD1
LEU N CA C O CB CG CD1 CD2          LYS N CA C O CB CG CD CE NZ
MET N CA C O CB CG SD CE            PHE N CA C O CB CG CD1 CD2 CE1 CE2 CZ
PRO N CA C O CB CG CD               SER N CA C O CB OG
THR N CA C O CB OG1 CG2             TRP N CA C O CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2
TYR N CA C O CB CG CD1 CD2 CE1 CE2 CZ OH
VAL N CA C O CB CG1 CG2             UNK N CA C O CB
```

- **Heavy atoms only.** The canonical mols contain no hydrogens *(verified: 0 H in all 21)*,
  and `parse_polymer` calls `RemoveHs` anyway.
- **OXT is never included.** OXT is not in `ref_atoms`, even on the C-terminal residue.
- **UNK** gets 5 atoms. Its mol also has `CG`, which is dropped because only `ref_atoms`
  names are used.
- **Element, charge and chirality** of each atom are looked up by atom name in
  `molecules[res_name]`, the canonical RDKit `Mol`, at featurization time. `StructureV2`
  atoms store only `name, coords, is_present, bfactor, plddt`.

## 5. Reference conformer features

All are computed in `process_atom_features`.

### `ref_pos` `[A_pad, 3]` float32

- **Source.** Each `~/.boltz/mols/<RES>.pkl` holds **10 precomputed conformers**, ids 0–9
  with no name property *(verified)*.
  - They are **not** the CCD ideal coordinates: pairwise distances differ from CCD
    `Ideal` by up to 0.78 Å for ALA. The package does not say how they were generated
    (see Gaps).
  - The conformer that `parse_polymer`'s `get_conformer` selects at parse time is discarded
    for Boltz-2.
- **Conformer choice.** One conformer per residue, keyed by `(asym_id, res_idx)`:
  ```python
  conf_id = int(random.choice(conf_ids))   # random = np.random.default_rng(42), featurizerv2.py:1196
  ```
  - Draws happen in token order, and no earlier step draws from `random`.
  - So the conformer ids are `default_rng(42).integers(0, 10)` drawn `T` times. This is
    identical to `choice(range(10))` *(verified)*. The fixtures store them as
    `ref_conformer_id`.
  - Upstream reseeds per input, so the same residue position always gets the same
    conformer whatever the system.
- **Per-residue random rigid motion** (`featurizerv2.py:1495`):
  ```python
  for i in range(torch.max(ref_space_uid)):          # NOTE: excludes the last uid
      included = ref_space_uid == i
      if torch.sum(included) > 0 and torch.any(resolved_mask[included]):
          ref_pos[included] = center_random_augmentation(ref_pos[included][None],
                                  resolved_mask[included][None], centering=True)[0]
  ```
  `center_random_augmentation` (`model/modules/utils.py`) does three things, in order:
  1. Subtracts the mean of the residue's atoms.
  2. Applies a uniformly random rotation, `x @ R` with `R` from `random_quaternions`: a
     `torch.randn(1, 4)` quaternion with nonnegative real part.
  3. Adds a random translation `torch.randn(1, 1, 3) * 1.0` Å.

  Notes:
  - Both draws use torch's **global** RNG, not the seeded numpy generator.
  - **Off-by-one:** the residue with the largest `ref_space_uid` (the last residue of the
    last chain) is **neither centered nor rotated nor translated**. Its `ref_pos` is the
    raw conformer *(verified)*.
  - For a one-residue input nothing is augmented.
- **Padding.** Padded atoms get `ref_pos = 0`.

### Other reference features

- **`ref_charge`** `[A_pad]` float32: `GetFormalCharge()` from the canonical mol. The
  canonical mols carry these nonzero charges *(verified)*:
  - ARG `NH2` +1
  - HIS `ND1` +1
  - LYS `NZ` +1

  Everything else is 0, so ASP and GLU are neutral.
- **`ref_element`** `[A_pad, 128]` int64: `one_hot(atomic_number, 128)`. The index is the
  atomic number itself: C=6, N=7, O=8, S=16. Padded rows are all zero.
- **`ref_atom_name_chars`** `[A_pad, 4, 64]` int64. Each character becomes `ord(c) - 32`.
  The name is stripped, right-padded with 0 to 4 characters, then one-hot over 64 classes.
  - Example: `"CA"` → `[35, 33, 0, 0]` *(verified)*.
  - Real atoms have exactly one hot entry per slot; the padding character is class 0, the
    same class as a space.
  - Padded atoms are all zero.
- **`ref_space_uid`** `[A_pad]` int64: a counter over distinct `(asym_id, res_idx)` in
  token order. For proteins this equals the token index of the atom's residue. Padded
  atoms get 0.
- **`ref_chirality`** `[A_pad]` int64 (not consumed by jfold): the index in
  `const.chirality_types`. CA is `CHI_TETRAHEDRAL_CCW` = 2 for every residue except GLY.
  ILE/THR CB is also 2. All else is 0.
- **`atom_backbone_feat`** `[A_pad, 17]` int64 (not consumed by jfold):
  `one_hot(idx, 17)`, where idx is 1..4 for N, CA, C, O and 0 for everything else.

## 6. Atom/token maps

- **`atom_to_token`** `[A_pad, T]` int64: `one_hot(token index of each atom, T)`. Padded
  rows are all zero, so `argmax` maps them to token 0.
- **`token_to_rep_atom`** `[T, A_pad]` int64: one-hot of the **distogram atom**, which is
  `CB` (`CA` for GLY). This is the representative atom jfold uses (`rep_atom`).
- **`token_to_center_atom`** `[T, A_pad]` int64: one-hot of `CA`. `r_set_to_rep_atom` is
  the same thing for resolved tokens.
- **Atom position within a token.** It is `arange(A) - first_atom[atom_to_token]`, the
  index into `const.ref_atoms[res_name]`. This is jfold's `atom_position`.
- **`frames_idx`** `[1, T, 3]` int64 and **`frame_resolved_mask`** `[1, T]` bool. Neither
  is consumed by jfold.
  - `frames_idx` holds the global atom indices of `(N, CA, C)` for standard protein
    residues. For UNK it is `(0, 0, 0)` with mask False.
  - `frame_resolved_mask` is **False everywhere at inference**. The coordinates are all
    zero, so `compute_collinear_mask` rejects every frame *(verified)*.

## 7. Chain, entity and position ids

All are `[T]` int64.

- **`asym_id`**: chain index 0, 1, …, in the entity-grouped order of §3.
- **`entity_id`**: index of the `(type, sequence)` group in order of first appearance.
  Identical protein sequences share an entity.
- **`sym_id`**: 0-based copy number of the chain within its entity, counted in asym order.
- **`residue_index`**: **0-based**, restarting at 0 for every chain. It is the residue's
  position in the input sequence (`enumerate`). Upstream applies no offset.
- **`token_index`**: `arange(T)`, global.
- **`mol_type`**: 0 (`PROTEIN`) for every token.

## 8. Bonds

- **`token_bonds`** `[T, T, 1]` float32 and **`type_bonds`** `[T, T]` int64 are **all zero**
  for proteins *(verified)*.
  - Standard residues are parsed with `bonds=[]`, and peptide bonds are never recorded.
  - Without a `bond` constraint there are no `connections`.
- **Encoding when a bond exists** (ligands or covalent constraints only): `type_bonds` =
  `bond_type + 1`, where `bond_type` indexes `["OTHER", "SINGLE", "DOUBLE", "TRIPLE",
  "AROMATIC", "COVALENT"]`. 0 means no bond, giving 7 classes, which matches jfold's
  `NUM_BOND_TYPES`.

## 9. MSA

### a3m parsing (`parse/a3m.py:_parse_a3m`, csv is identical)

- **Line handling.** Lines are stripped. Empty lines and `#…` lines are skipped. A line
  beginning `>` is a header, used only for taxonomy, which is always -1 for an a3m given
  through YAML. **Every other line is one complete sequence**: multi-line wrapped sequences
  are not supported.
- **Deduplication.** A row is dropped if `line.replace("-", "").upper()` was seen before,
  and the query counts as seen. So a hit identical to the query is dropped, and so is a hit
  that differs from an earlier one only in insertion case or gap placement *(verified:
  `hit1` == query is dropped; the fixture yields `S = 5` from 6 records)*. `max_seqs`
  (8192) counts kept rows.
- **Insertions.** Lowercase letters are insertions. They are counted, not emitted, and the
  count attaches to the **next** match column, i.e. the next uppercase letter or `-`:
  `deletion.append((res_idx, count))`. A trailing run of lowercase at the end of the line is
  dropped.
  - Verified for the fixture: `hit2` gives `(8, 2)` and `hit3` gives `(17, 2)`.
- **Validation.** Match columns are not checked against the query length during parsing.
  Rows must have exactly `L` match columns.
- **Query check** (`construct_paired_msa`). Row 0 must match the chain's residues.
  - If the lengths differ, or mismatches are not all (input MET vs MSA UNK) and the input is
    not all UNK at the mismatches, the MSA is replaced by `dummy_msa`, a single query row.
  - In the allowed mismatch cases, row 0 is overwritten with the input residues.

### Row assembly (`construct_paired_msa`, unpaired only)

- **Chains without an MSA.** `msa: empty`, i.e. `msa_id = -1`, gives `dummy_msa` (the query
  only).
- **Row 0** is every chain's query, with `is_paired = 1`.
- **Pairing.** With no taxonomy there is no pairing. Rows `1..` are filled by **zipping**
  the chains' remaining sequences in file order: row `i` holds chain c's `i`-th non-query
  sequence, or all gaps (`-` = 1) once chain c runs out. `is_paired = 0`.
  - This is **not** a block-diagonal layout *(verified with a 3-chain probe)*.
- **Row count.** `S = min(1 + max_c(n_c - 1), 16384)`, where `n_c` counts the rows kept for
  chain c (deduplicated, ≤ 8192). The truncation is deterministic (`random_subset=False` at
  inference).

### MSA features (`process_msa_features`)

| key | shape, dtype | rule |
|---|---|---|
| `msa` | `[S, T]` int64 | token ids (§2). Gap = 1, unknown letters = 22. |
| `msa_paired` | `[S, T]` float32 | 1 on row 0, else 0 (unpaired) |
| `msa_mask` | `[S, T]` int64 | all ones (`ones_like(msa)`) |
| `has_deletion` | `[S, T]` bool | `deletion > 0` |
| `deletion_value` | `[S, T]` float32 | `np.pi / 2 * np.arctan(deletion / 3)`, i.e. **(π/2)·atan(d/3)**, not AF3's (2/π)·atan(d/3) |
| `profile` | `[T, 33]` float32 | `one_hot(msa, 33).float().mean(0)`: mean over **all** S rows, including the query and gap rows (gaps count as class 1) |
| `deletion_mean` | `[T]` float32 | `deletion_value.mean(0)`, using the transformed value |

**Deletions are always zero in practice.** This is a bug in `construct_paired_msa`
(`featurizerv2.py:428-437`):

```python
for chain_id, chain_msa in msa.items():
    chain_deletions = chain_msa.deletions
    for sequence in chain_msa.sequences:
        ...
        chain_deletions = chain_deletions[del_start:del_end]   # re-slices the slice
        for deletion_data in chain_deletions:
            deletions[(chain_id, seq_idx, res_idx)] = deletion_values
```

Row 0, the query, has `del_start = del_end = 0`, so `chain_deletions` becomes empty and
every later row finds nothing. In practice `has_deletion`, `deletion_value` and
`deletion_mean` are therefore **all zero** *(verified on `monomer_msa`, whose a3m has two
insertion runs)*. The view reproduces the slicing (`_deletions`), so its deletion features
are zero too. The parsed deletion counts survive only in `processed/msa/*.npz`. (If row 0
itself had insertions, later rows would index into the re-sliced array and get arbitrary
values; a valid query cannot have insertions.)

For `monomer_msa`, `profile[0]` = 0.8 MET + 0.2 gap, because row 3 starts with gaps.

For a chain with no MSA in a system with `S` rows, the profile is `1/S` on the residue and
`(S-1)/S` on the gap.

## 10. Masks and padding

- **Atom padding.** `max_atoms=None`, so atoms are padded to the next multiple of 32
  (`atoms_per_window_queries`):
  `pad_len = ((A - 1) // 32 + 1) * 32 - A`.
  - Fixtures: monomer A = 186 → 192; dimer A = 253 → 256.
  - Padding is applied with `pad_dim` (value 0) to every atom-axis tensor: `ref_*`,
    `atom_to_token` rows, `token_to_*` columns, `coords`, `bfactor`, `plddt`.
- **`atom_pad_mask`** `[A_pad]` float32: 1 for real atoms, 0 for padding.
- **`atom_resolved_mask`** `[A_pad]` bool: True for real atoms.
- **No token or MSA padding** (`max_tokens=None`, `pad_to_max_seqs=False`).
  - `token_pad_mask` `[T]` float32 is all ones. `token_resolved_mask` and
    `token_disto_mask` `[T]` float32 are also all ones.
  - If `max_tokens` is set, as jfold's checkpoint test does, every token axis is
    zero-padded except `msa`, whose token axis is padded with gap (1).
- **Batch axis.** `collate` adds a leading axis of size 1 to every tensor.

## 11. Model-specific extras: defaults for plain protein input

| key | shape, dtype | value |
|---|---|---|
| `method_feature` | `[T]` int64 | 1 everywhere: `"x-ray diffraction"` in `const.method_types_ids`. `record.structure.method` is None at inference. |
| `modified` | `[T]` int64 | 0 |
| `cyclic_period` | `[T]` float32 | 0. With YAML `cyclic: true` it is the chain length, on every token of that chain. It is the max of the flag and a head-to-tail bond, and proteins have no bonds. |
| `mol_type` | `[T]` int64 | 0 |
| `contact_conditioning` | `[T, T, 5]` int64 | one-hot class 0 (`UNSPECIFIED`) everywhere. When all pairs are `UNSELECTED` (1), i.e. there are no constraints, they are rewritten to `UNSPECIFIED`. With any pocket or contact constraint, the remaining pairs stay `UNSELECTED`. |
| `contact_threshold` | `[T, T]` float32 | 0 |
| `affinity_token_mask` | `[T]` float32 | 0 (not consumed) |
| `template_restype` | `[1, T, 33]` int64 | **one-hot of class 0 (`<pad>`)**, not all zero |
| `template_frame_rot` | `[1, T, 3, 3]` float32 | zeros (not identity) |
| `template_frame_t`, `template_cb`, `template_ca` | `[1, T, 3]` float32 | zeros |
| `template_mask_cb`, `template_mask_frame`, `template_mask` | `[1, T]` float32 | zeros |
| `visibility_ids` | `[1, T]` float32 | zeros |
| `query_to_template` | `[1, T]` int64 | zeros |

**Not consumed by jfold**, all zero or empty at inference:
- `coords [1, A_pad, 3]`
- `disto_center [T, 3]`
- `disto_target`
- `disto_coords_ensemble`
- `bfactor`, `plddt`
- `ensemble_ref_idxs = [0]`
- the residue, chain and contact constraint index arrays (`rdkit_bounds_*`, `chiral_*`,
  `stereo_*`, `planar_*`, `connected_*`, `symmetric_chain_index`, `contact_pair_index`,
  …), which are empty.

## 11a. Templates

How upstream builds the `template_*` features from a template file given in the YAML
(`templates: - cif: …, chain_id: [...], template_id: [...]`, chains matched to template
chains), as mosaic passes them: one mmCIF holding every templated chain, renamed to its
query chain, each an entity whose full sequence is the template's residues. jfold's
`Chain.template` (a `jfold.featurize.Template`: a sequence and each residue's atoms by
name, unresolved residues with none) is that template. *(verified on the five systems in
`template_systems.json`)*

1. **Parsing** (`parse/schema.py`, `parse/mmcif.py:parse_polymer`). The template chain's
   sequence is its entity's full sequence as one-letter codes (`gemmi.one_letter_code`;
   UNK is `X`). Each residue's atoms are its reference atoms (`const.ref_atoms`) found in
   the file, by name; a residue in the sequence but not the file is unresolved (no atoms).
   Coordinates are float32.
2. **Alignment** (`get_template_records_from_matching` → `get_local_alignments`). The
   query chain's sequence against the template's, with Biopython's
   `PairwiseAligner(scoring="blastp")`, local, gap scores -1000: every optimal ungapped
   alignment, one template record each. The "blastp" matrix is BLOSUM62 with `X` scoring
   -1 against everything (so all-`X` sequences align nowhere). Ties come in order of where
   they end. `jfold.featurize.alignment.local_alignments` reproduces it (fuzzed against
   Biopython). With no alignment for any chain, upstream fails; the view raises.
3. **Template tokens** (`tokenize/boltz2.py:tokenize_structure` on the template): one per
   template residue, unresolved ones included: residue type; a frame from N, CA and C when
   all three are present (`compute_frame`, in float32: `e1` along CA→C, `e2` the part of
   CA→N orthogonal to it, `e3 = e1 × e2`, as the rotation's columns, CA the origin), else
   the identity and a zero origin; CA (`center_coords`) and the distogram atom (CB, CA for
   glycine; `disto_coords`) at the origin when missing; `disto_mask` when it is present.
4. **Placement** (`process_template_features`). For each record, `offset = template_st -
   query_st`, and **every** template token `t` with `t.res_idx - offset` a residue of the
   query chain is placed there, not only the aligned window: a template with an internal
   deletion (e.g. one listing only its resolved residues) puts the residues past the
   deletion off by its length. Later records overwrite earlier ones.
5. **Features** (`compute_template_features`), one row (`tdim = 1`) for the template file:

| key | shape, dtype | value |
|---|---|---|
| `template_restype` | `[1, T, 33]` int64 | one-hot of the placed template residue's type; class 0 (`<pad>`) elsewhere |
| `template_frame_rot` | `[1, T, 3, 3]` float32 | the frame's rotation (identity without a frame); zeros elsewhere |
| `template_frame_t`, `template_ca`, `template_cb` | `[1, T, 3]` float32 | frame origin, CA, distogram atom; zeros elsewhere |
| `template_mask_frame`, `template_mask_cb` | `[1, T]` float32 | the token's frame and distogram-atom masks |
| `template_mask` | `[1, T]` float32 | 1 on every placed position, unresolved residues included |
| `visibility_ids` | `[1, T]` float32 | 0 on chains that received template residues, `-1 - asym_id` on the others |
| `query_to_template` | `[1, T]` int64 | zeros (never set) |
| `template_force` | `[1]` bool | False (only with templates) |
| `template_force_threshold` | `[1]` float32 | inf (only with templates) |

## 12. Randomness

| What | Upstream generator | View |
|---|---|---|
| Conformer id per residue | `np.random.default_rng(42)`, seeded by upstream on every `__getitem__` | deterministic, same draws: `default_rng(42).choice(10)` × T in token order (`conformer_ids`) |
| Per-residue rotation and translation of `ref_pos` | torch global RNG, inside `center_random_augmentation` | `key=None`: unposed (upstream's `ref_pos` before the motion). With a `key`: the same distributions from JAX, every residue but the last (`pose`). |
| MSA subsampling (`random_subset`), contact or pocket sampling, training-only `max_seqs` | numpy `random` | disabled at inference |

**Model side, outside the featurizer:** the Boltz-2 model itself subsamples the MSA
(`MSAModuleArgs(subsample_msa=True, num_subsampled_msa=1024)`): every trunk pass reads
`torch.randperm(S)[:1024]`, 1024 rows uniformly with the query like any other (every row
when `S <= 1024`). That is outside the feature dict; jfold's `MSAModule` draws the same way
(`max_rows=1024, keep_query=False`; see `README.md`).

The fixtures are generated with `torch.manual_seed(0)` and saved with two extra keys,
`ref_pos_raw` (the unposed conformer coordinates) and `ref_conformer_id`.
`tools/fixtures/dump_boltz2.py` checks that `ref_pos_raw` equals the replayed conformer
coordinates exactly. The parity test compares the view's unposed `ref_pos` with
`ref_pos_raw`, and checks posed output per residue up to a rigid motion.

## 13. Dependencies by step

| Step | Needs |
|---|---|
| YAML parse (`parse_boltz_schema`) | PyYAML, numpy, RDKit (unpickling `Mol`s, `RemoveHs`, conformers), click. Also imported though unused for proteins: Bio (Biopython), chembl_structure_pipeline, scipy. gemmi is needed only for templates (`parse_mmcif`). biotite is not used anywhere. |
| Canonical residues | `~/.boltz/mols/{ALA..VAL,UNK}.pkl`: pickled RDKit `Mol`s from the Boltz-2 `mols.tar` download, holding atom names, elements, charges, chirality and 10 conformers. `ccd.pkl` is not used by Boltz-2. |
| a3m/csv parse | numpy (pandas for csv) |
| Tokenizer | numpy |
| Featurizer | torch (tensors, `one_hot`, global-RNG augmentation), numpy, numba (`_prepare_msa_arrays_inner`), RDKit (atom lookups on the canonical `Mol`s), scipy (import only) |
| Data module | torch `DataLoader`, pytorch_lightning |

Beyond `const.tokens`, the view needs only per-residue constants, which
`tools/tables/extract_boltz2.py` extracts from the 21 canonical pickles into
`src/jfold/featurize/data/boltz2.npz`: the `ref_atoms` order, each atom's element and formal
charge, and the 10 conformers per residue.

---

## Key-by-key summary

The keys the view returns. Shapes are unbatched.

| key | shape | dtype | rule (section) |
|---|---|---|---|
| `ref_pos` | `[A_pad, 3]` | f32 | conformer from default_rng(42), then per-residue center, rotation and N(0,1) translation, except the last residue (§5) |
| `ref_charge` | `[A_pad]` | f32 | CCD-mol formal charge; ARG NH2, HIS ND1 and LYS NZ = +1 (§5) |
| `ref_element` | `[A_pad, 128]` | i64 | one-hot of the atomic number (§5) |
| `ref_atom_name_chars` | `[A_pad, 4, 64]` | i64 | one-hot of `ord(c)-32`, zero-padded to 4 (§5) |
| `ref_space_uid` | `[A_pad]` | i64 | residue counter; padding 0 (§5) |
| `atom_to_token` | `[A_pad, T]` | i64 | one-hot; padded rows zero (§6) |
| `atom_pad_mask` | `[A_pad]` | f32 | 1 real / 0 pad, padded to a multiple of 32 (§10) |
| `token_to_rep_atom` | `[T, A_pad]` | i64 | one-hot of CB (CA for GLY) (§6) |
| `res_type` | `[T, 33]` | i64 | one-hot token id (§2) |
| `profile` | `[T, 33]` | f32 | mean one-hot over all MSA rows (§9) |
| `deletion_mean` | `[T]` | f32 | 0 (bug; §9) |
| `method_feature` | `[T]` | i64 | 1 (§11) |
| `modified` | `[T]` | i64 | 0 |
| `cyclic_period` | `[T]` | f32 | 0, or the chain length if cyclic |
| `mol_type` | `[T]` | i64 | 0 |
| `asym_id`, `entity_id`, `sym_id` | `[T]` | i64 | §7; asym order is entity-grouped |
| `residue_index` | `[T]` | i64 | 0-based per chain |
| `token_index` | `[T]` | i64 | `arange(T)` |
| `token_pad_mask` | `[T]` | f32 | ones |
| `token_bonds` | `[T, T, 1]` | f32 | zeros |
| `type_bonds` | `[T, T]` | i64 | zeros |
| `contact_conditioning` | `[T, T, 5]` | i64 | one-hot class 0 |
| `contact_threshold` | `[T, T]` | f32 | zeros |
| `msa` | `[S, T]` | i64 | token ids; rows zipped across chains; gap = 1 (§9) |
| `has_deletion` | `[S, T]` | bool | False (bug) |
| `deletion_value` | `[S, T]` | f32 | 0 (bug; the formula would be (π/2)·atan(d/3)) |
| `msa_paired` | `[S, T]` | f32 | row 0 = 1, else 0 |
| `msa_mask` | `[S, T]` | i64 | ones |
| `template_restype` | `[1, T, 33]` | i64 | one-hot class 0 |
| `template_frame_rot` | `[1, T, 3, 3]` | f32 | zeros |
| `template_frame_t`, `template_cb`, `template_ca` | `[1, T, 3]` | f32 | zeros |
| `template_mask_frame`, `template_mask_cb`, `template_mask` | `[1, T]` | f32 | zeros |
| `visibility_ids` | `[1, T]` | f32 | zeros |

---

## Shared vs specific

### Generic: another AF3-style pipeline computes these the same way

- **Tokens.** One token per standard residue. `token_index` = `arange(T)`.
- **Ids.** `asym_id` is the chain index. `entity_id` is shared by identical sequences.
  `sym_id` is the copy index within the entity.
- **`residue_index`** is the per-chain residue number; only the base (0 vs 1) varies
  between pipelines.
- **Atom features.** Heavy atoms only, and atom features come from a per-residue reference
  conformer:
  - element as a one-hot of the atomic number (AF3 uses 128 classes);
  - atom name as 4 characters of `ord(c) - 32`, one-hot over 64;
  - `ref_space_uid` = one id per residue;
  - formal charge.
- **Atom maps.** `atom_to_token` is a residue-major contiguous atom layout. The
  representative atom is CB, with CA for GLY (the AF3 distogram atom).
- **MSA parsing.**
  - a3m lowercase letters are insertions, counted onto the following match column.
  - Deletion features are `has_deletion`, a transformed `deletion_value`, and
    `deletion_mean`.
  - `profile` is the mean one-hot over the MSA, with the query in row 0.
- **Masks.** `msa_mask` and token/atom pad masks.
- **Bonds.** No polymer peptide bonds in the token-bond feature (AF3 also excludes polymer
  backbone links).
- **Relative position features** are computed from `asym_id`, `residue_index`, `entity_id`,
  `token_index` and `sym_id` (`jfold.blocks.inputs.relative_position_features`).

### Boltz-2-specific conventions

1. **Vocabulary.** A 33-token vocabulary: `<pad>` = 0, `-` = 1, the 20 amino acids at 2–21
   (AF order + 2), UNK = 22, then RNA and DNA. `res_type`, `msa` and `profile` all use it;
   the amino-acid order itself is shared.
2. **Atom order** comes from `const.ref_atoms`, which is AF-like but has no OXT and gives
   UNK = `N CA C O CB`.
3. **Reference conformers** are 10 precomputed RDKit conformers per residue, picked by
   `default_rng(42)`, rather than CCD ideal coordinates. The pickled mols' formal charges
   make ARG, HIS and LYS +1.
4. **Augmentation.** Per-residue centering, random rotation and **N(0, 1 Å) translation**
   of `ref_pos` from torch's RNG, with the **last residue skipped** (off-by-one).
5. **Deletion transform** is `(π/2)·atan(d/3)`, not AF3's `(2/π)·atan(d/3)`. And it is
   moot, because the re-slicing bug makes **all deletion features zero**.
6. **a3m deduplication** of rows identical to the query or to an earlier row, after
   removing gaps and uppercasing. No multi-line sequences.
7. **Unpaired rows are zipped across chains**, not block-diagonal, with gap fill when a
   chain runs out. Row 0 is flagged `msa_paired = 1`. There is a `msa_paired` feature at
   all.
8. **Chain order is entity-grouped.** Chains with the same sequence are made adjacent
   regardless of YAML order.
9. **Atom padding** to a multiple of 32 even without `max_atoms`.
10. **Extra token features:**
    - `method_feature` = 1 (x-ray), one-hot over 12 classes;
    - `modified` (2 classes);
    - `cyclic_period`, which also wraps relative positions;
    - `mol_type` (4 classes);
    - 5-class `contact_conditioning` plus `contact_threshold`, default `UNSPECIFIED`;
    - `type_bonds` with 7 classes.
11. **Template placeholders** when there are no templates (§11a with them): `tdim = 1` with
    `template_restype` one-hot on `<pad>` (class 0) and zero masks. `visibility_ids` is
    the template visibility scheme.
12. **`frame_resolved_mask`** is all False at inference, because of the collinearity check
    on zero coordinates. jfold does not consume it.

---

## Gaps

- **Conformer provenance.** The package does not record how the 10 conformers in
  `~/.boltz/mols/*.pkl` were generated, e.g. the ETKDG version, seeds, or whether they
  were derived from the full-hydrogen CCD mol. The view uses them as extracted constants
  rather than regenerating them.
- **Torch RNG.** The torch-RNG rotations and translations cannot be reproduced bit-for-bit
  without reimplementing torch's CPU `randn`, so posed `ref_pos` is compared per residue up
  to a rigid motion (§12).
- **Not covered.** Paired MSAs, csv taxonomy keys and the colabfold server path. The csv
  parser is line-for-line identical to the a3m parser apart from taxonomy. Templates:
  several templates per chain (more rows), the YAML's search mode (no `template_id`:
  chains assigned by global alignment score), force/threshold steering, and modified
  residues (MSE and others) in templates.
