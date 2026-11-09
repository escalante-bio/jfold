# ESMFold2 featurization (proteins, unpaired MSAs)

This spec covers how the upstream `esm` package (3.4.1.post1, as installed in jfold's
environment) turns a protein-only system into ESMFold2's forward inputs. It covers every
key that `jfold.models.esmfold2.features`, `lm_layout`, `lm_input` and
`token_hidden_states` consume, and is the reference for `jfold.featurize.views.esmfold2`.
The two model types share one featurizer:

- the release type: `EsmFold2Model` (`biohub/ESMFold2`, `biohub/ESMFold2-Fast`);
- the experimental type: `EsmFold2ExperimentalModel` (`biohub/ESMFold2-Experimental*`).

They differ only in how the forward pass treats the MSA profile (§12).

Scope: standard amino acids (plus `X`), one or more chains, unpaired MSAs. No ligands,
nucleic acids, modifications, covalent bonds, distogram conditioning or templates.

Every statement here was checked by running the upstream code. Paths are relative to
`~/jfold/.venv/lib/python3.12/site-packages/`. The parity fixtures come from
`tools/fixtures/dump_esmfold2.py` (§17).

Notation:
- `T`: tokens (one per residue for proteins, summed over chains);
- `A`: real atoms; `A_pad`: atoms after padding;
- `M`: MSA rows;
- `Lp`: length of the packed ESMC input.

---

## 1. Call chain

How it is driven (`tests/test_esmfold2.py::test_esmfold2_checkpoint`,
`tools/fixtures/dump_esmfold2.py`):

```python
from esm.models.esmfold2.processor import ESMFold2InputBuilder
from esm.utils.structure.input_builder import ProteinInput, StructurePredictionInput
from esm.utils.msa import MSA

msa = MSA.from_a3m("chain_A.a3m", remove_insertions=True)        # or None: single sequence
chains = [ProteinInput(id="A", sequence="MKTA...", msa=msa), ProteinInput(id="B", sequence="GSHM...")]
batched, chain_infos = ESMFold2InputBuilder().prepare_input(
    StructurePredictionInput(sequences=chains), seed=0)
f = {k: v[0].numpy() for k, v in batched.items() if isinstance(v, torch.Tensor)}  # unbatched
features = jfold.models.esmfold2.features(f | {"lm_hidden_states": hidden})       # see §13
```

1. `esm/models/esmfold2/processor.py:ESMFold2InputBuilder.__init__`. Calls
   `conformers.py:load_ccd()`, which unpickles the CCD (§16).
2. `processor.py:ESMFold2InputBuilder.prepare_input(input, seed=None, device=None)`. It
   runs the following:
   1. `processor.py:clean_esmfold2_input`. This splits a `ProteinInput` whose sequence
      contains `:` or `|` chain breaks into one chain per piece. Otherwise the input
      passes through unchanged, and separate `ProteinInput`s are never merged.
   2. Inside `processor.py:_seed_context(seed)` (only when `seed is not None`), it calls
      `prepare_input.py:prepare_esmfold2_input(input, seed)`:
      - `build_chains_from_input` assigns entity, sym and asym ids (§9). For each protein
        chain it calls `tokenize_protein` (§5, §6).
      - `build_feature_tensors` does the following:
        - token and atom arrays, plus atom padding (§6–§8, §14);
        - `compute_frame_indices` (`frames_idx`, unused by jfold);
        - `compute_token_bonds` (§10);
        - `compute_representative_atoms` (`distogram_atom_idx`);
        - `compute_msa_features`, which calls `paired_msa.py:construct_paired_msa`
          and `msa_to_res_type_and_deletions` (§11);
        - `compute_distogram_conditioning`, which gives all-zero `disto_cond` and an
          all-False `disto_cond_mask`.
   3. Every tensor gets a leading batch axis of 1 (`v[None]`).
3. The model forward (`esm/models/esmfold2/model.py:EsmFold2Model.forward`, and
   `experimental.py:EsmFold2ExperimentalModel.forward`) derives further inputs from these
   keys:
   - `profile` (§12);
   - the `disable_msa_features` zeroing (§12);
   - one-hots of `res_type`, `ref_element` and `ref_atom_name_chars`, zeroed on padding;
   - `atom_to_token * atom_attention_mask`;
   - `lm_hidden_states` from `input_ids` when none are passed
     (`layers.py:compute_lm_hidden_states`, §13).

   `forward` drops `pocket_feature`, `gt_coords`, `is_resolved` and `frames_idx`
   (`model.py:_IGNORED_FEATURE_KEYS`).

**Precomputed a3m (no MSA server).** Pass an `esm.utils.msa.MSA` as `ProteinInput.msa`.
Build it with `MSA.from_a3m(path_or_buffer, remove_insertions=True, max_sequences=None)`;
the view's `max_sequences` argument is the same cut. `remove_insertions` does not change the features,
because the deletion counts are taken from the raw rows either way (§11). A chain with
`msa=None` gets a one-row MSA of its own sequence and a "No MSA provided" warning.

---

## 2. Output keys

These are the unbatched shapes, as jfold's `features` takes them. "jfold" marks keys read
by `features`, `lm_layout` and `token_hidden_states`.

| key | shape | dtype | read by | section |
|---|---|---|---|---|
| `token_index` | `[T]` | int64 | jfold | §9 |
| `residue_index` | `[T]` | int64 | jfold | §9 |
| `asym_id` | `[T]` | int64 | jfold | §9 |
| `entity_id` | `[T]` | int64 | jfold | §9 |
| `sym_id` | `[T]` | int64 | jfold | §9 |
| `mol_type` | `[T]` | int64 | jfold (`lm_layout` protein mask) | §9 |
| `res_type` | `[T]` | int64 | jfold | §4 |
| `input_ids` | `[T]` | int64 | upstream ESMC (jfold rebuilds it from `res_type`) | §13 |
| `token_bonds` | `[T, T, 1]` | float32 | jfold | §10 |
| `token_attention_mask` | `[T]` | bool | jfold | §14 |
| `ref_pos` | `[A_pad, 3]` | float32 | jfold | §7 |
| `ref_element` | `[A_pad]` | int64 | jfold | §7 |
| `ref_charge` | `[A_pad]` | **int8** | jfold | §7 |
| `ref_atom_name_chars` | `[A_pad, 4]` | int64 | jfold | §7 |
| `ref_space_uid` | `[A_pad]` | int64 | jfold | §7 |
| `atom_attention_mask` | `[A_pad]` | bool | jfold | §8 |
| `atom_to_token` | `[A_pad]` | int64 | jfold | §8 |
| `distogram_atom_idx` | `[T]` | int64 | jfold (`rep_atom`) | §8 |
| `msa` | `[M, T]` | int64 | jfold | §11 |
| `has_deletion` | `[M, T]` | bool | jfold | §11 |
| `deletion_value` | `[M, T]` | float32 | jfold | §11 |
| `deletion_mean` | `[T]` | float32 | jfold | §11 |
| `msa_attention_mask` | `[M, T]` | bool | jfold | §11 |
| `lm_hidden_states` | `[T, n_layers+1, D]` | float (ESMC dtype) | jfold (optional); **not** made by `prepare_input` | §13 |
| `pocket_feature` | `[T]` | int64 | none (zeros) | |
| `gt_coords` | `[1, A_pad, 3]` | float32 | none (zeros) | |
| `is_resolved` | `[A_pad]` | bool | none (all False) | |
| `frames_idx` | `[T, 3]` | int64 | none | N, CA, C atom indices. An UNK token uses its first atom three times. |
| `disto_cond` | `[T, T]` | int64 | none (zeros) | |
| `disto_cond_mask` | `[T, T]` | bool | none (False) | |

jfold's `features` derives `atom_position` (each atom's index within its token) itself:
`atom_index - first_atom[atom_to_token]`.

There is no `profile` key. The model computes the profile from `msa` (§12).

The view returns every key above except `lm_hidden_states`, plus ESMC's packed inputs as
upstream builds them inside the model (§13): `lm_input_ids` `[Lp]`, `lm_sequence_id`
`[Lp]` and `lm_positions` `[T]`.

---

## 3. Inputs: sequences and chains

- Each `ProteinInput.sequence` is read **one character at a time with no normalisation**.
  `PROTEIN_1TO3.get(c, "UNK")` maps the 20 uppercase one-letter codes to residue names.
  **Anything else becomes `UNK`**: `X`, `B`, `Z`, `U`, `O`, digits, and **lowercase**
  letters (`"mkA"` gives `res_type = [22, 22, 2]`).
- `ProteinInput.id` may be a list. `id=["A", "B"]` gives two chains with the same
  sequence and MSA, `sym_id` 0 and 1.
- Chains are emitted in input order. Nothing is sorted or reordered.

## 4. Token vocabulary (`res_type`, `msa`; 33 classes)

Source: `esm/models/esmfold2/constants.py`. `NUM_RES_TYPES = 33` (`layers.py`).

| id | meaning |
|---|---|
| 0 | pad (`MSA_PAD_TOKEN_ID`); never emitted for proteins |
| 1 | MSA gap `-` (`MSA_GAP_TOKEN_ID`) |
| 2–21 | ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL (alphabetical by three-letter code) |
| 22 | protein UNK (`PROTEIN_UNK_RES_TYPE`) |
| 23–27 | RNA A G C U, RNA unknown |
| 28–32 | DNA DA DG DC DT, DNA unknown (`GAP_RES_TYPE = 32` is defined but unused on the protein path) |

This is the Boltz token order (`<pad>`, `-`, the 20 amino acids, UNK, `A G C U N`,
`DA DG DC DT DN`).

How unknowns map:
- In the query sequence, an unknown residue becomes `UNK`, which is 22.
- In the MSA, `X` and every other unrecognised uppercase-able character (`B Z U O J *`
  ...) map to 22. A gap `-` maps to 1 (§11).

`MSE` maps to 14 (MET), but a one-letter sequence cannot produce MSE.

## 5. Tokenization rules for proteins

Source: `prepare_input.py:tokenize_protein`.

- **One token per residue** for the 20 standard residues **and for `UNK`**. `UNK` is in
  `PROTEIN_HEAVY_ATOMS`, so it takes the standard branch, not the atom-tokenized
  modified-residue branch.
- The token fields are:
  - `res_type = PROTEIN_RESIDUE_TO_RES_TYPE.get(name, 22)`;
  - `input_id = _compute_esm_input_id(name)` (§13);
  - `mol_type = 0`;
  - `residue_index` = position within the chain.
- The atom-tokenized branch (one token per atom) runs only for `Modification`s. That is
  outside scope.

## 6. Per-residue atoms

Source: `constants.py:PROTEIN_HEAVY_ATOMS`. Atoms are listed in this order, backbone
first:

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
TYR N CA C O CB CG CD1 CD2 CE1 CE2 CZ OH   VAL N CA C O CB CG1 CG2
UNK N CA C O
```

- Heavy atoms only: **no hydrogens**.
- **No OXT**, not even on the C-terminal residue. Any residue-level leaving-atom logic
  applies only to the atom-tokenized branch.
- `UNK` (and therefore `X`) has 4 atoms. The CCD `UNK` has a CB, but it is not used.
- `A` is the sum of these counts over all residues. Atoms are laid out residue by
  residue in token order, so `atom_to_token` is non-decreasing.

## 7. Reference conformer features

**`ref_pos`** `[A_pad, 3]` float32 (Å).
- It comes from `conformers.py:get_idealized_atom_pos(res_type, name)`, which looks up
  `RES_TYPE_TO_CCD[res_type]` (22 gives `"UNK"`).
- `get_ccd_conformer(comp_id)` takes the CCD RDKit molecule from the ESMFold2 `ccd.pkl`.
  It picks the conformer whose `name` property is `"Computed"`, else `"Ideal"`, else
  conformer 0.
- It then runs `Chem.RemoveHs(mol, sanitize=False)` and reads each named atom's position.
- For all 20 amino acids and UNK, every atom equals the CCD `Computed` conformer exactly.
  *(verified)*
- **No centring, no random rotation and no augmentation**, at any seed. The coordinates
  are the pickle's raw frame, roughly centred on the full CCD molecule with its
  hydrogens and OXT. For example, the mean ALA heavy atom is `[0.16, -0.10, -0.60]`.
- The conformer is identical for every copy of a residue type, whatever its position or
  chain. *(verified)*
- Padding atoms are `0`.
- **The ESMFold2 `ccd.pkl` (md5 `5045106b0d95c4a4bbad5105a7107b06`) is not Boltz's
  `ccd.pkl` (md5 `aa43b8ba…`).** Their `Computed` conformers differ (ALA atom 0:
  `[-0.91, -1.32, -0.05]` vs `[-0.01, -1.21, -1.06]`). The view's table
  (`src/jfold/featurize/data/esmfold2.npz`) is dumped from ESMFold2's own pickle by
  `tools/tables/extract_esmfold2.py`.

**`ref_element`** `[A_pad]` int64: the atomic number of `_infer_element(name)`.
- For the standard residues this is the first character of the atom name: C 6, N 7,
  O 8, S 16. These match the CCD symbols for all 20 amino acids and UNK. *(verified)*
- Padding atoms are 0.
- The model one-hots it into 128 classes (`MAX_ATOMIC_NUMBER`) and zeroes padding.

**`ref_charge`** `[A_pad]` **int8**: `CHARGED_ATOMS.get((res, name), 0)`.
- For proteins the nonzero entries are LYS NZ +1, ARG NH2 +1 and HIS ND1 +1.
- These equal the CCD formal charges of every standard-residue heavy atom. *(verified)*
- The model uses it as a raw float feature, with no `arcsinh`
  (`layers.py:~868`; jfold's `atom_features`).

**`ref_atom_name_chars`** `[A_pad, 4]` int64: `encode_atom_name(name)`.

```python
padded = name.ljust(4)[:4]; [ord(c) - 32 if c != " " else 0 for c in padded]
```

- Names are left-justified, with no PDB-style leading space. For example `"N"` gives
  `[46, 0, 0, 0]`, `"CA"` gives `[35, 33, 0, 0]`, `"NH1"` gives `[46, 40, 17, 0]` and
  `"OXT"` would give `[47, 56, 52, 0]`.
- Padding atoms are 0.
- The model one-hots it into 64 classes (`CHAR_VOCAB_SIZE`) per character and zeroes
  padding.

**`ref_space_uid`** `[A_pad]` int64: one id per residue (conformer).
- A global counter starts at 0. Each standard residue adds 1. Each chain's offset adds
  `len(set(uids of the chain))`.
- For proteins it therefore equals `atom_to_token` on real atoms. *(verified)*
- Padding atoms get **0**, not a fresh id.
- jfold and upstream use it only in the atom transformer's rotary angles.

## 8. Atom/token maps and masks

- **`atom_to_token`** `[A_pad]` int64: the owning token index of each real atom.
  Padding atoms get **0**, and the model multiplies by the atom mask.
- **`atom_attention_mask`** `[A_pad]` bool: True for the `A` real atoms, False for the
  padding tail.
- **`token_attention_mask`** `[T]` bool: all True. Tokens are never padded.
- **`distogram_atom_idx`** `[T]` int64: the global atom index of the residue's `CB`,
  else its `CA`.
  - GLY and UNK/X use CA.
  - From `compute_representative_atoms`: `ad.get("CB", ad.get("CA", first))`.
  - jfold uses it as `rep_atom` for the confidence head's distances.

## 9. Chain, entity and position ids

From `build_chains_from_input` and `tokenize_protein`. All are `[T]` int64:

- `asym_id`: the chain's position in emission order, 0-based.
- `entity_id`: 0-based, in order of first appearance of the entity key
  `input_builder.py:_entity_key`, which is `("PROTEIN", sequence, frozenset(modifications))`.
  - Chains with an identical sequence share an entity, even if they are separate
    `ProteinInput`s or carry different MSAs.
  - The MSA is not part of the key.
- `sym_id`: the copy index within the entity, 0-based. A homodimer gives
  `[0,0,0,0,1,1,1,1]`.
- `residue_index`: the position within the chain, **0-based**.
- `token_index`: the global token position, `arange(T)`.
- `mol_type`: 0 (protein).

For the `dimer` fixture: `asym_id`, `entity_id` = 0×22 then 1×9; `sym_id` all 0;
`residue_index` = `0..21`, `0..8`.

## 10. Bonds

`token_bonds` `[T, T, 1]` float32 (`compute_token_bonds`) is **all zero for
standard-residue proteins**. Peptide connectivity is implicit in `residue_index`.

**Exception: `UNK`/`X` tokens get peptide bonds.**
- A token with `res_type == 22` counts as "atom-tokenized" in `compute_token_bonds`.
- Its N side is bonded to the previous residue's token, and its C side to the next
  residue's token, symmetric with value 1.0.
- Example: `"MKXAB"` gives bonds (1,2), (2,3) and (3,4). *(verified)*
- Intra-residue CCD bonds collapse to self-pairs and are dropped.

## 11. MSA

### 11.1 a3m parsing (`esm/utils/parsing.py:parse_fasta`, `esm/utils/msa/msa.py:MSA.from_a3m`)

- Parsing:
  - Blank lines and lines starting with `#` are skipped.
  - `>` starts a record, and the header is `line[1:].strip()`.
  - Sequence lines are concatenated without stripping.
- The first `max_sequences` records are kept. **No deduplication or filtering** is done.
- Row 0 (the a3m's first record) is the query row.
  - **The pipeline never checks it against `ProteinInput.sequence`.** A mismatched a3m
    query silently becomes MSA row 0.
  - `res_type` still comes from the sequence.
- Insertions are lowercase `a–z` and `.`. A match column is any other character.
- Deletions (`a3m_deletion_counts`): for each match column, the number of insertion
  characters since the previous match column.
  - Leading insertions count toward column 0.
  - **Trailing insertions after the last match column are dropped.**
- Every row must have the same number of match columns as row 0, else `ValueError`.
- `MSA.deletions` is set from the raw rows whether or not `remove_insertions` is used.
- The view turns `.` into a lowercase insertion before the shared a3m parser
  (`jfold.featurize.a3m.parse_a3m`), which otherwise matches these rules.

### 11.2 Residue mapping (`paired_msa.py:msa_to_res_type_and_deletions`)

- `L_c` = the number of match columns in the query row.
- For each row and match column:
  - `-` → 1;
  - otherwise `protein_letter_to_res_type().get(ch.upper(), 22)`, which maps the 20
    amino acids to 2–21, `X` to 22, and **anything else to 22**.
- Deletion counts come from `msa.deletions` (float32).

### 11.3 Merging chains into rows (`paired_msa.py:construct_paired_msa`)

**Headers with `key=N` are paired**, even within a single chain.
- Rows (other than the query) whose header has a `key=<int>` appearing at least twice
  across the system are moved to the front, grouped by taxon. *(verified)*
- The view reproduces this rule (`_msa_rows`). For purely unpaired rows, headers must not
  contain `key=`.

With no keys, the rows are:
- row 0 is every chain's query;
- then `max_c(depth_c - 1)` rows, where row `i` holds **every chain's `i`-th hit side by
  side**: chain A's hit `i` in A's columns and chain B's hit `i` in B's columns.
  - This is not block-diagonal, despite the module docstring.
  - A chain that has run out of hits gets gap (1) and deletion 0 there. *(verified)*
- A chain with no MSA has a one-row MSA of its own sequence, so its columns are gap in
  every row ≥ 1. That is the `dimer` fixture case.

Caps:
- `max_pairs=8192`, `max_total=16384`, `max_seqs=16384` (`compute_msa_features`).

Column gather:
- Each token reads column `min(residue_index, L_c - 1)` of its chain's MSA. **A query
  shorter than the chain silently repeats its last column**. No error is raised.

### 11.4 Features (`prepare_input.py:compute_msa_features`)

```python
msa            = msa_res                                   # [M, T] int64
has_deletion   = del_counts > 0                            # [M, T] bool
deletion_value = (np.pi / 2) * torch.arctan(del_counts / 3) # [M, T] float32
deletion_mean  = deletion_value.mean(dim=0)                # [T] float32, over ALL M rows
msa_attention_mask = torch.ones_like(msa, dtype=torch.bool)  # [M, T]
```

- **`deletion_value` uses `π/2 · arctan(d/3)`, not AF3's `2/π · arctan(d/3)`.** It
  ranges over `[0, π²/4 ≈ 2.467)`; d = 2 gives 0.92363. *(verified)*
- `deletion_mean` is an unmasked mean over every row, query included. With no MSA it is
  zero.
- `msa_attention_mask` is all True at featurization, including gap-filled rows of chains
  without an MSA.
- The query row (row 0) of a chain without an a3m equals its `res_type`. For a chain
  with an a3m it is the a3m's first record.
- `max_depth`: featurization keeps all rows up to the caps above. The 1024-row limit is
  applied at model time (§15).

The `monomer_msa` fixture has 6 rows. Columns 8 and 17 have `deletion_value` 0.92363 in
rows 2 and 3, and `deletion_mean` 0.15394 there. Column 6 of row 5 (`X`) is 22. Gaps are
1.

## 12. Profile and `disable_msa_features` (model side)

`profile` is not a featurizer output. Both forwards compute it before the embedder
(`model.py:~1082`, `experimental.py:~944`):

```python
msa_oh = one_hot(msa, 33) * msa_attention_mask[..., None]
profile = msa_oh.sum(rows) / msa_attention_mask.sum(rows).clamp(min=1)   # [T, 33]
# no msa key at all -> profile = one_hot(res_type) * token_mask
```

- Gaps (class 1) count in the profile. So do the gap-filled rows of chains without an
  MSA: chain B's profile in a two-chain system where only A has 6 rows is
  `1/6 · onehot(res) + 5/6 · onehot(gap)`.
- `res_type` is one-hot encoded and multiplied by the token mask.
- The profile uses the **featurized** `msa_attention_mask`, before the model-time
  column masking (§15).
- jfold's `features` recomputes the same formula from `msa` and `msa_attention_mask`. It
  matches the captured `embed_profile` to 1 ulp (summation order).

**`disable_msa_features`.**
- `experimental.py:972` replaces `profile` and `deletion_mean` with zeros before the
  input embedder. Only the experimental type does this.
- The release forward never reads the flag. `biohub/ESMFold2-Fast`'s config sets it to
  `True`, but its profile is still the MSA profile.
- jfold covers this with the model's `msa_profile = not disable_msa_features` for the
  experimental type, and always `True` for release (`jfold/convert/esmfold2.py`); `embed`
  zeroes the profile and deletion mean without it.

| checkpoint (resolved config) | type | `disable_msa_features` | MSA encoder | profile / deletion_mean in embedder |
|---|---|---|---|---|
| `biohub/ESMFold2` | release | False | yes | MSA profile |
| `biohub/ESMFold2-Fast` | release | True (ignored) | no | MSA profile |
| `biohub/ESMFold2-Experimental`, `-Cutoff2025` | experimental | False | yes | MSA profile |
| `biohub/ESMFold2-Experimental-Fast`, `-Fast-Cutoff2025`, `-Fast-base{300M,600M,6B}-step1500k` | experimental | True | no | zeros |

`res_type`, profile and `deletion_mean` form the last 67 channels of `s_inputs`
(jfold's `token_features`).

## 13. ESMC language-model input

Upstream builds this in `layers.py:compute_lm_hidden_states(esmc, input_ids, asym_id,
residue_index, mol_type, token_mask, pad_to_multiple=None, lm_mask_pct=0.0)`. The model
calls it when no `lm_hidden_states` are passed (`model.py:_compute_lm_hidden_states`).

**ESMC vocabulary** (`esm/models/esmc/tokenizer.py:SEQUENCE_VOCAB`; the embedding has
64 rows):
- `0 <cls>/BOS`, `1 <pad>`, `2 <eos>`, `3 <unk>`;
- `4 L, 5 A, 6 G, 7 V, 8 S, 9 E, 10 R, 11 T, 12 I, 13 D, 14 P, 15 K, 16 Q, 17 N, 18 F, 19 Y, 20 M, 21 H, 22 W, 23 C`;
- `24 X, 25 B, 26 U, 27 Z, 28 O, 29 ., 30 -, 31 |`;
- `32 <mask>`.

**`input_ids`** `[T]` (from `prepare_input.py:_compute_esm_input_id`):
- Each standard residue maps to `ESM_PROTEIN_VOCAB[letter]`. From `res_type` 2–21 these
  are `5 10 17 13 23 16 9 6 21 12 4 15 20 18 14 8 11 22 19 7`.
- `UNK`/`X` maps to **24 (`X`)**. `PROTEIN_3TO1` has no `UNK`, so the code falls through
  to `DNA_RNA_LIGAND_INPUT_ID = 24`. It is **not** 3. *(verified: `"MKXAB"` gives
  `[20, 15, 24, 5, 24]`)*
- jfold's `esmc_residues` (`_ESMC_COLUMN`) also maps class 22 to 24, and the view's
  `input_ids` give `X` 24.

**Packing** (B = 1):
- Protein tokens are those with `(mol_type == 0) & token_mask`. They are collapsed to
  one row per `(asym_id, residue_index)`, keeping the first token in input order. This
  is a no-op for standard proteins.
- Chains are ordered by **sorted `asym_id`**, and residues by first appearance within
  each chain.
- Each chain is wrapped as `[BOS] residues [EOS]`, and the chains are concatenated:
  `[0, c1..., 2, 0, c2..., 2]`. So `Lp = Σ(L_c + 2)`.
- There is no padding for B = 1. With B > 1, PAD = 1 and `sequence_id = -1` there.
- `sequence_id = cumsum(ids == 0) - 1`: a chain's BOS, residues and EOS all carry its
  chain number 0, 1, …
- Attention is block-diagonal: `sequence_id_i == sequence_id_j`.
- Rotary positions are **absolute** over the packed sequence, `0..Lp-1`, and are not
  reset per chain. (jfold `ESMC` uses `arange(Lp)`; this is verified by
  `test_esmfold2_checkpoint` on the dimer.)
- The expand map (`lm_positions` in the fixtures, `LMLayout.positions` in jfold) gives
  each token's packed row, or -1 off-protein.
- Dimer fixture:
  - `lm_input_ids = [0, 20,15,…,16, 2, 0, 6,8,21,20,4,9,13,14,7, 2]` (Lp = 35);
  - `sequence_id = [0]*24 + [1]*11`;
  - `lm_positions = 1..22, 25..33`.
- The view's `lm_inputs`, and jfold's `lm_layout` and `lm_input`, reproduce all three
  exactly on every fixture (`tests/test_featurize_esmfold2.py`).

**Masking.** With `lm_mask_pct > 0`, residues (not ids 0, 1 or 2) are replaced by 32
where `torch.rand < lm_mask_pct`. Every released config resolves `lm_mask_pct = 0.0`, so
nothing is masked.

**`lm_hidden_states`** `[T, n_layers+1, D]`, in ESMC's dtype (bf16 unless loaded in
fp32):
- `result[t] = hs[:, positions[t], :]`, where `hs` holds ESMC's `hidden_states`
  (`n_layers+1`).
  - Index 0 is the embedding output, i.e. the input to block 0.
  - Index `k` is the input to block `k`.
  - The last index is the final LayerNorm output (jfold `ESMC` returns the same stack).
- Off-protein tokens are zero.
- Sizes: ESMC-6B has 80 layers, so 81 × 2560. ESMC-300M has 31 × 960 and ESMC-600M has
  37 × 1152 (the `base300M/600M` experimental checkpoints).
- jfold transposes this to `[layers, T, D]` in `features`, and `token_hidden_states`
  builds the upstream layout from its own ESMC output.

## 14. Padding and dtypes

- Tokens are **not padded**. `token_attention_mask` is all True.
- **Atoms are padded** to `A_pad = ceil(A / 32) * 32` (minimum 32). For example, the
  monomer has 186 atoms padded to 192 and the dimer 253 padded to 256. Padding atoms
  have:
  - `ref_pos` 0, `ref_element` 0, `ref_charge` 0, `ref_atom_name_chars` 0;
  - `ref_space_uid` 0, `atom_to_token` 0;
  - `atom_attention_mask` False.
- The MSA is not padded: `M` is the true row count.
- dtypes are as in §2. Every integer array is int64 except `ref_charge` (int8). Masks are
  bool. `token_bonds`, `ref_pos`, `deletion_value` and `deletion_mean` are float32.
- jfold's `features` tolerates arbitrary token, atom and MSA-row padding, as long as the
  masks are False there.

## 15. Randomness

**Featurization draws no random numbers for proteins.**
- `prepare_input(seed)` wraps `prepare_esmfold2_input` in `_seed_context`. That context
  seeds `random`, `numpy` and `torch` (and CUDA) and then restores their states.
- The seed is only consumed by RDKit ETKDG for SMILES ligands
  (`tokenize_ligand_smiles`), which is out of scope.
- Results are identical for seed `None`, 0 and 1, and with arbitrary global RNG states.
  *(verified)*
- The dump script asserts this. The fixtures are byte-identical across runs.

Model-time randomness: not features, but parity tests must pin it.

| what | where | default | in jfold |
|---|---|---|---|
| MSA column mask: a column outside row 0 is kept iff `torch.rand(T) >= rate`, drawn once per forward | `layers.py:maybe_apply_msa_column_masking` (only with an MSA encoder) | config `column_mask_rate = 0.1`; `fold(msa_column_mask_rate=0.1)` | `msa_column_mask_rate`, `mask_msa_columns` (tests set 0) |
| MSA row subsample to `max_depth` (query kept, others `randperm`, sorted), fresh each loop, only when `M > max_depth` | `layers.py:maybe_subsample_msa` | config `max_depth = 1024`; `fold(msa_max_depth=1024)`, which overrides the config | `MSAModule.max_rows = 1024`, `keep_query` (`draw_rows`, each loop) |
| ESMC input masking | `compute_lm_hidden_states` | `lm_mask_pct = 0.0` | n/a |
| LM pair dropout | release: per loop, `lm_encoder.lm_dropout = 0.25`; experimental: `lm_dropout = 0.0`; `fold(lm_dropout=0.3)` overrides both | | `lm_dropout` (`Dropout`) |
| initial pair state (release: truncated normal), diffusion noise | forward | | `init_state`, the sampler |

For a deterministic upstream forward, pass `msa_column_mask_rate=0.0` and
`msa_max_depth=None`. Set the LM dropout to 0 (`config.lm_encoder.lm_dropout = 0`,
`fold(lm_dropout=None)`), and seed torch.

## 16. Dependencies

- **torch**: every `prepare_input` output is a torch tensor. The logic is plain
  Python/numpy, plus `torch.arctan` for `deletion_value`, which is float32.
- **RDKit**:
  - `pickle.load` of `ccd.pkl` needs RDKit, since the pickle holds `Chem.Mol`s.
  - `get_ccd_conformer` calls `Chem.RemoveHs(sanitize=False)`.
  - For proteins nothing else is used: no embedding and no sanitisation.
  - The view needs only the 21 residue templates (names, elements, charges, `Computed`
    conformer coordinates), dumped once into its table (§7), so it needs no RDKit.
- **CCD**: `biohub/ESMFold2`'s `ccd.pkl`, resolved through `hf_hub_download`
  (`<HF_HUB_CACHE>/models--biohub--ESMFold2/snapshots/69869f73…/ccd.pkl`),
  or `$ESMCFOLD_CCD_PATH`. It is loaded once per process (50 256 components). It is not
  Boltz's pickle (§7).
- **Biopython / scipy**: imported by `esm.utils.msa.msa`, but not used on this path.
- **gemmi, biotite**: not used by featurization.
- **huggingface_hub**: CCD download.

## 17. Parity fixtures

`tools/fixtures/dump_esmfold2.py` (run as `uv run python tools/fixtures/dump_esmfold2.py`
from `~/jfold`) writes `tests/featurize/data/esmfold2_{monomer,dimer,monomer_msa}.npz`. Each file
holds:

- every tensor `prepare_input(..., seed=0)` returns, unbatched, in upstream's dtypes;
- `lm_input_ids` `[Lp]`, `lm_sequence_id` `[Lp]` and `lm_positions` `[T]`, captured from
  upstream's `compute_lm_hidden_states` by a recording stand-in for ESMC (no weights);
- `embed_res_type`, `embed_profile` `[T, 33]` and `embed_deletion_mean` `[T]`. These are
  what the release forward (`biohub/ESMFold2`) hands its input embedder, captured on a
  weightless meta-device model;
- `embed_profile_no_msa_features` and `embed_deletion_mean_no_msa_features`, the same for
  `biohub/ESMFold2-Experimental-Fast`. Both are zeros.

`lm_hidden_states` is not included.

| system | T | A / A_pad | M |
|---|---|---|---|
| monomer | 22 | 186 / 192 | 1 |
| dimer | 31 | 253 / 256 | 1 (chain B has no a3m in any fixture) |
| monomer_msa | 22 | 186 / 192 | 6 |

Nothing in them is random. The script needs the CCD and the two configs in the HF cache.
`jfold.models.esmfold2.features(np.load(...))` reproduces `embed_*` exactly (profile to
1 ulp); the `*_no_msa_features` captures are zero, as a model without `msa_profile` makes
them. *(verified)*

---

## Shared vs specific

**Generic AF3-style (shared with the other pipelines):**
- Chain, entity and copy bookkeeping:
  - `asym_id` in input order;
  - `entity_id` by sequence identity, in order of first appearance;
  - `sym_id` as the copy index;
  - 0-based `residue_index`;
  - `token_index = arange(T)`;
  - `mol_type = 0`.
- One token per standard residue. Heavy atoms only, in the CCD/PDB order N, CA, C, O, CB,
  … with no hydrogens. The atom → token map. The representative atom is CB, or CA for
  GLY.
- The 33-class residue vocabulary (Boltz order: pad, gap, 20 amino acids, UNK, RNA, DNA).
- Reference conformer features from a CCD conformer: element as atomic number, formal
  charge, 4-character name codes `ord(c) - 32`, and one space uid per residue.
- a3m parsing: lowercase and `.` are insertions, deletion counts sit at the next match
  column, `-` is a gap.
- Unpaired MSA rows, `has_deletion = d > 0`, and a masked-mean profile.
- Token bonds empty for standard polymers.

**ESMFold2-specific conventions:**
- **`deletion_value = π/2 · arctan(d/3)`**, not `2/π · arctan(d/3)`. `deletion_mean` is
  an unmasked mean over all rows, query included.
- The profile is computed in the model from `msa` + `msa_attention_mask`, gaps included.
  `disable_msa_features` zeroes the profile and `deletion_mean` for the experimental type
  only.
- Multi-chain unpaired MSAs are stacked **row by row** across chains, not block-diagonal.
  Exhausted chains and chains without an MSA are gap-filled, and the mask stays all True.
- `key=N` headers trigger pairing and reordering. There is no query/sequence consistency
  check, and residue columns are gathered with `min(residue_index, L_c - 1)`.
- **No OXT** on C-terminal residues. `UNK`/`X` is a single 4-atom (N, CA, C, O) token
  with `res_type` 22, **peptide-bonded** to its neighbours in `token_bonds`.
- Lowercase and non-standard letters in the sequence become UNK.
- `ref_pos` is the raw `Computed` conformer from **ESMFold2's own `ccd.pkl`**, with no
  centring or rotation. It differs from Boltz's pickle.
- `ref_charge` is int8 from a fixed table (it matches the CCD for proteins). It is used
  raw.
- `ref_atom_name_chars` are left-justified.
- Padding atoms carry `atom_to_token = 0` and `ref_space_uid = 0`.
- Atom padding is to a multiple of 32. Tokens and MSA rows are unpadded.
- No `profile`, `msa_paired` or template keys. Extra unused keys (`frames_idx`,
  `disto_cond*`, `pocket_feature`, `gt_coords`, `is_resolved`).
- ESMC input:
  - ESMC vocabulary ids;
  - `UNK`/`X` → 24 (`X`);
  - per-chain BOS/EOS wrapping;
  - chains packed in sorted `asym_id` order into one sequence with block-diagonal
    `sequence_id` and absolute rotary positions;
  - `lm_hidden_states` `[T, n_layers+1, D]`, zero off-protein.
