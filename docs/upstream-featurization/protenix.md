# Protenix lineage featurization (Protenix, Mol-1, OpenDDE): protein-only spec

Scope: standard amino acids, one or more chains, no ligands, nucleic acids or modified
residues, unpaired MSAs only, and one template per chain for Protenix (§1.10) and Mol-1
(§2.3); OpenDDE takes no given template (§3.1). This document specifies what each upstream
pipeline emits for such inputs, key by key. It is the reference for the views
`jfold.featurize.views.protenix`, `.mol1` and `.opendde`. The fixtures are
`tests/featurize/data/{protenix,mol1,opendde}_<system>.npz` (§5), written by
`tools/fixtures/dump_protenix.py`, `tools/fixtures/dump_mol1.py` and
`tools/fixtures/dump_opendde.py`.

Package paths are relative to `~/jfold/.venv/lib/python3.12/site-packages/`.

- `protenix` is the `protenij` fork of Protenix (git `escalante-bio/protenij`
  7cad393). The fixtures and jfold's Protenix checkpoint test come from its
  `protenix.data.template.featurize`.
- `mol` is Mol-1 (`molence` 1.0.0), forked from upstream Protenix bd54a05.
- `opendde` is OpenDDE 1.0.0, from the same upstream Protenix base.

Contents:

1. Protenix (full spec)
2. Mol-1 (differences only)
3. OpenDDE (differences only)
4. Shared vs specific
5. Fixtures, determinism, gaps

---

## 1. Protenix

### 1.1 Entry point and call chain

```python
from protenix.data.template import ChainInput, featurize
feats, atom_array, token_array = featurize([ChainInput(seq, compute_msa=False), ...])
```

A chain's template is `ChainInput(..., template=gemmi_chain)` (§1.10).

**MSA input**

- **Single sequence:** `ChainInput(seq, compute_msa=False)` writes a dummy MSA
  directory. Both `non_pairing.a3m` and `pairing.a3m` contain only `>query\n{seq}`.
- **Precomputed a3m:** `ChainInput(seq, compute_msa=False, precomputed_msa_dir=d)`,
  where `d/non_pairing.a3m` is the a3m.
  - For a heteromer (more than one distinct protein sequence), every protein chain
    also needs `d/pairing.a3m`, or an assertion fails.
  - Unpaired-only: make `pairing.a3m` the query alone. `tools/fixtures/dump_protenix.py` does this.
- No MSA server is contacted unless `compute_msa=True` and no directory is given.

**Call chain**

All in `protenix/data/`, unless noted.

1. `template/template_featurizer.py:featurize` builds the sample dict
   `{"name", "sequences": [{"proteinChain": {"sequence", "count": 1, "msa":
   {"precomputed_msa_dir", "pairing_db": "uniref100"}}}]}`.
   - Each `ChainInput` is its own entity with `count=1`, even when sequences are identical.
2. `json_to_feature.py:SampleDictToFeatures.__init__` calls
   `json_parser.py:add_entity_atom_array`, then `build_polymer`, then
   `_build_polymer_atom_array`. The steps:
   - `ccd.py:get_component_atom_array(code, keep_leaving_atoms=True, keep_hydrogens=False)`
     for each residue, with `PROTEIN_1to3`; `X` maps to `UNK`.
   - `ccd.py:_connect_inter_residue` adds the C(i)-N(i+1) bonds.
   - `json_parser.py:remove_leaving_atoms` drops one leaving group per inter-residue
     bond. It calls `random.sample`, which has only one candidate for amino acids.
   - `_remove_non_std_ccd_leaving_atoms`.
   - `add_reference_features` reads `ccd.py:get_ccd_ref_info` into
     `ref_pos`/`ref_charge`/`ref_mask` annotations.
3. `json_to_feature.py:SampleDictToFeatures.get_feature_dict` runs these steps:
   - `get_atom_array`:
     - `build_full_atom_array`: chain ids `A`, `B`, ... (`utils.int_to_letters`);
       `label_entity_id = str(i+1)`.
     - `add_bonds_between_entities` (no-op).
     - `mse_to_met`.
     - `add_atom_array_attributes`. These are `parser.py:AddAtomArrayAnnot.*`:
       `add_token_mol_type`, `add_centre_atom_mask`, `add_atom_mol_type_mask`,
       `add_distogram_rep_atom_mask`, `add_plddt_m_rep_atom_mask`,
       `add_cano_seq_resname`, `add_tokatom_idx`, `add_modified_res_mask`,
       `unique_chain_and_add_ids`, `find_equiv_mol_and_assign_ids`,
       `add_ref_space_uid`.
   - `tokenizer.py:AtomArrayTokenizer.get_token_array`.
   - `constraint_featurizer.py:ConstraintFeatureGenerator.generate_from_json`, which
     yields zeros with no constraints.
   - `featurizer.py:Featurizer.get_all_input_features`:
     - `get_token_features`
     - `get_bond_features`
     - `get_reference_features`, which uses `protenix/utils/geometry.py:random_transform`
       and `get_token_frame`
     - `get_extra_features`
     - `get_chain_perm_features`
     - `get_mask_features`
   - `Featurizer.get_token_frame` gives `has_frame` and `frame_atom_index`.
4. `featurize` adds `distogram_rep_atom_mask` (int64).
5. `msa_featurizer.py:InferenceMSAFeaturizer.make_msa_feature` →
   `get_inference_prot_msa_features_for_assembly`:
   - For each unique sequence:
     - `process_prot_single_sequence` → `process_single_sequence` →
       `msa_utils.py:make_sequence_features`.
     - `load_and_process_msa` → `parse_prot_msa_data` (`parse_a3m`) →
       `make_msa_features` (`_make_msa_profile`).
     - Then `convert_monomer_features`.
   - `merge_all_chain_features`:
     - `add_assembly_features`
     - `pair_and_merge`:
       - `process_unmerged_features`
       - heteromers only: `openfold_local/data/msa_pairing.py:create_paired_features`
         and `deduplicate_unpaired_sequences`
       - `crop_chains`
       - `merge_chain_features`
       - `process_prot_final`: `correct_msa_restypes`, `final_transform`,
         `filter_features`
     - `clip_msa`
   - `tokenize_msa` maps MSA columns to tokens.
6. `utils.py:data_type_transform` casts the index keys to int64.
7. `template/template_featurizer.py:build_templates_from_chains` (with
   `_extract_chain_template` for a chain with a template, `_empty_chain_template` for one
   without) → `Templates.as_protenix_dict`.
8. `featurize` adds `atom_rep_atom_idx = nonzero(distogram_rep_atom_mask)`.

`protenix predict` uses a separate path, `infer_data_pipeline.py`, with JSON input and
optional ESM. It is not covered here (see §5).

### 1.2 Token vocabulary (`protenix/data/constants.py`)

`STD_RESIDUES_WITH_GAP` has 32 classes:

| idx | 0-19 | 20 | 21-25 | 26-30 | 31 |
|---|---|---|---|---|---|
| | ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL | UNK | A G C U N (RNA) | DA DG DC DT DN (DNA) | `-` gap |

- The amino acids are in AF2 one-letter order `ARNDCQEGHILKMFPSTWYV`
  (`jfold.featurize.system.AMINO_ACIDS`).
- Unknown residue: sequence letter `X` becomes CCD `UNK`, restype 20.
  - It gets UNK's CCD atoms: `N CA C O CB CG` (+`OXT` at the C-terminus).
  - Any other letter is a `KeyError` in `PROTEIN_1to3`.
- Gap: 31. It appears only in MSA rows and the profile.

### 1.3 Tokenization (`tokenizer.py:AtomArrayTokenizer.tokenize`)

- **One token per residue.** A standard residue with `mol_type != "ligand"` is one
  token holding all of its atoms.
- **Token order is atom-array order:**
  - chains in input order;
  - residues in sequence order.
- **Centre atom:** CA, from `centre_atom_mask` (= `pae_rep_atom_mask`).
- **`restype`** `[T,32]` float32: the one-hot of `STD_RESIDUES_WITH_GAP[cano_seq_resname]`.
  For proteins, `cano_seq_resname` is the residue name (UNK for X).

### 1.4 Atom set and order

**Source**

- CCD file `~/.protenix/components.v20240608.cif`
  (`protenix.configs.configs_data.data_configs["ccd_components_file"]`), parsed
  with biotite: `pdbx.get_component(use_ideal_coord=True)`.
- Hydrogens and deuterium are removed (`keep_hydrogens=False`).

**Order**

- Per residue: the CCD `chem_comp_atom` order of the heavy atoms.
- This equals the key order of `constants.RES_ATOMS_DICT`. Checked for all 21
  residue types; for example MET is `N CA C O CB CG SD CE`.

**Leaving atoms**

- For amino acids, the only heavy leaving atom is `OXT` (group `C: [[OXT]]`).
- Every residue with a C-N bond to the next residue loses `OXT`. So `OXT` survives
  only on the last residue of each chain.
- N-terminus: nothing extra (`H2`/`H` are hydrogens).
- Atoms per residue = CCD heavy atoms − 1, +1 for the C-terminal residue. Examples:
  - Trp: 14 internal, 15 at the C-terminus.
  - Gly: 4 internal, 5 at the C-terminus.

**Atom/token maps**

| key | shape, dtype | rule |
|---|---|---|
| `atom_to_token_idx` | `[N]` int64 | token of the atom (residue order) |
| `atom_to_tokatom_idx` | `[N]` int64 | `RES_ATOMS_DICT[res_name][atom_name]`, which is the position within the residue (`OXT` = last index, e.g. 5 for ALA, 9 for GLN) |

### 1.5 Reference conformer features (`featurizer.py:get_reference_features`)

**Conformer source**

- `ccd.py:get_ccd_ref_info(code)` reads the RDKit pickle
  `~/.protenix/components.v20240608.cif.rdkit_mol.pkl`.
- It uses conformer `mol.ref_conf_id`, which is 2 (`ref_conf_type = "rdkit"`) for all
  amino acids.
- The pickle holds three conformers:
  - 0 = CCD ideal coordinates;
  - 1 = CCD model coordinates;
  - 2 = an RDKit-generated conformer (with hydrogens), the one used.
- Atoms are looked up by name through `mol.atom_map`.
- The coordinates are stored as float32 in `atom_array.ref_pos`. The fixtures save
  this as `aux_ref_pos_conformer`.

**`ref_pos`** `[N,3]` float32. For each `ref_space_uid` in `np.unique` (ascending)
order, with `p` the float32 conformer of that residue's atoms:

```python
p = p - p.mean(axis=0)                     # centre over the residue's present atoms (incl. OXT)
t = np.random.uniform(-1.0, 1.0, size=3)   # legacy global numpy RNG
R = Rotation.random().as_matrix()          # = quaternion (x,y,z,w) = normalize(np.random.normal(size=4))
ref_pos_res = np.dot(p + t, R.T)           # Protenix/Mol-1: translate, THEN rotate
```

- The results are concatenated and cast to float32.
- A numpy-only replay is bit-exact against all fixtures:
  `rs = np.random.RandomState(seed)`, then per residue `t = rs.uniform(-1,1,3)`,
  `q = rs.normal(size=4)`, `q /= |q|`, and the standard quaternion→matrix formula
  with `q = (x,y,z,w)`.
- `Featurizer(ref_pos_augment=True)` is hard-wired in `featurize`. The augmentation
  cannot be switched off from the API.

**The other conformer keys**

| key | shape, dtype | rule |
|---|---|---|
| `ref_mask` | `[N]` int64 | `mol.ref_mask[atom]`, 1 for every standard amino-acid atom |
| `ref_charge` | `[N]` int64 | formal charge of the RDKit atom: **ARG NH2 = +1, HIS ND1 = +1, LYS NZ = +1**, 0 for every other atom (all residues, all positions). The adapter uses `arcsinh(ref_charge)` |
| `ref_element` | `[N,128]` int64 | one-hot over `get_all_elems()`: the RDKit symbols for Z=1..118 in upper case, then `UNK_ELEM_119..128`. Index = Z−1: C=5, N=6, O=7, S=15 |
| `ref_atom_name_chars` | `[N,4,64]` int64 | `atom_name.ljust(4)`, each char one-hot at `ord(c)-32` (space = 0). Example: `CA` → `[35,33,0,0]` |
| `ref_space_uid` | `[N]` int64 | index of `(asym_id_int, res_id)` in `np.unique(..., axis=0)`. For protein-only inputs it equals `atom_to_token_idx` |

### 1.6 Representative atoms and frames

All per atom unless stated.

| key | shape, dtype | rule |
|---|---|---|
| `distogram_rep_atom_mask` | `[N]` int64 | 1 at CB, or at CA for GLY. UNK uses CB |
| `atom_rep_atom_idx` | `[T]` int64 | `nonzero(distogram_rep_atom_mask)` (Protenix only). The jfold adapter computes the same: `rep_atom = nonzero(distogram_rep_atom_mask, size=T)` |
| `pae_rep_atom_mask` | `[N]` int64 | 1 at CA (`centre_atom_mask`) |
| `plddt_m_rep_atom_mask` | `[N]` int64 | 1 at CA |
| `has_frame` | `[T]` int64 | 1 |
| `frame_atom_index` | `[T,3]` int64 | global atom indices of N, CA, C |

### 1.7 Chain, entity and position ids (`parser.py:unique_chain_and_add_ids`, `featurizer.py:get_token_features`)

All are `[T]` int64.

- **`asym_id`:** chain index 0, 1, 2, ... in input order.
- **`entity_id`:** the rank of `label_entity_id` in **string-sorted** order.
  - With `featurize`, every chain is its own entity (`label_entity_id = str(i+1)`),
    even when sequences repeat. A homodimer gets `entity_id` 0, 1 and `sym_id` 0, 0.
  - At 10 or more chains the string sort reorders the ids. Verified for 11 chains:
    `[0,3,4,5,6,7,8,9,10,1,2]`.
  - The model only compares entity ids for equality, so the reordering does not
    change the relative-position features. The view reproduces it anyway, for exact
    dict parity.
- **`sym_id`:** the copy index within an entity, counting chains of the same entity
  in order. With `featurize` it is always 0.
- **`residue_index`:** `res_id`, **1-based** within each chain.
- **`token_index`:** `arange(T)`.

**Per-atom extras**

These are not read by `jfold.models.protenix.features`; the Protenix view returns them
(§1.12).

- `mol_id`: covalently connected component; one per chain here.
- `entity_mol_id`: identical molecules share it; here one per chain.
- `mol_atom_index`: index within the mol.
- `is_protein`/`is_ligand`/`is_dna`/`is_rna`: int64, by the chain's majority
  `mol_type`.
- `modified_res_mask`: 0.

**Not a key: relative positions.** The model computes them as AF3 Algorithm 3 with
`r_max=32`, `s_max=2`, giving 139 features. `jfold.blocks.inputs.relative_position_features`
matches upstream `generate_relp` exactly; this was checked against `aux_relp` in the
Mol-1/OpenDDE fixtures.

### 1.8 Bonds

- **`token_bonds`** `[T,T]` float32 is **all zeros** for protein-only inputs.
  - `get_bond_features` drops standard-standard polymer bonds.
  - The peptide bonds therefore never appear, and inter-chain bonds do not exist.
  - The adapter feeds it as `bonds = token_bonds[..., None]`.
- **`bond_mask`** `[N,N]` int64 is all zero (ligand-polymer bonds only).

### 1.9 MSA

**Parsing** (`msa_utils.py:parse_a3m`, `parse_prot_msa_data`)

- Lines starting with `>` start a record; `#` lines and blank lines are skipped.
- Sequence lines of a record are concatenated.
- Per row, a deletion count is attached to each uppercase or `-` column. It is the
  number of lowercase letters (`str.islower`) immediately before that column.
  Lowercase letters after the last column are dropped.
- The aligned sequence is the row with lowercase letters removed. All rows must have
  the query length.

**Alphabet** (`HHBLITS_AA_TO_ID`, then `correct_msa_restypes`)

- The 20 amino acids map to their restype index.
- `B`→ASP(3), `Z`→GLU(6), `U`→CYS(4).
- `X`, `J`, `O` → UNK(20).
- `-` → 31.
- Any other character (`.`, `*`, ...) raises a `KeyError`.

**Query row**

- The a3m's **first record is used as the query row as is**.
- The chain sequence is never inserted or substituted; it is only used to build the
  single-sequence fallback.
- The first a3m record must therefore be the query.

**Deduplication** (`make_msa_features`)

- Rows whose aligned sequence (insertions stripped) was already seen are dropped.
- A hit identical to the query is dropped. In the fixture, `hit1` disappears and
  5 of the 6 rows remain.

**Profile and deletion mean**

These are computed per chain on the deduplicated unpaired rows, before any merging.

- `profile[i,c]` = fraction of rows with class `c` in column `i`, over the 32
  classes. Gap (31) and UNK (20) are counted. The query row is counted once.
- `deletion_mean[i]` = mean over the same rows of the **raw** deletion count.
- Example from the fixture: column 8 has a deletion of 2 in 1 of 5 rows, so
  `deletion_mean = 0.4`.

**Per-row features** (`final_transform`)

```python
has_deletion   = np.clip(deletion_count, 0, 1).astype(bool)          # [S,T] bool
deletion_value = (2 / np.pi) * np.arctan(deletion_count / 3)         # [S,T] float32
```

**Assembly**

- **Monomer or homomer** (one distinct protein sequence): no pairing.
  - The rows are exactly the deduplicated unpaired rows.
  - The query appears **once**, as row 0.
  - Several chains are concatenated along the columns. Chains with fewer rows are
    padded with gap rows (31, deletion 0) — `dense_max`.
- **Heteromer:**
  - Rows are the paired block (from `pairing.a3m`; query only for unpaired-only use),
    then each chain's unpaired rows minus rows equal to a paired row. This removes
    the query.
  - The unpaired blocks are padded with gaps to the deepest chain.
  - Checked: chain A with the fixture a3m plus chain B with none gives 5 rows. Row 0
    is the joint query; rows 1-4 are hits 2-5 in A's columns and gaps in B's columns.

**Limits**

- `MSA_MAX_SIZE = 16384` rows (`crop_chains`, `clip_msa`).
- For heteromers, at most 8192 paired rows; the unpaired crop is reduced by the
  number of non-gap paired rows.
- No row limit at parse time (`seq_limit = -1`).
- No random subsampling in the featurizer.

**Tokenization** (`tokenize_msa`): for proteins, each token gets the column of its
`(asym_id, residue_index)`.

**Single sequence**

- The dummy a3m gives `msa = [restype argmax]` (1 row).
- `profile` = one-hot of the restype, float64.
- `deletion_mean = 0`; `has_deletion` and `deletion_value` are 0.

| key | shape, dtype | adapter use |
|---|---|---|
| `msa` | `[S,T]` int64 | `probabilities(msa, 32)` (one-hot) |
| `has_deletion` | `[S,T]` **bool** | cast to float32 |
| `deletion_value` | `[S,T]` float32 | |
| `profile` | `[T,32]` **float64** | cast to float32 |
| `deletion_mean` | `[T]` **float64** | cast to float32 |

### 1.10 Templates (`template/template_featurizer.py`)

`featurize` always emits **one** template slot (`build_templates_from_chains(chains,
num_templates=1)`), holding every chain's template side by side.

**Input.** `ChainInput.template`, a `gemmi.Chain` (mosaic passes its target chain's). jfold's
`Chain.template` (a `jfold.featurize.Template`: a sequence and each residue's atoms by name,
unresolved residues with none) is that chain: one gemmi residue per template residue, an
unresolved one without atoms.

**Placement** (`_extract_chain_template`):

- The template chain's amino-acid residues (`gemmi.find_tabulated_residue(name)
  .is_amino_acid()`; the 20 and UNK all are) must be exactly as many as the chain's
  residues, or upstream raises `ValueError("Template chain has n residues but expected
  m")`; the view raises too. Template residue `i` stands for the chain's residue `i`: there
  is no alignment, and no check that the residue types agree.
- `aatype`: the **template's** residue type (`restype_order` of gemmi's one-letter code,
  UNK = 20 for `X`), unresolved residues included.
- Atoms: atom37 (`residue_constants.atom_types`) positions and masks by atom name, float32
  as read; other names (hydrogens) are dropped.
- A chain without a template (`_empty_chain_template`): `aatype` 20 (UNK), no atoms.

**Features** (`Templates.as_protenix_dict`; the view's `protenix_templates` and
`template_pair_features`, which repeat its float32 operations, so the results are bit-exact):

- Pseudo-beta: CB, or CA for glycine (by the template's `aatype`, so UNK uses CB), with
  that atom's mask.
- `template_distogram`: one-hot of the pseudo-beta pair's squared distance in 39 bins with
  lower edges `linspace(3.25, 50.75, 39)²` (exact in float32 and float64), each bin
  `lower < d² < upper`, the last bin's upper edge `1e8`; times the pair mask.
- Unit vectors: per token a frame from N, CA, C (`e1 = (C - CA) / (|C - CA| + 1e-6)`, `e2`
  the part of `N - CA` orthogonal to it, normalized the same way, `e3 = e1 × e2`); the
  vector `CA_j - CA_i` in token `i`'s frame, normalized (`+ 1e-6`); times the backbone pair
  mask (N, CA and C all present on both tokens). Missing atoms are zero vectors.
- Pair features are computed over the whole system, between chains too (the model masks
  those, `jfold.models.protenix.template_features`).

| key | shape, dtype | value |
|---|---|---|
| `template_aatype` | `[1,T]` int64 | template residue types; 20 (UNK) on chains without a template |
| `template_distogram` | `[1,T,T,39]` float32 | as above |
| `template_pseudo_beta_mask` | `[1,T,T]` float32 | pseudo-beta mask, outer product |
| `template_unit_vector` | `[1,T,T,3]` float32 | as above |
| `template_backbone_frame_mask` | `[1,T,T]` float32 | backbone mask, outer product |

**Without templates** this is 20 (UNK) everywhere and zeros (`empty_templates`).

- Because `template_aatype` is present, the jfold adapter builds template features
  with `template_mask = ones(1)`, and protenij runs its template embedder.
- Upstream configs:
  - `template_embedder.n_blocks = 0` for mini and tiny, which ignore templates.
  - `n_blocks = 2` for base v1.0.0, base 2025-06-30 and v2, which see the UNK
    one-hot template when there is none.

### 1.11 Model-specific extras

**`constraint_feature`** is a nested dict of float32 zeros:

- `contact` `[T,T,2]`
- `contact_atom` `[T,T,2]`
- `pocket` `[T,T,1]`
- `substructure` `[T,T,4]`

The upstream model feeds it to `ConstraintEmbedder`. Every sub-embedder is disabled
(`enable: False`) except in `protenix_base_constraint_v0.5.0`, so for the jfold
models the result is `None` and the key is a no-op. The view does not emit it. The
fixtures flatten it as `constraint_feature__<name>`.

**`resolution`** `[1]` float32 = −1. Not consumed, and not emitted by the view.

**ESM:** `featurize` never emits `esm_token_embedding`. Only
`protenix_mini_esm/ism_v0.5.0` use it (through `infer_data_pipeline`), and jfold
does not support them.

**Padding:** none. No token or atom padding, and the MSA has as many rows as it
has.

**Optional pad masks:** `atom_pad_mask` `[N]` and `token_pad_mask` `[T]` (bool) are
not upstream featurizer outputs, and the view does not emit them. When present they
mark padded atoms and tokens; `jfold.models.protenix.features` and protenij both honour
them.

### 1.12 What the view returns, and what jfold reads

**`jfold.models.protenix.features`** reads (the view's `JFOLD_KEYS`, plus the optional
pad masks):

- Atom features:
  - `ref_pos`, `ref_charge`, `ref_mask`, `ref_element`, `ref_atom_name_chars`.
  - They are concatenated as `[pos(3), arcsinh(charge), mask, element(128), name(256)]`
    into 389 channels, × `ref_mask`.
- `ref_space_uid`, `atom_to_token_idx`.
- Token features: `restype`, `profile`, `deletion_mean`, giving 65 channels.
- Relative positions: `asym_id`, `residue_index`, `entity_id`, `token_index`,
  `sym_id`.
- `distogram_rep_atom_mask` (rep atom), `atom_to_tokatom_idx`, `token_bonds`.
- MSA: `msa`, `has_deletion`, `deletion_value`, giving 34 channels.
- Templates: `template_*`, giving 108 channels.
- Optionally `atom_pad_mask` and `token_pad_mask`.

**The Protenix view** returns `JFOLD_KEYS` (the list above, without the pad masks) plus
these further keys of upstream's dict (`EXTRA_KEYS`):

- `atom_rep_atom_idx` (§1.6);
- `has_frame`, `frame_atom_index` (§1.6);
- `pae_rep_atom_mask`, `plddt_m_rep_atom_mask` (§1.6);
- `is_protein`, `is_ligand`, `is_dna`, `is_rna`, `mol_id`, `mol_atom_index`,
  `entity_mol_id`, `modified_res_mask` (§1.7);
- `bond_mask` (§1.8).

It omits `constraint_feature` and `resolution` (§1.11).

### 1.13 Randomness

For protein-only inputs, the **only** random step is the per-residue `ref_pos`
augmentation, from numpy's global legacy RNG (`np.random.uniform`, then scipy
`Rotation.random()`, which draws `np.random.normal(size=4)`).

- Seed with `np.random.seed(s)` right before `featurize`.
- `random.seed(s)` also covers `remove_leaving_atoms`' `random.sample`, which is
  deterministic for amino acids.
- The draw order is `ref_space_uid` order: one uniform(3) and one normal(4) per
  residue.

The views draw the same distributions from a JAX `key` (`random_pose`); with
`key=None`, `ref_pos` is the centred, unposed conformer. The parity tests compare the
unposed `ref_pos` with the centred `aux_ref_pos_conformer` exactly, and check that
replaying upstream's numpy draws on it gives upstream's posed `ref_pos`.

### 1.14 Dependencies

- **numpy**.
- **biotite**: AtomArray, BondList, CIF parsing of the ~430 MB CCD file. Slow on
  first use; cached with `lru_cache`.
- **rdkit**: unpickling the CCD mol pickle, the periodic table for `get_all_elems`,
  and `SanitizeMol`/substructure permutations in `get_ccd_ref_info`.
- **scipy**: `Rotation`.
- **scikit-learn**: `KDTree`, for ligand frames only, but imported at module level.
- **gemmi**: `ChainInput.template` type.
- **torch**: imported by `constraint_featurizer.py`, but no torch tensor reaches the
  output.
- **The CCD**:
  - `components.v20240608.cif` (atoms, leaving flags, one-letter codes, mol type).
  - `components.v20240608.cif.rdkit_mol.pkl` (conformers, charges, atom maps).

The views need only a per-residue table for the 21 residue types: heavy atom names in
CCD order, elements, formal charges, `ref_mask`, and the conformer coordinates **of the
pipeline's own pickle** (§1.5). `tools/tables/extract_{protenix,mol1,opendde}.py`
extract them with rdkit into `src/jfold/featurize/data/{protenix,mol1,opendde}.npz`.

---

## 2. Mol-1 (differences from Protenix)

### 2.1 Entry point and call chain

```python
from mol.data.inference.infer_dataloader import InferenceDataset
ds = InferenceDataset.__new__(InferenceDataset)   # as tests/test_checkpoint_mol1.py and tools/fixtures/dump_mol1.py
ds.use_msa, ds.msa_pair_as_unpair, ds.use_rna_msa, ds.use_template = use_msa, True, False, False
ds.online_template_featurizer = None
data, atom_array, _ = ds.process_one({"name": ..., "sequences": [{"proteinChain": {"sequence": s, "count": 1, "unpairedMsaPath": a3m}}]})
feats = {k: v.numpy() for k, v in data["input_feature_dict"].items()}
```

**CCD data.** The CCD comes from `$MOL_ROOT_DIR/common/components.cif` and
`components.cif.rdkit_mol.pkl`. jfold's checkpoint test and `dump_mol1.py` patch
`mol.data.core.ccd.COMPONENTS_FILE` and `RKDIT_MOL_PKL`.

**Call chain** (paths under `mol/data/`)

1. `inference/infer_dataloader.py:InferenceDataset.process_one`.
2. `inference/json_to_feature.py:SampleDictToFeatures.get_feature_dict`:
   - The same steps as Protenix, from `inference/json_parser.py`,
     `core/parser.py`, `core/ccd.py`, `tokenizer.py` and `core/featurizer.py`.
   - No constraint generator.
   - `Featurizer(include_discont_poly_poly_bonds=True)`, which has no effect on
     linear proteins.
3. `msa/msa_featurizer.py:InferenceMSAFeaturizer.make_msa_feature` →
   `FeatureAssemblyLine.assemble`, using `msa/msa_utils.py:RawMsa`, `MSACore`,
   `MSAPairingEngine` and `map_to_standard`.
   - This runs only if `use_msa`.
4. `template/template_featurizer.py:InferenceTemplateFeaturizer.make_template_feature`
   → `TemplateFeatureAssemblyLine`, with `template_utils.py:TemplateFeatures.empty_template_features`
   (and, with templates, `TemplateHitFeaturizer.parse_json_templates`, §2.3).
5. `utils.py:make_dummy_feature`: the MSA when `use_msa=False`.
6. `utils.py:data_type_transform`.

Everything is a torch tensor; `.numpy()` gives the arrays below.

**Same as Protenix.** Checked equal on all three fixtures:

- tokenization, vocabulary, atom set and order, OXT rule;
- `ref_mask`, `ref_charge`, `ref_element`, `ref_atom_name_chars`, `ref_space_uid`;
- the atom/token maps, rep masks and frames;
- all ids, `token_bonds`;
- the MSA values (profile, deletion features, and the rows apart from the duplicated query row; see below).

### 2.2 Differences

**Conformer coordinates**

- Mol-1 has its own pickle, still `ref_conf_id = 2`, `ref_conf_type = "rdkit"`, with
  the same `atom_map`. Its conformers are different, independently generated ones.
- Kabsch RMSD of the heavy atoms against Protenix's ranges from 0.03 Å (ALA) to
  1.6 Å (LEU).
- So `ref_pos` differs from Protenix even with the same seed.
- The `ref_pos` formula and RNG order are identical to Protenix: translate, then
  rotate. Verified bit-exact from `aux_ref_pos_conformer`.

**dtypes**

- `profile` `[T,32]` float32, `deletion_mean` `[T]` float32, `has_deletion` `[S,T]`
  float32.
- No float64 anywhere.

**MSA input:** `proteinChain.unpairedMsaPath`, or `unpairedMsa` as a string, plus
`use_msa=True`.

**MSA query handling** (`RawMsa`, `FeatureAssemblyLine.assemble`):

- Every protein chain always gets an unpaired MSA and a "paired" MSA.
  - With no a3m, each is `[query]`.
  - The final `msa` is `concat([paired block, unpaired block])`.
- **No pairing needed** (monomer or homomer, 1 distinct sequence):
  - The paired block is the query alone.
  - The unpaired block (deduplicated, including the query) follows unchanged.
  - **The query appears twice** (rows 0 and 1).
  - `monomer_msa` therefore has 6 rows (Protenix has 5). The profile and
    deletion_mean still come from the 5 unpaired rows and match Protenix.
- **Heteromer:**
  - `MSAPairingEngine.cleanup_unpaired_features` removes unpaired rows equal to a
    paired row, so the query appears once.
  - Same rows as Protenix: verified 5×31 with A-only MSA.
- `use_msa=True` with no a3m: 2 rows (monomer/homomer) or 1 row (heteromer).
- `use_msa=False` is the jfold test path: `make_dummy_feature` gives
  - `msa = argmax(restype)[None]` (1 row);
  - `profile = restype` (float32);
  - zeros for `has_deletion`, `deletion_value`, `deletion_mean`;
  - `*_num_alignments = 0` (int32).
- The fixtures run `use_msa=True` exactly when some chain has an a3m. The view's
  `use_msa` argument defaults to the same rule; pass `use_msa=True` to get upstream's
  two-row query-only MSA for a system without a3ms.

**Chains of 4 residues or fewer** get no MSA, whatever is given; `skip` in
`assemble` leaves only the query.

**Deletion counts** (`MSACore.sequences_to_array`)

- A column is any character in `MSA_PROTEIN_SEQ_TO_ID`: 20 amino acids, `BJOUXZ`
  and `-`. The mapping is the same as Protenix: `B→3`, `Z→6`, `U→4`,
  `J/O/X→20`, `-→31`.
- **Every other character counts as an insertion**: lowercase, and also `.` or `*`,
  where Protenix raises.
- The deletion matrix is clipped to int8: counts > 127 become 127. Protenix does
  not clip.
- `deletion_value = arctan(d/3) * 2/π` and `has_deletion = clip(d, 0, 1)` are
  computed on the clipped int8 values. `deletion_mean` uses the unclipped counts.
- Deduplication ignores lowercase letters, as in Protenix. The query is never
  injected: `_verify_query` only checks that the shape matches.

**Limits:** `max_msa_size = 16384`; paired rows ≤ 8192; unpaired rows ≤ 16384 −
paired rows.

**Extra keys** (not read by jfold, not returned by the view):

- `msa_mask` `[S,T]` bool (ones);
- `deletion_matrix` `[S,T]` int8;
- `prot_paired_num_alignments`, `prot_unpaired_num_alignments`;
- `prot_pair_num_alignments`, `prot_unpair_num_alignments` (int64 scalars);
- the `rna_*` equivalents.
- With `use_msa=False`, only the four `*_pair/unpair_num_alignments` keys appear,
  as int32 zeros.

**Templates when absent:** `TemplateFeatures.empty_template_features`, padded to
**4** slots with zeros:

| key | shape, dtype | value |
|---|---|---|
| `template_aatype` | `[4,T]` int64 | row 0 = 31 (gap), rows 1-3 = 0 (**ALA**, from zero padding) |
| `template_atom_positions` | `[4,T,24,3]` float32 | 0 |
| `template_atom_mask` | `[4,T,24]` **bool** | False |
| `template_distogram`, `template_pseudo_beta_mask`, `template_unit_vector`, `template_backbone_frame_mask` | `[4,T,T,39]`, `[4,T,T]`, `[4,T,T,3]`, `[4,T,T]` float32 | 0 |

- jfold's adapter (`jfold.models.mol1.features` is `protenix.features`) consumes all
  4 slots, including the gap and ALA one-hots. The view emits the 5 keys it reads
  (§1.10's keys, with 4 slots), not `template_atom_positions` or `template_atom_mask`.
- Templates are also skipped for chains of 4 residues or fewer (§2.3).

**Absent keys:** no `constraint_feature`, no `atom_rep_atom_idx`.

**What the view returns:** `JFOLD_KEYS` (§1.12) plus `is_protein`.

**Randomness:** the same single source as Protenix. `process_one` touches no torch
RNG. Upstream Mol-1 also draws a random subset of MSA rows per trunk pass, as Protenix
does; that happens in the model, not the featurizer (§5).

**Dependencies:** like Protenix, plus **torch**, which is required: every feature is
built as a tensor.

### 2.3 Templates

Mol-1's inference takes templates per chain through `proteinChain.templatesPath`
(`InferenceTemplateFeaturizer.make_template_feature`), with `use_template=True` and the
`TemplateHitFeaturizer` that `InferenceDataset.__init__` builds. Two kinds of file:

- `.a3m` / `.hhr`: template-search hits naming PDB entries (`pdbid_chain`), whose mmCIFs
  come from `prot_template_mmcif_dir` or PDBe, filtered by release date (2021-09-30) and as
  near-duplicates of the query, and realigned to the query with kalign
  (`TemplateHitFeaturizer.get_templates`). This is search output, not a given structure;
  jfold does not reproduce it.
- **`.json`**: AF3-style templates, `[{"mmcif": str, "queryIndices": [...],
  "templateIndices": [...]}]` (`TemplateHitFeaturizer.parse_json_templates`), a given
  structure with a given residue mapping. This is the path the view reproduces.

**jfold's `Template` as the JSON** (the view's convention; `tools/fixtures/dump_mol1.py`
writes it so): the mmCIF holds the template's resolved residues (those with atoms), in
order; each maps to the chain residue at its position in the template, so the template must
have the chain's length (the view raises otherwise, as for Protenix): `queryIndices` = the
resolved residues' positions, `templateIndices` = `0, 1, ...`. Unresolved residues are not
in the mmCIF and map to nothing.

**Parsing** (`template_parser.py:TemplateParser.parse_simple_cif`, then
`TemplateHitProcessor._get_atom_positions`):

- Biopython's `MMCIFParser`, first model, **first chain**; its residues in file order are
  the template sequence (`protein_letters_3to1`, `X` for anything else, e.g. UNK).
- Atom37 positions and masks by atom name (`MSE` `SE` as `SD`). An arginine with CD, NH1
  and NH2 has NH1 and NH2 swapped when NH1 is farther from CD than NH2.
- **Centring** (`_zero_center_positions=True`): every atom37 position of the template minus
  their mean (float32, residues in order, atom37 order within each, after the swap). The
  pair features barely depend on it, but its float32 rounding reaches every coordinate, so
  the view repeats it (`_mol1_centre`).
- `_check_residue_distances`: consecutive CA-bearing residues more than 150 Å apart make the
  template fail. **Upstream then drops it silently** (`parse_json_templates` catches every
  error, and nothing logs it), as it does a template with no atoms; **the view raises
  instead**.

**Placement** (`parse_json_templates`): query position `q` gets template residue `t`'s
atoms and one-letter code for each `(q, t)` pair; every other position is `-`.
`template_aatype` = `TEMPLATE_PROTEIN_SEQ_TO_ID` (the 20 amino acids in AF2 order, `X` →
20, `-` → 31 gap).

**Assembly** (`TemplateFeatureAssemblyLine.assemble`, `TemplateFeatures.fix_template_features`,
`Templates.as_mol_dict`):

- Up to 4 templates per chain (jfold gives one), padded to 4 slots with zeros (ALA) and
  concatenated over chains; a chain without a template gets the gap slot of
  `empty_template_features`. **Chains of 4 residues or fewer get empty features even with a
  template** (`skip_chain`), so the view ignores theirs.
- Atom37 → the 24 dense atom slots by `template_aatype`
  (`PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37`; positions times mask).
- Pseudo-beta: dense slot `RESTYPE_PSEUDOBETA_INDEX[aatype]`: CB, CA for glycine, and slot
  0 for UNK and gap. Backbone: `RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX[aatype, 0]` (C, CA, N);
  slots `(0, 0, 0)` for UNK and gap. **Every dense slot of UNK maps to atom37 index 0, its
  N**: an UNK template residue's pseudo-beta is its N, its "CA" its N, and its frame N-N-N
  (zero vectors, so zero unit vectors from it), all masked by N alone.
- The distogram and unit vectors are §1.10's (`TemplateFeatures.dgram_from_positions`,
  `compute_template_unit_vector`). They are written with `einsum`, float32 bin edges and a
  stacked frame, but give the same bits as Protenix's code (checked on random inputs and on
  every fixture), so the view computes both with `template_pair_features`.

| key | shape, dtype | value |
|---|---|---|
| `template_aatype` | `[4,T]` int64 | slot 0: template residue types, 31 where unresolved or untemplated; slots 1-3: 0 |
| `template_distogram`, `template_pseudo_beta_mask`, `template_unit_vector`, `template_backbone_frame_mask` | `[4,T,T,39]`, `[4,T,T]`, `[4,T,T,3]`, `[4,T,T]` float32 | slot 0 as above; slots 1-3: 0 |
| `template_atom_positions`, `template_atom_mask` | `[4,T,24,3]` float32, `[4,T,24]` bool | dense atoms (not read by jfold, not returned by the view) |

---

## 3. OpenDDE (differences from Mol-1)

### 3.1 Entry point

`opendde/data/inference/infer_dataloader.py:InferenceDataset(configs).process_one(entry)`,
with `configs = build_inference_config(model_name="opendde_v1", fill_required_with_null=True)`
and:

- `use_template = use_rna_msa = False`;
- `use_msa` = whether any chain has `unpairedMsaPath`. The fixtures and the view's
  default set it this way; jfold's checkpoint test uses `False`.

`get_inference_dataloader` wraps the same call in a `DataLoader`
(`num_workers = 0` by default). `tools/fixtures/dump_opendde.py` calls `process_one` directly.

**Templates: none given.** OpenDDE's `InferenceTemplateFeaturizer.make_template_feature`
reads `templatesPath` only as an `.a3m` or `.hhr` of template-search hits (Mol-1's first kind,
§2.3): PDB entries by id, their mmCIFs from `prot_template_mmcif_dir` or PDBe, filtered by a
release-date table and as near-duplicates of the query (a template with the query's own
sequence is dropped), and realigned to the query with the kalign binary. It has no
`parse_json_templates`, so it cannot take a given structure, and the view refuses templates
(`System.without_templates`). (mosaic's own OpenDDE wrapper builds template features itself
from upstream's `Templates.as_opendde_dict`; that is not OpenDDE's inference path.)

**CCD data:** `configs.data.ccd_components_file`, by default
`~/.cache/opendde/common/components.cif` and its `.rdkit_mol.pkl`.
`InferenceDataset.__init__` installs it with `ccd.set_ccd_cache_paths`.

**biotite 1.7:** the featurizer uses `BondList` private attributes. Run
`_biotite_compat()`, copied from `tests/test_checkpoint_opendde.py` and
`tools/fixtures/dump_opendde.py`, first.

**Call chain:** the same as Mol-1 under `opendde/data/`.
`core/featurizer.py:Featurizer.get_all_input_features` additionally calls
`get_structural_token_features`, which uses `tokenizer.py:AtomArrayTokenizer.get_structural_token_array`,
`get_token_frame`, `inherit_parent_backbone_frames` and `get_polymer_residue_graph`.

**Same as Mol-1:** the residue-level dict, every key equal on all fixtures, including
the MSA convention with the duplicated query and the 4-slot empty templates. The
exceptions:

- **Conformers:** a third, distinct RDKit pickle (`ref_conf_id = 2`). On the residues checked (ALA, ARG, GLY, LYS, TRP, UNK), heavy-atom RMSD
  is 0.05-1.3 Å against Mol-1 and Protenix, and even ALA differs by 1.3 Å.
- **Augmentation order — rotate, THEN translate**
  (`opendde/utils/geometry.py:random_transform`), with the same draws:

  ```python
  ref_pos_res = np.dot(p, R.T) + t
  ```

  Verified bit-exact.
- **`template_atom_mask`** is int64, not bool.
- **Extra scalar:** `rna_paired_num_alignments`.

### 3.2 Structural tokens (`opendde/data/tokenizer.py:AtomArrayTokenizer.tokenize_structural`)

Each residue token is split by atom name. The atom array and its order are
unchanged. `PROTEIN_BACKBONE_ATOMS = {N, CA, C, O, OXT}`.

- **Non-Gly protein residue:** two structural tokens, in this order:
  - **backbone**: `role = protein_bb = 1`; its atoms are the backbone names, in atom
    order;
  - **side chain**: `role = protein_sc = 2`; all other atoms.
- **Glycine** (and any residue with no side-chain or backbone atoms): **one** token,
  `protein_bb`, holding all of its atoms.
- The C-terminal backbone token is `N CA C O OXT`. Its atoms are **not contiguous**
  in the atom array, because `OXT` comes after the side chain.

Role ids (`STRUCTURAL_TOKEN_ROLES`):

- `atom` 0
- `protein_bb` 1
- `protein_sc` 2
- `dna_bb` 3
- `dna_base` 4
- `rna_bb` 5
- `rna_base` 6

Token counts: `M = 2·T − (#Gly)`. The monomer has 44 tokens; the dimer has 61
(31 residues, one Gly).

| key | shape, dtype | rule |
|---|---|---|
| `structural_token_index` | `[M]` int64 | `arange(M)` |
| `parent_residue_idx` | `[M]` int64 | residue token index |
| `residue_token_group_id` | `[M]` int64 | `== parent_residue_idx` |
| `subtoken_role_id` | `[M]` int64 | role id as above |
| `subtoken_role` | `[M]` int64 | identical copy of `subtoken_role_id` |
| `twin_token_idx` | `[M]` int64 | the other structural token of the same residue (bb↔sc), −1 for Gly |
| `prev_parent_residue_idx`, `next_parent_residue_idx` | `[M]` int64 | the parent's predecessor and successor residue token, −1 at chain ends. Found from the C-N backbone bonds in the bond graph, falling back to same chain with `res_id` + 1. For proteins: `parent−1` / `parent+1` within the chain |
| `atom_to_structural_token_idx` | `[N]` int64 | structural token of each atom |
| `atom_to_structural_tokatom_idx` | `[N]` int64 | position within the structural token. bb: N0 CA1 C2 O3 OXT4; sc: CB0, CG1, ... |
| `structural_distogram_rep_atom_mask` | `[N]` int64 | the token's centre atom: **CA for bb tokens (incl. Gly), CB for sc tokens** (the first side-chain atom if there is no CB) |
| `structural_pae_rep_atom_mask` | `[N]` int64 | equal to `structural_distogram_rep_atom_mask` |
| `structural_has_frame` | `[M]` int64 | 1 |
| `structural_frame_atom_index` | `[M,3]` int64 | bb: global N, CA, C. sc tokens **inherit** their bb twin's frame (`inherit_parent_backbone_frames`) |
| `structural_is_polymer` | `[M]` int64 | 1 |
| `structural_polymer_type` | `[M]` int64 | 1 (protein) |
| `structural_seq_pos` | `[M]` int64 | `res_id` (1-based) |

The residue-level rep atoms (`distogram_rep_atom_mask` CB/CA-Gly,
`pae_rep_atom_mask` CA) and `atom_to_tokatom_idx` are unchanged.

### 3.3 What jfold derives from them (`jfold.models.opendde.features`)

**Residue branch:** `protenix.features(f)`.

**`StructuralTokens`**

- `parent = parent_residue_idx`
- `role = subtoken_role_id`
- `prev_parent`, `next_parent` from the keys of the same names
- `mask = ones(M)`
- `atoms` = the residue atoms, with `atom_to_token` replaced by
  `atom_to_structural_token_idx`. `atom_position` and the rep atom stay
  residue-level.

**Structural relative positions**

- AF3 Algorithm 3 on `asym_id[parent]`, `residue_index[parent]`,
  `entity_id[parent]`, `sym_id[parent]`, with `token_index = structural_token_index`.
- So the bb and sc tokens of one residue share a residue offset and differ by a
  token offset of ±1. Tokens of different residues get the "different residue"
  token bin.
- This matches upstream `expand_to_structural_tokens` → `generate_relp` exactly,
  checked against `aux_structural_relp`.

**Pair features** (`StructuralPairFeatures.build`, computed in the model, not
featurized)

- Role class: `ROLE_CLASS = (3,0,1,0,2,0,2)`, indexed by role:
  - 0 backbone
  - 1 protein side chain
  - 2 nucleotide base
  - 3 atom
- `same_parent`: same `parent`.
- `twin`: same parent, with one token backbone and the other side chain or base.
- `prev_backbone[i,j]`: both tokens are backbone, on the same chain, and
  `prev_parent[i] == parent[j]`.
- `next_backbone[i,j]`: both tokens are backbone, on the same chain, and
  `next_parent[i] == parent[j]`.
- `role_pair = ROLE_PAIR[class_i][class_j]`, 0-6, or 7 for any other pair.

### 3.4 What the view returns

`JFOLD_KEYS` (§1.12), every structural key of §3.2, and `has_frame`,
`frame_atom_index`, `is_ligand` and `pae_rep_atom_mask`. Model-time keys that upstream
adds in `update_input_feature_dict` (`relp`, `d_lm`, `v_lm`, `pad_info`) are not
featurizer outputs.

---

## 4. Shared vs specific

### Generic AF3-style (identical in all three; one implementation)

- **Vocabulary:** 32 classes, AA order `ARNDCQEGHILKMFPSTWYV`, UNK = 20, RNA 21-25,
  DNA 26-30, gap = 31. `X` in a sequence becomes `UNK` (6 heavy atoms + `OXT` at
  the C-terminus).
- **Tokenization:** one token per standard residue; centre atom CA.
- **Atoms:**
  - CCD heavy atoms in CCD order (= `RES_ATOMS_DICT` order);
  - no hydrogens;
  - `OXT` only on each chain's last residue;
  - `atom_to_tokatom_idx` = position in that order.
- **Conformer encodings:**
  - `ref_mask = 1`;
  - `ref_charge` = CCD formal charge (ARG NH2, HIS ND1, LYS NZ = +1);
  - `ref_element` = one-hot over 128 at Z−1;
  - `ref_atom_name_chars` = 4×64 one-hot of `ord(c)−32`;
  - `ref_space_uid` = residue index in `(asym, res_id)` order.
- **`ref_pos` recipe:**
  - the RDKit conformer `ref_conf_id = 2` from the pipeline's CCD pickle;
  - centred per residue (over present atoms, incl. OXT);
  - random rigid augmentation with `t ~ U(-1,1)^3` and a rotation from a
    normalized `N(0,1)^4` quaternion;
  - both drawn from numpy's global legacy RNG, per residue in uid order.
- **Ids:**
  - `asym_id` = chain order;
  - `entity_id` = rank of the string-sorted `label_entity_id`, one entity per input
    entry with `count = 1`;
  - `sym_id` = copy index within the entity;
  - `residue_index` 1-based;
  - `token_index = arange`.
  - The relative-position features are AF3 Algorithm 3 (`r_max = 32`, `s_max = 2`,
    139 channels), computed in the model.
- **Representative atoms:**
  - distogram CB (Gly CA);
  - PAE and pLDDT CA;
  - frames N-CA-C, `has_frame = 1`.
- **Bonds:** `token_bonds = 0` and `bond_mask = 0` for proteins.
- **MSA values:**
  - the a3m's first record is the query row, used verbatim;
  - lowercase letters are insertions, counted onto the next column;
  - deduplicated on the insertion-stripped sequence;
  - `B/Z/U` → `D/E/C`; `X/J/O` → UNK(20); `-` → 31;
  - `has_deletion = d > 0`; `deletion_value = 2/π·arctan(d/3)`;
  - `profile` and `deletion_mean` (raw counts) over the chain's own deduplicated
    unpaired rows, gap and UNK included, the query counted once;
  - multi-chain MSAs are gap-padded per chain (`dense_max`);
  - heteromer: paired block (query) first, unpaired rows minus paired duplicates;
  - cap 16384 rows.
- **No padding** of tokens or atoms.

### Per-pipeline conventions (the thin "views")

| | Protenix (`featurize`) | Mol-1 | OpenDDE |
|---|---|---|---|
| CCD and conformer pickle | `components.v20240608` | Mol-1 release `common/` | OpenDDE `common/` |
| `ref_pos` composition | `(p + t) @ R.T` | `(p + t) @ R.T` | `p @ R.T + t` |
| MSA input | `precomputed_msa_dir/non_pairing.a3m` (+`pairing.a3m` for heteromers) | `unpairedMsaPath`, `use_msa=True` | `unpairedMsaPath`, `use_msa=True` |
| Query rows, monomer/homomer with MSA | 1 (row 0) | **2** (paired query + unpaired query) | **2** |
| No MSA | 1 row (dummy a3m) | `use_msa=False`: 1 row; `use_msa=True` without a3m: 2 rows (1 for heteromers) | same as Mol-1 |
| Chains ≤ 4 residues | MSA used | MSA ignored, templates skipped | same as Mol-1 |
| Deletion count | unbounded; non-letters raise | int8-clipped (127); any non-alphabet char is an insertion | same as Mol-1 |
| dtypes | `profile`, `deletion_mean` float64; `has_deletion` bool | all float32 | all float32 |
| Empty templates | 1 slot, aatype 20 (UNK) | 4 slots, aatype `[31,0,0,0]`, plus `template_atom_positions/mask` `[4,T,24]` (mask bool) | 4 slots as Mol-1 (mask int64) |
| Given template | `ChainInput.template`, residue by residue, same length; template residue types, unresolved ones included | template JSON (jfold: residue by residue, same length); unresolved = gap; coordinates centred; UNK pseudo-beta and frame from N; ignored on chains ≤ 4 residues | not taken |
| Extra keys | `constraint_feature` (zeros, no-op), `atom_rep_atom_idx` | `msa_mask`, `deletion_matrix`, `*_num_alignments` | Mol-1's + structural-token keys (§3.2) |
| Output type | numpy | torch tensors | torch tensors |

### Model-specific additions

- **OpenDDE structural tokens** (§3.2): bb/sc split by atom name, Gly unsplit, roles,
  twins, prev/next parents, and per-token centre atoms CA/CB. The sc frame is
  inherited from bb.

**How the views are split**

- `views/protenix.py` holds everything the three share: the atom layout (`layout`),
  the residue-level features (`residue_features`), the pose (`random_pose`, `pose`),
  both MSA conventions (`protenix_msa`, and `mol1_msa` for Mol-1 and OpenDDE), the
  empty templates, the template pair features (`template_pair_features`) and both template
  conventions (`protenix_templates`, `mol1_templates`).
- Each view picks its residue table (`data/{protenix,mol1,opendde}.npz`), the pose
  order, the MSA convention, the template slots and its keys. `views/opendde.py` adds
  the structural tokens.

---

## 5. Fixtures, determinism, gaps

### Fixtures and scripts

Each script writes one file per system from `tests/featurize/data/systems.json`, and
`dump_protenix.py` and `dump_mol1.py` also for the templated systems of
`template_systems.json` (`tools/fixtures/make_templates.py`). Templates without their
chain's length (`dimer_template`, `dimer_template_covering`, `monomer_template_gap`) get no
file: Protenix refuses them, and they have no residue-by-residue mapping for Mol-1. Each
script takes system names to dump only those.

| script | outputs | arrays per file |
|---|---|---|
| `tools/fixtures/dump_protenix.py` | `tests/featurize/data/protenix_*.npz` | 50 |
| `tools/fixtures/dump_mol1.py` | `tests/featurize/data/mol1_*.npz` | 52, 57 with an MSA |
| `tools/fixtures/dump_opendde.py` | `tests/featurize/data/opendde_*.npz` | 74 / 74 / 76 |

Run them from `~/jfold` with `LAYERNORM_TYPE=torch uv run python tools/fixtures/dump_<model>.py`.
They need no GPU.

**What the files contain**

- Every upstream array under its own name.
- Protenix's nested `constraint_feature` is flattened to `constraint_feature__<name>`.
- `aux_*` extras, which are not model inputs:
  - `aux_ref_pos_conformer`: the unaugmented conformer per atom;
  - `aux_atom_name`, `aux_res_name`, `aux_element`: unicode arrays, no pickle needed;
  - `aux_seed`;
  - Mol-1/OpenDDE: `aux_relp`;
  - OpenDDE: `aux_structural_relp`.

**Determinism**

- Each system is featurized right after seeding: `np.random.seed(0)`,
  `random.seed(0)`, and `torch.manual_seed(0)` for Mol-1/OpenDDE.
- Repeated runs give byte-identical files (md5 checked).
- Nothing else is random for protein-only inputs. With a different seed, or none,
  only `ref_pos` changes.

**MSA fixture result** (`monomer_msa`)

- Protenix gives 5 rows: query, hit2, hit3, hit4, hit5 (`hit1` is a duplicate of
  the query).
- Mol-1 and OpenDDE give 6 rows: the query twice, then hits 2-5.
- `hit2` (`...AKqqQRQ...`) gives deletion 2 at column 8; `hit3` (`...KShhHF...`)
  gives deletion 2 at column 17.
- `deletion_value = 0.37433`; `deletion_mean[8] = deletion_mean[17] = 0.4`.
- `X` in hit5 maps to 20. Leading and trailing `-` map to 31.

### Gaps and caveats

- **Protenix's other path.** `protenix predict`'s own pipeline
  (`protenix/data/infer_data_pipeline.py`, JSON input, optional ESM) was not dumped.
  The view follows `featurize`; the two may differ, for example in templates.
- **Mol-1's template mapping is jfold's.** Upstream's template JSON carries its own
  query-to-template mapping; jfold's `Template` has none, so the view maps residue by
  residue, as Protenix does. Templates from search hits (`.a3m`/`.hhr`) are not reproduced.
- **The arginine NH1/NH2 swap** in Mol-1's centring changes only the order of a float32
  sum; the fixtures' arginines trigger it, but no fixture's centre depends on it.
- **Conformer extraction needs rdkit.** The conformer coordinates exist only inside
  the three RDKit pickles, so the extract scripts need rdkit. They cannot be
  re-derived from the CCD CIF, because conformer 2 is an RDKit-generated
  conformer, not the CCD ideal one.
- **Empty-template channels.** The empty-template aatype channels (Protenix UNK;
  Mol-1/OpenDDE gap + 3×ALA) reach the template embedder whenever `n_blocks > 0`, so
  the views reproduce them exactly.
- **Model-side MSA sampling.** Every trunk pass, the MSA modules draw rows inside the
  model; the views return all rows. Protenix and Mol-1
  (`sample_msa_feature_dict_random_without_replacement`, `strategy="random"`): a random
  number of rows, `k ~ U{min(min_size, S)..S}` with the test `min_size` 1, uniformly
  (the query like any other), at most the test `sample_cutoff` (16384); so any MSA of
  more than one row is cut afresh on every pass. jfold computes each pass on at most 1024
  of them (`k` rows when `k ≤ 1024`, else 1024), bounding the cost. OpenDDE
  (`subsample_msa_feature_dict_valid_first`): `msa_depth` (1280) rows uniformly, rows
  with a residue first. jfold's `MSAModule` draws the same way (`max_rows`, `min_rows`,
  `keep_query=False`; see `README.md`).
- **Not covered by fixtures.** Heteromers with real paired MSAs, chains of 4 residues
  or fewer with MSAs, and the int8 deletion clip follow from the code only. The
  heteromer + unpaired-MSA layout (§1.9) was checked for Protenix and Mol-1 outside
  the fixtures: both give the same 5×31 MSA.
