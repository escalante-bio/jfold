# ProMERA featurization (proteins)

This spec covers how ProMERA's upstream pipeline turns a protein-only input into the feature
dict that `jfold.models.promera.features` consumes. That pipeline is tinyprot's
`AF3Featurizer`, then `promera.inference.utils.finalize_feats`, then
`promera.data.utils.collate`. It is the reference for `jfold.featurize.views.promera`.
Scope: standard amino acids (plus `X`), one or more chains, unpaired MSAs, no ligands,
nucleic acids, modifications or templates.

Every statement below was checked by running the upstream code: tinyprot 0.1.0 and promera
0.1.0 in jfold's environment, with scipy 1.18.1 and numpy 2.4.1. File references are relative
to `~/jfold/.venv/lib/python3.12/site-packages/`.

Parity fixtures: `tests/featurize/data/promera_{monomer,dimer,monomer_msa}.npz`, written by
`tools/fixtures/dump_promera.py`. Each is the collated batch without its batch axis, which is exactly what
`features()` takes. It also has `aux_*` keys that record the conformer randomness (see
[Randomness](#11-randomness)).

## 1. Call chain

ProMERA's inference driver is `promera/inference/cofolding.py:Cofolding.__getitem__`. The
design driver, `promera/inference/design.py`, uses the same steps.

1. `tinyprot/structure.py:Structure.from_schema(schema)`. The schema maps a chain id to
   `{"type": "protein", "sequence": str}`.
   - For each chain this calls `tinyprot/chain.py:Chain.from_sequence`, which maps each
     letter to a residue name with `tinyprot/ccd.py:_letter_to_std_rname`.
   - It builds the per-residue CCD atom-name grid `chain.aname` using `_get_attr_grid`
     with `leaving=False`.
   - It sets `chain.ridx = arange(L) + 1`.
   - It sets `chain.entity_id = schema[key].get("entity_id", i + 1)`, `chain.asym_id = key`
     and `chain.seq` (one-letter code, `X` for UNK).
2. `tinyprot/msa.py:load_msa_from_dir(cfg.msa_dir, struct.chains)`. Looks up one MSA per
   chain (see [MSA](#10-msa)), falling back to `make_dummy_msa(chain.seq)`.
   - `use_msa: false` in the schema forces the dummy.
   - `assert_msa: true` (the cofolding default) raises if a protein chain has no a3m.
3. `tinyprot/msa.py:construct_paired_msa(msas)` returns `(pairing, is_paired)`. The
   default strategy is `"protenix"`, with `max_pairs=8192` and `max_total=16384`.
4. `tinyprot/feature.py:AF3Featurizer(struct, msas, pairing).featurize(compute_frames=True)`.
   - `tinyprot/tokenize.py:AF3Tokenizer.tokenize` sets `chain.tidx`, `chain.aidx`,
     `chain.is_token` and `struct.tokens`.
   - `_add_atom_features` adds the atom keys, `token_to_rep_atom` and the frames.
   - `_add_token_features` adds the token keys.
   - `_add_connection_features` does nothing here: there are no `connections`.
   - `_add_msa_features` adds the MSA, `profile` and `deletion_mean`.
5. `promera/inference/utils.py:finalize_feats(feats, struct, name, seed_idx)` adds the pad
   masks, `is_epitope`, the dense `token_bonds` and `atom_is_*`.
6. `promera/data/utils.py:collate([feats])` pads atoms to a multiple of 32, adds the batch
   axis and casts dtypes.
   - `promera/__main__.py` uses `collate(token_multiple=cfg.pad_tokens_to_multiple or 1)`.
   - The default is 1, so tokens are not padded.
7. The model reads the batch. jfold's adapter is `jfold/models/promera.py:features(...)`,
   applied to `{k: v[0].numpy()}`; see `~/jfold/tests/test_checkpoint_promera.py:_batch`.

The view returns the keys the adapter reads plus `frames_mask` and `msa_paired`, unbatched,
in upstream's (post-`collate`) dtypes.

## 2. Token vocabulary (`tinyprot/feature.py:_tokens`, 34 entries)

```
 0 <pad>   1 <gap>   2 ALA  3 ARG  4 ASN  5 ASP  6 CYS  7 GLN  8 GLU  9 GLY 10 HIS
11 ILE    12 LEU    13 LYS 14 MET 15 PHE 16 PRO 17 SER 18 THR 19 TRP 20 TYR 21 VAL
22 UNK    23 A      24 G   25 C   26 U   27 N   28 DA  29 DG  30 DC  31 DT  32 DN  33 <other>
```

- The amino acids are in alphabetical order of their 3-letter codes, in slots 2..21. The
  one-letter order is `ARNDCQEGHILKMFPSTWYV`, so slots `[2:22]` are the AF2 order.
- Unknown residues: an input `X` becomes `UNK`, which is 22.
- Gaps: in the MSA, `-` is 1 (`<gap>`).
- Rows of the paired layout where a chain is absent hold 0 (`<pad>`), not 1. See §10.
- In a sequence, only the 20 standard letters and `X` are accepted.
  - `_letter_to_std_rname` raises `KeyError` for anything else, including `B`, `Z`, `U`, `O`
    and lowercase letters.
  - `M` maps to `MET`. The inverted `_std_aa` table would give `MSE`, so `MET` is special-cased.

## 3. Tokenization (proteins)

There is one token per residue. Tokens follow schema chain order, then residue order. Every
residue from a sequence is "standard" (`_is_std(rname, "polymer:polypeptide(L)")` is True
for the 20 amino acids, `UNK` and `MSE`), so no residue is split into per-atom tokens.

| key | shape | dtype | rule |
|---|---|---|---|
| `restype` | `[T]` | int64 | `_tokens.index(rname)`. `MSE→MET`; a non-standard name would give 33. |
| `is_std` | `[T]` | bool | True for every protein residue, **including UNK**. |
| `token_index` | `[T]` | int64 | `0..T-1`, global across chains. |
| `residue_index` | `[T]` | int64 | `chain.ridx`, **1-based** and restarting at 1 for each chain. |
| `token_pad_mask` | `[T]` | int64 | `ones_like(restype)` from `finalize_feats`. There is no token padding. |
| `is_protein` | `[T]` | int64 | 1 |
| `is_rna`, `is_dna`, `is_ligand` | `[T]` | int64 | 0 |
| `residue_name`, `asym_id_` | `[T]` | `<U3`, `<U1` | Strings. The model does not read them. |

## 4. Atoms

### Atom set and order

Each residue's atoms are the heavy atoms of its CCD entry, in CCD `_chem_comp_atom` order,
minus atoms flagged `pdbx_leaving_atom_flag = Y`. The source is
`tinyprot/ccd.py:_parse_cif_block`, called with `leaving=False` because the chain type
starts with `polymer`.
- Hydrogens and deuterium (`type_symbol` H or D) are dropped when the entry is parsed.
- OXT is a leaving atom, so it is never present, not even on the C-terminal residue.
- Atoms run residue by residue and chain by chain. `atom_to_token` is therefore sorted and
  each token's atoms are contiguous.

Per residue: token id, atom count, representative atom (with its slot), and atoms in order.
Each atom shows `(hydrogen count[, formal charge])`.

| res | id | n | rep (slot) | atoms |
|---|---|---|---|---|
| ALA | 2 | 5 | CB (4) | N(2H) CA(1H) C(0H) O(0H) CB(3H) |
| ARG | 3 | 11 | CB (4) | N(2H) CA(1H) C(0H) O(0H) CB(2H) CG(2H) CD(2H) NE(1H) CZ(0H) NH1(2H) NH2(2H,+1) |
| ASN | 4 | 8 | CB (4) | N(2H) CA(1H) C(0H) O(0H) CB(2H) CG(0H) OD1(0H) ND2(2H) |
| ASP | 5 | 8 | CB (4) | N(2H) CA(1H) C(0H) O(0H) CB(2H) CG(0H) OD1(0H) OD2(1H) |
| CYS | 6 | 6 | CB (4) | N(2H) CA(1H) C(0H) O(0H) CB(2H) SG(1H) |
| GLN | 7 | 9 | CB (4) | N(2H) CA(1H) C(0H) O(0H) CB(2H) CG(2H) CD(0H) OE1(0H) NE2(2H) |
| GLU | 8 | 9 | CB (4) | N(2H) CA(1H) C(0H) O(0H) CB(2H) CG(2H) CD(0H) OE1(0H) OE2(1H) |
| GLY | 9 | 4 | CA (1) | N(2H) CA(2H) C(0H) O(0H) |
| HIS | 10 | 10 | CB (4) | N(2H) CA(1H) C(0H) O(0H) CB(2H) CG(0H) ND1(1H,+1) CD2(1H) CE1(1H) NE2(1H) |
| ILE | 11 | 8 | CB (4) | N(2H) CA(1H) C(0H) O(0H) CB(1H) CG1(2H) CG2(3H) CD1(3H) |
| LEU | 12 | 8 | CB (4) | N(2H) CA(1H) C(0H) O(0H) CB(2H) CG(1H) CD1(3H) CD2(3H) |
| LYS | 13 | 9 | CB (4) | N(2H) CA(1H) C(0H) O(0H) CB(2H) CG(2H) CD(2H) CE(2H) NZ(3H,+1) |
| MET | 14 | 8 | CB (4) | N(2H) CA(1H) C(0H) O(0H) CB(2H) CG(2H) SD(0H) CE(3H) |
| PHE | 15 | 11 | CB (4) | N(2H) CA(1H) C(0H) O(0H) CB(2H) CG(0H) CD1(1H) CD2(1H) CE1(1H) CE2(1H) CZ(1H) |
| PRO | 16 | 7 | CB (4) | N(1H) CA(1H) C(0H) O(0H) CB(2H) CG(2H) CD(2H) |
| SER | 17 | 6 | CB (4) | N(2H) CA(1H) C(0H) O(0H) CB(2H) OG(1H) |
| THR | 18 | 7 | CB (4) | N(2H) CA(1H) C(0H) O(0H) CB(1H) OG1(1H) CG2(3H) |
| TRP | 19 | 14 | CB (4) | N(2H) CA(1H) C(0H) O(0H) CB(2H) CG(0H) CD1(1H) CD2(0H) NE1(1H) CE2(0H) CE3(1H) CZ2(1H) CZ3(1H) CH2(1H) |
| TYR | 20 | 12 | CB (4) | N(2H) CA(1H) C(0H) O(0H) CB(2H) CG(0H) CD1(1H) CD2(1H) CE1(1H) CE2(1H) CZ(0H) OH(1H) |
| VAL | 21 | 7 | CB (4) | N(2H) CA(1H) C(0H) O(0H) CB(1H) CG1(3H) CG2(3H) |
| UNK | 22 | 6 | **CA (1)** | N(2H) CA(1H) C(0H) O(0H) CB(2H) CG(3H) |

UNK has six atoms (it carries `CB` and `CG`), and its representative atom is **CA**, not CB.

### Atom keys

`A` is the number of atoms after padding; `N_real = atom_pad_mask.sum()`. Padded rows
(`N_real <= a < A`) are all zero (False for bool). That includes `atom_to_token = 0` and
`ref_space_uid = 0`, so padded atoms "point at" token 0 and must be excluded with the mask.

| key | shape | dtype | rule |
|---|---|---|---|
| `ref_pos` | `[A,3]` | float32 | Reference conformer; see below. |
| `ref_element` | `[A]` | int64 | Atomic number (C 6, N 7, O 8, S 16). The adapter takes `one_hot(ref_element, 128)`. |
| `ref_charge` | `[A]` | int64 | CCD `_chem_comp_atom.charge` (formal charge); 0 if absent or non-numeric. Only ARG NH2, HIS ND1 and LYS NZ are +1. The adapter casts it to float. |
| `ref_hydrogens` | `[A]` | int64 | The number of H/D atoms bonded to this atom in the CCD entry, **counting leaving hydrogens**. So the backbone N is 2 (H and H2) in every residue, PRO N is 1, and there is no adjustment for peptide bonds. These are the free-amino-acid CCD protonation states: ASP OD2 and GLU OE2 carry an H, HIS is doubly protonated. The adapter takes `one_hot(ref_hydrogens, 8)`. |
| `ref_atom_name_chars` | `[A,4]` | int64 | `[ord(c) - 32 for c in name] + [0] * (4 - len(name))`. For example CA is `[35,33,0,0]` and OD1 is `[47,36,17,0]`. The adapter takes `one_hot(·, 64)` per character and multiplies it by `atom_is_std`. |
| `ref_atom_name` | `[N_real]` | `<U3` | The names themselves, unpadded; not consumed. |
| `ref_space_uid` | `[A]` | int64 | A per-residue counter, global across chains and starting at 0, so it equals `atom_to_token` on real protein atoms. |
| `atom_to_token` | `[A]` | int64 | `chain.tidx` of the atom, the token index of its residue. |
| `atom_pad_mask` | `[A]` | int64 | `ones_like(ref_element)` from `finalize_feats`; collate pads it with 0. |
| `atom_is_std` | `[A]` | bool | `is_std[atom_to_token]`. All True for real protein atoms (including UNK), False on padding. |
| `atom_is_protein` | `[A]` | int64 | Gathered from the token flags through `atom_to_token`. |
| `atom_is_rna`, `atom_is_dna`, `atom_is_ligand` | `[A]` | int64 | Gathered from the token flags through `atom_to_token`. |
| `atom_coords` | `[A,3]` | float32 | Zeros; the schema has no coordinates. |
| `atom_resolved_mask` | `[A]` | bool | False; the schema has no coordinates. |

### Token/atom maps

| key | shape | dtype | rule |
|---|---|---|---|
| `token_to_rep_atom` | `[T]` | int64 | The global atom index of the residue's representative atom (`aidx[is_token]`): **CA for GLY and UNK, CB otherwise.** That is the residue's first atom index plus the slot in the table above. |
| `frames_idx` | `[T,3]` | int64 | Global atom indices of `N, CA, C` (`compute_frames=True`). Not read by the adapter. |
| `frames_mask` | `[T]` | bool | True for every protein residue, including UNK. Not read by the adapter; the view returns it. |

The jfold adapter derives `atom_position` (an atom's slot within its token) itself, as
`atom_index - first_atom[atom_to_token]`.

### Reference conformer: `ref_pos`

The source is `tinyprot/ccd.py:_get_conformer(rname, leaving=False)`, with its defaults
`center=True`, `jitter=1.0`, `randrot=True`. It is called once per residue, in token order:

```python
ref_pos = ideal_xyz[non_leaving_heavy_atoms]          # float64, CCD order
ref_pos = ref_pos - ref_pos.mean(0)                    # centre on this residue's atoms (no OXT)
ref_pos = ref_pos + np.random.randn(3) * 1.0           # random translation, N(0, 1 Å^2) per axis
ref_pos = ref_pos @ R.random().as_matrix().T           # scipy uniform random rotation
```

- `ideal_xyz` is `_chem_comp_atom.pdbx_model_Cartn_{x,y,z}_ideal`. If those fail to parse,
  tinyprot falls back to `model_Cartn_{x,y,z}`; this never happens for the 21 residues here.
- The translation is applied **before** the rotation, so the residue centroid ends up at
  `R @ t`. That is still isotropic N(0, I).
- Each residue gets independent draws. There are no other random steps.
- The CCD is tinyprot's prebuilt LMDB (`$TINYPROT_CACHE` or `~/.cache/tinyprot/ccd.lmdb`,
  downloaded from the Hugging Face dataset `bjing-mit/tinyprot`).
- Biotite's bundled CCD (`biotite.structure.info.residue`, biotite 1.7.1) gives the same
  results for these 21 residues: the same heavy-atom order without OXT, the same H counts,
  the same formal charges, and ideal coordinates equal to within 2.3e-7 Å.
- The ALA canonical (centred, no randomness) conformer, as a fingerprint:
  ```
  N  [-0.835   0.5678  1.1302]
  CA [ 0.388   0.4928  0.3222]
  C  [ 0.037   0.0918 -1.0858]
  O  [-0.925  -0.6072 -1.2928]
  CB [ 1.335  -0.5452  0.9262]
  ```

## 5. Chain, entity and position ids

`asym_ids`, `entity_ids` and `sym_ids` are built in `AF3Featurizer.featurize` in
first-appearance order over `struct.chains`, which follows schema order.

| key | shape | dtype | rule |
|---|---|---|---|
| `asym_id` | `[T]` | int64 | Index of the chain id in schema order, 0-based. |
| `entity_id` | `[T]` | int64 | `entity_ids.index(chain.entity_id)`, 0-based. **By default every chain is its own entity, even when sequences are identical**, because `from_schema` sets `entity_id = schema.get("entity_id", i + 1)`. Equal entities only arise from an explicit `"entity_id"` in the schema. |
| `sym_id` | `[T]` | int64 | The chain's index among the chains of its entity. It is 0 everywhere unless entity ids are given explicitly. Verified: two `GG` chains with `entity_id: 1` give `entity_id = [0,0,0,0]` and `sym_id = [0,0,1,1]`. |
| `entity_id_` | `[T]` | int64 | The raw `chain.entity_id` (1-based); not consumed. |
| `residue_index` | `[T]` | int64 | 1-based within the chain (§3). |
| `token_index` | `[T]` | int64 | 0-based, global (§3). |

The view never sets `entity_id`, so it always gives upstream's default: `entity_id =
asym_id` and `sym_id = 0`, and a homodimer is two distinct entities.

## 6. Bonds: `token_bonds`

`[T,T]` int64, **all zeros for proteins**.
- `_add_atom_features` adds intra-residue bonds only for non-standard residues.
- `_add_connection_features` keeps only connections that involve a ligand.
- So the `(K, 2)` list is empty, and `finalize_feats` turns it into a symmetric dense
  matrix with `bond_mat[i, j] = bond_mat[j, i] = 1`.
- There are no polymer backbone bonds. The adapter uses `token_bonds[..., None]` as float.

## 7. Epitope: `is_epitope`

`[T]` int64, set to zeros by `finalize_feats`. Only the design driver changes it
(`promera/inference/design.py`, after `finalize_feats`):
- `feats["is_epitope"][idx] = 1` for the epitope residues, given as a chain plus author
  residue numbers.
- For VHH binders with `paratope_from_cdrs`, the same for the binder's `X` (CDR) positions.

The adapter casts it to float. It is optional in `features()` and defaults to zeros.

## 8. Padding and dtypes (`collate`)

- Every numeric ndarray value gets a leading batch axis, then a cast: float to float32, int
  to int64, bool unchanged. Strings and objects stay as Python lists.
- Atom keys pad dimension 0 with zeros up to the next multiple of 32. If the count is
  already a multiple of 32, nothing is added. The atom keys are `ref_pos`, `ref_element`,
  `ref_charge`, `ref_space_uid`, `ref_hydrogens`, `atom_to_token`, `ref_atom_name_chars`,
  `atom_pad_mask`, `atom_coords`, `atom_resolved_mask`, `atom_is_*` and `atom_is_std`.
  - monomer: 186 atoms, padded to 192.
  - dimer: 253 atoms, padded to 256.
- Token keys and MSA keys are padded only to the batch maximum, which is a no-op for a
  batch of one, unless `token_multiple > 1`.
- The fixtures drop the batch axis; `features()` takes unbatched arrays.

## 9. Not consumed but present

`token_pos` `[T,3]` float32 (zeros), `token_pos_mask` `[T]` bool (False), `msa_chars`
`[S,T]` `<U1`, plus `name`, `seed_idx` and `struct` (Python objects). The fixtures omit the
Python objects.

## 10. MSA

### Supplying a precomputed a3m

ProMERA has no per-chain MSA path in the schema. It uses `cfg.msa_dir`, a directory or a list
of directories, and `load_msa_from_dir` looks each chain up by content hash:

```python
key = sha256((chain.seq + "\n").encode()).hexdigest()   # tinyprot.msa.hash_sequence
path = f"{dir}/{key}.a3m.gz" or f"{dir}/{key}.a3m"         # .a3m.gz checked first
MSA.from_a3m(path, std_letter_type=chain.type)
```

`chain.seq` is the one-letter sequence with `X` for UNK. If `msa_dir` is unset (`null` in
`cofolding.yaml`), it defaults to `~/.cache/tinyprot/msa`. `tools/fixtures/dump_promera.py` writes the
fixture a3m into a private temporary directory under that hash name, so the user cache is
never consulted.

A chain with no hit gets `make_dummy_msa(seq)`: one row (the query), with deletions 0 and
taxonomy -1.

The view takes each chain's a3m from `Chain.msa` directly, with no hash lookup; a chain
without one gets the dummy MSA.

### Parsing an a3m (`MSA.from_a3m`)

- Lines are stripped. Empty lines and lines starting with `#` are skipped.
- A `>` line sets the taxonomy for the following sequence. Taxonomy comes from the
  taxonomy LMDB if the header's first token starts with `>UniRef100`, otherwise -1.
- Deduplication: `key = line.replace("-", "").upper()`. A row is skipped if its key was
  already seen.
  - The query is always kept.
  - **A hit identical to the query is dropped.** In the fixture, `hit1` is dropped.
  - Because insertions are upper-cased into the key, a row that differs from the query
    only by case is also dropped.
- For each character `c`:
  - If `c != "-"` and `c.islower()`, it is an insertion: `count += 1` and nothing is
    emitted.
  - Otherwise, emit `_letter_to_std_letter(c, "polymer:polypeptide(L)")`, then
    `dels.append(count)` and `count = 0`.
  - `_letter_to_std_letter` keeps `-` and the letters of `_std_aa` (`ACDEFGHIKLMNPQRSTVWXY`)
    and maps any other character to `X`.
- So a deletion count is attached to the **next** aligned column. Insertions before the
  first column attach to column 0. **Insertions after the last column are dropped.**
- `seqs` is `[S_c, L_aligned]` (characters), `dels` is `[S_c, L_aligned]` (int) and `taxs` is
  `[S_c]`.
- The query row of the MSA is the a3m's first sequence. It is **not** checked against
  the chain sequence.
- Columns are selected as `[:, ridx - 1]`. A longer alignment is silently truncated; a
  shorter one raises `IndexError`.

The fixture `chain_A.a3m` parses to 5 rows:

```
MKTAYIAKQRQISFVKSHFSRQ   query
MKSAYIAKQRQLSFV--HFSRQ   hit2: "qq" -> deletion 2 at column 8
MRTAFIAKERQISFVKSHFSKQ   hit3: "hh" -> deletion 2 at column 17
--TAYIAKQRQISFVKSHF---   hit4
MKTAYIXKQRQISFVKSHFSRE   hit5
```

### Row layout (`construct_paired_msa`, unpaired case)

Row 0 is the query of every chain, with `is_paired = 1` for every chain.

**Pairing tier.** Any hit whose taxonomy is not -1, and is shared by at least two `(chain,
row)` entries, goes into the "paired" tier with `msa_paired = 1`.
- This happens **even within a single chain**. A monomer MSA with `>UniRef100_…` headers
  is reordered: hits that share a species with another hit move first, ordered by log rank.
- Checked on a cached ColabFold MSA: 7706 of 12167 rows were moved and flagged.
- Without taxonomy, i.e. with no header starting `>UniRef100`, the tier is empty and the
  original row order is kept, minus duplicates.
- The view reproduces the tier (`pair_rows`). Since tinyprot reads taxonomy from its
  LMDB, `features` takes a `taxonomy` lookup (UniRef100 id → taxonomy id), and raises on a
  UniRef100 header when none is given.

**Unpaired block.** Each remaining row `k = 1, 2, …` takes the k-th remaining row of every
chain, side by side and *not* block-diagonal. A chain that has run out of rows gets `-1`.

**Limits.**
- Total rows: `S = 1 + min(max_total - 1, max_c(S_c - 1)) = min(16384, max_c S_c)` in the
  unpaired case.
- The paired tier is capped at 8192 rows, including the query.
- There is no subsampling at featurization. The model subsamples rows on every trunk pass
  (`PromeraModel.subsample_msa`: `min(S, 1024)` real rows uniformly, `msas_per_trunk_iter`,
  the query like any other). That is model-side and random; jfold's `MSAModule` draws the
  same way (`max_rows=1024, keep_query=False`).

### MSA keys (`AF3Featurizer._add_msa_features`)

`S` is the number of rows. `seq_idx[r, c]` is `pairing[r][chain c]`.

| key | shape | dtype | rule |
|---|---|---|---|
| `msa` | `[S,T]` | int64 | `_msa_letter_to_index(msa_c.seqs)[seq_idx, ridx - 1]` for each chain. The letter map is the 20 amino acids to 2..21, `X` to 22, `-` to 1. **Where `seq_idx == -1` the value stays 0 (`<pad>`).** |
| `msa_mask` | `[S,T]` | bool | **All True**, including the `-1` (`<pad>`) entries. |
| `has_deletion` | `[S,T]` | bool | `raw_deletions > 0`, computed before the transform. |
| `deletion_value` | `[S,T]` | float32 | `(2 / pi) * arctan(raw_deletions / 3)`, with `raw_deletions = msa_c.dels[seq_idx, ridx - 1]` and 0 where `seq_idx == -1`. |
| `msa_paired` | `[S,T]` | bool | `is_paired[r][chain]`: 1 on row 0, 0 on unpaired rows. Not read by the adapter; the view returns it. |
| `msa_chars` | `[S,T]` | `<U1` | Raw letters, `-` where `seq_idx == -1`. Not consumed. |
| `profile` | `[T,34]` | float32 | `np.eye(34)[msa_as_number[:, ridx - 1]].mean(0)` over **all rows of that chain's own MSA** (`msa_c.seqs`, after deduplication, query included, gaps counted in slot 1). It is independent of the pairing layout and of the `max_total` cut. With a dummy MSA it equals `one_hot(restype)`. |
| `deletion_mean` | `[T]` | float32 | `msa_c.dels.mean(0)[ridx - 1]`: the mean of **raw counts**, not arctan-transformed, over all rows of the chain's own MSA. For the fixture it is 0.4 at columns 8 and 17, 0 elsewhere. |

The adapter builds `msa_features = [one_hot(msa, 34), has_deletion, deletion_value]` (36
channels) and `token_features = [one_hot(restype, 34), profile, deletion_mean, is_epitope]`
(70 channels).

## 11. Randomness

**Random:** `ref_pos`, with one translation and one rotation per residue (§4). The draws
come from numpy's global legacy `RandomState`.
- `np.random.randn(3)`, then `scipy.spatial.transform.Rotation.random()`, which with no
  `rng` argument draws from the same global state.
- Per residue, in token order: the translation first, then the rotation.
- `np.random.seed(s)` immediately before `featurize()` makes the result reproducible, given
  the same numpy and scipy versions.
- `tools/fixtures/dump_promera.py` uses seed 0 and asserts that replaying the stream gives the same
  arrays. Two runs of the script gave identical arrays.

**Not random:** everything else. That covers MSA parsing and layout (`MSA.subsample`
and the `np.random.rand` in `promera.data.utils.mask_chain` are never called on this
path), tokenization, ids, masks and the profile.

**Fixtures and the view:**
- The fixtures store `aux_ref_pos_canonical` (centred ideal coordinates, padded like
  `ref_pos`), `aux_ref_translation[R,3]` and `aux_ref_rotation[R,3,3]`, indexed by
  `ref_space_uid`. Then `ref_pos = (aux_ref_pos_canonical + t) @ Rm.T` for each residue.
- The view's `key=None` gives the canonical conformers, which the parity test compares
  with `aux_ref_pos_canonical`. With a `key` it draws the same distributions, in the same
  order (translation, then rotation), from JAX. The test also applies the stored `t` and
  `Rm` to the view's canonical conformers (`pose`) and recovers upstream's `ref_pos`.

**Upstream itself is not seeded.** `Cofolding` sets no numpy seed, so production `ref_pos`
varies from run to run, and the released model was trained with this augmentation.

## 12. Dependencies

- **tinyprot**: pure numpy plus `scipy` (`Rotation`), `lmdb`, `pandas` (CIF block parsing),
  `rustworkx` (symmetry only, unused here), `requests`/`tqdm` (downloads). It needs two LMDBs:
  - the CCD;
  - the taxonomy database. `tinyprot.msa` **opens the taxonomy LMDB at import time** and
    raises if it is missing, even when no pairing will happen.

  `tinyprot.init.download_pretrained()` fetches both.
- **No RDKit, gemmi or biotite** in this path. `_CCDAtom` only mimics the RDKit atom API; the
  conformer is the CCD ideal geometry, not an RDKit embedding.
- **torch**: only `promera.data.utils.collate` (`torch.from_numpy`), plus the `promera`
  package import. Both `finalize_feats` and `collate` are trivial in numpy.
- **CCD content**: matched by biotite's bundled CCD for the 21 residues used (§4). The
  view's table (`src/jfold/featurize/data/promera.npz`) is extracted from tinyprot's own
  CCD by `tools/tables/extract_promera.py`, keeping the centred ideal coordinates in
  float64 so that posing them reproduces upstream's float64 arithmetic.

## 13. Shared vs specific

**Generic AF3-style** (shared with the other AF3-style pipelines):
- One token per standard residue; atoms contiguous per token; `atom_to_token`.
- `ref_space_uid` per residue, equal to the token index for proteins.
- Heavy atoms only, CCD order, leaving atoms (OXT) removed.
- Element as atomic number, one-hot to 128.
- Atom-name characters as `ord(c) - 32`, 4 slots, one-hot to 64.
- Formal charge from the CCD.
- Reference conformer randomly rotated and translated per residue.
- `asym_id`, `entity_id`, `sym_id`, `residue_index`, `token_index` with the AF3 meanings.
- MSA one-hot over the token vocabulary, `has_deletion`, and
  `deletion_value = 2/π·arctan(d/3)`.
- `profile` as the per-column token distribution over the MSA; `deletion_mean`.
- Query as MSA row 0; deletion counts attached to the following column.
- Representative atom CB (CA for GLY).
- Atoms padded to a multiple of 32.

**ProMERA/tinyprot-specific:**
1. The 34-token vocabulary: Boltz's 33-token list (`<pad>`, `-`, 20 amino acids, UNK,
   A G C U N, DA DG DC DT DN, in the same slots) plus a 34th `<other>` slot (33). The 20
   amino acids and UNK sit in the same slots 2..22 as Boltz's.
2. `ref_pos` comes from CCD **ideal** coordinates, centred on the residue's non-leaving
   heavy atoms. It is then translated by N(0, 1 Å) **before** a uniform rotation. There is
   no RDKit conformer, and the randomness comes from numpy's global RNG.
3. `ref_hydrogens`: a CCD bonded-H count that includes leaving hydrogens (N has 2), one-hot
   to 8. This feature is not in AF3's list.
4. UNK uses the CCD `UNK` entry (6 atoms including CB and CG), and its representative atom
   is **CA**. UNK counts as standard (`is_std = True`), so its atom aggregate is dropped and
   its name characters are kept.
5. `is_std` drives `mask_std_feats`. The atom-encoder aggregate is zeroed for standard
   residues (`keep_aggregate = ~is_std`), and the name-character channels are multiplied by
   `atom_is_std`.
6. `residue_index` is 1-based.
7. Every chain is its own entity unless `entity_id` is given, so `sym_id` is 0 everywhere
   by default.
8. `token_bonds` are ligand-only, so they are all zero for proteins.
9. `is_epitope` is an extra token channel: 0 at inference, set by the design driver.
10. Token feature layout `[restype(34), profile(34), deletion_mean(1), is_epitope(1)]` (70
    channels). MSA feature layout `[msa(34), has_deletion, deletion_value]` (36 channels).
    `deletion_mean` uses raw counts, not arctan-transformed.
11. MSA input conventions:
    - MSAs are found by the sha256 of `seq + "\n"` in `msa_dir`.
    - Deduplication drops hits identical to the query, using the upper-cased ungapped line.
    - Non-standard letters map to `X`, and trailing insertions are dropped.
12. MSA layout conventions:
    - Unpaired rows sit side by side across chains.
    - Absent-chain entries are **`<pad>` (0) with `msa_mask = True`**, not a gap with the
      mask False.
    - Taxonomy pairing (and reordering) applies even to a single chain whenever there are
      UniRef100 headers.
    - `profile` and `deletion_mean` are computed over each chain's full MSA, independent of
      the layout.
13. No MSA cropping at featurization, up to 16384 rows; the model subsamples 1024 rows on
    each trunk pass.
14. Pad masks are int64 `ones` and `msa_mask` is bool. Tokens are not padded (token
    multiple 1).

## 14. Fixture contents

| system | T | atoms (real/padded) | MSA rows | notes |
|---|---|---|---|---|
| `monomer` | 22 | 186/192 | 1 | dummy MSA (query only) |
| `dimer` | 31 | 253/256 | 1 | both chains dummy; asym/entity 0,1; sym 0 |
| `monomer_msa` | 22 | 186/192 | 5 | `chain_A.a3m`; hit1 deduplicated away |

Each npz holds every numeric key of the collated batch without the batch axis. It also holds
the string keys (`ref_atom_name`, `residue_name`, `asym_id_`, `msa_chars`, unpadded) and the
`aux_ref_*` keys. All of it loads with `allow_pickle=False`. All three files load through
`jfold.models.promera.features` (atom features `[A, 397]`, token features `[T, 70]`, MSA
features `[S, T, 36]`).

Not covered by these fixtures, though checked separately and described above:
- `<pad>` entries for a chain missing from a multi-chain MSA row;
- UNK residues;
- taxonomy-driven reordering.
