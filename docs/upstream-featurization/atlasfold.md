# AtlasFold featurization (proteins)

How AtlasFold's runners turn protein chains into the feature dict that
`jfold.models.atlasfold.features` consumes; the reference for `jfold.featurize.views.atlasfold`.
Upstream: `atlasfold` 1.0.4 (`SeonghwanSeo/atlasfold` at `6df9f458`), files relative to its
`src/`. Scope: standard amino acids (plus `X`), one sequence per chain: AtlasFold reads no
MSA, and its runners take no templates (AtlasFold-M's template module reads `template.*` keys
only through the Python API), so the view refuses both.

Parity fixtures: `tests/featurize/data/atlasfold_m_{monomer,dimer,homomer}.npz` (AtlasFold-M's
runner) and `atlasfold_monomer.npz` (AtlasFold's), written by `tools/fixtures/dump_atlasfold.py`:
each runner's batch without its batch axis. Nothing is random.

## Call chain

- AtlasFold-M: `atlasfold/runner_multimer.py:MultimerFoldingRunner._make_batch_features` on
  `common/protein.py:ProteinMultimer.get_empty(name, chains)`, which calls
  `common/featurize.py:featurize_complex(sequences, entity_ids, asym_ids, sym_ids)` and pads
  with `_pad_to_lengths(residue_length=bucket, lm_length=bucket + 32)`.
- AtlasFold: `atlasfold/runner.py:FoldingRunner._make_batch_features`, which calls
  `featurize(sequence)` (entity, chain and copy 1) and pads with
  `_pad_to_length(bucket)` (AtlasLM's tokens to `bucket + 2`).
- The bucket (`_get_length_bucket`): the first of `featurize.DEFAULT_BUCKETS` (32, 64, 128,
  192, 256, 384, ..., 1024, 1152, ..., 2048) that holds the residues, else the next multiple
  of 128.

## Keys (unbatched; residues `L`, AtlasLM tokens `S`)

| key | shape, dtype | rule |
|---|---|---|
| `entity_id` | `[L]` int64 | per chain: AtlasFold-M numbers sequences by first appearance, from 1 |
| `asym_id` | `[L]` int64 | chain index from 1 |
| `sym_id` | `[L]` int64 | copy of the chain's sequence, from 1 |
| `aatype` | `[L, 21]` float32 | one-hot over `ARNDCQEGHILKMFPSTWYVX` |
| `aatype_int` | `[L]` int64 | its index |
| `res_idx` | `[L]` int64 | 1-based within each chain |
| `seq_tok_idx` | `[L]` int64 | each residue's row in AtlasLM's tokens |
| `seq_mask` | `[L]` bool | real residues |
| `atom14_mask` | `[L, 14]` bool | the residue type's first `n` atom14 slots (`residue_atoms`: N CA C O CB ..., UNK as N CA C O CB) |
| `atom37_mask` | `[L, 37]` bool | the same atoms in AlphaFold's atom37 order (the model does not read it) |
| `pseudo_beta` | `[L]` int64 | atom14 slot of CB (4), CA (1) for glycine |
| `lm.input_ids` | `[S]` int64 | each chain as `<cls>` (0), its residues (ESM-3's vocabulary, `L` 4 ... `C` 23, `X` 24), `<eos>` (2), chains concatenated; padding `<pad>` (1) |
| `lm.pos_id` | `[S]` int64 | rotary positions: 0 at `<cls>`, then `res_idx`, then the chain length + 1 at `<eos>` |
| `lm.seq_id` | `[S]` int64 | `asym_id` of each chain's tokens (attention stays within a chain) |

Padding is zero everywhere except `lm.input_ids` (`<pad>`); padded residues' `seq_tok_idx`
are 0. The view's atom sites (`atom_sites`) list each residue's atoms in atom14 order: their
positions are the model's at `atom14_mask`.

## Model side

Every trunk pass masks a share of AtlasLM's residue tokens (`input_ids` in 4–31) with
`<mask>` (32), drawn afresh: 20% for AtlasFold-M, 15% for AtlasFold (the runners'
`mlm_prob`; `AtlasFoldTrunk.mlm_rate`). `jfold.models.atlasfold.features` puts `aatype`'s
probability vectors into AtlasLM's residue rows, so a soft sequence reaches the language
model.
