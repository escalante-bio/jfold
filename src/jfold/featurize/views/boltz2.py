"""Boltz-2's features, as `boltz` 2.2.1's inference featurization produces them for proteins
(see `docs/upstream-featurization/boltz2.md`): the unbatched dict
`jfold.models.boltz2.features` takes, plus the rest of upstream's dict.

Upstream's quirks are reproduced: entity-grouped chain order, Boltz's own a3m rules (one
line per sequence, deduplication), unpaired MSA rows zipped across chains, and the
deletion-slicing bug that leaves every deletion feature zero.

Reference conformers: one of each residue's 10 stored conformers, drawn from
`np.random.default_rng(42)` in token order, as upstream. With `key=None` they are left
unposed (upstream's `ref_pos` before its random per-residue motion); with a JAX `key`, each
residue but the last is centred, randomly rotated and translated by N(0, 1) Å, as
upstream does with torch's global RNG."""

import jax
import jax.numpy as jnp
import numpy as np

from jfold.featurize.alignment import local_alignments
from jfold.featurize.padding import pad_to, padded_length
from jfold.featurize.residues import ResidueTable
from jfold.featurize.sites import AtomSites
from jfold.featurize.sites import atom_sites as _atom_sites
from jfold.featurize.system import THREE_LETTER, Chain, System, Template

# `boltz.data.const.tokens`: <pad>, gap, the 20 amino acids by three-letter code, UNK, ...
NUM_TOKENS = 33
GAP, UNK = 1, 22
_AMINO_ACIDS = sorted(name for name in THREE_LETTER.values() if name != "UNK")
TOKEN = {k: 2 + _AMINO_ACIDS.index(v) for k, v in THREE_LETTER.items() if k != "X"}
TOKEN["X"] = UNK
MET = TOKEN["M"]
# `const.prot_letter_to_token`, for a3m rows: other characters are an error.
MSA_TOKEN = TOKEN | dict.fromkeys("XJBZOU", UNK) | {"-": GAP}
MAX_SEQS_PER_CHAIN = 8192  # `parse_a3m(max_seqs=...)` in `process_input`
MAX_MSA_ROWS = 16384  # `const.max_msa_seqs`
ATOM_WINDOW = 32
NUM_ELEMENTS = 128
METHOD_XRAY = 1  # `const.method_types_ids["x-ray diffraction"]`
CONFORMER_SEED = 42


def residues() -> ResidueTable:
    return ResidueTable.load("boltz2")


def chain_order(system: System) -> list[tuple[int, int, int]]:
    """(chain index, entity, copy) in upstream's asym order (`System.chain_order`); the
    chains of an entity must share their MSA."""
    order = system.chain_order()
    for entity in {e for _, e, _ in order}:
        members = [i for i, e, _ in order if e == entity]
        if len({system.chains[i].msa for i in members}) > 1:
            raise ValueError(f"chains {members}: one sequence, different MSAs")
    return order


def _parse_a3m(text: str) -> tuple[list[list[int]], list[list[tuple[int, int]]]]:
    """`boltz.data.parse.a3m._parse_a3m`: every non-header line is a whole sequence; a line
    whose gapless uppercase form was seen before is dropped; lowercase letters are
    insertions, counted onto the next column (a trailing run is dropped). Rows of token ids
    and each row's `(column, count)` deletions, at most `MAX_SEQS_PER_CHAIN` rows."""
    seen, rows, deletions = set(), [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ">")):
            continue
        key = line.replace("-", "").upper()
        if key in seen:
            continue
        seen.add(key)
        row, deletion, count = [], [], 0
        for c in line:
            if c != "-" and c.islower():
                count += 1
                continue
            if c not in MSA_TOKEN:
                raise ValueError(f"a3m: unsupported character {c!r}")
            if count:
                deletion.append((len(row), count))
                count = 0
            row.append(MSA_TOKEN[c])
        rows.append(row)
        deletions.append(deletion)
        if len(rows) >= MAX_SEQS_PER_CHAIN:
            break
    return rows, deletions


def _chain_msa(chain: Chain) -> tuple[list[list[int]], list[list[tuple[int, int]]]]:
    """A chain's MSA rows after upstream's query check (`construct_paired_msa`): the query
    alone when there is no MSA or row 0 does not match the sequence; row 0 overwritten
    by the sequence when the mismatches are all MET vs UNK, or all at UNK residues."""
    query = [TOKEN[c] for c in chain.sequence]
    dummy = [query], [[]]
    if chain.msa is None:
        return dummy
    rows, deletions = _parse_a3m(chain.msa)
    if not rows:
        raise ValueError(f"chain {chain.id}: empty a3m")
    for row in rows:
        if len(row) != len(rows[0]):
            raise ValueError(f"chain {chain.id}: a3m rows of {len(row)} and {len(rows[0])}")
    if len(rows[0]) != len(query):
        return dummy
    mismatches = [(q, m) for q, m in zip(query, rows[0], strict=True) if q != m]
    if mismatches:
        met_vs_unk = all(q == MET and m == UNK for q, m in mismatches)
        if not (met_vs_unk or all(q == UNK for q, _ in mismatches)):
            return dummy
        rows[0] = query
    return rows, deletions


def _deletions(deletions: list[list[tuple[int, int]]]) -> dict[tuple[int, int], int]:
    """`(row, column) -> count`, as `construct_paired_msa` collects them, slicing bug
    included: each row re-slices the previous row's slice (with offsets into the full
    list), so after the query (no insertions) nothing is left."""
    flat = [d for row in deletions for d in row]
    out, start, remaining = {}, 0, flat
    for seq_idx, row in enumerate(deletions):
        remaining = remaining[start : start + len(row)]
        start += len(row)
        for res_idx, count in remaining:
            out[(seq_idx, res_idx)] = count
    return out


def _msa(chains: list[Chain]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Token ids, deletion counts and paired flags `[S, T]`: row 0 every chain's query
    (paired), then the chains' remaining rows zipped in file order, gaps once a chain
    runs out."""
    per_chain = [_chain_msa(chain) for chain in chains]
    depth = min(1 + max(len(rows) - 1 for rows, _ in per_chain), MAX_MSA_ROWS)
    msa, deletion = [], []
    for chain, (rows, dels) in zip(chains, per_chain, strict=True):
        length = len(chain.sequence)
        block = np.full((depth, length), GAP, np.int64)
        counts = np.zeros((depth, length), np.int64)
        for i, row in enumerate(rows[:depth]):
            block[i] = row
        for (i, j), count in _deletions(dels).items():
            if i < depth:
                counts[i, j] = count
        msa.append(block)
        deletion.append(counts)
    paired = np.zeros((depth, sum(len(c.sequence) for c in chains)), np.float32)
    paired[0] = 1
    return np.concatenate(msa, axis=1), np.concatenate(deletion, axis=1), paired


def conformer_ids(num_residues: int, num_conformers: int = 10) -> np.ndarray:
    """Upstream's conformer choice: `default_rng(42).choice(conf_ids)` once per residue,
    in token order."""
    rng = np.random.default_rng(CONFORMER_SEED)
    return np.array([int(rng.choice(num_conformers)) for _ in range(num_residues)], np.int64)


def _quaternion_to_matrix(q: jax.Array) -> jax.Array:
    r, i, j, k = jnp.unstack(q, axis=-1)
    two_s = 2.0 / (q * q).sum(-1)
    return jnp.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def pose(ref_pos: np.ndarray, residue: np.ndarray, key: jax.Array) -> jax.Array:
    """Upstream's per-residue `center_random_augmentation`: each residue's atoms centred,
    rotated by a uniform random rotation (`x @ R`, `R` from a normal quaternion with
    nonnegative real part) and translated by N(0, 1) Å, every residue but the last (the
    `range(max(ref_space_uid))` off-by-one). `ref_pos` `[A, 3]` real atoms only;
    `residue` `[A]` their residue (`ref_space_uid`)."""
    n = int(residue.max()) + 1
    key_rotation, key_translation = jax.random.split(key)
    q = jax.random.normal(key_rotation, (n, 4), jnp.float32)
    q = q / jnp.copysign(jnp.sqrt((q * q).sum(-1)), q[:, 0])[:, None]
    rotation = _quaternion_to_matrix(q)
    translation = jax.random.normal(key_translation, (n, 3), jnp.float32)
    x = jnp.array(ref_pos, jnp.float32)
    count = jax.ops.segment_sum(jnp.ones(len(residue), jnp.float32), residue, n)
    mean = jax.ops.segment_sum(x, residue, n) / count[:, None]
    posed = ((x - mean[residue])[:, :, None] * rotation[residue]).sum(1) + translation[residue]
    return jnp.where((residue < n - 1)[:, None], posed, x)


def compute_frame(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A residue's frame, as Boltz-2's tokenizer computes it (float32): `e1` along CA->C,
    `e2` the part of CA->N orthogonal to it, `e3 = e1 x e2`, as columns; origin CA."""
    v1, v2 = c - ca, n - ca
    e1 = v1 / (np.linalg.norm(v1) + 1e-10)
    u2 = v2 - e1 * np.dot(e1.T, v2)
    e2 = u2 / (np.linalg.norm(u2) + 1e-10)
    e3 = np.cross(e1, e2)
    return np.column_stack([e1, e2, e3]), ca


def _template_tokens(template: Template) -> list[dict]:
    """Each template residue's token, as Boltz-2's tokenizer builds it from the parsed
    template: residue type, frame (identity without N, CA and C), CA and the distogram atom
    (CB; CA for glycine), each with its mask; missing atoms at the origin."""
    tokens = []
    for letter, atoms in zip(template.sequence, template.atoms, strict=True):
        origin = np.zeros(3, np.float32)
        disto = "CA" if letter == "G" else "CB"
        frame = all(a in atoms for a in ("N", "CA", "C"))
        rot, t = (
            compute_frame(atoms["N"], atoms["CA"], atoms["C"])
            if frame
            else (np.eye(3), np.zeros(3))
        )
        tokens.append(
            {
                "res_type": TOKEN[letter],
                "frame_rot": rot,
                "frame_t": t,
                "frame_mask": frame,
                "ca": atoms.get("CA", origin),
                "cb": atoms.get(disto, origin),
                "cb_mask": disto in atoms,
            }
        )
    return tokens


def template_features(system: System, num_tokens: int) -> dict[str, np.ndarray]:
    """The `template_*` features: one template row holding every chain's `template`, as
    upstream builds them from a template file covering those chains, each matched to its
    chain (`chain_id` and `template_id` in the input YAML). Without templates, upstream's
    placeholder row.

    Each template is aligned to its chain (`local_alignments`), and for every optimal
    alignment *all* the template's residues are placed with that alignment's offset, not
    only the aligned ones (later alignments overwrite earlier ones). `visibility_ids` is 0
    on chains that received template residues and `-1 - asym_id` on the others."""
    order = chain_order(system)
    zeros = lambda *shape: np.zeros(shape, np.float32)
    restype = np.zeros((1, num_tokens), np.int64)
    out = {
        "template_frame_rot": zeros(1, num_tokens, 3, 3),
        "template_frame_t": zeros(1, num_tokens, 3),
        "template_cb": zeros(1, num_tokens, 3),
        "template_ca": zeros(1, num_tokens, 3),
        "template_mask_cb": zeros(1, num_tokens),
        "template_mask_frame": zeros(1, num_tokens),
        "template_mask": zeros(1, num_tokens),
        "query_to_template": np.zeros((1, num_tokens), np.int64),
        "visibility_ids": zeros(1, num_tokens),
    }
    if all(c.template is None for c in system.chains):
        return {"template_restype": np.eye(NUM_TOKENS, dtype=np.int64)[restype]} | out

    # Each chain's asym id and first token, in upstream's chain order.
    asym_of, start_of, start = {}, {}, 0
    for asym, (i, _, _) in enumerate(order):
        asym_of[i], start_of[i] = asym, start
        start += len(system.chains[i].sequence)
    templated = set()
    for i, chain in enumerate(system.chains):  # the YAML's chain_id order: input order
        if chain.template is None:
            continue
        tokens = _template_tokens(chain.template)
        for alignment in local_alignments(chain.sequence, chain.template.sequence):
            offset = alignment.template_start - alignment.query_start
            for j, token in enumerate(tokens):
                if not 0 <= j - offset < len(chain.sequence):
                    continue
                q = start_of[i] + j - offset
                restype[0, q] = token["res_type"]
                out["template_frame_rot"][0, q] = token["frame_rot"]
                out["template_frame_t"][0, q] = token["frame_t"]
                out["template_cb"][0, q] = token["cb"]
                out["template_ca"][0, q] = token["ca"]
                out["template_mask_cb"][0, q] = token["cb_mask"]
                out["template_mask_frame"][0, q] = token["frame_mask"]
                out["template_mask"][0, q] = 1.0
                templated.add(i)
    if not templated:
        raise ValueError("no template aligns to its chain")
    for i in range(len(system.chains)):
        first = start_of[i]
        stop = first + len(system.chains[i].sequence)
        out["visibility_ids"][0, first:stop] = 0 if i in templated else -1 - asym_of[i]
    return (
        {"template_restype": np.eye(NUM_TOKENS, dtype=np.int64)[restype]}
        | out
        | {
            "template_force": np.zeros(1, bool),
            "template_force_threshold": np.full(1, np.inf, np.float32),
        }
    )


def features(system: System, *, key: jax.Array | None = None) -> dict[str, np.ndarray]:
    """The feature dict for `system`; `key` poses the reference conformers (see the module
    docstring), `None` leaves them as stored."""
    table = residues()
    order = chain_order(system)
    chains = [system.chains[i] for i, _, _ in order]

    # Tokens: one per residue, chains in asym order.
    token, asym_id, entity_id, sym_id, residue_index = [], [], [], [], []
    for asym, (chain, (_, entity, copy)) in enumerate(zip(chains, order, strict=True)):
        n = len(chain.sequence)
        token += [TOKEN[c] for c in chain.sequence]
        asym_id += [asym] * n
        entity_id += [entity] * n
        sym_id += [copy] * n
        residue_index += range(n)
    num_tokens = len(token)

    # Atoms: each residue's `ref_atoms`, with its drawn conformer.
    letters = [c for chain in chains for c in chain.sequence]
    conformer = conformer_ids(num_tokens)
    coords, charge, element, names, atom_token, rep_atom = [], [], [], [], [], []
    for t, (letter, conf) in enumerate(zip(letters, conformer, strict=True)):
        r = table[THREE_LETTER[letter]]
        rep_atom.append(len(atom_token) + r.atoms.index("CA" if letter == "G" else "CB"))
        coords.append(r.conformers[conf])
        charge.append(r.charges)
        element.append(r.elements)
        names += r.atoms
        atom_token += [t] * len(r.atoms)
    num_atoms = len(atom_token)
    padded = padded_length(num_atoms, ATOM_WINDOW)
    atom_token = np.array(atom_token, np.int64)
    pad = lambda x: pad_to(x, padded)

    ref_pos = np.concatenate(coords).astype(np.float32)
    if key is not None:
        ref_pos = np.asarray(pose(ref_pos, atom_token, key), np.float32)
    name_chars = np.array([[ord(c) - 32 for c in n.ljust(4)] for n in names])  # pad: 0
    real = np.zeros(padded, bool)
    real[:num_atoms] = True

    msa, deletion, msa_paired = _msa(chains)
    deletion_value = (np.pi / 2 * np.arctan(deletion.astype(np.float32) / 3)).astype(np.float32)
    msa_one_hot = np.eye(NUM_TOKENS, dtype=np.float32)[msa]

    zeros = lambda *shape: np.zeros(shape, np.float32)
    contacts = np.zeros((num_tokens, num_tokens, 5), np.int64)
    contacts[..., 0] = 1
    return {
        "ref_pos": pad(ref_pos),
        "ref_charge": pad(np.concatenate(charge)).astype(np.float32),
        "ref_element": pad(np.eye(NUM_ELEMENTS, dtype=np.int64)[np.concatenate(element)]),
        "ref_atom_name_chars": pad(np.eye(64, dtype=np.int64)[name_chars]),
        "ref_space_uid": pad(atom_token),
        "atom_to_token": pad(np.eye(num_tokens, dtype=np.int64)[atom_token]),
        "atom_pad_mask": real.astype(np.float32),
        "token_to_rep_atom": np.eye(padded, dtype=np.int64)[rep_atom],
        "res_type": np.eye(NUM_TOKENS, dtype=np.int64)[token],
        "profile": msa_one_hot.mean(0),
        "deletion_mean": deletion_value.mean(0),
        "method_feature": np.full(num_tokens, METHOD_XRAY, np.int64),
        "modified": np.zeros(num_tokens, np.int64),
        "cyclic_period": zeros(num_tokens),
        "mol_type": np.zeros(num_tokens, np.int64),
        "asym_id": np.array(asym_id, np.int64),
        "entity_id": np.array(entity_id, np.int64),
        "sym_id": np.array(sym_id, np.int64),
        "residue_index": np.array(residue_index, np.int64),
        "token_index": np.arange(num_tokens, dtype=np.int64),
        "token_pad_mask": np.ones(num_tokens, np.float32),
        "token_bonds": zeros(num_tokens, num_tokens, 1),
        "type_bonds": np.zeros((num_tokens, num_tokens), np.int64),
        "contact_conditioning": contacts,
        "contact_threshold": zeros(num_tokens, num_tokens),
        "msa": msa,
        "has_deletion": deletion > 0,
        "deletion_value": deletion_value,
        "msa_paired": msa_paired,
        "msa_mask": np.ones_like(msa),
        **template_features(system, num_tokens),
    }


def atom_sites(system: System) -> AtomSites:
    """What each real atom of `features(system)` is (chains in Boltz-2's entity-grouped order)."""
    return _atom_sites([system.chains[i] for i, _, _ in chain_order(system)], residues())
