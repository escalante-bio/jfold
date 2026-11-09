"""a3m parsing shared by the views: aligned (match) columns and insertion counts.

Lowercase letters are insertions relative to the query; every other character is an
aligned column (`-` a gap). Each row's insertions are counted onto the next aligned column;
insertions after the last aligned column are dropped. How characters map to residue
classes, and which rows are kept, are each view's rules."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class A3M:
    """`rows[i]`: row i's aligned columns, as written (uppercase, `-`, `.`); `deletions[i]`:
    the insertion count before each column; `headers[i]`: its header line, without `>`."""

    rows: tuple[str, ...]
    deletions: np.ndarray  # [S, L] int
    headers: tuple[str, ...]


def parse_a3m(text: str) -> A3M:
    """FASTA records (sequences may wrap over lines); every row must have as many aligned
    columns as the first (the query)."""
    headers, sequences = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            headers.append(line[1:])
            sequences.append([])
        elif sequences:
            sequences[-1].append(line)
    rows, deletions = [], []
    for header, parts in zip(headers, sequences, strict=True):
        aligned, counts, pending = [], [], 0
        for c in "".join(parts):
            if c.islower():
                pending += 1
            else:
                aligned.append(c)
                counts.append(pending)
                pending = 0
        rows.append("".join(aligned))
        deletions.append(counts)
        if len(aligned) != len(rows[0]):
            raise ValueError(
                f"a3m row {header!r}: {len(aligned)} columns, query has {len(rows[0])}"
            )
    return A3M(
        tuple(rows), np.array(deletions, dtype=np.int64).reshape(len(rows), -1), tuple(headers)
    )
