"""Parse a3m alignments into aligned columns and per-column insertion counts."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class A3M:
    """A parsed a3m alignment, where `deletions[i]` counts the insertions before each column
    of row i."""

    rows: tuple[str, ...]
    deletions: np.ndarray  # [S, L] int
    headers: tuple[str, ...]


def parse_a3m(text: str) -> A3M:
    """Parse a3m text whose rows all have as many aligned columns as the query."""
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
