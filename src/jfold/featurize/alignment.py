"""Pairwise sequence alignment as upstream pipelines use it to place templates.

`local_alignments` is Biopython's `PairwiseAligner(scoring="blastp")` in local mode with
gaps forbidden (open and extend scores of -1000), as Boltz-2 aligns a chain to its
template: every optimal alignment, each an ungapped segment, without Biopython."""

from typing import NamedTuple

import numpy as np

_LETTERS = "ARNDCQEGHILKMFPSTWYVX"
# Biopython's "blastp" substitution matrix: BLOSUM62 over the amino acids, X scoring -1
# against everything.
_BLOSUM62 = {
    "A": (4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -1),
    "R": (-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1),
    "N": (-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, -1),
    "D": (-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, -1),
    "C": (0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -1),
    "Q": (-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, -1),
    "E": (-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, -1),
    "G": (0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1),
    "H": (-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, -1),
    "I": (-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -1),
    "L": (-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -1),
    "K": (-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, -1),
    "M": (-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -1),
    "F": (-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -1),
    "P": (-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -1),
    "S": (1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, -1),
    "T": (0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1),
    "W": (-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -1),
    "Y": (-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -1),
    "V": (0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -1),
    "X": (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
}  # fmt: skip
BLOSUM62 = np.array([_BLOSUM62[a] for a in _LETTERS], np.int64)


class Alignment(NamedTuple):
    """An ungapped segment: `query[query_start:query_end]` against
    `template[template_start:template_end]` (both the same length)."""

    query_start: int
    query_end: int
    template_start: int
    template_end: int


def _indices(sequence: str) -> np.ndarray:
    unknown = set(sequence) - set(_LETTERS)
    if unknown:
        raise ValueError(f"cannot align residues {sorted(unknown)}")
    return np.array([_LETTERS.index(c) for c in sequence], np.int64)


def local_alignments(query: str, template: str) -> list[Alignment]:
    """Every optimal ungapped local alignment of `query` and `template` under BLOSUM62, as
    Biopython's aligner enumerates them: a segment starts after the last place its
    diagonal's running score (reset at zero) was zero, and ends where that score first
    reaches the best; ties ordered by where they end (query, then template). None when no
    pair of residues scores above zero."""
    scores = BLOSUM62[_indices(query)][:, _indices(template)]
    n, m = scores.shape
    runs = []  # (score, query start, query end, template start) per run of each diagonal
    for d in range(-(n - 1), m):  # template index minus query index
        diagonal = np.diagonal(scores, offset=d)
        q0 = max(0, -d)
        h, start, best_in_run = 0, 0, None
        for k, s in enumerate(diagonal.tolist()):
            h = max(0, h + s)
            if h == 0:
                start, best_in_run = k + 1, None
                continue
            if best_in_run is None or h > best_in_run[0]:
                best_in_run = (h, start, k + 1)
                runs.append((h, q0 + start, q0 + k + 1, q0 + start + d))
    if not runs:
        return []
    best = max(r[0] for r in runs)
    # A run's segment ends where its score first reaches the best.
    first = {}
    for score, q_start, q_end, t_start in runs:
        if score == best:
            first.setdefault((q_start, t_start), q_end)
    segments = [Alignment(q, q_end, t, t + (q_end - q)) for (q, t), q_end in first.items()]
    return sorted(segments, key=lambda a: (a.query_end, a.template_end))
