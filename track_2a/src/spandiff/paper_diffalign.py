"""Unclamped DiffAlign score from ACL 2026 section 4.1.

The submitted run does not import this module. It keeps clamp(1 - max cosine, 0, 1).
"""

from __future__ import annotations

from spandiff.align import cosine


def paper_diff(source: list[list[float]], other: list[list[float]]) -> list[float]:
    """1 - max cosine, with no clamp. One score per source vector."""
    scores: list[float] = []
    for vector in source:
        if not other:
            scores.append(1.0)
            continue
        best = max(cosine(vector, candidate) for candidate in other)
        scores.append(1.0 - best)
    return scores
