#!/usr/bin/env python3
"""Fake-vector check of the paper formula. No model and no graded file."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from spandiff.align import diff_against  # noqa: E402
from spandiff.paper_diffalign import paper_diff  # noqa: E402


def main() -> int:
    same = [[1.0, 0.0]]
    if paper_diff(same, same) != [0.0]:
        raise SystemExit("identical vectors were not 0")
    if paper_diff([[1.0, 0.0]], [[0.0, 1.0]]) != [1.0]:
        raise SystemExit("orthogonal vectors were not 1")

    opposite = paper_diff([[1.0, 0.0]], [[-1.0, 0.0]])
    submitted = diff_against([[1.0, 0.0]], [[-1.0, 0.0]])
    if abs(opposite[0] - 2.0) > 1e-9:
        raise SystemExit(f"paper formula on a negative cosine returned {opposite[0]}, expected 2")
    if submitted != [1.0]:
        raise SystemExit(f"submitted clamp returned {submitted}, expected 1")
    if opposite[0] == submitted[0]:
        raise SystemExit("the unclamped score matched the clamp")

    best_of_two = paper_diff([[1.0, 0.0]], [[-1.0, 0.0], [1.0, 0.0]])
    if best_of_two != [0.0]:
        raise SystemExit("the maximum cosine was not the one that counted")
    print("paper diffalign: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
