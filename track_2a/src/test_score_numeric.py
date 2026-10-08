#!/usr/bin/env python3
"""Numeric checks for the submitted score. No model and no graded file."""

from __future__ import annotations

import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
REPO = SRC.parents[1]

from spandiff.align import diff_against, mean_pool, scores_after_stitch, stitch_chunks  # noqa: E402
from spandiff.score import dumps_line, encoder_line  # noqa: E402


def _graded_files(repo: Path) -> set[Path]:
    found = set()
    for path in repo.rglob("*"):
        if path.is_file() and path.name.startswith("SpanDiff_admin_") and path.name.endswith(".jsonl.jsonl"):
            found.add(path.resolve())
    return found


def main() -> int:
    before = _graded_files(REPO)

    identical = diff_against([[1.0, 0.0]], [[1.0, 0.0]])
    if identical != [0.0]:
        raise SystemExit(f"identical unit vectors scored {identical}, expected 0")

    orthogonal = diff_against([[1.0, 0.0]], [[0.0, 1.0]])
    if orthogonal != [1.0]:
        raise SystemExit(f"orthogonal vectors scored {orthogonal}, expected 1")

    negative = diff_against([[1.0, 0.0]], [[-1.0, 0.0]])
    if negative != [1.0] or negative[0] > 1.0:
        raise SystemExit(f"negative cosine scored {negative}, expected 1")

    # Two chunks. The best match for the first source token sits in the second other-side chunk.
    source_chunks = [[[1.0, 0.0]], [[0.0, 1.0]]]
    other_chunks = [[[0.0, 1.0]], [[1.0, 0.0]]]
    stitched_source = stitch_chunks(source_chunks)
    stitched_other = stitch_chunks(other_chunks)
    if stitched_source != [[1.0, 0.0], [0.0, 1.0]]:
        raise SystemExit("source chunks were not concatenated in order")
    if stitched_other != [[0.0, 1.0], [1.0, 0.0]]:
        raise SystemExit("other-side chunks were not concatenated in order")
    full = scores_after_stitch(source_chunks, other_chunks)
    per_chunk = diff_against(source_chunks[0], other_chunks[0])
    if per_chunk != [1.0]:
        raise SystemExit("the per-chunk case no longer contrasts with the full sequence")
    if full[0] != 0.0:
        raise SystemExit(f"full-sequence max scored {full[0]}, expected 0")
    if full[0] == per_chunk[0]:
        raise SystemExit("full-sequence score matched the wrong per-chunk max")
    if len(full) != len(stitched_source):
        raise SystemExit("stitched score length does not match the concatenated tokens")

    record = {
        "id": "length-check",
        "text_a": "one two",
        "text_b": "trois",
        "labels_a": [0.0, 0.0],
        "labels_b": [0.0],
    }
    line = encoder_line(record, [[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0]])
    cosine = 0.5 / math.sqrt(0.5)
    expected = round(min(1.0, max(0.0, 1.0 - cosine)), 6)
    rounded = encoder_line(
        {
            "id": "round-check",
            "text_a": "aa",
            "text_b": "bb",
            "labels_a": [0.0],
            "labels_b": [0.0],
        },
        [[1.0, 0.0]],
        [[0.5, 0.5]],
    )
    if rounded["labels_a"] != [expected] or expected == 1.0 - cosine:
        raise SystemExit(f"score was not rounded to 6 decimals: {rounded['labels_a']}")
    written = dumps_line(
        {
            "id": "round-check",
            "text_a": "aa",
            "text_b": "bb",
            "labels_a": [0.0],
            "labels_b": [0.0],
        },
        [[1.0, 0.0]],
        [[0.5, 0.5]],
    )
    if "0.292893" not in written or "0.2928932" in written:
        raise SystemExit(f"written score is not 6 decimals: {written}")
    from spandiff.schema import dumps_encoder

    rejected = {
        "id": "reject",
        "text_a": "one two",
        "text_b": "trois",
        "labels_a": [-1, 0.0],
        "labels_b": [0.0],
    }
    try:
        dumps_encoder(rejected)
    except ValueError as exc:
        if "outside 0..1" not in str(exc):
            raise
        print("rejected label -1")
        print("rejected label \u22121")
    else:
        raise SystemExit("writer accepted -1")
    rejected["labels_a"] = [float("nan"), 0.0]
    try:
        dumps_encoder(rejected)
    except ValueError as exc:
        if "outside 0..1" not in str(exc):
            raise
        print("rejected NaN")
    else:
        raise SystemExit("writer accepted NaN")
    try:
        dumps_line(
            {
                "id": "nan-score",
                "text_a": "aa",
                "text_b": "bb",
                "labels_a": [0.0],
                "labels_b": [0.0],
            },
            [[float("nan"), 1.0]],
            [[1.0, 0.0]],
        )
    except ValueError as exc:
        if "rejected NaN" not in str(exc):
            raise
        print("rejected NaN")
    else:
        raise SystemExit("score writer accepted a NaN cosine")
    if len(line["labels_a"]) != len(record["text_a"].split()):
        raise SystemExit("labels_a length is not the whitespace token count")
    if len(line["labels_b"]) != len(record["text_b"].split()):
        raise SystemExit("labels_b length is not the whitespace token count")
    pooled = mean_pool([[1.0, 1.0], [3.0, 5.0]], [(0, 1), (1, 2)], [(0, 2)])
    if pooled != [[2.0, 3.0]]:
        raise SystemExit(f"equal-width rows pooled to {pooled}")
    try:
        mean_pool([[1.0, 0.0], [1.0]], [(0, 1), (1, 2)], [(0, 2)])
    except ValueError:
        pass
    else:
        raise SystemExit("a short hidden row was padded with zeros")
    try:
        diff_against([[1.0, 0.0]], [[1.0]])
    except ValueError:
        pass
    else:
        raise SystemExit("a short cosine vector was truncated")
    # One subword covers the apostrophe and the following s. The earlier span keeps it.
    possessive = mean_pool([[4.0, 6.0]], [(59, 61)], [(59, 60), (60, 61)])
    if possessive != [[4.0, 6.0], [0.0, 0.0]]:
        raise SystemExit(f"possessive s was not a zero vector: {possessive}")
    possessive_score = diff_against([possessive[1]], [[4.0, 6.0]])
    if possessive_score != [1.0]:
        raise SystemExit(f"an empty span scored {possessive_score}, expected 1")

    after = _graded_files(REPO)
    if after != before:
        raise SystemExit("numeric test wrote a graded prediction file")
    print("score numeric: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
