#!/usr/bin/env python3
"""Group fake subword pieces onto whitespace tokens.

The splitter below is a local stand-in. It is not Apertus, and this test
does not write a graded prediction file.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
REPO = SRC.parents[1]

from spandiff.align import assign_subwords, locate_tokens  # noqa: E402
from spandiff.devdata import LANGS, dev_store, local_dev_file  # noqa: E402

# Controlled piece width. Not a property of Apertus.
PIECE_CHARS = 3


def fake_piece_offsets(text: str) -> list[tuple[int, int]]:
    """Split each whitespace word into marked pieces. Not the Apertus tokenizer."""
    offsets = [(0, 0)]
    for match in re.finditer(r"\S+", text):
        start, end = match.span()
        cursor = start
        while cursor < end:
            nxt = min(end, cursor + PIECE_CHARS)
            offsets.append((cursor, nxt))
            cursor = nxt
    return offsets


def _graded_files(repo: Path) -> set[Path]:
    found = set()
    for path in repo.rglob("*"):
        if path.is_file() and path.name.startswith("SpanDiff_admin_") and path.name.endswith(".jsonl.jsonl"):
            found.add(path.resolve())
    return found


def _snippet(record: dict, side: str, limit: int = 8) -> str:
    tokens = record[f"text_{side}"].split()[:limit]
    return " ".join(tokens)


def main() -> int:
    tied = assign_subwords([(0, 4)], [(0, 2), (2, 4)])
    if tied != [[0], []]:
        raise SystemExit(f"equal overlap did not stay on the earlier token: {tied}")
    # Recorded admin_de_0 offsets. The tokenizer is not called.
    # One subword covers the apostrophe and the following s, one character each.
    possessive = assign_subwords([(59, 61)], [(59, 60), (60, 61)])
    if possessive != [[0], []]:
        raise SystemExit(f"possessive s did not stay empty: {possessive}")
    before = _graded_files(REPO)
    store = dev_store(REPO)
    for lang in LANGS:
        path = local_dev_file(store, lang)
        with path.open(encoding="utf-8") as handle:
            record = json.loads(handle.readline())
        for side in ("a", "b"):
            snippet = _snippet(record, side)
            tokens = snippet.split()
            spans = locate_tokens(snippet, tokens)
            offsets = fake_piece_offsets(snippet)
            groups = assign_subwords(offsets, spans)
            if len(groups) != len(tokens):
                raise SystemExit(
                    f"{record['id']} side {side}: {len(groups)} groups != {len(tokens)} whitespace tokens"
                )
            if any(not group for group in groups):
                raise SystemExit(f"{record['id']} side {side}: a whitespace token received no piece")
            for group_index, group in enumerate(groups):
                span_start, span_end = spans[group_index]
                for piece_index in group:
                    start, end = offsets[piece_index]
                    if start < span_start or end > span_end:
                        raise SystemExit(
                            f"{record['id']} side {side}: piece left its whitespace token"
                        )
            print(
                f"{lang} {record['id']} side {side} whitespace={len(tokens)} "
                f"fake_pieces={len(offsets) - 1}"
            )
    after = _graded_files(REPO)
    if after != before:
        raise SystemExit("grouping test wrote a graded prediction file")
    print("fake piece grouping: passed")
    print("equal overlap stays on the earlier token")
    print("tokenizer: not used. This stand-in is not Apertus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
