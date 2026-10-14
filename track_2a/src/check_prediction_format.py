#!/usr/bin/env python3
"""Check test and full prediction files against the encoder schema.

Full-set order is ids 0 through 223. That is the published full-set order.
This script does not open sealed gold.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[1]
sys.path.insert(0, str(SRC))

from spandiff.schema import check_object  # noqa: E402

LANGS = ("de", "fr", "it")


def _rows(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise SystemExit(f"{path}:{number}: blank line")
        if line[:1] in ("\ufeff", "\x1e"):
            line = line[1:]
        try:
            obj = json.loads(line)
            check_object(obj)
        except (json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(f"{path}:{number}: {exc}") from exc
        for side in ("a", "b"):
            for value in obj[f"labels_{side}"]:
                if not math.isfinite(float(value)):
                    raise SystemExit(f"{path}:{number}: non-finite label")
        rows.append(obj)
    return rows


def main() -> int:
    for lang in LANGS:
        dev = _rows(REPO / "track_2a/predictions/dev" / f"SpanDiff_admin_{lang}.jsonl.jsonl")
        test = _rows(REPO / "track_2a/predictions/test" / f"SpanDiff_admin_{lang}.jsonl.jsonl")
        full = _rows(REPO / "track_2a/predictions/full" / f"SpanDiff_admin_{lang}.jsonl.jsonl")
        full_ids = [row["id"] for row in full]
        expected = [f"admin_{lang}_{index}" for index in range(224)]
        if full_ids != expected:
            raise SystemExit(f"{lang}: full file is not ids 0 through 223")
        dev_ids = [row["id"] for row in dev]
        test_ids = [row["id"] for row in test]
        if len(dev_ids) != 168 or len(set(dev_ids)) != 168:
            raise SystemExit(f"{lang}: development file is not 168 distinct ids")
        if set(dev_ids) & set(test_ids):
            raise SystemExit(f"{lang}: development and test ids overlap")
        if set(dev_ids) | set(test_ids) != set(expected):
            raise SystemExit(f"{lang}: development plus test is not the full id set")
        by_id = {row["id"]: row for row in full}
        for row in dev + test:
            match = by_id[row["id"]]
            if row["text_a"] != match["text_a"] or row["text_b"] != match["text_b"]:
                raise SystemExit(f"{row['id']}: text differs from the full file")
            if len(row["labels_a"]) != len(match["labels_a"]) or len(row["labels_b"]) != len(match["labels_b"]):
                raise SystemExit(f"{row['id']}: token count differs from the full file")
        print(f"{lang} ok dev {len(dev)} test {len(test)} full {len(full)}")
    print("format check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
