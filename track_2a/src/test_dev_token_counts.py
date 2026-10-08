#!/usr/bin/env python3
"""One real development record per language. Placeholder scores stay in memory."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
REPO = SRC.parents[1]

from spandiff.devdata import LANGS, dev_store, ensure_gold, local_dev_file, read_jsonl  # noqa: E402
from spandiff.schema import check_object  # noqa: E402
from spandiff.score import encoder_line, placeholder_encoder_line  # noqa: E402


def _graded_files(repo: Path) -> set[Path]:
    found = set()
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name.startswith("SpanDiff_admin_") and name.endswith(".jsonl.jsonl"):
            found.add(path.resolve())
    return found


def main() -> int:
    short = {
        "id": "short",
        "text_a": "one two",
        "text_b": "trois",
        "labels_a": [0.0],
        "labels_b": [0.0],
    }
    try:
        check_object(short)
    except ValueError as exc:
        if "whitespace tokens" not in str(exc):
            raise
    else:
        raise SystemExit("a short label list was accepted")
    try:
        check_object({**short, "labels_a": [0.0, 1.1], "text_a": "one two"})
    except ValueError as exc:
        if "outside 0..1" not in str(exc):
            raise
    else:
        raise SystemExit("a label above 1 was accepted")
    try:
        check_object({**short, "labels_a": [0.0, -1], "text_a": "one two"})
    except ValueError as exc:
        if "outside 0..1" not in str(exc):
            raise
    else:
        raise SystemExit("a gold-style -1 label was accepted")
    store = dev_store(REPO)
    before = _graded_files(REPO)
    english = {}
    for lang in LANGS:
        path = local_dev_file(store, lang)
        with path.open(encoding="utf-8") as handle:
            record = json.loads(handle.readline())
        english[lang] = record["text_a"]
        line = placeholder_encoder_line(record, fill=0.0)
        check_object(line)
        if line["labels_a"] != [0.0] * len(record["labels_a"]):
            raise SystemExit(f"{record['id']}: labels_a length does not match gold")
        if line["labels_b"] != [0.0] * len(record["labels_b"]):
            raise SystemExit(f"{record['id']}: labels_b length does not match gold")
        if len(line["labels_a"]) != len(record["text_a"].split()):
            raise SystemExit(f"{record['id']}: labels_a is not the whitespace token count")
        if len(line["labels_b"]) != len(record["text_b"].split()):
            raise SystemExit(f"{record['id']}: labels_b is not the whitespace token count")
        try:
            encoder_line(record, [[0.0]], [[0.0]])
        except RuntimeError as exc:
            if "pad or truncate" not in str(exc):
                raise
        else:
            raise SystemExit(f"{record['id']}: a short vector list was accepted")
        print(
            f"{lang} {record['id']} tokens_a={len(line['labels_a'])} "
            f"tokens_b={len(line['labels_b'])}"
        )
    if not english["de"] == english["fr"] == english["it"]:
        raise SystemExit("the first English page is not the same in de, fr, and it")
    if len(english["de"].split()) != 344:
        raise SystemExit("the first English page is not 344 whitespace tokens")
    pages = {}
    for lang in LANGS:
        pages[lang] = [row["text_a"] for row in read_jsonl(local_dev_file(store, lang))]
    if len(pages["de"]) != 168 or pages["de"] != pages["fr"] or pages["de"] != pages["it"]:
        raise SystemExit("English pages are not the same 168 texts in de, fr, and it")
    english_tokens = english_dropped = 0
    drop_masks = {}
    for lang in LANGS:
        gold_rows = read_jsonl(ensure_gold(REPO / ".swissgov", lang))
        dev_rows = read_jsonl(local_dev_file(store, lang))
        if len(dev_rows) != len(gold_rows):
            raise SystemExit(f"{lang} development input does not match the gold line count")
        for dev_row, gold_row in zip(dev_rows, gold_rows):
            if (
                dev_row["id"] != gold_row["id"]
                or dev_row["text_a"] != gold_row["text_a"]
                or dev_row["text_b"] != gold_row["text_b"]
                or dev_row["labels_a"] != gold_row["labels_a"]
                or dev_row["labels_b"] != gold_row["labels_b"]
            ):
                raise SystemExit(
                    f"{lang} development input does not match gold at {dev_row.get('id')}"
                )
        if lang == "de":
            for gold_row in gold_rows:
                english_tokens += len(gold_row["text_a"].split())
                english_dropped += sum(1 for label in gold_row["labels_a"] if label == -1)
        drop_masks[lang] = [
            [label == -1 for label in gold_row["labels_a"]] for gold_row in gold_rows
        ]
    if english_tokens != 65320 or english_dropped != 8431:
        raise SystemExit(
            f"English token totals changed: {english_tokens} tokens, {english_dropped} dropped"
        )
    if drop_masks["de"] != drop_masks["fr"] or drop_masks["de"] != drop_masks["it"]:
        raise SystemExit("English drop positions are not the same in de, fr, and it")
    same = differ = wide = 0
    label_rows = {}
    for lang in LANGS:
        label_rows[lang] = [
            row["labels_a"] for row in read_jsonl(ensure_gold(REPO / ".swissgov", lang))
        ]
    for left, mid, right in zip(label_rows["de"], label_rows["fr"], label_rows["it"]):
        for x, y, z in zip(left, mid, right):
            if x == -1:
                continue
            if x == y == z:
                same += 1
            else:
                differ += 1
                if max(x, y, z) - min(x, y, z) >= 0.8:
                    wide += 1
    if same != 42301 or differ != 14588 or wide != 6138:
        raise SystemExit(f"English agreement changed: same {same} differ {differ} wide {wide}")
    kept = {}
    for lang in LANGS:
        total = 0
        for row in read_jsonl(ensure_gold(REPO / ".swissgov", lang)):
            total += sum(1 for label in row["labels_a"] if label != -1)
            if not all(label == -1 for label in row["labels_b"]):
                total += sum(1 for label in row["labels_b"] if label != -1)
        kept[lang] = total
    if kept != {"de": 108945, "fr": 124262, "it": 118735}:
        raise SystemExit(f"official kept-token counts changed: {kept}")
    dropped = 0
    for lang in LANGS:
        for row in read_jsonl(ensure_gold(REPO / ".swissgov", lang)):
            for side in ("a", "b"):
                if side == "b" and all(label == -1 for label in row["labels_b"]):
                    continue
                dropped += sum(1 for label in row[f"labels_{side}"] if label == -1)
    if dropped != 53712:
        raise SystemExit(f"gold -1 positions changed: {dropped}")
    if sum(kept.values()) + dropped != 405654:
        raise SystemExit(
            f"kept tokens {sum(kept.values())} plus gold -1 {dropped} "
            "are not the 405654 whitespace tokens"
        )
    architecture = (REPO / "track_2a" / "docs" / "architecture.md").read_text(encoding="utf-8")
    if "53712" not in architecture or "351942" not in architecture or "405654" not in architecture:
        raise SystemExit("architecture does not reconcile kept tokens, gold -1, and whitespace tokens")
    after = _graded_files(REPO)
    if after != before:
        raise SystemExit("placeholder scores were written to a graded prediction file")
    print("dev token counts: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
