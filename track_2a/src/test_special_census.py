#!/usr/bin/env python3
"""Recompute the development special-token census without loading weights."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
REPO = SRC.parents[1]

from spandiff.census import (  # noqa: E402
    census_development,
    lock_line,
    parse_census_file,
    tokenizer_json,
    _omni_offset,
)
from spandiff.devdata import LANGS, dev_store, local_dev_file, read_jsonl  # noqa: E402
from spandiff.schema import dumps_encoder  # noqa: E402
from spandiff.score import dumps_line  # noqa: E402

GRADED_SHA = {
    "de": "a34b70adf48e9eb2748eb1cd6b189192fd5665790e08e047ad9218954c14b018",
    "fr": "199bc15eadb56ca290117cd04101d6d2bc37fe681577f8f5513a6c0fb4aaf0da",
    "it": "446efc6e5cf6d851a5dbab0ea3c40f803043d19edaf98db66f03364ae2d4c965",
}


def _graded_hashes() -> dict[str, str]:
    pred = REPO / "track_2a" / "predictions" / "dev"
    found = {}
    for lang in LANGS:
        path = pred / f"SpanDiff_admin_{lang}.jsonl.jsonl"
        found[lang] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


def _reject_labels() -> None:
    base = {
        "id": "reject",
        "text_a": "one two",
        "text_b": "trois",
        "labels_a": [0.0, 0.0],
        "labels_b": [0.0],
    }
    try:
        dumps_encoder({**base, "labels_a": [-1, 0.0]})
    except ValueError as exc:
        if "outside 0..1" not in str(exc):
            raise
        print("rejected label -1")
        print("rejected label \u22121")
    else:
        raise SystemExit("writer accepted -1")
    try:
        dumps_encoder({**base, "labels_a": [float("nan"), 0.0]})
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


def _refuse_short_copy() -> None:
    """Drive grade_dev.sh against a copy. The graded files stay put."""
    before = _graded_hashes()
    tmp = Path(tempfile.mkdtemp(prefix="spandiff-short-"))
    try:
        for lang, count in (("de", 167), ("fr", 168), ("it", 168)):
            path = tmp / f"SpanDiff_admin_{lang}.jsonl.jsonl"
            path.write_text("\n".join(["{}"] * count) + "\n", encoding="utf-8")
        env = os.environ.copy()
        env["SPAN_DIFF_PRED_PREFIX"] = str(tmp / "SpanDiff_admin_")
        proc = subprocess.run(
            ["bash", str(REPO / "track_2a" / "grade_dev.sh")],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
    finally:
        for path in tmp.glob("*"):
            path.unlink()
        tmp.rmdir()
    if proc.returncode == 0:
        raise SystemExit("a 167-line copy was graded")
    combined = proc.stdout + proc.stderr
    if "need 168" not in combined:
        raise SystemExit("a 167-line copy was not refused:\n" + combined)
    if "weight load" in combined:
        raise SystemExit("the short-file refusal loaded weights")
    after = _graded_hashes()
    if after != before:
        raise SystemExit("the short-file refusal changed a graded file")


def main() -> int:
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        print("tokenizer not available")
        return 0
    try:
        tokenizer_json()
    except SystemExit:
        print("tokenizer not available")
        return 0
    before = _graded_hashes()
    if before != GRADED_SHA:
        raise SystemExit("graded file hash changed before the census")
    recorded = parse_census_file(
        (REPO / "track_2a" / "docs" / "special_tokens.txt").read_text(encoding="utf-8")
    )
    _reject_labels()
    from tokenizers import Tokenizer

    path = tokenizer_json()
    if path.suffix != ".json" or "safetensors" in path.name:
        raise SystemExit("census refused to open a weight file")
    tokenizer = Tokenizer.from_file(str(path))
    store = dev_store(REPO)
    rows = {lang: read_jsonl(local_dev_file(store, lang)) for lang in LANGS}
    stats = census_development(tokenizer, rows, _omni_offset(path))
    comparable = {key: stats[key] for key in recorded}
    if comparable != recorded:
        mismatch = {key: (recorded[key], comparable.get(key)) for key in recorded if recorded[key] != comparable.get(key)}
        raise SystemExit(f"census does not match special_tokens.txt: {mismatch}")
    if stats["bos_pooled"] != 0 or stats["only_leading_bos"] != stats["windows"]:
        raise SystemExit("BOS was pooled or was not the only special")
    if stats["eos_count"] or stats["pad_count"] or stats["unk_count"] or stats["multimodal_special_count"]:
        raise SystemExit("a non-BOS special was counted")
    print(lock_line(stats))
    _refuse_short_copy()
    if _graded_hashes() != GRADED_SHA:
        raise SystemExit("census changed a graded file")
    print("special census: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
