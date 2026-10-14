#!/usr/bin/env python3
"""Regenerate test and full predictions from public SwissGov text fields.

The Hub publishes development and complement text files. It has no separate
full file. Full-set order is ids 0..223 from those two public files. Label
fields are dropped before scoring. score_split.py uses the shipped constants.
Committed prediction files are only read for id order and the byte compare.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[1]
sys.path.insert(0, str(SRC))

from score_split import text_only  # noqa: E402
from spandiff.devdata import LANGS, PUBLISHED_SHA256  # noqa: E402

HF_ROOT = "https://huggingface.co/datasets/ZurichNLP/SwissGov-RSD/resolve/main/data/"
SCRATCH = Path("/tmp/spandiff-predict-test")


def _public_url(lang: str, kind: str) -> str:
    return HF_ROOT + lang + "/" + kind + "_" + lang + ".jsonl"


def _fetch(lang: str, kind: str) -> tuple[bytes, list[dict]]:
    url = _public_url(lang, kind)
    request = urllib.request.Request(url, headers={"User-Agent": "spandiff"})
    with urllib.request.urlopen(request, timeout=120) as response:
        downloaded = response.read()
    if kind == "dev":
        digest = hashlib.sha256(downloaded).hexdigest()
        expected = PUBLISHED_SHA256[f"{lang}/dev_{lang}.jsonl"]
        if digest != expected:
            raise SystemExit(f"{lang} development text is not the published file")
    raw = downloaded[3:] if downloaded.startswith(b"\xef\xbb\xbf") else downloaded
    records = []
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            records.append(text_only(line))
    return raw, records


def _ids(split: str, lang: str) -> list[str]:
    path = REPO / "track_2a" / "predictions" / split / f"SpanDiff_admin_{lang}.jsonl.jsonl"
    return [
        json.loads(line)["id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_texts(directory: Path, lang: str, records: dict[str, dict], order: list[str]) -> None:
    missing = [item for item in order if item not in records]
    if missing:
        raise SystemExit(f"{lang}: {len(missing)} ids are absent from the public text")
    path = directory / f"{lang}.jsonl"
    lines = []
    for item in order:
        record = records[item]
        kept = {
            "id": record["id"],
            "text_a": record["text_a"],
            "text_b": record["text_b"],
        }
        for key in ("text_a_untokenized", "text_b_untokenized"):
            if key in record:
                kept[key] = record[key]
        kept["labels_a"] = record["labels_a"]
        kept["labels_b"] = record["labels_b"]
        lines.append(json.dumps(kept, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _max_delta(got: Path, expected: Path) -> float:
    worst = 0.0
    for left, right in zip(
        got.read_text(encoding="utf-8").splitlines(),
        expected.read_text(encoding="utf-8").splitlines(),
    ):
        a = json.loads(left)
        b = json.loads(right)
        if a.get("id") != b.get("id"):
            return float("inf")
        for side in ("a", "b"):
            for one, two in zip(a[f"labels_{side}"], b[f"labels_{side}"]):
                worst = max(worst, abs(float(one) - float(two)))
    return worst


def _compare(split: str, lang: str, got_dir: Path) -> bool:
    name = f"SpanDiff_admin_{lang}.jsonl.jsonl"
    got = got_dir / name
    expected = REPO / "track_2a" / "predictions" / split / name
    if got.read_bytes() == expected.read_bytes():
        print(f"byte-identical {split} {lang}")
        return True
    print(f"DIFF {split} {lang} max_abs {_max_delta(got, expected)}")
    return False


def main() -> int:
    if SCRATCH.is_relative_to(REPO):
        raise SystemExit("refusing to write the scratch directory inside the repository")
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    text_dir = SCRATCH / "texts"
    out_dir = SCRATCH / "full"
    test_dir = SCRATCH / "test"
    text_dir.mkdir(parents=True)
    test_dir.mkdir()
    for lang in LANGS:
        dev_records = {row["id"]: row for row in _fetch(lang, "dev")[1]}
        complement = {row["id"]: row for row in _fetch(lang, "test")[1]}
        overlap = set(dev_records) & set(complement)
        if overlap:
            raise SystemExit(f"{lang}: public text ids overlap")
        records = dev_records | complement
        full_order = _ids("full", lang)
        test_order = _ids("test", lang)
        if len(complement) != 56 or set(complement) != set(test_order):
            raise SystemExit(f"{lang}: public complement is not the committed 56 ids")
        if len(records) != 224 or full_order != [f"admin_{lang}_{index}" for index in range(224)]:
            raise SystemExit(f"{lang}: full text is not ids 0 through 223")
        _write_texts(text_dir, lang, records, full_order)
    subprocess.run(
        [
            sys.executable,
            str(SRC / "apply_ridge.py"),
            "--texts",
            str(text_dir),
            "--out",
            str(out_dir),
        ],
        check=True,
    )
    ok = True
    for lang in LANGS:
        scored = {
            json.loads(line)["id"]: line
            for line in (out_dir / f"SpanDiff_admin_{lang}.jsonl.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        test_lines = [scored[item] for item in _ids("test", lang)]
        (test_dir / f"SpanDiff_admin_{lang}.jsonl.jsonl").write_text(
            "\n".join(test_lines) + "\n",
            encoding="utf-8",
        )
        ok = _compare("full", lang, out_dir) and ok
        ok = _compare("test", lang, test_dir) and ok
    print("scratch", SCRATCH)
    if not ok:
        raise SystemExit("scratch predictions are not byte-identical to the committed files")
    print("RESULT byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
