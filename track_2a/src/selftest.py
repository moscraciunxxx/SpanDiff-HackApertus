#!/usr/bin/env python3
"""Invariants for the committed development grade. No document-wording locks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[1]
sys.path.insert(0, str(SRC))

from spandiff.leakage import needles, scan  # noqa: E402

LANGS = ("de", "fr", "it")
KEYS = {"id", "text_a", "text_b", "labels_a", "labels_b"}
GRADED_SHA = {
    "de": "bb0c1cac606d6ee8e407f968f99797640311040e1e86b506cb93f708e75952f6",
    "fr": "56b58418d67c22e7114cc8457647f02d65aeae117d974f64edf6c13a74f0cee9",
    "it": "c83223ce12938f255615e3b8c491d79449987b057024e152c1ebe9083fc70ddb",
}
PRED = REPO / "track_2a" / "predictions" / "dev"


def _lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    rows = text.splitlines()
    if any(not row.strip() for row in rows):
        raise SystemExit(f"{path.name} has a blank line")
    return rows


def _check_file(path: Path, lang: str) -> None:
    rows = _lines(path)
    if len(rows) != 168:
        raise SystemExit(f"{path.name} has {len(rows)} lines, need 168")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != GRADED_SHA[lang]:
        raise SystemExit(f"{path.name} is not the frozen development file")
    for raw in rows:
        obj = json.loads(raw)
        if set(obj) != KEYS:
            raise SystemExit(f"{path.name} {obj.get('id')} keys are not the encoder set")
        for side in ("a", "b"):
            labels = obj[f"labels_{side}"]
            if len(labels) != len(str(obj[f"text_{side}"]).split()):
                raise SystemExit(f"{path.name} {obj.get('id')} label count")
            for value in labels:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise SystemExit(f"{path.name} label type")
                number = float(value)
                if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                    raise SystemExit(f"{path.name} label outside 0..1")


def _check_shape(path: Path, count: int) -> None:
    rows = _lines(path)
    if len(rows) != count:
        raise SystemExit(f"{path.name} has {len(rows)} lines, need {count}")
    for raw in rows:
        obj = json.loads(raw)
        if set(obj) != KEYS:
            raise SystemExit(f"{path.name} keys")
        for side in ("a", "b"):
            labels = obj[f"labels_{side}"]
            if len(labels) != len(str(obj[f"text_{side}"]).split()):
                raise SystemExit(f"{path.name} label count")
            for value in labels:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise SystemExit(f"{path.name} label type")
                number = float(value)
                if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                    raise SystemExit(f"{path.name} label outside 0..1")


def _check_predictions() -> None:
    for lang in LANGS:
        visible = PRED / f"SpanDiff_admin_{lang}.jsonl.jsonl"
        if not visible.is_file():
            raise SystemExit(f"missing {visible}")
        _check_file(visible, lang)
        hidden = (
            REPO
            / ".swissgov"
            / "data"
            / "evaluation"
            / "encoder_predictions"
            / "dev"
            / visible.name
        )
        if hidden.is_file() and hidden.read_bytes() != visible.read_bytes():
            raise SystemExit(f"{visible.name} does not match the checkout copy")
        _check_shape(REPO / "track_2a" / "predictions" / "test" / visible.name, 56)
        _check_shape(REPO / "track_2a" / "predictions" / "full" / visible.name, 224)


def _check_command() -> None:
    grader = (REPO / "track_2a" / "grade_dev.sh").read_text(encoding="utf-8")
    if "--split dev" not in grader:
        raise SystemExit("grade_dev.sh does not pass the development split")
    if "track_2a/predictions/dev/SpanDiff_admin_" not in grader:
        raise SystemExit("grade_dev.sh does not grade the committed predictions")
    for sealed in needles():
        if sealed in grader:
            raise SystemExit("grade_dev.sh names a sealed path")
    makefile = (REPO / "track_2a" / "Makefile").read_text(encoding="utf-8")
    run_recipe = makefile.split("\nrun:", 1)[1].split("\ncheck:", 1)[0]
    if "run_image.sh" not in run_recipe:
        raise SystemExit("make run does not use the image")
    root_make = (REPO / "Makefile").read_text(encoding="utf-8")
    if "$(MAKE) -C track_2a run" not in root_make:
        raise SystemExit("root make run does not delegate")


def _selector_path(argv: list[str]) -> str | None:
    probe = subprocess.run(argv, check=False, capture_output=True, text=True, cwd=SRC)
    if probe.returncode != 0:
        return None
    text = probe.stdout.strip()
    return text or None


def _check_selectors() -> None:
    shell = _selector_path(["bash", str(REPO / "track_2a" / "find_grader.sh")])
    py = _selector_path(
        [sys.executable, "-c", "import run; print(run._grader_python())"],
    )
    if shell != py:
        raise SystemExit(
            f"find_grader.sh and the Python selector disagree: {shell} vs {py}"
        )


def _check_leakage() -> None:
    hits = scan(REPO)
    if hits:
        raise SystemExit("leakage fence failed:\n" + "\n".join(hits))


def main() -> int:
    _check_predictions()
    _check_command()
    _check_leakage()
    _check_selectors()
    print("self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
