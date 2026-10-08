"""Encoder JSON lines the official loader accepts."""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED = ("id", "text_a", "text_b", "labels_a", "labels_b")
REJECTED = ("api_request", "prompt")


def check_object(obj: dict) -> None:
    if not isinstance(obj, dict):
        raise ValueError("encoder line is not an object")
    for key in REJECTED:
        if key in obj:
            raise ValueError(f"encoder line contains {key}")
    missing = [key for key in REQUIRED if key not in obj]
    if missing:
        raise ValueError("encoder line missing " + ", ".join(missing))
    if not isinstance(obj["id"], str) or not obj["id"]:
        raise ValueError("id must be a non-empty string")
    for side in ("a", "b"):
        text = obj[f"text_{side}"]
        labels = obj[f"labels_{side}"]
        if not isinstance(text, str):
            raise ValueError(f"text_{side} must be a string")
        if not isinstance(labels, list):
            raise ValueError(f"labels_{side} must be a list")
        tokens = text.split()
        if len(labels) != len(tokens):
            raise ValueError(
                f"labels_{side} length {len(labels)} != whitespace tokens {len(tokens)}"
            )
        for value in labels:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"labels_{side} must be numbers")
            number = float(value)
            if number != number or not 0.0 <= number <= 1.0:
                raise ValueError(f"label {number} is outside 0..1")


def check_jsonl(path: Path) -> int:
    """Count encoder records. A blank line is refused, matching jsonlines.

    One leading BOM or record separator is stripped, which is what the
    evaluator's reader does before ``json.loads``.
    """
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(
                    f"{path}:{line_number}: blank line. "
                    "The evaluator's jsonlines reader rejects it."
                )
            if line[:1] in ("\x1e", "\ufeff"):
                line = line[1:]
            try:
                obj = json.loads(line)
                check_object(obj)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            count += 1
    if count == 0:
        raise ValueError(f"{path} has no encoder lines")
    return count


def dumps_encoder(obj: dict) -> str:
    check_object(obj)
    payload = {key: obj[key] for key in REQUIRED}
    return json.dumps(payload, ensure_ascii=False)
