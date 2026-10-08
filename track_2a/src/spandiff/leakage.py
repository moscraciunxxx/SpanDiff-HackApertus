"""Fail the build if a sealed-split name appears in the tree."""

from __future__ import annotations

import sys
from pathlib import Path

# Pieces stay split so this file does not contain the sealed names.
_PAIRS = (
    ("gold_labels/", "test"),
    ("--split ", "test"),
    ("--split ", "full"),
    ("test_", "de.jsonl"),
    ("test_", "fr.jsonl"),
    ("test_", "it.jsonl"),
)

SKIP_DIRS = {".git", ".swissgov", "__pycache__", ".venv", "venv"}


def needles() -> list[str]:
    return [left + right for left, right in _PAIRS]


def _skipped(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def scan(root: Path) -> list[str]:
    """Return human-readable hits. Empty means the tree is clean."""
    hits: list[str] = []
    banned = needles()
    root = root.resolve()
    for path in root.rglob("*"):
        if _skipped(path):
            continue
        label = str(path.relative_to(root))
        for needle in banned:
            if needle in label:
                hits.append(f"{label}: name")
                break
        if not path.is_file():
            continue
        # Development jsonl can be a few megabytes. A sealed filename still fails above.
        if path.stat().st_size > 32_000_000:
            hits.append(f"{label}: file exceeds 32 MB")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in banned:
            if needle in text:
                hits.append(f"{label}: text")
                break
    hits.extend(_sealed_directories(root))
    return hits


def _sealed_directories(root: Path) -> list[str]:
    """Directory names only under the local data tree. Do not read those files."""
    hits: list[str] = []
    swiss = root / ".swissgov"
    if not swiss.is_dir():
        return hits
    banned = needles()
    for path in swiss.rglob("*"):
        if not path.is_dir():
            continue
        label = path.relative_to(root).as_posix()
        for needle in banned:
            if needle in label:
                hits.append(f"{label}: sealed directory")
                break
    return hits


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: leakage.py ROOT", file=sys.stderr)
        return 2
    hits = scan(Path(argv[1]))
    if hits:
        print("\n".join(hits))
        return 1
    print(f"clean: {argv[1]}")
    return 0


def assert_path_allowed(path: Path) -> None:
    """Refuse a filesystem path that names the sealed split."""
    text = str(path)
    for needle in needles():
        if needle in text:
            raise SystemExit(f"refusing sealed path: {path.name}")
    if path.name.startswith("test_"):
        raise SystemExit(f"refusing sealed file: {path.name}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
