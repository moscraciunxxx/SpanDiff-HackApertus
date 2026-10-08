"""Development-split files only. The sealed split is never requested."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

from spandiff.leakage import assert_path_allowed, needles

LANGS = ("de", "fr", "it")
HF_DATASET = "https://huggingface.co/datasets/ZurichNLP/SwissGov-RSD/resolve/main/"
GOLD_BASE = "https://raw.githubusercontent.com/ZurichNLP/SwissGov-RSD/master/"

# Suffix of the destination path, then the published SHA256. scripts/__init__.py
# is the upstream file. The local copy is blanked after download.
PUBLISHED_SHA256 = {
    "de/dev_de.jsonl": "4b5141414cd06cc2277d7d56c1c794fb8d92a1be092f36b250dfc3f7d7780ccd",
    "fr/dev_fr.jsonl": "732ccdb2d3d1a755dde3fab10d9b44061cbee71829aab917c599962f9a7587dd",
    "it/dev_it.jsonl": "e8f0dc72f66a24b7e244e1cc5cfab2c9f5ba14193603094e0e1a4b8da507c698",
    "gold_labels/dev/gold_admin_de.jsonl": "68db3a17b77d49813da32b16573f5f79ba29db51c8296ec5d16f823bc1e98e78",
    "gold_labels/dev/gold_admin_fr.jsonl": "19d08d5ad56118c94a8608dc3f46e417fe2e3ee7145d31ee966c349550940c46",
    "gold_labels/dev/gold_admin_it.jsonl": "55bfe711ece79274ee81b25f7d46d77a11179dfe3f7a08f74fa6bc47f7499885",
    "list_to_drop.txt": "3bd489505875ecc0800685f1b732f84dad7b7c25380f63612cadcfbe2732857a",
    "scripts/__init__.py": "e1a623a06cf6ccd7b11f6f4a88d0c846cb2e2461699b2073fd627b18e8d0bf9e",
    "scripts/evaluate_predictions_admin.py": "3e8101661621c51938e62d8d064e737e3e7d9bb16088dad754ba289c0aa083fa",
    "evaluation/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "evaluation/utils.py": "8da64aa6ec33f316eec2c65d0cbc3fe36d247c585a033d2ed04a7bebc175df0b",
    "evaluation/predictions.py": "40a914ce039804fa8858600e992844489aea5292841ccb6666ef183c30c32fbc",
    "rsd/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "rsd/recognizers/__init__.py": "95e2aef525eb41688ab5f0d10a4a5def10cd6d8ff2a7b0800302fc38adfcd2eb",
    "rsd/recognizers/utils.py": "e69f014a83b4c731b2b98fd06eb7b8758fefa9df9e3440410002aa3bae84831a",
}


def _check_relative(name: str) -> str:
    for needle in needles():
        if needle in name:
            raise SystemExit("refusing a sealed relative path")
    if name.startswith("test_") or "/test_" in name:
        raise SystemExit("refusing a sealed relative path")
    return name


def dev_input_name(lang: str) -> str:
    if lang not in LANGS:
        raise SystemExit(f"unknown language {lang}")
    return _check_relative(f"data/{lang}/dev_{lang}.jsonl")


def dev_gold_name(lang: str) -> str:
    if lang not in LANGS:
        raise SystemExit(f"unknown language {lang}")
    return _check_relative(f"data/evaluation/gold_labels/dev/gold_admin_{lang}.jsonl")


def prediction_name(lang: str) -> str:
    if lang not in LANGS:
        raise SystemExit(f"unknown language {lang}")
    # The official script appends this suffix when the prefix has no llm_predictions substring.
    return _check_relative(
        f"data/evaluation/encoder_predictions/dev/SpanDiff_admin_{lang}.jsonl.jsonl"
    )


def read_jsonl(path: Path) -> list[dict]:
    """JSON objects in file order, the way the evaluator's reader parses them.

    A blank line is refused. One leading BOM or record separator is stripped.
    Skipping a blank line would shift every later record.
    """
    assert_path_allowed(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if any(not line.strip() for line in lines):
        raise SystemExit(
            "refusing a blank line. The evaluator's jsonlines reader rejects it."
        )
    rows: list[dict] = []
    for line in lines:
        if line[:1] in ("\x1e", "\ufeff"):
            line = line[1:]
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                "refusing a line the evaluator's jsonlines reader would reject."
            ) from exc
        if not isinstance(obj, dict):
            raise SystemExit("refusing a line that is not a JSON object")
        if not isinstance(obj.get("id"), str) or not obj["id"]:
            raise SystemExit("refusing a record whose id is not a string")
        rows.append(obj)
    return rows


def load_ids(path: Path) -> list[str]:
    """Gold ids in file order."""
    ids: list[str] = []
    for obj in read_jsonl(path):
        if "id" not in obj:
            raise SystemExit("refusing a gold line that is not an id record")
        ids.append(obj["id"])
    return ids


def clear_partials(store: Path) -> None:
    """Drop interrupted downloads. A leftover would fail the store check."""
    if not store.is_dir():
        return
    for partial in store.rglob("*.partial"):
        partial.unlink()


def _download(url: str, dest: Path) -> None:
    assert_path_allowed(dest)
    for needle in needles():
        if needle in url:
            raise SystemExit("refusing a sealed URL")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "spandiff"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
        temporary.replace(dest)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if dest.name.endswith(".jsonl"):
        raw = dest.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]
        first = raw.splitlines()[0] if raw else b""
        if not first.startswith(b"{") or b'"id"' not in first:
            dest.unlink()
            raise SystemExit(
                f"refusing a download that is not a development JSONL record: {dest.name}"
            )
    _require_published_hash(dest)


def _published_digest(dest: Path) -> str | None:
    text = dest.as_posix()
    found: tuple[str, str] | None = None
    for suffix, digest in PUBLISHED_SHA256.items():
        if text.endswith(suffix) and (found is None or len(suffix) > len(found[0])):
            found = (suffix, digest)
    return None if found is None else found[1]


def _require_published_hash(dest: Path) -> None:
    expected = _published_digest(dest)
    if expected is None:
        return
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != expected:
        dest.unlink()
        raise SystemExit(
            f"refusing a download that does not match the published file: {dest.name}"
        )


def dev_store(repo: Path) -> Path:
    return repo / "track_2a" / "data" / "swissgov"


def local_dev_file(store: Path, lang: str) -> Path:
    if lang not in LANGS:
        raise SystemExit(f"unknown language {lang}")
    dest = store / lang / f"dev_{lang}.jsonl"
    assert_path_allowed(dest)
    _check_relative(dest.name)
    return dest


def assert_no_sealed_files(root: Path) -> None:
    """Fail when a sealed file is present, including an empty placeholder."""
    for path in root.rglob("*"):
        assert_path_allowed(path)


DEV_SOURCE_NOTE = "\n".join(
    [
        "ZurichNLP/SwissGov-RSD development split only. The sealed split is not included.",
        "Authors: Michelle Wastl, Jannis Vamvas, and Rico Sennrich.",
        "Source: https://huggingface.co/datasets/ZurichNLP/SwissGov-RSD",
        "Licence: CC-BY-4.0, https://creativecommons.org/licenses/by/4.0/",
        "These files keep that licence. They are not Apache-2.0 and not CDLA-Permissive-2.0.",
        "de/dev_de.jsonl",
        "fr/dev_fr.jsonl",
        "it/dev_it.jsonl",
        "Printed Spearman de 0,245; fr 0,165; it 0,282. Mean of those printed values: 0.2307. The script does not print the mean.",
        "",
    ]
)


def ensure_dev_inputs(store: Path) -> list[Path]:
    """Download only the three development jsonl files. Never a sealed sibling."""
    store.mkdir(parents=True, exist_ok=True)
    clear_partials(store)
    written: list[Path] = []
    for lang in LANGS:
        name = dev_input_name(lang)
        dest = local_dev_file(store, lang)
        if not dest.is_file() or dest.stat().st_size == 0:
            _download(HF_DATASET + name, dest)
        if dest.name != f"dev_{lang}.jsonl":
            raise SystemExit(f"refusing unexpected development filename: {dest.name}")
        written.append(dest)
    sources = store / "SOURCES.txt"
    sources.write_text(DEV_SOURCE_NOTE, encoding="utf-8")
    allowed = {path.name for path in written}
    allowed.add("SOURCES.txt")
    for path in store.rglob("*"):
        if path.is_file() and path.name not in allowed:
            raise SystemExit(f"refusing unexpected file in the development store: {path.name}")
    assert_no_sealed_files(store)
    return written


def ensure_gold(root: Path, lang: str) -> Path:
    name = dev_gold_name(lang)
    dest = root / name
    if not dest.is_file():
        _download(GOLD_BASE + name, dest)
    return dest


def iter_dev_records(lang: str, store: Path):
    """Read one local development file. The sealed split is not a fallback.

    A blank line is refused. One leading BOM or record separator is stripped.
    Skipping a blank line would drop that row and keep scoring the rest.
    """
    path = local_dev_file(store, lang)
    if not path.is_file():
        ensure_dev_inputs(store)
    yield from read_jsonl(path)


EVAL_FILES = (
    "list_to_drop.txt",
    "scripts/__init__.py",
    "scripts/evaluate_predictions_admin.py",
    "evaluation/__init__.py",
    "evaluation/utils.py",
    "evaluation/predictions.py",
    "rsd/__init__.py",
    "rsd/recognizers/__init__.py",
    "rsd/recognizers/utils.py",
)


def ensure_eval_kit(root: Path) -> None:
    """Download the official grader and the dev gold layout. No sealed filenames."""
    for name in EVAL_FILES:
        _check_relative(name)
        dest = root / name
        if dest.is_file():
            continue
        try:
            _download(GOLD_BASE + name, dest)
        except urllib.error.HTTPError as exc:
            if name.endswith("__init__.py") and exc.code == 404:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text("", encoding="utf-8")
                continue
            raise


def first_allowlisted(lang: str, allow_ids: set[str], store: Path) -> dict:
    for record in iter_dev_records(lang, store):
        if record.get("id") in allow_ids:
            return record
    raise SystemExit(f"no allow-listed development id in {lang}")
