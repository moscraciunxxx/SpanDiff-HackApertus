#!/usr/bin/env python3
"""SpanDiff entrypoint. The default run is a one-document probe, not the full dev set."""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
# parents[0] is track_2a. parents[1] is the repository root, which holds .swissgov.
REPO = SRC.parents[1]

from spandiff import GRADED_COMMAND, MODEL_ID  # noqa: E402
from spandiff.align import (  # noqa: E402
    cosine,
    difference,
    diff_against,
    locate_tokens,
    mean_pool,
    pack_spans,
)
from spandiff.devdata import (  # noqa: E402
    LANGS,
    assert_no_sealed_files,
    dev_store,
    ensure_dev_inputs,
    ensure_eval_kit,
    ensure_gold,
    first_allowlisted,
    iter_dev_records,
    load_ids,
    local_dev_file,
    prediction_name,
    read_jsonl,
)
from spandiff.gate import probe_config  # noqa: E402
from spandiff.leakage import assert_path_allowed, needles, scan  # noqa: E402
from spandiff.report_pdf import write_method, write_report  # noqa: E402
from spandiff.schema import REQUIRED, check_jsonl, check_object, dumps_encoder  # noqa: E402
from spandiff.study_guard import refuse_graded_load  # noqa: E402
from spandiff.score import dumps_line, score_record  # noqa: E402


def repo_root() -> Path:
    return REPO


def _self_test() -> int:
    """Schema, hashes, leakage, and grader-selector agreement. No README locks."""
    from selftest import main as run_self_test

    return run_self_test()


def _note_weight_gate(error: str) -> None:
    """Record a failed weight check without erasing the accepted-run note."""
    gate_path = REPO / "track_2a" / "docs" / "weight_gate.txt"
    prior = gate_path.read_text(encoding="utf-8") if gate_path.is_file() else ""
    block = error.strip()
    if block and block in prior:
        return
    gate_path.write_text(prior.rstrip() + "\n\nLater check:\n" + block + "\n", encoding="utf-8")


def _has_prediction(swissgov: Path) -> bool:
    return any((swissgov / prediction_name(lang)).is_file() for lang in LANGS)


def _run_action(complete: bool, has_file: bool) -> str:
    """grade, refuse, or score. A short file is not a prefix to continue."""
    if complete:
        return "grade"
    if has_file:
        return "refuse"
    return "score"


def _after_probe(complete: bool, has_file: bool) -> str:
    """What follows a probe that returned 0. A partial set is not called absent."""
    if complete:
        return "grade"
    if has_file:
        return "refuse"
    return "absent"


def _probe() -> int:
    hits = scan(REPO)
    if hits:
        print("leakage fence failed:", file=sys.stderr)
        print("\n".join(hits), file=sys.stderr)
        return 1
    swissgov = REPO / ".swissgov"
    action = _run_action(
        swissgov.is_dir() and _complete(swissgov),
        swissgov.is_dir() and _has_prediction(swissgov),
    )
    if action == "grade":
        print("Development prediction files are complete. The model is not loaded.")
        return 0
    if action == "refuse":
        print(
            "A prediction file is present but is not the 168-line development set. "
            "The model is not loaded."
        )
        print("Official dev stdout: not yet printed")
        return 1
    loaded, error = probe_config()
    if not loaded:
        _note_weight_gate(error)
        print("Weights did not load.")
        print(error)
        print("One-document probe did not run a forward pass.")
        print("Official dev stdout: not yet printed")
        return 1

    print(f"config.json for {MODEL_ID} is readable. Loading the model for one development document.")
    try:
        from spandiff.score import load_apertus
    except SystemExit as exc:
        print(exc)
        print("Official dev stdout: not yet printed")
        return 1
    try:
        model, tokenizer, device, max_length = load_apertus()
    except SystemExit as exc:
        print(exc)
        print("Official dev stdout: not yet printed")
        return 1

    swissgov = REPO / ".swissgov"
    allow = set(load_ids(ensure_gold(swissgov, "de")))
    record = first_allowlisted("de", allow, dev_store(REPO))
    if record["id"] not in allow:
        raise SystemExit("refusing an id that is not on the dev gold allow-list")
    line, input_tokens = score_record(model, tokenizer, record, device, max_length)
    probe_dir = REPO / "track_2a" / "data" / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    out = probe_dir / "one_dev_document.jsonl"
    out.write_text(line + "\n", encoding="utf-8")
    check_jsonl(out)
    print(f"Scored one development document: {record['id']}")
    print(f"Forward-pass input tokens: {input_tokens}")
    print("Generated tokens: 0")
    print("Stopped after one document. Official dev stdout: not yet printed")
    return 0


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _prediction_ids(path: Path) -> list[str]:
    """Ids in file order. Invalid JSON is an empty list, not a traceback.

    A leading BOM or record separator is stripped, matching jsonlines.
    An empty list is not a complete development file, so the model is not loaded.
    """
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if any(not line.strip() for line in lines):
        raise SystemExit(
            "refusing a blank line. The evaluator's jsonlines reader rejects it."
        )
    found: list[str] = []
    for line in lines:
        if line[:1] in ("\x1e", "\ufeff"):
            line = line[1:]
        try:
            obj = json.loads(line)
            found.append(obj["id"])
        except (json.JSONDecodeError, TypeError, KeyError, IndexError):
            return []
    return found


def _complete(swissgov: Path) -> bool:
    for lang in LANGS:
        dest = swissgov / prediction_name(lang)
        ids = load_ids(ensure_gold(swissgov, lang))
        if _prediction_ids(dest) != ids:
            return False
    return True


def _grade_ready(swissgov: Path) -> bool:
    return _complete(swissgov)


def _blank_eval_package_init(swissgov: Path) -> None:
    """Keep the grade from importing an API client or a missing recognizer.

    scripts/__init__.py is blanked whenever it imports project_labels, even if
    that helper file is present. The recognizer init is blanked only when
    diff_align.py was not downloaded. A full checkout that has that file keeps
    its recognizer init. The evaluator file stays as published.
    """
    init = swissgov / "scripts" / "__init__.py"
    if init.is_file() and "project_labels" in init.read_text(encoding="utf-8"):
        init.write_text("", encoding="utf-8")
    recognizers = swissgov / "rsd" / "recognizers" / "__init__.py"
    if (
        recognizers.is_file()
        and "diff_align" in recognizers.read_text(encoding="utf-8")
        and not (swissgov / "rsd" / "recognizers" / "diff_align.py").is_file()
    ):
        recognizers.write_text("", encoding="utf-8")


def _commit_checkpoint(counts: dict[str, int]) -> None:
    """A scoring run does not write a commit."""
    print(
        "checkpoint skipped: de "
        f"{counts['de']}/168, fr {counts['fr']}/168, it {counts['it']}/168.",
        flush=True,
    )


def _keep_valid_prefix(path: Path, ids: list[str], records: dict[str, dict]) -> int:
    """Keep lines whose ids and texts match gold order.

    This does not check that the labels are hidden state 8. --probe and
    --full return before calling it when a prediction file exists and the
    three files are not the complete development set.
    """
    if not path.is_file():
        return 0
    text = path.read_text(encoding="utf-8")
    raw_lines = text.splitlines()
    if any(not line.strip() for line in raw_lines):
        raise SystemExit(
            "refusing a blank line. The evaluator's jsonlines reader rejects it."
        )
    kept: list[str] = []
    for raw in raw_lines:
        if len(kept) >= len(ids):
            break
        line = raw[1:] if raw[:1] in ("\x1e", "\ufeff") else raw
        try:
            obj = json.loads(line)
            check_object(obj)
        except (json.JSONDecodeError, ValueError):
            break
        if set(obj) != set(REQUIRED):
            break
        item_id = ids[len(kept)]
        record = records[item_id]
        if obj["id"] != item_id or obj["text_a"] != record["text_a"] or obj["text_b"] != record["text_b"]:
            break
        kept.append(raw)
    if kept == raw_lines and path.read_text(encoding="utf-8").endswith("\n"):
        return len(kept)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = ("\n".join(kept) + "\n") if kept else ""
    with path.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return len(kept)


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _seed_scored_document(path: Path, ids: list[str], records: dict[str, dict]) -> None:
    """Do not copy the one-document probe into a prediction file.

    On admin_de_0 the probe matches the last-block labels on both sides.
    Copying it would start a file with those scores.
    """
    del path, ids, records


def _comma(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _record_official_stdout(stdout: str) -> None:
    """Keep saved stdout files unchanged. A scoring run must not edit them."""
    del stdout


_EVALUATOR_IMPORT = "import nlpstats, numpy, jsonlines, torch, tokenizers, transformers"


def _grader_python() -> str:
    """Absolute path of a PATH interpreter that can import the evaluator."""
    seen: list[str] = []
    for candidate in (
        sys.executable,
        shutil.which("python3") or "",
        shutil.which("python") or "",
    ):
        if not candidate or candidate in seen:
            continue
        seen.append(candidate)
        if not os.path.isfile(candidate):
            continue
        probe = subprocess.run(
            [candidate, "-c", _EVALUATOR_IMPORT + "; import sys; print(sys.executable)"],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            lines = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
            if lines:
                return lines[-1]
    raise SystemExit(
        "refusing: no interpreter can import the evaluator "
        "(nlpstats, numpy, jsonlines, torch, tokenizers, transformers). The model is not loaded."
    )


def _record_lines(text: str) -> list[str]:
    """Lines the evaluator will try to parse.

    ``str.splitlines`` keeps a final record that has no newline.
    jsonlines does not skip a blank line unless ``skip_empty`` is set, and
    the evaluator leaves that default off, so a blank line is refused here.
    """
    lines = text.splitlines()
    if any(not line.strip() for line in lines):
        raise SystemExit(
            "refusing a blank line. The evaluator's jsonlines reader rejects it."
        )
    return lines


def _json_line(line: str) -> dict:
    """Parse one record the way jsonlines does, including a leading BOM."""
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
    return obj


def _assert_graded_files(swissgov: Path, predictions: Path | None = None) -> None:
    """Refuse a file the evaluator would mis-pair or send toward the full split.

    Gold is read from `swissgov`. Prediction files are read from `predictions`
    when the evaluator checkout is not the tree that holds those files.
    """
    pred_root = predictions or swissgov
    for lang in LANGS:
        path = pred_root / prediction_name(lang)
        if not path.is_file():
            raise SystemExit(f"refusing to grade: {path.name} is missing")
        lines = _record_lines(path.read_text(encoding="utf-8"))
        if len(lines) != 168:
            raise SystemExit(
                f"refusing to grade: {path.name} has {len(lines)} lines, need 168. "
                "A count mismatch makes the evaluator look for the full gold split."
            )
        gold_rows = [
            _json_line(line)
            for line in _record_lines(ensure_gold(swissgov, lang).read_text(encoding="utf-8"))
        ]
        if len(gold_rows) != len(lines):
            raise SystemExit(f"refusing to grade: {path.name} does not match the gold line count")
        for raw, gold in zip(lines, gold_rows):
            obj = _json_line(raw)
            check_object(obj)
            if set(obj) != set(REQUIRED):
                raise SystemExit(f"refusing to grade: {path.name} {obj.get('id')} has extra keys")
            if (
                obj["id"] != gold["id"]
                or obj["text_a"] != gold["text_a"]
                or obj["text_b"] != gold["text_b"]
            ):
                raise SystemExit(f"refusing to grade: {path.name} does not follow gold at {obj.get('id')}")
        backup = path.parent / "last_layer" / path.name
        if backup.is_file() and filecmp.cmp(path, backup, shallow=False):
            raise SystemExit(f"refusing to grade: {path.name} is the last-block backup")
        assert_path_allowed(path)


def printed_spearman_values(stdout: str) -> list[str] | None:
    """The three printed Spearman cells. Kendall is not this row."""
    lines = stdout.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "Spearman:":
            continue
        cells = [item.strip() for item in lines[index + 1 : index + 4] if item.strip()]
        if len(cells) < 2 or cells[0].split() != ["de", "fr", "it"]:
            return None
        values = cells[1].split()
        if len(values) != 3:
            return None
        return values
    return None


def printed_spearman_mean(stdout: str) -> str | None:
    """Mean of the three printed Spearman cells, to four decimals."""
    values = printed_spearman_values(stdout)
    if values is None:
        return None
    try:
        numbers = [float(cell.replace(",", ".")) for cell in values]
    except ValueError:
        return None
    return f"{sum(numbers) / 3:.4f}"


def _graded_run(swissgov: Path) -> int:
    _assert_graded_files(swissgov)
    ensure_eval_kit(swissgov)
    _blank_eval_package_init(swissgov)
    script = (swissgov / "scripts" / "evaluate_predictions_admin.py").read_text(encoding="utf-8")
    if "prediction file absent" in script or "scored ids" in script:
        raise SystemExit("refusing a modified evaluator")
    print(GRADED_COMMAND, flush=True)
    completed = subprocess.run(
        [
            _grader_python(),
            "-m",
            "scripts.evaluate_predictions_admin",
            str((REPO / "track_2a" / "predictions" / "dev" / "SpanDiff_admin_").resolve()),
            "--split",
            "dev",
        ],
        cwd=swissgov,
        check=False,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode == 0:
        _record_official_stdout(completed.stdout)
        mean = printed_spearman_mean(completed.stdout)
        if mean:
            print(
                f"Primary score, mean of those three printed Spearmans: {mean}. "
                "The script does not print it.",
                flush=True,
            )
    else:
        print("Official dev stdout: not yet printed", flush=True)
    return completed.returncode


def _full() -> int:
    hits = scan(REPO)
    if hits:
        print("\n".join(hits), file=sys.stderr)
        return 1
    swissgov = REPO / ".swissgov"
    action = _run_action(
        swissgov.is_dir() and _complete(swissgov),
        swissgov.is_dir() and _has_prediction(swissgov),
    )
    if action == "grade":
        print("Development prediction files are already complete. Refusing to replace them.", flush=True)
        return _graded_run(swissgov)
    if action == "refuse":
        print(
            "A prediction file is present but is not the 168-line development set. "
            "The model is not loaded.",
            flush=True,
        )
        print("Official dev stdout: not yet printed", flush=True)
        return 1
    loaded, error = probe_config()
    if not loaded:
        _note_weight_gate(error)
        print(error)
        print("Official dev stdout: not yet printed")
        return 1
    swissgov = REPO / ".swissgov"
    ensure_eval_kit(swissgov)
    _blank_eval_package_init(swissgov)
    pid_path = swissgov / "dev_run.pid"
    if pid_path.is_file():
        try:
            old = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            old = 0
        if old and old != os.getpid() and _pid_alive(old):
            print(f"already running: {old}", flush=True)
            return 0
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    from spandiff.score import load_apertus

    model, tokenizer, device, max_length = load_apertus()
    counts = {lang: 0 for lang in LANGS}
    forward_tokens = 0
    scoring_seconds = 0.0
    for lang in LANGS:
        ids = load_ids(ensure_gold(swissgov, lang))
        if len(ids) != 168:
            raise SystemExit(f"{lang}: expected 168 development ids, found {len(ids)}")
        allow = set(ids)
        found: dict[str, dict] = {}
        for record in iter_dev_records(lang, dev_store(REPO)):
            item_id = record.get("id")
            if item_id in allow:
                found[item_id] = record
        missing = [item for item in ids if item not in found]
        if missing:
            raise SystemExit(
                f"{lang}: {len(missing)} dev-gold ids are absent from the dev input. "
                "Refusing to write a partial prediction file."
            )
        dest = swissgov / prediction_name(lang)
        assert_path_allowed(dest)
        if lang == "de":
            _seed_scored_document(dest, ids, found)
        done = _keep_valid_prefix(dest, ids, found)
        counts[lang] = done
        if done == 1 or (done and done % 10 == 0):
            _commit_checkpoint(counts)
        for item_id in ids[done:]:
            started = time.perf_counter()
            line, tokens = score_record(model, tokenizer, found[item_id], device, max_length)
            _append_line(dest, line)
            counts[lang] += 1
            forward_tokens += tokens
            elapsed = time.perf_counter() - started
            scoring_seconds += elapsed
            print(f"scored {item_id} tokens {tokens} seconds {elapsed:.1f}", flush=True)
            if counts[lang] == 1 or counts[lang] % 10 == 0:
                _commit_checkpoint(counts)
        check_jsonl(dest)
        _commit_checkpoint(counts)
    if forward_tokens:
        (REPO / "track_2a" / "docs" / "forward_pass.txt").write_text(
            "Documents already on disk are not counted again.\n"
            f"forward_pass_tokens {forward_tokens}\n"
            f"scoring_seconds {scoring_seconds:.1f}\n",
            encoding="utf-8",
        )
    if not _complete(swissgov):
        print("Official dev stdout: not yet printed", flush=True)
        return 0
    code = _graded_run(swissgov)
    _commit_checkpoint({lang: 168 for lang in LANGS})
    return code


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--checkout")
    parser.add_argument("--predictions")
    parser.add_argument("--blank-inits")
    parser.add_argument("--primary-mean")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Score every development id, checkpoint each document, then run the official development command.",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()
    if args.blank_inits:
        _blank_eval_package_init(Path(args.blank_inits))
        return 0
    if args.primary_mean:
        mean = printed_spearman_mean(Path(args.primary_mean).read_text(encoding="utf-8"))
        if mean:
            print(
                f"Primary score, mean of those three printed Spearmans: {mean}. "
                "The script does not print it."
            )
        return 0
    if args.check_files:
        _assert_graded_files(
            Path(args.checkout) if args.checkout else REPO / ".swissgov",
            Path(args.predictions) if args.predictions else None,
        )
        print("graded files match development gold")
        return 0
    if args.full:
        return _full()
    code = _probe()
    if code != 0:
        return code
    swissgov = REPO / ".swissgov"
    follow = _after_probe(
        swissgov.is_dir() and _grade_ready(swissgov),
        swissgov.is_dir() and _has_prediction(swissgov),
    )
    if follow == "grade":
        return _graded_run(swissgov)
    if follow == "refuse":
        return 1
    print("Prediction files are absent. Official dev stdout: not yet printed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
