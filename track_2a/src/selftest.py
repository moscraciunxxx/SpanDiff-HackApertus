#!/usr/bin/env python3
"""Invariants for the committed development grade. No document-wording locks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
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
    "de": "1098404f7ad70afeac8f39d42d36a5a726fd796ca4a297b056a14458f7206e68",
    "fr": "4672894bd3bb10dee8bf3ffe8b9d4ea16df4743cac272d51718a27d8f9324271",
    "it": "a689c9367e1e71c45b0cfe97d83444b063c7ad29b54f589399697f569b33659a",
}
L15_WORD_SHA = {
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


def _ids(folder: str, lang: str) -> list[str]:
    path = REPO / "track_2a" / "predictions" / folder / f"SpanDiff_admin_{lang}.jsonl.jsonl"
    return [json.loads(line)["id"] for line in _lines(path)]


def _check_id_partition() -> None:
    """predictions/test is the 56-id set inside 0..223 that is absent from development. predictions/full is ids 0..223. make run does not grade them."""
    for lang in LANGS:
        dev = _ids("dev", lang)
        test = _ids("test", lang)
        full = _ids("full", lang)
        if dev != _ids("l15", lang):
            raise SystemExit(f"{lang} development ids differ from the layer-15 copy")
        if dev != _ids("ablation_l8", lang) or dev != _ids("baseline_final_layer", lang):
            raise SystemExit(f"{lang} comparison files are not the development ids")
        if set(dev) & set(test):
            raise SystemExit(f"{lang} development ids overlap the test predictions")
        if set(dev) | set(test) != set(full):
            raise SystemExit(f"{lang} development plus test predictions is not the 224-id file")
        nums = [int(item.rsplit("_", 1)[-1]) for item in full]
        if nums != list(range(224)):
            raise SystemExit(f"{lang} 224-id file is not admin ids 0 through 223")
        l15_bytes = (
            REPO / "track_2a" / "predictions" / "l15" / f"SpanDiff_admin_{lang}.jsonl.jsonl"
        ).read_bytes()
        if hashlib.sha256(l15_bytes).hexdigest() != L15_WORD_SHA[lang]:
            raise SystemExit(f"{lang} predictions/l15 is not the frozen word-mean-pool L15 ablation")
    for rel in ("README.md", "track_2a/README.md", "track_2a/technical_report.md"):
        if "predictions/l15" not in (REPO / rel).read_text(encoding="utf-8"):
            raise SystemExit(f"{rel} does not retain predictions/l15 as an ablation row")


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
    if "primary grade on this machine is run-local" in makefile:
        raise SystemExit("Makefile calls make run-local the judge command")
    root_make = (REPO / "Makefile").read_text(encoding="utf-8")
    if "$(MAKE) -C track_2a run" not in root_make:
        raise SystemExit("root make run does not delegate")
    if "run-local" not in root_make:
        raise SystemExit("root make run-local is missing")
    if "--out track_2a/predictions/rerun" not in makefile or "--out track_2a/predictions/dev" in makefile:
        raise SystemExit("make predict writes into the graded development files")
    if "\npredict-test:\n\tpython3 src/predict_public_texts.py\n" not in makefile:
        raise SystemExit("make predict-test does not call the public-text scorer")
    if "$(MAKE) -C track_2a predict-test" not in root_make:
        raise SystemExit("root make predict-test does not delegate")
    dockerfile = (REPO / "track_2a" / "Dockerfile").read_text(encoding="utf-8")
    if 'CMD ["bash", "track_2a/docker_grade.sh"]' not in dockerfile:
        raise SystemExit("Docker entry point is not the development grade")
    if "--split " + "test" in dockerfile or "--split " + "full" in dockerfile:
        raise SystemExit("Dockerfile names a sealed split")
    image_script = REPO / "track_2a" / "run_image.sh"
    resolve = 'SCRIPT_DIR="$(cd "$(dirname "$1")" && pwd)"; ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"; printf "%s\\n" "$ROOT"'
    for script in (
        image_script,
        REPO / "track_2a" / "grade_dev.sh",
        REPO / "track_2a" / "docker_grade.sh",
    ):
        body = script.read_text(encoding="utf-8")
        if "SPAN_DIFF_PRINT_ROOT" in body:
            raise SystemExit(f"{script.name} can exit before it grades")
        for cwd in (Path("/tmp"), REPO, REPO / "track_2a"):
            found = subprocess.run(
                ["bash", "-c", resolve, "_", str(script)],
                check=False,
                capture_output=True,
                text=True,
                cwd=cwd,
            )
            if found.returncode != 0 or found.stdout.strip() != str(REPO):
                raise SystemExit(f"{script.name} resolved the wrong root from {cwd}: {found.stdout} {found.stderr}")
    image = image_script.read_text(encoding="utf-8")
    if "/tmp/SwissGov-RSD" in (REPO / "track_2a" / "grade_dev.sh").read_text(encoding="utf-8"):
        raise SystemExit("the grade can use a stray checkout outside the repository")
    if "docker is not on PATH" not in image:
        raise SystemExit("make run does not say the image was not run when docker is missing")
    if ".cache/huggingface" in image or "HF_TOKEN" in image:
        raise SystemExit("make run mounts a weight cache the grade does not use")
    entry = (REPO / "track_2a" / "docker_grade.sh").read_text(encoding="utf-8")
    if "grade_dev.sh" not in entry or "Does not load Apertus weights." not in entry:
        raise SystemExit("docker_grade.sh does not grade without loading weights")
    architecture = (REPO / "track_2a" / "docs" / "architecture.md").read_text(encoding="utf-8")
    if "make run-local grades the three 168-line files" not in architecture:
        raise SystemExit("architecture does not name make run-local as the grade")
    if "does not start Docker when a local interpreter" in architecture:
        raise SystemExit("architecture still describes the old Docker fallback")


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


def _check_printed_mean() -> None:
    """The primary score is the four-decimal mean of the three printed cells."""
    import run

    sample = "Spearman:\nde\tfr\tit\n0,373\t0,231\t0,398\n"
    if run.printed_spearman_mean(sample) != "0.3340":
        raise SystemExit("printed mean of the submission cells is not 0.3340")
    subword_l15 = "Spearman:\nde\tfr\tit\n0,309\t0,199\t0,333\n"
    if run.printed_spearman_mean(subword_l15) != "0.2803":
        raise SystemExit("printed mean of the unsmoothed subword-first L15 cells is not 0.2803")
    layer15_word = "Spearman:\nde\tfr\tit\n0,296\t0,191\t0,324\n"
    if run.printed_spearman_mean(layer15_word) != "0.2703":
        raise SystemExit("printed mean of the word-mean-pool L15 cells is not 0.2703")
    layer8 = "Spearman:\nde\tfr\tit\n0,245\t0,165\t0,282\n"
    final = "Spearman:\nde\tfr\tit\n0,053\t0,056\t0,106\n"
    if run.printed_spearman_mean(layer8) != "0.2307":
        raise SystemExit("printed mean of the hidden-state-8 cells is not 0.2307")
    if run.printed_spearman_mean(final) != "0.0717":
        raise SystemExit("printed mean of the final-layer cells is not 0.0717")
    saved_dir = REPO / "track_2a" / "docs"
    saved = (saved_dir / "official_dev_stdout_tier1_combo.txt").read_text(encoding="utf-8")
    if run.printed_spearman_mean(saved) != "0.3340":
        raise SystemExit("saved development stdout does not mean 0.3340")
    if "0.3340" in saved:
        raise SystemExit("saved development stdout already contains the mean the official script does not print")
    for name, expected in (
        ("official_dev_stdout_subword_l15.txt", "0.2803"),
        ("official_dev_stdout_L15.txt", "0.2703"),
        ("official_dev_stdout_l12_17.txt", "0.2613"),
        ("official_dev_stdout_combined.txt", "0.2717"),
        ("official_dev_stdout_qwen_comparison.txt", "0.3163"),
        ("official_dev_stdout.txt", "0.2307"),
        ("official_dev_stdout_last.txt", "0.0717"),
    ):
        got = run.printed_spearman_mean((saved_dir / name).read_text(encoding="utf-8"))
        if got != expected:
            raise SystemExit(f"{name} means {got}, not {expected}")
    for rel in (
        "README.md",
        "track_2a/README.md",
        "track_2a/docs/method.md",
        "track_2a/technical_report.md",
        "track_2a/docs/architecture.md",
    ):
        text = (REPO / rel).read_text(encoding="utf-8")
        if "0.3340" not in text or "0.2803" not in text or "0.2703" not in text:
            raise SystemExit(f"{rel} does not state 0.3340, 0.2803, and the 0.2703 L15 ablation")
        if "to 3 decimals" in text:
            raise SystemExit(f"{rel} still rounds the printed mean to 3 decimals")
    for rel in ("README.md", "track_2a/README.md", "track_2a/technical_report.md"):
        if "The official script does not print that mean." not in (REPO / rel).read_text(encoding="utf-8"):
            raise SystemExit(f"{rel} does not say the official script omits the mean")
    for rel in ("README.md", "track_2a/README.md"):
        readme = (REPO / rel).read_text(encoding="utf-8")
        first = readme.split("```bash", 1)[1].split("```", 1)[0]
        if first.strip() != "make run":
            raise SystemExit(f"{rel} does not lead with make run")
    root_readme = (REPO / "README.md").read_text(encoding="utf-8")
    if "does not replace `track_2a/predictions/dev`" not in root_readme:
        raise SystemExit("README lets make predict replace the graded files")
    for name in (
        "track_2a/docs/official_dev_stdout_tier1_combo.txt",
        "track_2a/docs/official_dev_stdout_subword_l15.txt",
        "track_2a/docs/official_dev_stdout_L15.txt",
        "track_2a/docs/official_dev_stdout_l12_17.txt",
        "track_2a/docs/official_dev_stdout_combined.txt",
        "track_2a/docs/official_dev_stdout_qwen_comparison.txt",
        "track_2a/docs/official_dev_stdout.txt",
        "track_2a/docs/official_dev_stdout_last.txt",
    ):
        if f"`{name}`" not in root_readme or not (REPO / name).is_file():
            raise SystemExit(f"README stdout path is missing: {name}")
    submitted = "We never graded the test split. make run grades the development files only."
    report = (REPO / "track_2a/technical_report.md").read_text(encoding="utf-8")
    if submitted not in report or "are not graded" in report:
        raise SystemExit("report still says the organizer files are not graded")
    baseline = "The UZH challenge asks for a SwissGov-paper baseline on this same Apertus model."
    for rel in ("track_2a/docs/method.md", "track_2a/technical_report.md"):
        if baseline not in (REPO / rel).read_text(encoding="utf-8"):
            raise SystemExit(f"{rel} does not name the paper baseline")
    phrase = (
        "Docker image verified under podman on linux/amd64 at an earlier commit; "
        "`make run-local` is the equivalent local path"
    )
    stale = "Docker is not installed on this Mac"
    if stale in root_readme or phrase not in root_readme:
        raise SystemExit("README still says Docker is not installed on this Mac")
    for rel in ("track_2a/README.md", "track_2a/technical_report.md"):
        text = (REPO / rel).read_text(encoding="utf-8")
        if stale in text or phrase not in text:
            raise SystemExit(f"{rel} still says Docker is not installed on this Mac")
    for name in ("track_2a/SpanDiff_Method.pdf", "track_2a/SpanDiff_Report.pdf"):
        blob = (REPO / name).read_bytes()
        if b"0.3340" not in blob or b"0.2803" not in blob or b"0.2703" not in blob:
            raise SystemExit(f"{name} does not contain the printed means 0.3340, 0.2803, and 0.2703")


def _check_prefix_labels() -> None:
    """The label check must read the prefix that will be graded, not a different directory."""
    import os
    import tempfile

    before = {
        lang: hashlib.sha256((PRED / f"SpanDiff_admin_{lang}.jsonl.jsonl").read_bytes()).hexdigest()
        for lang in LANGS
    }
    tmp = Path(tempfile.mkdtemp(prefix="spandiff-badlabel-"))
    try:
        row = json.dumps(
            {
                "id": "x",
                "text_a": "aa",
                "text_b": "bb",
                "labels_a": [1.1],
                "labels_b": [0.0],
            }
        )
        for lang in LANGS:
            path = tmp / f"SpanDiff_admin_{lang}.jsonl.jsonl"
            path.write_text((row + "\n") * 168, encoding="utf-8")
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
    combined = proc.stdout + proc.stderr
    if proc.returncode == 0 or "label outside 0..1" not in combined:
        raise SystemExit("a 168-line file with a label of 1.1 was not refused:\n" + combined)
    if "DEV TEST RESULTS" in combined:
        raise SystemExit("a label outside 0..1 reached the evaluator")
    after = {
        lang: hashlib.sha256((PRED / f"SpanDiff_admin_{lang}.jsonl.jsonl").read_bytes()).hexdigest()
        for lang in LANGS
    }
    if after != before:
        raise SystemExit("the bad-label refusal changed a graded file")


def _check_nondev_prefix() -> None:
    """Ablation files match the gold texts, so only predictions/dev may be graded."""
    env = os.environ.copy()
    env["SPAN_DIFF_PRED_PREFIX"] = str(
        REPO / "track_2a" / "predictions" / "ablation_l8" / "SpanDiff_admin_"
    )
    proc = subprocess.run(
        ["bash", str(REPO / "track_2a" / "grade_dev.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    combined = proc.stdout + proc.stderr
    if proc.returncode == 0 or "not the development grade" not in combined:
        raise SystemExit("the hidden-state-8 prefix was not refused:\n" + combined)
    if "DEV TEST RESULTS" in combined:
        raise SystemExit("the hidden-state-8 prefix reached the evaluator")


def _check_gold_order() -> None:
    """A 168-line file in the wrong order must not reach the evaluator."""
    import shutil
    import tempfile

    before = {
        lang: hashlib.sha256((PRED / f"SpanDiff_admin_{lang}.jsonl.jsonl").read_bytes()).hexdigest()
        for lang in LANGS
    }
    tmp = Path(tempfile.mkdtemp(prefix="spandiff-order-"))
    try:
        for lang in LANGS:
            src = PRED / f"SpanDiff_admin_{lang}.jsonl.jsonl"
            dest = tmp / src.name
            shutil.copyfile(src, dest)
        de = tmp / "SpanDiff_admin_de.jsonl.jsonl"
        rows = de.read_text(encoding="utf-8").splitlines()
        rows[0], rows[1] = rows[1], rows[0]
        de.write_text("\n".join(rows) + "\n", encoding="utf-8")
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
    combined = proc.stdout + proc.stderr
    if proc.returncode == 0 or "does not follow gold" not in combined:
        raise SystemExit("a reordered development file was not refused:\n" + combined)
    if "DEV TEST RESULTS" in combined:
        raise SystemExit("a reordered file reached the evaluator")
    after = {
        lang: hashlib.sha256((PRED / f"SpanDiff_admin_{lang}.jsonl.jsonl").read_bytes()).hexdigest()
        for lang in LANGS
    }
    if after != before:
        raise SystemExit("the order refusal changed a graded file")


def _check_blank_inits() -> None:
    """A fresh grader checkout imports an API helper and a missing recognizer."""
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="spandiff-init-"))
    scripts = tmp / "scripts"
    recognizers = tmp / "rsd" / "recognizers"
    try:
        scripts.mkdir(parents=True)
        recognizers.mkdir(parents=True)
        (scripts / "__init__.py").write_text(
            "from .project_labels import project_labels\n",
            encoding="utf-8",
        )
        (recognizers / "__init__.py").write_text(
            "from rsd.recognizers.diff_align import DiffAlign\n",
            encoding="utf-8",
        )
        proc = subprocess.run(
            [sys.executable, str(REPO / "track_2a" / "src" / "run.py"), "--blank-inits", str(tmp)],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise SystemExit("blank-inits failed:\n" + proc.stderr)
        if (scripts / "__init__.py").read_text(encoding="utf-8") != "":
            raise SystemExit("the published scripts init was not blanked")
        if (recognizers / "__init__.py").read_text(encoding="utf-8") != "":
            raise SystemExit("a missing diff_align init was not blanked")
        (recognizers / "__init__.py").write_text(
            "from rsd.recognizers.diff_align import DiffAlign\n",
            encoding="utf-8",
        )
        (recognizers / "diff_align.py").write_text("x = 1\n", encoding="utf-8")
        kept = subprocess.run(
            [sys.executable, str(REPO / "track_2a" / "src" / "run.py"), "--blank-inits", str(tmp)],
            check=False,
            capture_output=True,
            text=True,
        )
        if kept.returncode != 0 or "DiffAlign" not in (recognizers / "__init__.py").read_text(encoding="utf-8"):
            raise SystemExit("a full checkout lost its recognizer init")
    finally:
        shutil.rmtree(tmp)


def _check_default_entry() -> None:
    """python run.py must not load Apertus when the committed development files exist."""
    proc = subprocess.run(
        [sys.executable, str(REPO / "track_2a" / "src" / "run.py")],
        check=False,
        capture_output=True,
        text=True,
    )
    combined = proc.stdout + proc.stderr
    if proc.returncode != 0 or "The model is not loaded." not in proc.stdout:
        raise SystemExit("the default entry did not keep the model unloaded:\n" + combined)
    if "Loading the model" in combined:
        raise SystemExit("the default entry started a model load")
    full = subprocess.run(
        [sys.executable, str(REPO / "track_2a" / "src" / "run.py"), "--full"],
        check=False,
        capture_output=True,
        text=True,
    )
    full_text = full.stdout + full.stderr
    if full.returncode != 0 or "Refusing to replace them." not in full.stdout:
        raise SystemExit("a full run did not refuse the committed development files:\n" + full_text)
    if "Loading the model" in full_text:
        raise SystemExit("a full run started a model load")


def _check_ship_notes() -> None:
    for rel in (
        "track_2a/docs/layer_map_by_lang_avg.txt",
        "track_2a/docs/layer_map_by_lang.txt",
        "track_2a/docs/layer_map_dev.txt",
    ):
        body = (REPO / rel).read_text(encoding="utf-8")
        if "The submitted files are hidden state 15." not in body:
            raise SystemExit(f"{rel} still calls hidden state 8 the graded files")
    report_path = REPO / "track_2a" / "technical_report.md"
    report = report_path.read_text(encoding="utf-8")
    for image in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report):
        if not (report_path.parent / image).is_file():
            raise SystemExit(f"report image is missing: {image}")
    if "0.0484, 0.0564, and 0.0945" not in report:
        raise SystemExit("report drops the Italian final-state value")
    if "Those files are not copied into this repository." not in report:
        raise SystemExit("report implies the train and validation gold is in this repository")
    if "Together, train and validation are the 168 development documents, and they share no id." not in report:
        raise SystemExit("report does not say train and validation partition the development set")
    expected_rows = {
        "de train": (134, 134, 15, 0.1623, 0.1899, 0.0276),
        "de val": (33, 34, 15, 0.1636, 0.2003, 0.0367),
        "fr train": (132, 134, 15, 0.0430, 0.0798, 0.0367),
        "fr val": (33, 34, 15, 0.0461, 0.0801, 0.0340),
        "it train": (133, 134, 14, 0.1509, 0.1903, 0.0394),
        "it val": (33, 34, 15, 0.1796, 0.2066, 0.0270),
    }
    for line in report.splitlines():
        if not line.startswith("| ") or line.startswith("| Slice") or line.startswith("| ---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 6 or cells[0] not in expected_rows:
            continue
        used, total = (int(part) for part in cells[1].split("/"))
        best = int(cells[2])
        layer8 = float(cells[3])
        layer15 = float(cells[4])
        delta = float(cells[5])
        want = expected_rows.pop(cells[0])
        if (used, total, best, layer8, layer15, delta) != want:
            raise SystemExit(f"layer-selection row {cells[0]} changed: {cells[1:]}")
        if abs((layer15 - layer8) - delta) > 0.00015:
            raise SystemExit(f"layer-selection delta does not match {cells[0]}")
    if expected_rows:
        raise SystemExit(f"layer-selection table is missing {sorted(expected_rows)}")
    pooling = (REPO / "track_2a" / "docs" / "pooling_dev.txt").read_text(encoding="utf-8")
    if "submitted print is hidden state 15" not in pooling:
        raise SystemExit("pooling notes still treat hidden state 8 as the submission")
    pipeline = (REPO / "track_2a" / "docs" / "pipeline_check.txt").read_text(encoding="utf-8")
    if "0,373" not in pipeline or "0.3340" not in pipeline or "0.2803" not in pipeline or "0.2703" not in pipeline:
        raise SystemExit("pipeline check does not quote 0.3340, 0.2803, and the 0.2703 ablation")
    if "Hidden state 8 ablation" not in pipeline:
        raise SystemExit("pipeline check does not label the hidden-state-8 result as an ablation")
    sources = (REPO / "track_2a" / "data" / "swissgov" / "SOURCES.txt").read_text(encoding="utf-8")
    if "0.3340" not in sources or "0.2803" not in sources or "0.2703" not in sources or "not CDLA-Permissive-2.0" not in sources:
        raise SystemExit("SOURCES.txt does not keep the SwissGov licence and the submitted mean")
    if "Submitted development print, hidden state 15" not in sources:
        raise SystemExit("SOURCES.txt still quotes hidden state 8 as the submission")
    ignore = (REPO / ".dockerignore").read_text(encoding="utf-8")
    if "private/" not in ignore or ".swissgov/" not in ignore:
        raise SystemExit("Docker build context includes private notes or a local grader checkout")
    script = (REPO / "track_2a" / "docs" / "DEMO_SCRIPT.md").read_text(encoding="utf-8")
    spoken = script.split("## Say", 1)[1]
    words = spoken.split()
    if len(words) > 300:
        raise SystemExit(f"demo script is {len(words)} words")
    if "0.3340" not in spoken or "0,373" not in spoken or "0.2803" not in spoken or "0.2703" not in spoken:
        raise SystemExit("demo script does not speak the printed development score")
    if "Judges run make run." not in spoken:
        raise SystemExit("demo script does not speak the judge command")
    for rel in (
        "track_2a/docs/layer_curve_six.txt",
        "track_2a/docs/sentence_probe.txt",
        "track_2a/docs/bitext_map.txt",
    ):
        note = (REPO / rel).read_text(encoding="utf-8")
        if "Graded-file average-rank" in note or "Graded average-rank" in note:
            raise SystemExit(f"{rel} calls the ablation the graded file")
        if "not the submitted files" not in note:
            raise SystemExit(f"{rel} does not say those page scores are not the submission")
    if "SwissGov paper baseline" not in spoken:
        raise SystemExit("demo script does not name the paper baseline")
    if "gold" + ".ac.uk" in script or "@" in script:
        raise SystemExit("demo script contains an email")
    for rel in (
        "README.md",
        "track_2a/README.md",
        "track_2a/docs/method.md",
        "track_2a/technical_report.md",
    ):
        text = (REPO / rel).read_text(encoding="utf-8")
        for name in ("LLM_NAME", "LLM_BASE_URL", "LLM_API_KEY"):
            if name not in text:
                raise SystemExit(f"{rel} does not name {name}")


def _check_leakage() -> None:
    hits = scan(REPO)
    if hits:
        raise SystemExit("leakage fence failed:\n" + "\n".join(hits))


def main() -> int:
    _check_predictions()
    _check_id_partition()
    _check_command()
    _check_printed_mean()
    _check_prefix_labels()
    _check_nondev_prefix()
    _check_gold_order()
    _check_blank_inits()
    _check_default_entry()
    _check_ship_notes()
    _check_leakage()
    _check_selectors()
    print("self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
