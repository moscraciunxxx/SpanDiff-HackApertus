#!/bin/bash
# Grade the committed development predictions. Does not load model weights.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Committed predictions: track_2a/predictions/dev/SpanDiff_admin_
PRED_DIR="${ROOT}/track_2a/predictions/dev"
PREFIX="${SPAN_DIFF_PRED_PREFIX:-${PRED_DIR}/SpanDiff_admin_}"

case "$PREFIX" in
  *llm*|*Zurich*)
    echo "refusing: prediction path would change how the evaluator scores" >&2
    exit 1
    ;;
  */predictions/dev/*)
    ;;
  */predictions/*)
    echo "refusing: that prefix is not the development grade" >&2
    exit 1
    ;;
esac

for lang in de fr it; do
  file="${PREFIX}${lang}.jsonl.jsonl"
  if [[ ! -f "$file" ]]; then
    echo "refusing: ${file} is missing" >&2
    exit 1
  fi
  if grep -q '^[[:space:]]*$' "$file"; then
    echo "refusing: ${file} has a blank line. The evaluator's jsonlines reader rejects it." >&2
    exit 1
  fi
  lines="$(grep -c '^' "$file" || true)"
  if [ "$lines" != "168" ]; then
    echo "refusing: ${file} has ${lines} lines, need 168" >&2
    exit 1
  fi
done

python3 - "$PREFIX" << 'PY'
import json
import math
import sys
from pathlib import Path

prefix = sys.argv[1]
keys = {"id", "text_a", "text_b", "labels_a", "labels_b"}
for lang in ("de", "fr", "it"):
    path = Path(f"{prefix}{lang}.jsonl.jsonl")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            raise SystemExit(f"blank line in {path.name}")
        obj = json.loads(line)
        if set(obj) != keys:
            raise SystemExit(f"encoder keys in {path.name}")
        for side in ("a", "b"):
            labels = obj[f"labels_{side}"]
            if len(labels) != len(str(obj[f"text_{side}"]).split()):
                raise SystemExit(f"label count in {path.name} {obj.get('id')}")
            for value in labels:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise SystemExit("label type")
                number = float(value)
                if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                    raise SystemExit("label outside 0..1")
PY

if [[ ! -f "${ROOT}/.swissgov/scripts/evaluate_predictions_admin.py" ]]; then
  (
    cd "$ROOT"
    python3 - << 'PY'
import sys
from pathlib import Path

sys.path.insert(0, "track_2a/src")
from spandiff.devdata import LANGS, ensure_eval_kit, ensure_gold

root = Path(".swissgov")
root.mkdir(parents=True, exist_ok=True)
ensure_eval_kit(root)
for lang in LANGS:
    ensure_gold(root, lang)
PY
  )
fi

checkout=""
for candidate in "${ROOT}/.swissgov"; do
  script="${candidate}/scripts/evaluate_predictions_admin.py"
  if [[ ! -f "$script" || ! -f "${candidate}/list_to_drop.txt" ]]; then
    continue
  fi
  if grep -q "prediction file absent" "$script"; then
    continue
  fi
  checkout="$candidate"
  break
done

if [[ -z "$checkout" ]]; then
  echo "refusing: no unmodified SwissGov-RSD checkout was found" >&2
  exit 1
fi

python3 - "$PREFIX" "$checkout" << 'PY'
import json
import sys
from pathlib import Path

prefix, checkout = sys.argv[1], Path(sys.argv[2])
for lang in ("de", "fr", "it"):
    pred = Path(f"{prefix}{lang}.jsonl.jsonl")
    gold = checkout / "data" / "evaluation" / "gold_labels" / "dev" / f"gold_admin_{lang}.jsonl"
    if not gold.is_file():
        raise SystemExit(f"refusing: gold file missing for {lang}")
    def load_rows(path: Path) -> list[dict]:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line[:1] in ("\ufeff", "\x1e"):
                line = line[1:]
            if not line.strip():
                raise SystemExit(f"refusing: {path.name} has a blank line")
            rows.append(json.loads(line))
        return rows

    preds = load_rows(pred)
    golds = load_rows(gold)
    if len(preds) != len(golds):
        raise SystemExit(f"refusing: {pred.name} does not match the gold line count")
    for got, expected in zip(preds, golds):
        if (
            got.get("id") != expected.get("id")
            or got.get("text_a") != expected.get("text_a")
            or got.get("text_b") != expected.get("text_b")
        ):
            raise SystemExit(f"refusing: {pred.name} does not follow gold at {got.get('id')}")
PY

if ! PYTHON="$(bash "${ROOT}/track_2a/find_grader.sh")"; then
  echo "refusing: no interpreter can import the evaluator (nlpstats, numpy, jsonlines, torch, tokenizers, transformers). The model is not loaded." >&2
  exit 1
fi

cd "$checkout"
"$PYTHON" "${ROOT}/track_2a/src/run.py" --blank-inits "$checkout"
stdout_file="$(mktemp)"
echo "$PYTHON -m scripts.evaluate_predictions_admin ${PREFIX} --split dev"
"$PYTHON" -m scripts.evaluate_predictions_admin "$PREFIX" --split dev | tee "$stdout_file"
"$PYTHON" "${ROOT}/track_2a/src/run.py" --primary-mean "$stdout_file"
rm -f "$stdout_file"
