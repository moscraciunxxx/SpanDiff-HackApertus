#!/usr/bin/env python3
"""Sprint 3 ideas 6, 8, and 9. Development gold only. No new forward pass."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[1]
sys.path.insert(0, str(SRC))

from spandiff.score import PUNCT, _span_smooth_tri  # noqa: E402

LANGS = ("de", "fr", "it")
GOLD = Path(os.environ.get("SPAN_DIFF_DEV_GOLD", ""))
BONUS = 0.15
DAMP = 0.5
STOP = {
    "en": "the a an of and to in for on with at by from as or".split(),
    "de": "der die das den dem des ein eine und zu von mit auf für im in ist".split(),
    "fr": "le la les de des du un une et à au aux en dans pour par sur est".split(),
    "it": "il lo la i gli le di del della dei e a in per con su è".split(),
}
VALUE = re.compile(
    r"CHF\s*\d[\d'’.]*|\d[\d'’.]*\s*CHF|\d[\d'’.]*\s*%|\d{1,2}[./]\d{1,2}[./]\d{2,4}|\d{4}-\d{2}-\d{2}|(?:Art|Abs|al)\.\s*\d+|\d[\d'’.]*",
    re.IGNORECASE,
)


def _ids() -> dict:
    return json.loads((REPO / "track_2a/docs/sprint2_dev_ids.json").read_text(encoding="utf-8"))


def _load(path: Path) -> dict[str, dict]:
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            found[row["id"]] = row
    return found


def _digits(text: str) -> set[str]:
    found = set()
    for match in VALUE.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if digits:
            found.add(digits)
    return found


def _mismatch(tokens: list[str], other: set[str], scores: list[float]) -> list[float]:
    out = []
    for token, score in zip(tokens, scores):
        values = _digits(token)
        bumped = score + BONUS if values and any(item not in other for item in values) else score
        out.append(round(float(min(1.0, max(0.0, bumped))), 6))
    return out


def _damp(tokens: list[str], lang: str, scores: list[float]) -> list[float]:
    stops = set(STOP["en"]) | set(STOP[lang])
    out = []
    for token, score in zip(tokens, scores):
        hit = token.lower() in stops or (token != "" and all(ch in PUNCT for ch in token))
        value = score * DAMP if hit else score
        out.append(round(float(min(1.0, max(0.0, value))), 6))
    return out


def _adaptive(tokens_a, tokens_b, scores_a, scores_b):
    if max(len(tokens_a), len(tokens_b)) < 800:
        return scores_a, scores_b
    return (
        _span_smooth_tri(tokens_a, np.asarray(scores_a, dtype=np.float64), w=2),
        _span_smooth_tri(tokens_b, np.asarray(scores_b, dtype=np.float64), w=2),
    )


def _spearman(pred, gold, wanted: set[str]) -> float:
    from scipy.stats import spearmanr

    xs, ys = [], []
    for item in wanted:
        for side in ("a", "b"):
            gp = gold[item][f"labels_{side}"]
            pp = pred[item][f"labels_{side}"]
            if side == "b" and all(value == -1 for value in gp):
                continue
            for got, truth in zip(pp, gp):
                if truth != -1:
                    xs.append(float(got))
                    ys.append(float(truth))
    if len(xs) < 3:
        return float("nan")
    return float(spearmanr(xs, ys).statistic)


def _gate(name: str, table, golds, frozen, idmap) -> dict:
    dev_ids = {lang: idmap[lang]["train"] + idmap[lang]["val"] for lang in LANGS}

    def mean(split: str) -> float:
        vals = [
            _spearman(table[lang], golds[lang], set(idmap[lang][split]))
            for lang in LANGS
        ]
        return float(np.round(np.mean(vals), 4))

    train, val = mean("train"), mean("val")
    wins = 0
    folds = []
    for fold in range(5):
        cand, base = [], []
        for lang in LANGS:
            wanted = {item for item in dev_ids[lang] if int(item.rsplit("_", 1)[-1]) % 5 == fold}
            cand.append(_spearman(table[lang], golds[lang], wanted))
            base.append(_spearman(frozen[lang], golds[lang], wanted))
        cmean, bmean = float(np.mean(cand)), float(np.mean(base))
        win = cmean > bmean
        wins += int(win)
        folds.append({"candidate": round(cmean, 4), "frozen": round(bmean, 4), "win": win})
    adopted = train >= 0.3433 and val >= 0.3165 and wins >= 4
    print(f"{name} train {train} val {val} folds {wins}/5 adopt {adopted}", flush=True)
    return {"train": train, "val": val, "fold_wins": wins, "folds": folds, "adopt": adopted}


def main() -> int:
    idmap = _ids()
    golds = {lang: _load(GOLD / f"gold_admin_{lang}.jsonl") for lang in LANGS}
    frozen = {
        lang: _load(REPO / "track_2a/predictions/dev" / f"SpanDiff_admin_{lang}.jsonl.jsonl")
        for lang in LANGS
    }
    mismatch = {lang: {} for lang in LANGS}
    damp = {lang: {} for lang in LANGS}
    adaptive = {lang: {} for lang in LANGS}
    for lang in LANGS:
        for item, row in frozen[lang].items():
            tokens_a = str(row["text_a"]).split()
            tokens_b = str(row["text_b"]).split()
            other_a = _digits(row["text_b"])
            other_b = _digits(row["text_a"])
            mismatch[lang][item] = {
                "labels_a": _mismatch(tokens_a, other_a, row["labels_a"]),
                "labels_b": _mismatch(tokens_b, other_b, row["labels_b"]),
            }
            damp[lang][item] = {
                "labels_a": _damp(tokens_a, lang, row["labels_a"]),
                "labels_b": _damp(tokens_b, lang, row["labels_b"]),
            }
            la, lb = _adaptive(tokens_a, tokens_b, row["labels_a"], row["labels_b"])
            adaptive[lang][item] = {"labels_a": la, "labels_b": lb}
    results = {
        "mismatch": _gate("mismatch", mismatch, golds, frozen, idmap),
        "damping": _gate("damping", damp, golds, frozen, idmap),
        "adaptive": _gate("adaptive", adaptive, golds, frozen, idmap),
    }
    dest = REPO / "track_2a/docs/sprint3_post.json"
    dest.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("WROTE", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
