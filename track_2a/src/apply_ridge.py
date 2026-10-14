#!/usr/bin/env python3
"""Apply the saved development ridge. This script does not fit.

Weights, means, and standard deviations come from track_2a/docs/ridge_weights.json.
Predict time does not read development annotations and does not refit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[1]
sys.path.insert(0, str(SRC))

from spandiff.devdata import LANGS  # noqa: E402
from sprint3_stacker import (  # noqa: E402
    _featurize_doc,
    _read_cache,
    _write_cache,
)

WEIGHTS = REPO / "track_2a/docs/ridge_weights.json"


def load_weights() -> dict:
    payload = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    if payload.get("alpha") != 1.0:
        raise SystemExit("ridge weights are not the pre-registered alpha 1.0")
    if "fit" in payload or any(key.startswith("label") for key in payload):
        raise SystemExit("ridge weights file is not a coefficient file")
    return payload


def apply_weights(features: np.ndarray, weights: dict) -> np.ndarray:
    names = list(weights["feature_names"])
    if features.shape[1] != len(names):
        raise SystemExit("feature width does not match the saved ridge")
    mu = np.array([weights["mean"][name] for name in names], dtype=np.float64)
    sd = np.array([weights["std"][name] for name in names], dtype=np.float64)
    sd = np.where(sd < 1e-8, 1.0, sd)
    beta = np.array(
        [weights["intercept"], *[weights["weights"][name] for name in names]],
        dtype=np.float64,
    )
    scaled = (features - mu) / sd
    design = np.concatenate([np.ones((len(scaled), 1)), scaled], axis=1)
    return np.clip(design @ beta, 0.0, 1.0)


def _rows(path: Path) -> list[dict]:
    found = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            found.append(json.loads(line))
    return found


def _index(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in _rows(path)}


def _blank(record: dict) -> dict:
    raw = {
        "id": record["id"],
        "text_a": record["text_a"],
        "text_b": record["text_b"],
    }
    for key in ("text_a_untokenized", "text_b_untokenized"):
        if key in record:
            raw[key] = record[key]
    raw["labels_a"] = [0.0] * len(str(raw["text_a"]).split())
    raw["labels_b"] = [0.0] * len(str(raw["text_b"]).split())
    return raw


def _dump(record: dict, scores_a, scores_b) -> str:
    payload = {
        "id": record["id"],
        "text_a": record["text_a"],
        "text_b": record["text_b"],
        "labels_a": [round(float(value), 6) for value in scores_a],
        "labels_b": [round(float(value), 6) for value in scores_b],
    }
    return json.dumps(payload, ensure_ascii=False)


def _ensure_surprisal(records: list[tuple[str, dict]]) -> dict:
    from sprint3_surprisal import CACHE, NAMES, _cached, _load_lm, _side_surprisal
    from spandiff.align import locate_tokens, source_text

    cached = _cached()
    missing = [(lang, raw) for lang, raw in records if raw["id"] not in cached.get(lang, {})]
    if not missing:
        return cached
    model, tokenizer, device = _load_lm()
    with CACHE.open("a", encoding="utf-8") as handle:
        for lang, raw in missing:
            blank = _blank(raw)
            prefix_b = f"Translate English to {NAMES[lang]}:\n"
            prefix_a = f"Translate {NAMES[lang]} to English:\n"
            text_a, tokens_a = source_text(blank, "a")
            text_b, tokens_b = source_text(blank, "b")
            sur_b = _side_surprisal(
                model, tokenizer, device, prefix_b, text_a, text_b, locate_tokens(text_b, tokens_b)
            )
            sur_a = _side_surprisal(
                model, tokenizer, device, prefix_a, text_b, text_a, locate_tokens(text_a, tokens_a)
            )
            handle.write(json.dumps({"lang": lang, "id": raw["id"], "a": sur_a, "b": sur_b}) + "\n")
            handle.flush()
            print(f"surprisal {raw['id']}", flush=True)
    del model
    return _cached()


def _frozen_score(model, tokenizer, device, max_length, record: dict) -> dict:
    from spandiff.score import score_record

    line, _tokens = score_record(model, tokenizer, _blank(record), device, max_length)
    parsed = json.loads(line)
    return {"labels_a": parsed["labels_a"], "labels_b": parsed["labels_b"]}


def _features(model_pack, lang: str, record: dict, frozen_row: dict, sur: dict) -> np.ndarray:
    cached = _read_cache(record["id"])
    if cached is not None:
        return cached[0]
    if model_pack is None:
        raise SystemExit(f"no feature cache for {record['id']}")
    model, tokenizer, device, max_length = model_pack
    features, labels = _featurize_doc(
        model, tokenizer, device, max_length, _blank(record), frozen_row, sur, lang
    )
    _write_cache(record["id"], features, labels)
    return features


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--order-from", type=Path)
    parser.add_argument("--texts", type=Path)
    parser.add_argument("--frozen-from", type=Path)
    args = parser.parse_args(argv)
    if args.texts is None and args.order_from is None:
        raise SystemExit("pass --order-from or --texts")
    weights = load_weights()
    out_dir = args.out if args.out.is_absolute() else REPO / args.out
    if out_dir.resolve().is_relative_to((REPO / "track_2a/predictions/dev").resolve()):
        raise SystemExit("refusing to write the graded development files")
    out_dir.mkdir(parents=True, exist_ok=True)
    order_root = None if args.order_from is None else (
        args.order_from if args.order_from.is_absolute() else REPO / args.order_from
    )
    text_root = None if args.texts is None else (args.texts if args.texts.is_absolute() else REPO / args.texts)
    frozen_root = None if args.frozen_from is None else (
        args.frozen_from if args.frozen_from.is_absolute() else REPO / args.frozen_from
    )
    jobs = []
    for lang in LANGS:
        if text_root is not None:
            records = _rows(text_root / f"{lang}.jsonl")
        else:
            records = _rows(order_root / f"SpanDiff_admin_{lang}.jsonl.jsonl")
        jobs.extend((lang, record) for record in records)
    surprisal = _ensure_surprisal(jobs)
    frozen = {}
    if frozen_root is not None:
        for lang in LANGS:
            frozen[lang] = _index(frozen_root / f"SpanDiff_admin_{lang}.jsonl.jsonl")
    need_model = []
    for lang, record in jobs:
        if _read_cache(record["id"]) is None:
            need_model.append((lang, record))
    model_pack = None
    if need_model:
        import spandiff.score as score
        from spandiff.score import load_apertus

        score.LAYER_INDEX = 15
        model, tokenizer, device, max_length = load_apertus()
        model_pack = (model, tokenizer, device, min(int(max_length), 512))
    by_lang = {lang: [] for lang in LANGS}
    for lang, record in jobs:
        if model_pack is not None and frozen_root is None and _read_cache(record["id"]) is None:
            frozen_row = _frozen_score(*model_pack, record)
        elif frozen_root is not None:
            frozen_row = frozen[lang][record["id"]]
        else:
            frozen_row = None
        if _read_cache(record["id"]) is None and frozen_row is None:
            raise SystemExit(f"missing frozen score for {record['id']}")
        features = _features(model_pack, lang, record, frozen_row or {}, surprisal[lang][record["id"]])
        pred = apply_weights(features, weights)
        n_a = len(str(record["text_a"]).split())
        if len(pred) != n_a + len(str(record["text_b"]).split()):
            raise SystemExit(f"score length {record['id']}")
        by_lang[lang].append(_dump(record, pred[:n_a], pred[n_a:]))
        print(f"applied {record['id']}", flush=True)
    if model_pack is not None:
        del model_pack
    for lang in LANGS:
        dest = out_dir / f"SpanDiff_admin_{lang}.jsonl.jsonl"
        dest.write_text("\n".join(by_lang[lang]) + "\n", encoding="utf-8")
        print(f"wrote {dest} {len(by_lang[lang])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
