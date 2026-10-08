#!/usr/bin/env python3
"""Hidden-state-8 pooling comparison on the development split.

One forward per window. Mean-pool, last-subword, and subword-then-mean
are scored from that same state. Does not write the graded prediction
files and does not open the sealed test split.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "track_2a" / "src"
sys.path.insert(0, str(SRC))

from spandiff.align import assign_subwords, locate_tokens, pack_spans, source_text  # noqa: E402
from spandiff.devdata import read_jsonl  # noqa: E402
from spandiff.leakage import assert_path_allowed  # noqa: E402

GOLD = REPO / ".swissgov" / "data" / "evaluation" / "gold_labels" / "dev"
SUMMARY = REPO / "track_2a" / "docs" / "pooling_dev.txt"
LAYER_INDEX = 8


def average_rank_spearman(x, y) -> float:
    """Spearman with ties averaged. This is the contest formula."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 5 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranks


def one_minus_max(left: np.ndarray, right: np.ndarray, block: int = 256) -> np.ndarray:
    """Match align.diff_against, including a zero vector scoring 1."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size == 0:
        return np.zeros((0,), dtype=np.float64)
    if right.size == 0:
        return np.ones((left.shape[0],), dtype=np.float64)
    right_norm = np.linalg.norm(right, axis=1)
    out = np.empty(left.shape[0], dtype=np.float64)
    for start in range(0, left.shape[0], block):
        chunk = left[start : start + block]
        left_norm = np.linalg.norm(chunk, axis=1)
        denom = np.outer(left_norm, right_norm)
        cosines = np.zeros((chunk.shape[0], right.shape[0]), dtype=np.float64)
        np.divide(chunk @ right.T, denom, out=cosines, where=denom > 0.0)
        out[start : start + chunk.shape[0]] = np.clip(1.0 - cosines.max(axis=1), 0.0, 1.0)
    return out


def _reduce_window(layer: np.ndarray, offsets, spans):
    buckets = assign_subwords(offsets, spans)
    width = layer.shape[1]
    means = []
    lasts = []
    subs = []
    counts = []
    for bucket in buckets:
        if not bucket:
            means.append(np.zeros(width, dtype=np.float32))
            lasts.append(np.zeros(width, dtype=np.float32))
            counts.append(0)
            continue
        rows = layer[bucket]
        means.append(rows.mean(axis=0).astype(np.float32))
        lasts.append(rows[-1].astype(np.float32))
        subs.append(rows.astype(np.float32))
        counts.append(int(len(bucket)))
    return means, lasts, subs, counts


def encode_side(model, tokenizer, text, spans, device):
    if not spans:
        width = int(model.config.hidden_size)
        empty = np.zeros((0, width), dtype=np.float32)
        return empty, empty, empty, []

    def fits(piece: str) -> bool:
        return len(tokenizer.encode(piece, add_special_tokens=True)) <= 512

    means, lasts, subs, counts = [], [], [], []
    for group in pack_spans(text, spans, fits):
        start = group[0][0]
        piece = text[start : group[-1][1]]
        local = [(a - start, b - start) for a, b in group]
        if not fits(piece):
            raise RuntimeError("a single whitespace token exceeds the 512 window")
        batch = tokenizer(
            piece,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=False,
        )
        offsets = [(int(a), int(b)) for a, b in batch.pop("offset_mapping")[0].tolist()]
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states
        if len(hidden) <= LAYER_INDEX:
            raise RuntimeError(f"hidden state {LAYER_INDEX} is missing")
        layer = hidden[LAYER_INDEX][0].detach().float().cpu().numpy()
        del outputs, hidden
        window_means, window_lasts, window_subs, window_counts = _reduce_window(layer, offsets, local)
        means.extend(window_means)
        lasts.extend(window_lasts)
        subs.extend(window_subs)
        counts.extend(window_counts)
    if len(means) != len(spans):
        raise RuntimeError("pooled tokens do not match whitespace tokens")
    sub_matrix = np.concatenate(subs, axis=0) if subs else np.zeros((0, means[0].shape[0]), dtype=np.float32)
    return np.stack(means), np.stack(lasts), sub_matrix, counts


def subword_then_mean(subs: np.ndarray, counts: list[int], other_subs: np.ndarray) -> np.ndarray:
    if not counts:
        return np.zeros((0,), dtype=np.float64)
    if subs.size == 0:
        return np.ones(len(counts), dtype=np.float64)
    token_scores = one_minus_max(subs, other_subs)
    out = np.empty(len(counts), dtype=np.float64)
    cursor = 0
    for index, count in enumerate(counts):
        if count == 0:
            out[index] = 1.0
            continue
        out[index] = float(token_scores[cursor : cursor + count].mean())
        cursor += count
    if cursor != len(token_scores):
        raise RuntimeError("subword scores were not consumed in order")
    return out


def keep(gold_flags, scores) -> list[float]:
    return [float(score) for flag, score in zip(gold_flags, scores) if flag is not None]


def load_model():
    from spandiff.study_guard import refuse_graded_load

    refuse_graded_load()
    spec = importlib.util.spec_from_file_location(
        "worker_b", "/tmp/apertus_mps_score_b/apertus_mps_score.py"
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker.load_model("mps")


def write_summary(buckets: dict[str, dict[str, list[float]]], docs: int, complete: bool) -> None:
    if "SpanDiff_admin" in SUMMARY.name:
        raise SystemExit("refusing to write a graded prediction name")
    lines = [
        "Hidden state 8 pooling comparison on the development split.",
        "Not a replacement for the graded files. The contest score stays the printed hidden-state-8 Spearman.",
        f"documents {docs} complete {str(complete).lower()}",
        "Spearman uses average ranks, the contest formula. Mean-pool rounded to 6 decimals is the graded dump rule.",
    ]
    for name in ("mean_round6", "mean", "last", "subword_then_mean"):
        parts = []
        for lang in ("de", "fr", "it"):
            gold = buckets[lang]["gold"]
            pred = buckets[lang][name]
            parts.append(f"{lang} {average_rank_spearman(pred, gold):.4f} n {len(gold)}")
        lines.append(f"{name} " + " ".join(parts))
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check() -> None:
    ranks = _average_ranks(np.asarray([1.0, 2.0, 2.0]))
    if list(ranks) != [1.0, 2.5, 2.5]:
        raise SystemExit(f"average ranks are wrong: {ranks}")
    if one_minus_max(np.zeros((1, 2)), np.asarray([[1.0, 0.0]])).tolist() != [1.0]:
        raise SystemExit("zero vector did not score 1")
    if abs(one_minus_max(np.asarray([[1.0, 0.0]]), np.asarray([[1.0, 0.0]]))[0]) > 1e-9:
        raise SystemExit("exact match did not score 0")
    scored = subword_then_mean(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        [2],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    if abs(float(scored[0]) - 0.5) > 1e-9:
        raise SystemExit(f"subword-then-mean is wrong: {scored}")
    print("pooling check passed")


def main() -> int:
    if "--check" in sys.argv:
        check()
        return 0
    assert_path_allowed(GOLD)
    check()
    model, tokenizer, device, _ = load_model()
    buckets = {
        lang: {"gold": [], "mean": [], "mean_round6": [], "last": [], "subword_then_mean": []}
        for lang in ("de", "fr", "it")
    }
    docs = 0
    for lang in ("de", "fr", "it"):
        path = GOLD / f"gold_admin_{lang}.jsonl"
        assert_path_allowed(path)
        for record in read_jsonl(path):
            sides = []
            gold_flags = []
            for side in ("a", "b"):
                text, tokens = source_text(record, side)
                spans = locate_tokens(text, tokens)
                sides.append(encode_side(model, tokenizer, text, spans, device))
                gold_flags.extend(
                    None if value == -1 else float(value) for value in record[f"labels_{side}"]
                )
            mean_scores = np.concatenate(
                [one_minus_max(sides[0][0], sides[1][0]), one_minus_max(sides[1][0], sides[0][0])]
            )
            last_scores = np.concatenate(
                [one_minus_max(sides[0][1], sides[1][1]), one_minus_max(sides[1][1], sides[0][1])]
            )
            sub_scores = np.concatenate(
                [
                    subword_then_mean(sides[0][2], sides[0][3], sides[1][2]),
                    subword_then_mean(sides[1][2], sides[1][3], sides[0][2]),
                ]
            )
            if not (len(gold_flags) == len(mean_scores) == len(last_scores) == len(sub_scores)):
                raise RuntimeError(f"{record['id']}: score length does not match labels")
            gold_kept = [value for value in gold_flags if value is not None]
            buckets[lang]["gold"].extend(gold_kept)
            buckets[lang]["mean"].extend(keep(gold_flags, mean_scores))
            buckets[lang]["mean_round6"].extend(keep(gold_flags, np.round(mean_scores, 6)))
            buckets[lang]["last"].extend(keep(gold_flags, last_scores))
            buckets[lang]["subword_then_mean"].extend(keep(gold_flags, sub_scores))
            docs += 1
            print(f"scored {record['id']} docs {docs}", flush=True)
            if docs % 20 == 0:
                write_summary(buckets, docs, complete=False)
            torch.mps.empty_cache()
    write_summary(buckets, docs, complete=True)
    print(f"WROTE {SUMMARY}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
