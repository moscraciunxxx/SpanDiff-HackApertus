#!/usr/bin/env python3
"""Per-language layer map with average-rank Spearman.

Same forward pass as layer_map_by_lang.py. The rank formula matches
scipy.stats.spearmanr, which is the contest metric's rank handling.
On the frozen German file this formula reads 0.2451, the four-decimal
round of the unrounded coefficient. The script prints 0,245. It does not write
the graded SpanDiff_admin_*.jsonl.jsonl files and does not open the
sealed test split.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "track_2a" / "src"
sys.path.insert(0, str(SRC))

from spandiff.devdata import read_jsonl  # noqa: E402
from spandiff.leakage import assert_path_allowed  # noqa: E402

spec = importlib.util.spec_from_file_location("layer_map_dev", SRC / "layer_map_dev.py")
layer_map = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layer_map)

GOLD = REPO / ".swissgov" / "data" / "evaluation" / "gold_labels" / "dev"
SUMMARY = REPO / "track_2a" / "docs" / "layer_map_by_lang_avg.txt"


def average_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    n = len(values)
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 5 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    rx = average_rank(x)
    ry = average_rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def write_summary(buckets: dict[str, list[dict]], n_states: int) -> None:
    lines = [
        "Per-language development layer map, average-rank Spearman.",
        "Not a replacement for the graded hidden-state-8 files.",
        "This rank formula matches the unrounded German coefficient to four decimals (0.2451). The script prints 0,245.",
    ]
    for lang in ("de", "fr", "it"):
        rows = buckets[lang]
        lines.append(f"lang {lang} documents {len(rows)}")
        if not rows:
            continue
        ranked = []
        for idx in range(n_states):
            pred, gold = [], []
            for row in rows:
                pred.extend(row["pred"][idx])
                gold.extend(row["gold"])
            value = spearman(pred, gold)
            ranked.append((value, idx, len(gold)))
            lines.append(f"{lang} layer {idx} n {len(gold)} spearman {value:.4f}")
        finite = [item for item in ranked if np.isfinite(item[0])]
        if finite:
            best = max(finite)
            lines.append(
                f"{lang} peak layer {best[1]} n {best[2]} spearman {best[0]:.4f}"
            )
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    model, tokenizer, device, _ = layer_map.load_model()
    n_states = int(model.config.text_config.num_hidden_layers) + 1
    print(f"n_states {n_states}", flush=True)
    buckets = {lang: [] for lang in ("de", "fr", "it")}
    for lang in ("de", "fr", "it"):
        path = GOLD / f"gold_admin_{lang}.jsonl"
        assert_path_allowed(path)
        for record in read_jsonl(path):
            item_id = record["id"]
            sides = []
            gold = []
            for side in ("a", "b"):
                text, tokens = layer_map.source_text(record, side)
                spans = layer_map.locate_tokens(text, tokens)
                sides.append(
                    layer_map.encode_side(model, tokenizer, text, spans, device, n_states)
                )
                gold.extend(
                    None if value == -1 else float(value) for value in record[f"labels_{side}"]
                )
            pred_kept = []
            gold_kept = [value for value in gold if value is not None]
            for idx in range(n_states):
                scores = np.concatenate(
                    [
                        layer_map.one_minus_max(sides[0][idx], sides[1][idx]),
                        layer_map.one_minus_max(sides[1][idx], sides[0][idx]),
                    ]
                )
                pred_kept.append(
                    [
                        float(score)
                        for value, score in zip(gold, scores)
                        if value is not None
                    ]
                )
            buckets[lang].append({"pred": pred_kept, "gold": gold_kept})
            print(
                f"scored {item_id} n {len(gold_kept)} lang {lang} docs {len(buckets[lang])}",
                flush=True,
            )
            if len(buckets[lang]) % 20 == 0:
                write_summary(buckets, n_states)
            layer_map.torch.mps.empty_cache()
        write_summary(buckets, n_states)
    print(f"WROTE {SUMMARY}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
