#!/usr/bin/env python3
"""Per-language token-pooled layer map.

Same forward pass as layer_map_dev.py. Spearman is computed inside each
language, which is the shape of the contest metric. It is still not the
graded score: graded files are hidden state 8 only, and this script does
not write them. Does not open the sealed test split.
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
SUMMARY = REPO / "track_2a" / "docs" / "layer_map_by_lang.txt"


def write_summary(buckets: dict[str, list[dict]], n_states: int) -> None:
    lines = [
        "Per-language development layer map. Not a replacement for the graded hidden-state-8 files.",
        "Spearman is token-pooled inside one language and uses ordinal ranks. The average-rank pass is layer_map_by_lang_avg.txt.",
        "It is not the unweighted mean of the three printed contest Spearmans.",
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
            value = layer_map.spearman(pred, gold)
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
