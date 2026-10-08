#!/usr/bin/env python3
"""Development layer map for Apertus v1.5 8B.

One forward per window. Every hidden state is pooled and scored.
Does not write the graded SpanDiff_admin_*.jsonl.jsonl files.
Does not open the sealed test split.
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
OUT = REPO / "track_2a" / "docs" / "layer_map_dev.jsonl"
SUMMARY = REPO / "track_2a" / "docs" / "layer_map_dev.txt"


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 5 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def one_minus_max(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Match align.diff_against: a zero vector has cosine 0, so score 1.

    The layer-map curves already written used a 1e-8 norm floor. A later
    mean-pool pass with this rule still read 0.2495, 0.1662, and 0.2842,
    so the floor does not explain the gap to the four-decimal graded
    coefficients 0.2451, 0.1649, and 0.2819. The script prints 0,245, 0,165, and 0,282.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size == 0:
        return np.zeros((0,), dtype=np.float64)
    if right.size == 0:
        return np.ones((left.shape[0],), dtype=np.float64)
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denom = np.outer(left_norm, right_norm)
    cosines = np.zeros((left.shape[0], right.shape[0]), dtype=np.float64)
    np.divide(left @ right.T, denom, out=cosines, where=denom > 0.0)
    return np.clip(1.0 - cosines.max(axis=1), 0.0, 1.0)


def pool_states(states: np.ndarray, offsets, spans) -> np.ndarray:
    buckets = assign_subwords(offsets, spans)
    width = states.shape[1]
    rows = []
    for bucket in buckets:
        if not bucket:
            rows.append(np.zeros(width, dtype=np.float32))
        else:
            rows.append(states[bucket].mean(axis=0))
    return np.stack(rows).astype(np.float32)


def encode_side(model, tokenizer, text, spans, device, n_states: int):
    def fits(piece: str) -> bool:
        return len(tokenizer.encode(piece, add_special_tokens=True)) <= 512

    groups = pack_spans(text, spans, fits)
    chunks = [[] for _ in range(n_states)]
    for group in groups:
        start = group[0][0]
        piece = text[start : group[-1][1]]
        local = [(a - start, b - start) for a, b in group]
        batch = tokenizer(
            piece,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=False,
        )
        offsets = [(int(a), int(b)) for a, b in batch.pop("offset_mapping")[0].tolist()]
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states
        if len(hidden) != n_states:
            raise RuntimeError(f"expected {n_states} hidden states, got {len(hidden)}")
        for idx in range(n_states):
            layer = hidden[idx][0].detach().float().cpu().numpy()
            chunks[idx].append(pool_states(layer, offsets, local))
        del outputs, hidden
    return [np.concatenate(parts, axis=0) for parts in chunks]


def load_model():
    from spandiff.study_guard import refuse_graded_load

    refuse_graded_load()
    spec = importlib.util.spec_from_file_location(
        "worker_b", "/tmp/apertus_mps_score_b/apertus_mps_score.py"
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker.load_model("mps")


def done_ids() -> set[str]:
    if not OUT.is_file():
        return set()
    found = set()
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            found.add(json.loads(line)["id"])
    return found


def write_summary(rows: list[dict], n_states: int) -> None:
    lines = [
        "Development layer map. Not a replacement for the graded hidden-state-8 files.",
        "Spearman uses ordinal ranks. The contest formula is average rank. Do not compare these numbers with the printed score.",
        f"documents {len(rows)}",
    ]
    for idx in range(n_states):
        pred, gold = [], []
        for row in rows:
            pred.extend(row["pred"][idx])
            gold.extend(row["gold"])
        lines.append(f"layer {idx} n {len(gold)} spearman {spearman(pred, gold):.4f}")
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    model, tokenizer, device, _ = load_model()
    n_states = int(model.config.text_config.num_hidden_layers) + 1
    print(f"n_states {n_states}", flush=True)
    finished = done_ids()
    print(f"already {len(finished)}", flush=True)
    kept: list[dict] = []
    for lang in ("de", "fr", "it"):
        path = GOLD / f"gold_admin_{lang}.jsonl"
        assert_path_allowed(path)
        for record in read_jsonl(path):
            item_id = record["id"]
            if item_id in finished:
                continue
            sides = []
            gold = []
            for side in ("a", "b"):
                text, tokens = source_text(record, side)
                spans = locate_tokens(text, tokens)
                sides.append(encode_side(model, tokenizer, text, spans, device, n_states))
                gold.extend(
                    None if value == -1 else float(value) for value in record[f"labels_{side}"]
                )
            pred_kept = []
            gold_kept = [value for value in gold if value is not None]
            spearmans = []
            for idx in range(n_states):
                scores = np.concatenate(
                    [
                        one_minus_max(sides[0][idx], sides[1][idx]),
                        one_minus_max(sides[1][idx], sides[0][idx]),
                    ]
                )
                kept_scores = [
                    float(score)
                    for value, score in zip(gold, scores)
                    if value is not None
                ]
                pred_kept.append(kept_scores)
                spearmans.append(spearman(kept_scores, gold_kept))
            row = {"id": item_id, "lang": lang, "spearman": spearmans, "n": len(gold_kept)}
            with OUT.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": item_id, "lang": lang, "n": row["n"], "spearman": spearmans}) + "\n")
                handle.flush()
            kept.append({"pred": pred_kept, "gold": gold_kept})
            print(
                f"scored {item_id} n {row['n']} "
                f"layer8 {spearmans[8]:.4f} layer32 {spearmans[-1]:.4f} docs {len(kept)}",
                flush=True,
            )
            if len(kept) % 10 == 0:
                write_summary(kept, n_states)
            torch.mps.empty_cache()
    write_summary(kept, n_states)
    print(f"WROTE {SUMMARY}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
