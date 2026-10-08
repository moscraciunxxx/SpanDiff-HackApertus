#!/usr/bin/env python3
"""Figures for the report. Reads saved dev outputs. Does not load the model."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "track_2a" / "docs"
CURVE = DOCS / "layer_map_by_lang_avg.txt"
PRED = REPO / "track_2a" / "predictions" / "dev" / "SpanDiff_admin_de.jsonl.jsonl"
PAIR = "admin_de_105"


def layer_curve() -> None:
    series: dict[str, list[tuple[int, float]]] = {"de": [], "fr": [], "it": []}
    current = None
    for line in CURVE.read_text(encoding="utf-8").splitlines():
        match = re.match(r"lang (de|fr|it) ", line)
        if match:
            current = match.group(1)
            continue
        point = re.match(r"(de|fr|it) layer (\d+) n \d+ spearman ([0-9.]+)", line)
        if point and current == point.group(1):
            series[current].append((int(point.group(2)), float(point.group(3))))
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for lang, points in series.items():
        xs, ys = zip(*points)
        axis.plot(xs, ys, label=lang, linewidth=2)
    axis.axvline(15, color="black", linestyle="--", linewidth=1, label="submitted layer 15")
    axis.axvline(8, color="gray", linestyle=":", linewidth=1, label="ablation layer 8")
    axis.set_xlabel("Hidden state index")
    axis.set_ylabel("Average-rank Spearman")
    axis.set_title("Development Spearman by layer")
    axis.legend(frameon=False, ncol=2)
    axis.set_xlim(0, 32)
    figure.tight_layout()
    figure.savefig(DOCS / "layer_curve.png", dpi=150)
    plt.close(figure)


def _gold_row() -> dict | None:
    path = REPO / ".swissgov" / "data" / "evaluation" / "gold_labels" / "dev" / "gold_admin_de.jsonl"
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        if obj.get("id") == PAIR:
            return obj
    return None


def heatmap() -> None:
    record = None
    for line in PRED.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        if obj["id"] == PAIR:
            record = obj
            break
    if record is None:
        raise SystemExit(f"{PAIR} is not in the development predictions")
    gold = _gold_row()
    figure, axes = plt.subplots(2, 1, figsize=(10, 4.4), constrained_layout=True)
    image = None
    for axis, side, title in (
        (axes[0], "a", "English"),
        (axes[1], "b", "German"),
    ):
        tokens = record[f"text_{side}"].split()
        rows = [record[f"labels_{side}"]]
        names = ["prediction"]
        if gold is not None and len(gold[f"labels_{side}"]) == len(tokens):
            rows.append([float("nan") if value == -1 else float(value) for value in gold[f"labels_{side}"]])
            names.append("gold")
        image = axis.imshow(rows, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        axis.set_yticks(range(len(names)))
        axis.set_yticklabels([f"{title} {name}" for name in names])
        step = 5
        axis.set_xticks(range(0, len(tokens), step))
        axis.set_xticklabels(tokens[::step], rotation=60, ha="right", fontsize=7)
        axis.set_xlim(-0.5, len(tokens) - 0.5)
    figure.colorbar(image, ax=axes, fraction=0.03, pad=0.02, label="difference")
    figure.suptitle(f"{PAIR}: layer-15 prediction and development gold")
    figure.savefig(DOCS / "token_heatmap.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    layer_curve()
    heatmap()
    print("wrote", DOCS / "layer_curve.png")
    print("wrote", DOCS / "token_heatmap.png")
