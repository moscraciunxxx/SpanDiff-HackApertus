#!/usr/bin/env python3
"""Sprint 3 idea 4: rank-average Tier-1 scores from layers 12, 15, and 18."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[1]
sys.path.insert(0, str(SRC))

import spandiff.score as score  # noqa: E402
from spandiff.align import assign_subwords, locate_tokens, pack_spans, source_text  # noqa: E402
from spandiff.score import _span_smooth_tri, load_apertus  # noqa: E402
from sprint3_idea1 import _ids, _load_rows, _score_pair, _spearman  # noqa: E402
from sprint3_post import GOLD, LANGS  # noqa: E402

LAYERS = (12, 15, 18)


def _encode(model, tokenizer, text, spans, device, max_length):
    import torch

    empty = {layer: (np.zeros((0, 4096), dtype=np.float32), [], np.zeros((0, 2), dtype=np.int32)) for layer in LAYERS}
    if not spans:
        return empty

    def fits(piece: str) -> bool:
        return len(tokenizer.encode(piece, add_special_tokens=True)) <= max_length

    groups = pack_spans(text, spans, fits)
    counts: list[int] = []
    offsets: list[tuple[int, int]] = []
    chunks = {layer: [] for layer in LAYERS}
    for group in groups:
        start = group[0][0]
        piece = text[start : group[-1][1]]
        local = [(a - start, b - start) for a, b in group]
        batch = tokenizer(
            piece,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        offs = [(int(a), int(b)) for a, b in batch.pop("offset_mapping")[0].tolist()]
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states
        idx = []
        for bucket in assign_subwords(offs, local):
            counts.append(len(bucket))
            for item in bucket:
                idx.append(item)
                oa, ob = offs[item]
                offsets.append((oa + start, ob + start))
        for layer in LAYERS:
            arr = hidden[layer][0].detach().float().cpu().numpy()
            if idx:
                chunks[layer].append(arr[idx].astype(np.float16).astype(np.float32))
        del outputs
    off = np.asarray(offsets, dtype=np.int32)
    packed = {}
    for layer in LAYERS:
        stitched = np.concatenate(chunks[layer], axis=0) if chunks[layer] else np.zeros((0, 4096), dtype=np.float32)
        packed[layer] = (stitched, counts, off)
    return packed


def _rank01(values) -> np.ndarray:
    from scipy.stats import rankdata

    arr = np.asarray(values, dtype=np.float64)
    if len(arr) <= 1:
        return np.zeros(len(arr))
    return (rankdata(arr, method="average") - 1.0) / (len(arr) - 1.0)


def main() -> int:
    score.LAYER_INDEX = 18
    model, tokenizer, device, max_length = load_apertus()
    max_length = min(int(max_length), 512)
    # Reuse the Tier-1 math in idea 1, with the diagonal term forced to zero.
    # _score_pair always applies 0.25. Call the frozen score_record path per layer instead.
    from spandiff.score import encode_side_subwords

    table = {lang: {} for lang in LANGS}
    real_chosen = score.chosen_layer_tensor

    def choose(layer):
        def _chosen(outputs, expected=33):
            return outputs.hidden_states[layer]
        return _chosen

    for lang in LANGS:
        path = REPO / "track_2a/data/swissgov" / lang / f"dev_{lang}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            raw["labels_a"] = [0.0] * len(str(raw["text_a"]).split())
            raw["labels_b"] = [0.0] * len(str(raw["text_b"]).split())
            text_a, tokens_a = source_text(raw, "a")
            text_b, tokens_b = source_text(raw, "b")
            spans_a = locate_tokens(text_a, tokens_a)
            spans_b = locate_tokens(text_b, tokens_b)
            packed_a = _encode(model, tokenizer, text_a, spans_a, device, max_length)
            packed_b = _encode(model, tokenizer, text_b, spans_b, device, max_length)
            sides = []
            for layer in LAYERS:
                sa, ca, oa = packed_a[layer]
                sb, cb, ob = packed_b[layer]
                # Zero the diagonal by scoring through a local copy of Tier-1 (w=2).
                labels = _tier1(sa, ca, oa, sb, cb, ob, text_a, text_b, tokens_a, tokens_b)
                sides.append(labels)
            rank_a = np.mean([_rank01(pair[0]) for pair in sides], axis=0)
            rank_b = np.mean([_rank01(pair[1]) for pair in sides], axis=0)
            table[lang][raw["id"]] = {
                "labels_a": [round(float(x), 6) for x in rank_a],
                "labels_b": [round(float(x), 6) for x in rank_b],
            }
            print(f"scored {raw['id']}", flush=True)
    idmap = _ids()
    golds = {lang: _load_rows(GOLD / f"gold_admin_{lang}.jsonl") for lang in LANGS}
    frozen = {
        lang: _load_rows(REPO / "track_2a/predictions/dev" / f"SpanDiff_admin_{lang}.jsonl.jsonl")
        for lang in LANGS
    }
    from sprint3_post import _gate

    result = _gate("layers", table, golds, frozen, idmap)
    dest = REPO / "track_2a/docs/sprint3_layers.json"
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("WROTE", dest)
    score.chosen_layer_tensor = real_chosen
    encode_side_subwords  # silence unused if the import stays
    return 0


def _tier1(sa, ca, oa, sb, cb, ob, text_a, text_b, tokens_a, tokens_b):
    from spandiff.score import (  # noqa: E402
        _assign_to_sents,
        _pool_sub_scores_np,
        _sentence_spans,
    )

    if sa.size == 0 or sb.size == 0:
        return [1.0] * len(tokens_a), [1.0] * len(tokens_b)
    mean = np.concatenate([sa, sb], axis=0).mean(axis=0, keepdims=True)
    sa = sa - 0.50 * mean
    sb = sb - 0.50 * mean
    na = np.linalg.norm(sa, axis=1, keepdims=True)
    nb = np.linalg.norm(sb, axis=1, keepdims=True)
    ua = np.divide(sa, na, out=np.zeros_like(sa), where=na > 0)
    ub = np.divide(sb, nb, out=np.zeros_like(sb), where=nb > 0)
    sim = ua @ ub.T
    ka = min(5, sim.shape[1])
    kb = min(5, sim.shape[0])
    ra = np.partition(sim, -ka, axis=1)[:, -ka:].mean(axis=1, keepdims=True)
    rb = np.partition(sim, -kb, axis=0)[-kb:, :].mean(axis=0, keepdims=True)
    eff = sim - 0.50 * 0.5 * (ra + rb)
    max_a = eff.max(axis=1)
    max_b = eff.max(axis=0)
    ss_a = _sentence_spans(text_a)
    ss_b = _sentence_spans(text_b)
    if len(ss_a) > 1 and len(ss_b) > 1:
        sid_a = _assign_to_sents(oa, ss_a)
        sid_b = _assign_to_sents(ob, ss_b)
        sent_a = np.zeros((len(ss_a), sa.shape[1]), dtype=np.float32)
        sent_b = np.zeros((len(ss_b), sb.shape[1]), dtype=np.float32)
        for s_idx in range(len(ss_a)):
            mask = sid_a == s_idx
            if mask.any():
                vec = sa[mask].mean(axis=0)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    sent_a[s_idx] = vec / norm
        for s_idx in range(len(ss_b)):
            mask = sid_b == s_idx
            if mask.any():
                vec = sb[mask].mean(axis=0)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    sent_b[s_idx] = vec / norm
        mapping = sent_a @ sent_b.T
        target_b = mapping.argmax(axis=1)[sid_a]
        mask_ab = np.abs(target_b[:, None] - sid_b[None, :]) <= 1
        sent_max_a = np.where(mask_ab, eff, -10.0).max(axis=1)
        sent_max_a = np.where(sent_max_a < -5.0, max_a, sent_max_a)
        target_a = mapping.argmax(axis=0)[sid_b]
        mask_ba = np.abs(target_a[:, None] - sid_a[None, :]) <= 1
        sent_max_b = np.where(mask_ba, eff.T, -10.0).max(axis=1)
        sent_max_b = np.where(sent_max_b < -5.0, max_b, sent_max_b)
        max_a = 0.65 * max_a + 0.35 * sent_max_a
        max_b = 0.65 * max_b + 0.35 * sent_max_b
    word_a = _pool_sub_scores_np(np.clip(1.0 - max_a, 0.0, 1.0), ca)
    word_b = _pool_sub_scores_np(np.clip(1.0 - max_b, 0.0, 1.0), cb)
    return (
        _span_smooth_tri(tokens_a, word_a, w=2),
        _span_smooth_tri(tokens_b, word_b, w=2),
    )


if __name__ == "__main__":
    raise SystemExit(main())
