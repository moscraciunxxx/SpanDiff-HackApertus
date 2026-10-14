#!/usr/bin/env python3
"""Sprint 3 idea 7: sentence-similarity prior on the frozen scores. Development only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[1]
sys.path.insert(0, str(SRC))

import spandiff.score as score  # noqa: E402
from spandiff.align import locate_tokens, source_text  # noqa: E402
from spandiff.score import (  # noqa: E402
    _assign_to_sents,
    _sentence_spans,
    encode_side_subwords,
    load_apertus,
)
from sprint3_post import GOLD, LANGS, _gate, _ids, _load  # noqa: E402


def _sentence_vectors(centered, offsets, text):
    spans = _sentence_spans(text)
    if centered.size == 0:
        return np.zeros((len(spans), 1), dtype=np.float32), spans
    sids = _assign_to_sents(offsets, spans)
    dim = centered.shape[1]
    out = np.zeros((len(spans), dim), dtype=np.float32)
    for s_idx in range(len(spans)):
        mask = sids == s_idx
        if mask.any():
            vec = centered[mask].mean(axis=0)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                out[s_idx] = vec / norm
    return out, spans


def _word_sentence(counts, offsets, spans):
    mids = []
    cur = 0
    for count in counts:
        if count == 0:
            mids.append(0.0)
        else:
            start = int(offsets[cur, 0])
            end = int(offsets[cur + count - 1, 1])
            mids.append(0.5 * (start + end))
            cur += count
    ends = np.asarray([end for _, end in spans], dtype=np.float64)
    return np.clip(np.searchsorted(ends, np.asarray(mids), side="right"), 0, max(len(spans) - 1, 0))


def _apply(scores, sids, sent, other):
    if other.size == 0 or sent.size == 0:
        return [round(float(x), 6) for x in scores]
    sim = sent @ other.T
    best = sim.max(axis=1) if sim.size else np.zeros(len(sent))
    out = []
    for score, sid in zip(scores, sids):
        factor = 1.0 - float(best[sid]) if len(best) else 1.0
        out.append(round(float(min(1.0, max(0.0, score * factor))), 6))
    return out


def main() -> int:
    score.LAYER_INDEX = 15
    model, tokenizer, device, max_length = load_apertus()
    max_length = min(int(max_length), 512)
    frozen = {
        lang: _load(REPO / "track_2a/predictions/dev" / f"SpanDiff_admin_{lang}.jsonl.jsonl")
        for lang in LANGS
    }
    table = {lang: {} for lang in LANGS}
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
            sa, ca, oa, _ = encode_side_subwords(
                model, tokenizer, text_a, locate_tokens(text_a, tokens_a), device, max_length
            )
            sb, cb, ob, _ = encode_side_subwords(
                model, tokenizer, text_b, locate_tokens(text_b, tokens_b), device, max_length
            )
            if sa.size and sb.size:
                mean = np.concatenate([sa, sb], axis=0).mean(axis=0, keepdims=True)
                sa = sa - 0.50 * mean
                sb = sb - 0.50 * mean
            sent_a, spans_a = _sentence_vectors(sa, oa, text_a)
            sent_b, spans_b = _sentence_vectors(sb, ob, text_b)
            base = frozen[lang][raw["id"]]
            sid_a = _word_sentence(ca, oa, spans_a)
            sid_b = _word_sentence(cb, ob, spans_b)
            table[lang][raw["id"]] = {
                "labels_a": _apply(base["labels_a"], sid_a, sent_a, sent_b),
                "labels_b": _apply(base["labels_b"], sid_b, sent_b, sent_a),
            }
            print(f"scored {raw['id']}", flush=True)
    idmap = _ids()
    golds = {lang: _load(GOLD / f"gold_admin_{lang}.jsonl") for lang in LANGS}
    result = _gate("sentence", table, golds, frozen, idmap)
    dest = REPO / "track_2a/docs/sprint3_sentence.json"
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("WROTE", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
