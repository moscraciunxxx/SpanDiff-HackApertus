#!/usr/bin/env python3
"""Sprint 3 ideas 1 and 2. Development documents only. No sealed labels."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[1]
sys.path.insert(0, str(SRC))

from spandiff import MODEL_ID  # noqa: E402
from spandiff.align import assign_subwords, locate_tokens, pack_spans, source_text  # noqa: E402
from spandiff.score import _assign_to_sents, _sentence_spans  # noqa: E402
from sprint3_post import GOLD, LANGS, _digits, _gate, _ids, _load, _mismatch  # noqa: E402

CACHE = Path("/tmp/spandiff-surprisal.jsonl")
NAMES = {"de": "German", "fr": "French", "it": "Italian"}


def _load_lm():
    import json as _json
    from copy import deepcopy

    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.apertus1p5.modeling_apertus1p5 import (
        Apertus1p5TextForCausalLM,
        Apertus1p5TextRotaryEmbedding,
    )
    from transformers.utils.hub import cached_file

    torch.manual_seed(0)
    device = "mps"
    torch.mps.set_per_process_memory_fraction(0.80)
    snap = Path(cached_file(MODEL_ID, "model.safetensors.index.json")).parent
    full_cfg = AutoConfig.from_pretrained(str(snap), local_files_only=True)
    text_cfg = deepcopy(full_cfg.text_config)
    text_cfg._attn_implementation = "sdpa"
    text_cfg.dtype = torch.bfloat16
    with torch.device("meta"):
        model = Apertus1p5TextForCausalLM(text_cfg)
    idx_map = _json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    needed: dict[str, list[tuple[str, str]]] = {}
    for name in model.state_dict().keys():
        fk = f"model.language_model.{name}" if not name.startswith("lm_head.") else name
        if name.startswith("model."):
            fk = f"model.language_model.{name[len('model.'):]}"
        if fk in idx_map:
            needed.setdefault(idx_map[fk], []).append((name, fk))
    state = {}
    for shard, pairs in needed.items():
        with safe_open(str(snap / shard), framework="pt", device="cpu") as handle:
            for local, fk in pairs:
                state[local] = handle.get_tensor(fk).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(state, assign=True, strict=False)
    del state
    model.model.rotary_emb = Apertus1p5TextRotaryEmbedding(config=text_cfg)
    model.model.rotary_emb.inv_freq = model.model.rotary_emb.inv_freq.to(device)
    for module in model.modules():
        for key, buf in list(module._buffers.items()):
            if buf is not None and buf.device.type == "meta":
                module._buffers[key] = torch.zeros_like(buf, device=device)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    return model, tokenizer, device


def _rank01(values) -> np.ndarray:
    from scipy.stats import rankdata

    arr = np.asarray(values, dtype=np.float64)
    if len(arr) <= 1:
        return np.zeros(len(arr))
    return (rankdata(arr, method="average") - 1.0) / (len(arr) - 1.0)


def _blend(frozen: list[float], surprisal: list[float]) -> list[float]:
    mixed = 0.5 * _rank01(frozen) + 0.5 * _rank01(surprisal)
    return [round(float(x), 6) for x in mixed]


def _target_logprobs(model, device, ids: list[int], start: int):
    import torch

    tensor = torch.tensor([ids], device=device)
    with torch.no_grad():
        logits = model(tensor, use_cache=False).logits[0, start - 1 : -1].float()
    targets = tensor[0, start:]
    logp = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
    return logp.detach().cpu().tolist()


def _side_surprisal(model, tokenizer, device, prefix: str, source: str, target: str, spans):
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    source_ids = tokenizer.encode(source, add_special_tokens=False)
    newline = tokenizer.encode("\n", add_special_tokens=False)
    bos = int(tokenizer.bos_token_id)
    scores = [0.0] * len(spans)

    def fits(piece: str) -> bool:
        return 1 + len(prefix_ids) + len(tokenizer.encode(piece, add_special_tokens=False)) <= 512

    cursor = 0
    for group in pack_spans(target, spans, fits):
        start = group[0][0]
        piece = target[start : group[-1][1]]
        local = [(a - start, b - start) for a, b in group]
        encoded = tokenizer(
            piece,
            return_offsets_mapping=True,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        target_ids = list(encoded["input_ids"])
        offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
        budget = 512 - (1 + len(prefix_ids) + len(newline) + len(target_ids))
        kept = source_ids[-budget:] if budget > 0 else []
        cond = [bos] + prefix_ids + kept + newline + target_ids
        bare = [bos] + prefix_ids + target_ids
        cond_logp = _target_logprobs(model, device, cond, len(cond) - len(target_ids))
        bare_logp = _target_logprobs(model, device, bare, len(bare) - len(target_ids))
        surp = [bare - cond for bare, cond in zip(bare_logp, cond_logp)]
        for bucket in assign_subwords(offsets, local):
            if bucket:
                scores[cursor] = float(np.mean([surp[i] for i in bucket]))
            cursor += 1
    if cursor != len(spans):
        raise RuntimeError(f"surprisal covered {cursor} tokens, expected {len(spans)}")
    return scores


def _done() -> set[str]:
    if not CACHE.is_file():
        return set()
    found = set()
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            found.add(row["lang"] + ":" + row["id"])
    return found


def _score_all(model, tokenizer, device) -> None:
    done = _done()
    with CACHE.open("a", encoding="utf-8") as handle:
        for lang in LANGS:
            path = REPO / "track_2a/data/swissgov" / lang / f"dev_{lang}.jsonl"
            prefix_b = f"Translate English to {NAMES[lang]}:\n"
            prefix_a = f"Translate {NAMES[lang]} to English:\n"
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                raw = json.loads(line)
                key = lang + ":" + raw["id"]
                if key in done:
                    continue
                if os.environ.get("SPRINT3_CHECK") == "1" and raw["id"] != "admin_de_0":
                    continue
                raw["labels_a"] = [0.0] * len(str(raw["text_a"]).split())
                raw["labels_b"] = [0.0] * len(str(raw["text_b"]).split())
                text_a, tokens_a = source_text(raw, "a")
                text_b, tokens_b = source_text(raw, "b")
                sur_b = _side_surprisal(
                    model, tokenizer, device, prefix_b, text_a, text_b, locate_tokens(text_b, tokens_b)
                )
                sur_a = _side_surprisal(
                    model, tokenizer, device, prefix_a, text_b, text_a, locate_tokens(text_a, tokens_a)
                )
                handle.write(json.dumps({"lang": lang, "id": raw["id"], "a": sur_a, "b": sur_b}) + "\n")
                handle.flush()
                print(f"scored {raw['id']}", flush=True)
                if os.environ.get("SPRINT3_CHECK") == "1":
                    print("CHECK", len(sur_a), len(tokens_a), len(sur_b), len(tokens_b), flush=True)
                    return


def _cached() -> dict[str, dict[str, dict]]:
    found = {lang: {} for lang in LANGS}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found[row["lang"]][row["id"]] = row
    return found


def _evaluate(cached) -> None:
    idmap = _ids()
    golds = {lang: _load(GOLD / f"gold_admin_{lang}.jsonl") for lang in LANGS}
    frozen = {
        lang: _load(REPO / "track_2a/predictions/dev" / f"SpanDiff_admin_{lang}.jsonl.jsonl")
        for lang in LANGS
    }
    solo = {lang: {} for lang in LANGS}
    for lang in LANGS:
        for item, base in frozen[lang].items():
            sur = cached[lang][item]
            solo[lang][item] = {
                "labels_a": _blend(base["labels_a"], sur["a"]),
                "labels_b": _blend(base["labels_b"], sur["b"]),
            }
    result = {"idea1": _gate("surprisal", solo, golds, frozen, idmap)}
    if result["idea1"]["adopt"]:
        stacked = {lang: {} for lang in LANGS}
        for lang in LANGS:
            for item, row in solo[lang].items():
                base = frozen[lang][item]
                tokens_a = str(base["text_a"]).split()
                tokens_b = str(base["text_b"]).split()
                stacked[lang][item] = {
                    "labels_a": _mismatch(tokens_a, _digits(base["text_b"]), row["labels_a"]),
                    "labels_b": _mismatch(tokens_b, _digits(base["text_a"]), row["labels_b"]),
                }
        result["idea1_stack"] = _gate("surprisal_stack", stacked, golds, frozen, idmap)
    both = _bidirectional(cached, frozen)
    result["idea2"] = _gate("bidirectional", both, golds, frozen, idmap)
    if result["idea2"]["adopt"]:
        stacked = {lang: {} for lang in LANGS}
        for lang in LANGS:
            for item, row in both[lang].items():
                base = frozen[lang][item]
                tokens_a = str(base["text_a"]).split()
                tokens_b = str(base["text_b"]).split()
                stacked[lang][item] = {
                    "labels_a": _mismatch(tokens_a, _digits(base["text_b"]), row["labels_a"]),
                    "labels_b": _mismatch(tokens_b, _digits(base["text_a"]), row["labels_b"]),
                }
        result["idea2_stack"] = _gate("bidirectional_stack", stacked, golds, frozen, idmap)
    dest = REPO / "track_2a/docs/sprint3_surprisal.json"
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("WROTE", dest, flush=True)


def _bidirectional(cached, frozen):
    """Project the other side through the Tier-1 greedy sentence match, best sentence only."""
    import spandiff.score as score
    from spandiff.score import encode_side_subwords, load_apertus

    score.LAYER_INDEX = 15
    model, tokenizer, device, max_length = load_apertus()
    max_length = min(int(max_length), 512)
    table = {lang: {} for lang in LANGS}
    try:
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
                sur = cached[lang][raw["id"]]
                base = frozen[lang][raw["id"]]
                proj_a = _project_side(text_a, ca, oa, sur["b"], text_b, cb, ob, sa, sb)
                proj_b = _project_side(text_b, cb, ob, sur["a"], text_a, ca, oa, sb, sa)
                third_a = (_rank01(base["labels_a"]) + _rank01(sur["a"]) + _rank01(proj_a)) / 3.0
                third_b = (_rank01(base["labels_b"]) + _rank01(sur["b"]) + _rank01(proj_b)) / 3.0
                table[lang][raw["id"]] = {
                    "labels_a": [round(float(x), 6) for x in third_a],
                    "labels_b": [round(float(x), 6) for x in third_b],
                }
    finally:
        del model
    return table


def _sent_vecs(centered, offsets, text):
    spans = _sentence_spans(text)
    if centered.size == 0:
        return np.zeros((len(spans), 1), dtype=np.float32), spans
    sids = _assign_to_sents(offsets, spans)
    out = np.zeros((len(spans), centered.shape[1]), dtype=np.float32)
    for s_idx in range(len(spans)):
        mask = sids == s_idx
        if mask.any():
            vec = centered[mask].mean(axis=0)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                out[s_idx] = vec / norm
    return out, spans


def _word_sent(counts, offsets, spans):
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


def _project_side(text, counts, offsets, own_sur, other_text, other_counts, other_offsets, centered, other_centered):
    sent, spans = _sent_vecs(centered, offsets, text)
    other, other_spans = _sent_vecs(other_centered, other_offsets, other_text)
    if not len(spans) or not len(other_spans):
        return own_sur
    partner = (sent @ other.T).argmax(axis=1)
    word_sid = _word_sent(counts, offsets, spans)
    other_sid = _word_sent(other_counts, other_offsets, other_spans)
    buckets = {idx: [] for idx in range(len(other_spans))}
    for sid, value in zip(other_sid, own_sur if False else []):
        buckets[int(sid)].append(value)
    # own_sur is this side. The projected value comes from the other side's surprisal,
    # passed in as the fourth argument's counterpart by the caller via own_sur here
    # being the OTHER side's surprisal. See _bidirectional.
    buckets = {idx: [] for idx in range(len(other_spans))}
    for sid, value in zip(other_sid, own_sur):
        buckets[int(sid)].append(value)
    means = {idx: float(np.mean(vals)) if vals else 0.0 for idx, vals in buckets.items()}
    return [means[int(partner[int(sid)])] for sid in word_sid]


def main() -> int:
    if os.environ.get("SPRINT3_EVAL_ONLY") == "1":
        _evaluate(_cached())
        return 0
    model, tokenizer, device = _load_lm()
    _score_all(model, tokenizer, device)
    del model
    if os.environ.get("SPRINT3_CHECK") == "1" or os.environ.get("SPRINT3_SCORE_ONLY") == "1":
        return 0
    _evaluate(_cached())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
