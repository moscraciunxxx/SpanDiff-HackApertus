"""Tokenizer-only census of the development windows.

Opens tokenizer.json and the development JSONL. Does not open model weights.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from spandiff.align import assign_subwords, locate_tokens, pack_spans, source_text

WINDOW = 512


def tokenizer_json() -> Path:
    """Local Apertus tokenizer file. A missing file is a refusal, not a download."""
    override = os.environ.get("SPAN_DIFF_TOKENIZER_JSON")
    if override:
        path = Path(override)
    else:
        hf_home = os.environ.get("HF_HOME")
        hf = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
        hub = hf / "hub" / "models--swiss-ai--Apertus-v1.5-8B"
        ref = hub / "refs" / "main"
        if ref.is_file():
            revision = ref.read_text(encoding="utf-8").strip()
            path = hub / "snapshots" / revision / "tokenizer.json"
        else:
            found = sorted(hub.glob("snapshots/*/tokenizer.json"))
            path = found[-1] if found else Path()
    if path.name != "tokenizer.json" or not path.is_file():
        raise SystemExit(
            "census lock refused: tokenizer.json is not available. "
            "Model weights were not opened."
        )
    return path


def _omni_offset(tokenizer_path: Path) -> int:
    config_path = tokenizer_path.parent / "tokenizer_config.json"
    if not config_path.is_file():
        return 131072
    config = json.loads(config_path.read_text(encoding="utf-8"))
    omni = config.get("omnimodal_config") or {}
    return int(omni.get("omni_special_token_offset") or 131072)


def census_development(tokenizer, rows_by_lang: dict[str, list[dict]], omni_offset: int) -> dict[str, int]:
    """Count windows with the same packer the scorer uses. No hidden states."""
    bos = tokenizer.token_to_id("<s>")
    eos = tokenizer.token_to_id("</s>")
    pad = tokenizer.token_to_id("<pad>")
    unk = tokenizer.token_to_id("<unk>")
    if None in (bos, eos, pad, unk):
        raise SystemExit("census lock refused: tokenizer is missing a core special token")
    core = {bos, eos, pad, unk}
    stats = {
        "sides": 0,
        "untokenized_sides": 0,
        "joined_sides": 0,
        "windows": 0,
        "forward_pass_tokens": 0,
        "forward_pass_de": 0,
        "forward_pass_fr": 0,
        "forward_pass_it": 0,
        "windows_de": 0,
        "windows_fr": 0,
        "windows_it": 0,
        "max_len": WINDOW,
        "over512": 0,
        "single_overflow": 0,
        "encode_mismatch": 0,
        "only_leading_bos": 0,
        "first_not_bos": 0,
        "trailing_special": 0,
        "interior_special": 0,
        "empty_offsets": 0,
        "special_with_span": 0,
        "content_empty_offset": 0,
        "bos_id": bos,
        "eos_id": eos,
        "pad_id": pad,
        "unk_id": unk,
        "bos_count": 0,
        "eos_count": 0,
        "pad_count": 0,
        "unk_count": 0,
        "multimodal_special_count": 0,
        "bos_pooled": 0,
    }
    empty_per_window = set()

    def fits(piece: str) -> bool:
        return len(tokenizer.encode(piece).ids) <= WINDOW

    for lang, rows in rows_by_lang.items():
        for row in rows:
            for side in ("a", "b"):
                text, tokens = source_text(row, side)
                raw = row.get(f"text_{side}_untokenized") or ""
                stats["sides"] += 1
                if text == raw:
                    stats["untokenized_sides"] += 1
                elif text == " ".join(tokens):
                    stats["joined_sides"] += 1
                spans = locate_tokens(text, tokens)
                for group in pack_spans(text, spans, fits):
                    start = group[0][0]
                    piece = text[start : group[-1][1]]
                    local = [(span_start - start, span_end - start) for span_start, span_end in group]
                    encoded = tokenizer.encode(piece)
                    ids = list(encoded.ids)
                    offsets = [(int(a), int(b)) for a, b in encoded.offsets]
                    if len(ids) != len(offsets):
                        stats["encode_mismatch"] += 1
                    stats["windows"] += 1
                    stats["forward_pass_tokens"] += len(ids)
                    stats[f"windows_{lang}"] += 1
                    stats[f"forward_pass_{lang}"] += len(ids)
                    if len(ids) > WINDOW:
                        stats["over512"] += 1
                        if len(group) == 1:
                            stats["single_overflow"] += 1
                    special_positions = [
                        index
                        for index, token_id in enumerate(ids)
                        if token_id in core or token_id >= omni_offset
                    ]
                    if not ids or ids[0] != bos:
                        stats["first_not_bos"] += 1
                    elif all(index == 0 for index in special_positions) and ids.count(bos) == 1:
                        stats["only_leading_bos"] += 1
                    later = [index for index in special_positions if index > 0]
                    if later:
                        if ids[-1] in core or ids[-1] >= omni_offset:
                            stats["trailing_special"] += 1
                        if any(0 < index < len(ids) - 1 for index in later):
                            stats["interior_special"] += 1
                    empty = 0
                    for token_id, (start_off, end_off) in zip(ids, offsets):
                        is_special = token_id in core or token_id >= omni_offset
                        if end_off <= start_off:
                            empty += 1
                            if not is_special:
                                stats["content_empty_offset"] += 1
                        elif is_special:
                            stats["special_with_span"] += 1
                        if token_id == bos:
                            stats["bos_count"] += 1
                        elif token_id == eos:
                            stats["eos_count"] += 1
                        elif token_id == pad:
                            stats["pad_count"] += 1
                        elif token_id == unk:
                            stats["unk_count"] += 1
                        elif token_id >= omni_offset:
                            stats["multimodal_special_count"] += 1
                    stats["empty_offsets"] += empty
                    empty_per_window.add(empty)
                    pooled = {
                        index
                        for bucket in assign_subwords(offsets, local)
                        for index in bucket
                    }
                    if any(index in pooled for index, token_id in enumerate(ids) if token_id == bos):
                        stats["bos_pooled"] += 1
    if empty_per_window != {1}:
        stats["empty_per_window"] = -1
    else:
        stats["empty_per_window"] = 1
    return stats


def parse_census_file(text: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, value = stripped.rsplit(" ", 1)
        parsed[key] = int(value)
    return parsed


def lock_line(stats: dict[str, int]) -> str:
    """One line the self-test log must show. Numbers come from the census."""
    return (
        f"census lock: {stats['windows']} windows, leading BOS only, BOS not pooled, "
        f"EOS/PAD/UNK/multimodal counts {stats['eos_count']}/{stats['pad_count']}/"
        f"{stats['unk_count']}/{stats['multimodal_special_count']}, "
        f"{stats['forward_pass_tokens']} tokens"
    )
