#!/usr/bin/env python3
"""Score one Apertus hidden state into a new prefix.

Reads id and text fields only. Placeholder label lists are built from
whitespace tokens so the scorer can check lengths. Gold labels are not
read and the layer-8 prediction files are not opened for writing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[1]
sys.path.insert(0, str(SRC))

from spandiff.devdata import LANGS, dev_store, local_dev_file  # noqa: E402
from spandiff.score import dumps_line, load_apertus, score_record  # noqa: E402
import spandiff.score as score  # noqa: E402

TEXT_KEYS = ("id", "text_a", "text_b", "text_a_untokenized", "text_b_untokenized")


def text_only(line: str) -> dict:
    raw = json.loads(line)
    record = {key: raw[key] for key in TEXT_KEYS if key in raw}
    for side in ("a", "b"):
        text = record[f"text_{side}"]
        record[f"labels_{side}"] = [0.0] * len(text.split())
    return record


def load_text_records(path: Path) -> dict[str, dict]:
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = text_only(line)
        found[record["id"]] = record
    return found


def id_order(path: Path) -> list[str]:
    return [
        json.loads(line)["id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def done_ids(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [
        json.loads(line)["id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--order-from", type=Path)
    parser.add_argument("--texts", type=Path)
    args = parser.parse_args(argv)
    if not 0 <= args.layer <= 32:
        raise SystemExit(f"refusing layer {args.layer}")
    score.LAYER_INDEX = args.layer
    if args.texts is None and args.order_from is None:
        raise SystemExit("pass --order-from or --texts")
    out_dir = args.out if args.out.is_absolute() else REPO / args.out
    text_dir = None if args.texts is None else (args.texts if args.texts.is_absolute() else REPO / args.texts)
    order_dir = None if args.order_from is None else (args.order_from if args.order_from.is_absolute() else REPO / args.order_from)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device, max_length = load_apertus()
    total_tokens = 0
    started = time.perf_counter()
    for lang in LANGS:
        if text_dir is not None:
            source = text_dir / f"{lang}.jsonl"
            records = load_text_records(source)
            order = [
                json.loads(line)["id"]
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            records = load_text_records(local_dev_file(dev_store(REPO), lang))
            order = id_order(order_dir / f"SpanDiff_admin_{lang}.jsonl.jsonl")
        missing = [item for item in order if item not in records]
        if missing:
            raise SystemExit(f"{lang}: {len(missing)} ids are absent from the dev text")
        dest = out_dir / f"SpanDiff_admin_{lang}.jsonl.jsonl"
        finished = done_ids(dest)
        if finished != order[: len(finished)]:
            raise SystemExit(f"{lang}: existing {dest.name} is not a prefix of the id order")
        print(f"{lang} resume {len(finished)}/{len(order)}", flush=True)
        with dest.open("a", encoding="utf-8") as handle:
            for item_id in order[len(finished) :]:
                t0 = time.perf_counter()
                line, tokens = score_record(
                    model, tokenizer, records[item_id], device, max_length
                )
                handle.write(line + "\n")
                handle.flush()
                total_tokens += tokens
                print(
                    f"scored {item_id} layer {args.layer} tokens {tokens} "
                    f"seconds {time.perf_counter() - t0:.1f}",
                    flush=True,
                )
    print(
        f"forward_pass_tokens {total_tokens} "
        f"seconds {time.perf_counter() - started:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
