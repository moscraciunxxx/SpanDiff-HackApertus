#!/usr/bin/env python3
"""Sprint 3 idea 10. Ridge, alpha 1, doc-level out-of-fold predictions only."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[1]
sys.path.insert(0, str(SRC))

import spandiff.score as score  # noqa: E402
from spandiff.align import locate_tokens, source_text  # noqa: E402
from spandiff.score import PUNCT, encode_side_subwords, load_apertus  # noqa: E402
from sprint3_post import GOLD, LANGS, STOP, _digits, _gate, _ids, _load, _spearman  # noqa: E402
from sprint3_surprisal import _cached, _rank01  # noqa: E402
from sprint3_sentence import _sentence_vectors, _word_sentence  # noqa: E402

ALPHA = 1.0
FEATURE_NAMES = (
    "frozen_score",
    "surprisal_rank",
    "mismatch",
    "sentence_prior",
    "relative_position",
    "stopword",
    "punctuation",
    "token_char_length",
    "doc_length_ratio",
    "side",
)
CACHE = Path(os.environ.get("STACKER_CACHE", "/tmp/spandiff-stacker-feat"))
BASELINE = {"de": 0.373, "fr": 0.231, "it": 0.398}


def _fit_predict(train_x, train_y, test_x):
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    z_train = (train_x - mu) / sd
    z_test = (test_x - mu) / sd
    design_train = np.concatenate([np.ones((len(z_train), 1)), z_train], axis=1)
    design_test = np.concatenate([np.ones((len(z_test), 1)), z_test], axis=1)
    gram = design_train.T @ design_train
    pen = np.eye(gram.shape[0]) * ALPHA
    pen[0, 0] = 0.0
    beta = np.linalg.solve(gram + pen, design_train.T @ train_y)
    pred = design_test @ beta
    return np.clip(pred, 0.0, 1.0), beta, mu, sd


def _assert_features(doc_id: str, features: np.ndarray, labels: np.ndarray) -> list[str]:
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise SystemExit(f"feature width on {doc_id}")
    if features.shape[0] != labels.shape[0]:
        raise SystemExit(f"feature rows on {doc_id}")
    copied = []
    keep = labels != -1
    if int(keep.sum()) >= 3:
        gold = labels[keep]
        for index, name in enumerate(FEATURE_NAMES):
            column = features[keep, index]
            # Constant columns can match a page whose kept gold is the same constant.
            if float(np.std(column)) > 1e-8 and np.allclose(column, gold):
                copied.append(name)
    numeric = float(doc_id.rsplit("_", 1)[-1])
    # 0 and 1 are legal constant values for the binary features, so only larger ids can be tested this way.
    if numeric >= 2:
        for index, name in enumerate(FEATURE_NAMES):
            if np.allclose(features[:, index], numeric):
                raise SystemExit(f"{name} copies the document id on {doc_id}")
    return copied


def _features(raw, frozen_row, sur, sent_factor, lang: str):
    tokens_a = str(raw["text_a"]).split()
    tokens_b = str(raw["text_b"]).split()
    stops = set(STOP["en"]) | set(STOP[lang])
    digits_a = _digits(raw["text_b"])
    digits_b = _digits(raw["text_a"])
    ratio = max(len(tokens_a), len(tokens_b)) / max(min(len(tokens_a), len(tokens_b)), 1)
    rows = []
    golds = []
    for side, tokens, other_digits, factor in (
        ("a", tokens_a, digits_a, sent_factor[0]),
        ("b", tokens_b, digits_b, sent_factor[1]),
    ):
        scores = frozen_row[f"labels_{side}"]
        surprisal = sur[side]
        ranks = _rank01(surprisal)
        labels = raw[f"labels_{side}"]
        denom = max(len(tokens) - 1, 1)
        for index, token in enumerate(tokens):
            values = _digits(token)
            mismatch = 1.0 if values and any(item not in other_digits for item in values) else 0.0
            punct = 1.0 if token and all(ch in PUNCT for ch in token) else 0.0
            rows.append([
                float(scores[index]),
                float(ranks[index]),
                mismatch,
                float(factor[index]),
                index / denom,
                1.0 if token.lower() in stops else 0.0,
                punct,
                float(len(token)),
                float(ratio),
                0.0 if side == "a" else 1.0,
            ])
            golds.append(float(labels[index]))
    features = np.asarray(rows, dtype=np.float64)
    gold = np.asarray(golds, dtype=np.float64)
    copied = _assert_features(str(raw["id"]), features, gold)
    if copied:
        print(f"NONCONSTANT gold copy {raw['id']} {copied}", flush=True)
    return features, gold


def _cache_file(doc_id: str) -> Path:
    return CACHE / f"{doc_id}.npz"


def _read_cache(doc_id: str):
    path = _cache_file(doc_id)
    if not path.is_file():
        return None
    blob = np.load(path)
    features, gold = blob["x"], blob["y"]
    copied = _assert_features(doc_id, features, gold)
    if copied:
        print(f"NONCONSTANT gold copy {doc_id} {copied}", flush=True)
    return features, gold


def _write_cache(doc_id: str, features, gold) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(_cache_file(doc_id), x=features, y=gold)


def _featurize_doc(model, tokenizer, device, max_length, raw, frozen_row, sur, lang: str):
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
    from sprint3_sentence import _apply

    ones_a = [1.0] * len(tokens_a)
    ones_b = [1.0] * len(tokens_b)
    # _apply multiplies by (1 - max cosine). Feeding ones yields the prior itself.
    factor_a = _apply(ones_a, _word_sentence(ca, oa, spans_a), sent_a, sent_b)
    factor_b = _apply(ones_b, _word_sentence(cb, ob, spans_b), sent_b, sent_a)
    features, labels = _features(raw, frozen_row, sur, (factor_a, factor_b), lang)
    blank = dict(raw)
    blank["labels_a"] = [0.0] * len(raw["labels_a"])
    blank["labels_b"] = [0.0] * len(raw["labels_b"])
    alt, _ = _features(blank, frozen_row, sur, (factor_a, factor_b), lang)
    if not np.allclose(features, alt):
        raise SystemExit(f"features changed when gold labels were replaced on {raw['id']}")
    return features, labels


def _named(beta, mu, sd) -> dict:
    names = ("intercept",) + FEATURE_NAMES
    return {
        "intercept": float(beta[0]),
        "weights": {name: float(beta[index + 1]) for index, name in enumerate(FEATURE_NAMES)},
        "mean": {name: float(mu[index]) for index, name in enumerate(FEATURE_NAMES)},
        "std": {name: float(sd[index]) for index, name in enumerate(FEATURE_NAMES)},
        "beta": [float(value) for value in beta],
        "order": list(names),
    }


def main() -> int:
    if ALPHA != 1.0:
        raise SystemExit("alpha left the pre-registered value 1.0")
    cached = _cached()
    for lang, rows in cached.items():
        for doc_id in rows:
            if not doc_id.startswith(f"admin_{lang}_"):
                raise SystemExit(f"surprisal cache id is not a development id: {doc_id}")
    golds = {lang: _load(GOLD / f"gold_admin_{lang}.jsonl") for lang in LANGS}
    if "test" in str(GOLD):
        raise SystemExit("gold path is not the development split")
    frozen = {
        lang: _load(REPO / "track_2a/predictions/dev" / f"SpanDiff_admin_{lang}.jsonl.jsonl")
        for lang in LANGS
    }
    texts = []
    for lang in LANGS:
        path = REPO / "track_2a/data/swissgov" / lang / f"dev_{lang}.jsonl"
        if path.name.startswith("test"):
            raise SystemExit(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                texts.append((lang, json.loads(line)))
    missing = [(lang, raw) for lang, raw in texts if _read_cache(raw["id"]) is None]
    model = None
    if missing:
        score.LAYER_INDEX = 15
        model, tokenizer, device, max_length = load_apertus()
        max_length = min(int(max_length), 512)
        for lang, raw in missing:
            gold = golds[lang][raw["id"]]
            raw = dict(raw)
            raw["labels_a"] = gold["labels_a"]
            raw["labels_b"] = gold["labels_b"]
            features, labels = _featurize_doc(
                model, tokenizer, device, max_length, raw, frozen[lang][raw["id"]], cached[lang][raw["id"]], lang
            )
            _write_cache(raw["id"], features, labels)
            print(f"featurized {raw['id']}", flush=True)
        del model
    packed = {lang: {} for lang in LANGS}
    for lang, raw in texts:
        packed[lang][raw["id"]] = _read_cache(raw["id"])
        print(f"loaded {raw['id']}", flush=True)
    all_docs = [item for lang in LANGS for item in packed[lang]]
    table = {lang: {} for lang in LANGS}
    fold_models = []
    assigned = {}
    for fold in range(5):
        train_x, train_y = [], []
        test_keys = []
        train_docs, test_docs = [], []
        for lang in LANGS:
            for item, (features, labels) in packed[lang].items():
                bucket = int(item.rsplit("_", 1)[-1]) % 5
                keep = labels != -1
                if bucket == fold:
                    test_keys.append((lang, item, features))
                    test_docs.append(item)
                else:
                    train_docs.append(item)
                    if keep.any():
                        train_x.append(features[keep])
                        train_y.append(labels[keep])
        if set(train_docs) & set(test_docs):
            raise SystemExit(f"document overlap in fold {fold}")
        if sorted(train_docs + test_docs) != sorted(all_docs):
            raise SystemExit(f"fold {fold} does not partition the documents")
        for item in test_docs:
            if item in assigned:
                raise SystemExit(f"{item} is held out of more than one fold")
            assigned[item] = fold
        pred, beta, mu, sd = _fit_predict(
            np.vstack(train_x), np.concatenate(train_y), np.vstack([item[2] for item in test_keys])
        )
        cursor = 0
        for lang, item, features in test_keys:
            n_a = len(frozen[lang][item]["labels_a"])
            got = pred[cursor : cursor + len(features)]
            cursor += len(features)
            table[lang][item] = {
                "labels_a": [round(float(x), 6) for x in got[:n_a]],
                "labels_b": [round(float(x), 6) for x in got[n_a:]],
            }
        named = _named(beta, mu, sd)
        named["fold"] = fold
        named["n_train_docs"] = len(train_docs)
        named["n_test_docs"] = len(test_docs)
        named["n_train_tokens"] = int(sum(len(rows) for rows in train_y))
        fold_models.append(named)
        print(f"fold {fold} docs {len(test_keys)}", flush=True)
    if set(assigned) != set(all_docs):
        raise SystemExit("a document was never held out")
    idmap = _ids()
    for lang in LANGS:
        official_train = set(idmap[lang]["train"])
        official_val = set(idmap[lang]["val"])
        if official_train & official_val:
            raise SystemExit(f"official train and val share a document in {lang}")
        if (official_train | official_val) != set(packed[lang]):
            raise SystemExit(f"official ids do not match the development documents in {lang}")
    result = _gate("stacker", table, golds, frozen, idmap)
    matrix = np.vstack([row["beta"] for row in fold_models])
    order = ["intercept", *FEATURE_NAMES]
    flips = []
    weight_std = {}
    for index, name in enumerate(order):
        column = matrix[:, index]
        weight_std[name] = float(np.std(column, ddof=1))
        if float(column.min()) < -1e-8 and float(column.max()) > 1e-8:
            flips.append(name)
    per_language = {}
    for lang in LANGS:
        wanted = set(idmap[lang]["train"]) | set(idmap[lang]["val"])
        rho = _spearman(table[lang], golds[lang], wanted)
        drop = BASELINE[lang] - rho
        per_language[lang] = {
            "oof": round(float(rho), 4),
            "baseline": BASELINE[lang],
            "drop": round(float(drop), 4),
            "pass": bool(drop <= 0.01),
            "train": round(float(_spearman(table[lang], golds[lang], set(idmap[lang]["train"]))), 4),
            "val": round(float(_spearman(table[lang], golds[lang], set(idmap[lang]["val"]))), 4),
        }
    all_x, all_y = [], []
    for lang in LANGS:
        for features, labels in packed[lang].values():
            keep = labels != -1
            if keep.any():
                all_x.append(features[keep])
                all_y.append(labels[keep])
    _, all_beta, all_mu, all_sd = _fit_predict(np.vstack(all_x), np.concatenate(all_y), np.vstack(all_x)[:1])
    language_pass = all(row["pass"] for row in per_language.values())
    audit = {
        "alpha": ALPHA,
        "alpha_fixed_before_scoring": True,
        "feature_names": list(FEATURE_NAMES),
        "features_exclude": ["gold labels", "document ids", "test text statistics"],
        "standardize": "inside each training fold; prediction uses that fold's mean and std",
        "fold_unit": "document, numeric id modulo 5, same page held out in every language",
        "same_document_in_fit_and_heldout": False,
        "official_train_val_disjoint": True,
        "folds": [
            {key: value for key, value in row.items() if key != "beta"}
            for row in fold_models
        ],
        "weight_std_ddof1": weight_std,
        "sign_flips": flips,
        "per_language_oof": per_language,
        "language_pass": language_pass,
        "gate": result,
        "all_dev": _named(all_beta, all_mu, all_sd),
    }
    dest = REPO / "track_2a/docs/sprint3_stacker.json"
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    audit_path = REPO / "track_2a/docs/sprint3_stacker_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    oof_path = Path("/tmp/spandiff-stacker-oof.json")
    oof_path.write_text(json.dumps(table) + "\n", encoding="utf-8")
    print("WROTE", dest, flush=True)
    print("WROTE", audit_path, flush=True)
    print("LANGUAGE", json.dumps(per_language), flush=True)
    print("FLIPS", flips, flush=True)
    print("STD", json.dumps(weight_std), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
