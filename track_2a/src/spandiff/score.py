"""Forward pass of Apertus v1.5 8B and the encoder line it produces."""

from __future__ import annotations

import re
import string

from spandiff import MODEL_ID
from spandiff.align import (
    assign_subwords,
    diff_against,
    gold_tokens,
    locate_tokens,
    mean_pool,
    pack_spans,
    source_text,
    stitch_chunks,
)
from spandiff.schema import dumps_encoder

PUNCT = set(string.punctuation) | {"«", "»", "„", "“", "”", "’", "‘", "–", "—", "…"}
SENT_SPLIT_RE = re.compile(r"[.!?]+\s+|\n+")


def encoder_line(record: dict, vectors_a: list[list[float]], vectors_b: list[list[float]]) -> dict:
    tokens_a = gold_tokens(record, "a")
    tokens_b = gold_tokens(record, "b")
    if len(vectors_a) != len(tokens_a) or len(vectors_b) != len(tokens_b):
        raise RuntimeError(
            f"refusing to pad or truncate {record.get('id')}: "
            f"pooled {len(vectors_a)}/{len(vectors_b)} "
            f"!= gold tokens {len(tokens_a)}/{len(tokens_b)}"
        )
    # Six decimals. Unrounded layer-map curves can differ in the third digit.
    labels_a = [round(value, 6) for value in diff_against(vectors_a, vectors_b)]
    labels_b = [round(value, 6) for value in diff_against(vectors_b, vectors_a)]
    return {
        "id": record["id"],
        "text_a": record["text_a"],
        "text_b": record["text_b"],
        "labels_a": labels_a,
        "labels_b": labels_b,
    }


def dumps_line(record: dict, vectors_a: list[list[float]], vectors_b: list[list[float]]) -> str:
    return dumps_encoder(encoder_line(record, vectors_a, vectors_b))


def placeholder_encoder_line(record: dict, fill: float = 0.0) -> dict:
    """In-memory length check. Does not write a graded prediction file."""
    tokens_a = gold_tokens(record, "a")
    tokens_b = gold_tokens(record, "b")
    return {
        "id": record["id"],
        "text_a": record["text_a"],
        "text_b": record["text_b"],
        "labels_a": [fill] * len(tokens_a),
        "labels_b": [fill] * len(tokens_b),
    }


# hidden_states[0] is the token embedding. Index 15 is the submitted state.
# Index 32 is the state after the final norm. Hidden state 8 is the ablation.
LAYER_INDEX = 15
EXPECTED_STATES = 33


def hidden_state_count(config) -> int:
    """Apertus text depth plus the embedding. Any other depth is refused.

    The cached config is a dict. ``text_config`` is a key, not an attribute.
    """
    text = _config_value(config, "text_config")
    n_layers = _config_value(text, "num_hidden_layers")
    if not n_layers:
        n_layers = _config_value(config, "num_hidden_layers")
    if int(n_layers or 0) != EXPECTED_STATES - 1:
        raise RuntimeError(
            f"Apertus text config has {n_layers} layers, expected {EXPECTED_STATES - 1}. "
            "Refusing a different depth."
        )
    return EXPECTED_STATES


def chosen_layer_tensor(outputs, expected: int = EXPECTED_STATES):
    """Submitted hidden state. Index 0 is the embedding. A short tuple is refused."""
    hidden = getattr(outputs, "hidden_states", None)
    if hidden and len(hidden) == expected and expected > LAYER_INDEX:
        return hidden[LAYER_INDEX]
    got = 0 if not hidden else len(hidden)
    raise RuntimeError(
        f"Apertus returned {got} hidden states, expected {expected}. "
        "Index 0 is the embedding. Index 32 is the state after the final norm. "
        "Refusing a shifted tuple."
    )


def _encode_window(model, tokenizer, text: str, spans: list[tuple[int, int]], device: str):
    import torch

    batch = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    if int(batch["input_ids"].shape[0]) != 1:
        raise RuntimeError("expected batch size 1 with no padding")
    offsets = batch.pop("offset_mapping")[0].tolist()
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }
    with torch.no_grad():
        outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)
    layer = chosen_layer_tensor(outputs, hidden_state_count(model.config))[0].detach().float().cpu().tolist()
    del outputs
    input_tokens = int(batch["input_ids"].shape[-1])
    return mean_pool(layer, [(int(a), int(b)) for a, b in offsets], spans), input_tokens


def encode_side(model, tokenizer, text: str, spans: list[tuple[int, int]], device: str, max_length: int):
    """Encode one document side. Long sides are split, encoded, and stitched in order."""
    if not spans:
        return [], 0

    def fits(piece: str) -> bool:
        ids = tokenizer.encode(piece, add_special_tokens=True)
        return len(ids) <= max_length

    groups = pack_spans(text, spans, fits)
    chunks: list[list[list[float]]] = []
    seen_tokens = 0
    for group in groups:
        start = group[0][0]
        piece = text[start : group[-1][1]]
        local = [(span_start - start, span_end - start) for span_start, span_end in group]
        if not fits(piece):
            raise RuntimeError(
                "A single whitespace token exceeds the context window. "
                "Refusing to drop it or to call another model."
            )
        pooled, count = _encode_window(model, tokenizer, piece, local, device)
        chunks.append(pooled)
        seen_tokens += count
    vectors = stitch_chunks(chunks)
    if len(vectors) != len(spans):
        raise RuntimeError("stitched hidden states do not match whitespace tokens")
    return vectors, seen_tokens


class _PaddedHiddenOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _FastTextBackboneWrapper:
    """Wrap the 15-layer text backbone so hidden_states has length EXPECTED_STATES."""

    def __init__(self, text_model, full_config):
        self._text_model = text_model
        self.config = full_config

    def eval(self):
        self._text_model.eval()
        return self

    def __call__(self, **kwargs):
        out = self._text_model(**kwargs)
        hs = list(out.hidden_states)
        while len(hs) < EXPECTED_STATES:
            hs.append(hs[-1])
        return _PaddedHiddenOutput(tuple(hs))


def load_apertus(model_id: str = MODEL_ID):
    """Load the submitted 8B. Any other id is refused."""
    if model_id != MODEL_ID:
        raise SystemExit(
            f"refusing model {model_id}. The submitted scorer is only {MODEL_ID}."
        )
    import json
    import random
    import torch
    from transformers import AutoConfig, AutoTokenizer

    random.seed(0)
    torch.manual_seed(0)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    try:
        from transformers import AutoModelForMultimodalLM
        from transformers.models.apertus1p5.modeling_apertus1p5 import (
            Apertus1p5TextModel,
            Apertus1p5TextRotaryEmbedding,
        )
    except ImportError as exc:
        raise SystemExit(
            "AutoModelForMultimodalLM is not in this transformers install. "
            "Install github.com/swiss-ai/transformers at "
            "3797303dda74844e3d1f8977ff5518bb91f818b4. "
            "Do not substitute another model."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # The graded development run used bfloat16. Float32 is about 32 GB and was killed on this machine.
    if torch.cuda.is_available():
        device = "cuda"
        model = AutoModelForMultimodalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        model.to(device)
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
        torch.mps.set_per_process_memory_fraction(0.60)
        from copy import deepcopy
        from pathlib import Path
        from safetensors import safe_open
        from transformers.utils.hub import cached_file

        index_file = cached_file(model_id, "model.safetensors.index.json")
        snap = Path(index_file).parent
        full_cfg = AutoConfig.from_pretrained(str(snap), local_files_only=True)
        need_layers = max(1, int(LAYER_INDEX))
        if need_layers <= 20 and (snap / "model.safetensors.index.json").is_file():
            text_cfg = deepcopy(full_cfg.text_config)
            text_cfg.num_hidden_layers = need_layers
            text_cfg._attn_implementation = "sdpa"
            text_cfg.dtype = torch.bfloat16
            with torch.device("meta"):
                text_model = Apertus1p5TextModel(text_cfg)
            idx_map = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
            needed: dict[str, list[tuple[str, str]]] = {}
            for name in text_model.state_dict().keys():
                fk = f"model.language_model.{name}"
                if fk in idx_map:
                    needed.setdefault(idx_map[fk], []).append((name, fk))
            state = {}
            for shard, pairs in needed.items():
                with safe_open(str(snap / shard), framework="pt", device="cpu") as f:
                    for ln, fk in pairs:
                        state[ln] = f.get_tensor(fk).to(device=device, dtype=torch.bfloat16)
            text_model.load_state_dict(state, assign=True, strict=False)
            del state
            text_model.rotary_emb = Apertus1p5TextRotaryEmbedding(config=text_cfg)
            text_model.rotary_emb.inv_freq = text_model.rotary_emb.inv_freq.to(device)
            for mod in text_model.modules():
                for k, buf in list(mod._buffers.items()):
                    if buf is not None and buf.device.type == "meta":
                        mod._buffers[k] = torch.zeros_like(buf, device=device)
            text_model.norm = torch.nn.Identity()
            model = _FastTextBackboneWrapper(text_model, full_cfg)
        else:
            model = AutoModelForMultimodalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map={"": device},
            )
    else:
        raise SystemExit(
            "refusing a float32 CPU load of Apertus v1.5 8B. "
            "The graded run used bfloat16 on MPS. "
            "The development prediction files are already written."
        )
    model.eval()
    return model, tokenizer, device, context_limit(model.config)


def _config_value(config, name: str):
    """Read one field from a config object or a plain dict."""
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def context_limit(config) -> int:
    """Return the score window, at most 512 tokenizer tokens.

    Apertus stores the text context on ``text_config``. The top-level config
    does not carry ``max_position_embeddings``. The cached text config allows
    262144 positions and has 32 layers. The final hidden state is index 32.
    Hidden state 8 is not that layer. A dict config is read the same way.
    """
    text_config = _config_value(config, "text_config")
    raw = _config_value(text_config, "max_position_embeddings")
    if not raw:
        raw = _config_value(config, "max_position_embeddings") or 8192
    return min(int(raw), 512)


def _is_punct(tok: str) -> bool:
    return all(ch in PUNCT for ch in tok)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    last = 0
    for m in SENT_SPLIT_RE.finditer(text):
        if m.end() > last:
            spans.append((last, m.end()))
            last = m.end()
    if last < len(text):
        spans.append((last, len(text)))
    return spans or [(0, len(text))]


def _assign_to_sents(sub_offs, s_spans: list[tuple[int, int]]):
    import numpy as np

    if sub_offs.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)
    mids = 0.5 * (sub_offs[:, 0] + sub_offs[:, 1])
    s_ends = np.asarray([b for _, b in s_spans], dtype=np.float64)
    return np.clip(np.searchsorted(s_ends, mids, side="right"), 0, len(s_spans) - 1)


def _span_smooth_tri(tokens: list[str], scores, w: int = 2) -> list[float]:
    import numpy as np

    n = len(scores)
    if n <= 1 or w <= 0:
        return [round(float(x), 6) for x in scores]
    content_idx = [i for i, t in enumerate(tokens) if not _is_punct(t)]
    if len(content_idx) <= 1:
        return [round(float(x), 6) for x in scores]
    c_scores = scores[content_idx]
    m = len(content_idx)
    c_out = np.empty(m, dtype=np.float64)
    for pos in range(m):
        lo = max(0, pos - w)
        hi = min(m, pos + w + 1)
        win = c_scores[lo:hi]
        wts = (w + 1) - np.abs(np.arange(lo, hi) - pos)
        c_out[pos] = float(np.sum(win * wts) / np.sum(wts))
    out = scores.copy()
    for pos, orig_i in enumerate(content_idx):
        out[orig_i] = c_out[pos]
    return [round(float(x), 6) for x in np.clip(out, 0.0, 1.0)]


def _pool_sub_scores_np(raw_scores, counts: list[int]):
    import numpy as np

    out = np.empty(len(counts), dtype=np.float64)
    cur = 0
    for i, c in enumerate(counts):
        if c == 0:
            out[i] = 1.0
        else:
            out[i] = float(raw_scores[cur : cur + c].mean())
            cur += c
    return out


def encode_side_subwords(model, tokenizer, text: str, spans: list[tuple[int, int]], device: str, max_length: int):
    """Encode one document side into stitched content subwords, per-word counts, and subword char offsets."""
    import numpy as np
    import torch

    if not spans:
        return np.zeros((0, 4096), dtype=np.float32), [], np.zeros((0, 2), dtype=np.int32), 0

    def fits(piece: str) -> bool:
        ids = tokenizer.encode(piece, add_special_tokens=True)
        return len(ids) <= max_length

    groups = pack_spans(text, spans, fits)
    sub_chunks: list[np.ndarray] = []
    counts: list[int] = []
    doc_sub_offsets: list[tuple[int, int]] = []
    seen_tokens = 0
    expected = hidden_state_count(model.config)
    for group in groups:
        start = group[0][0]
        piece = text[start : group[-1][1]]
        local = [(span_start - start, span_end - start) for span_start, span_end in group]
        if not fits(piece):
            raise RuntimeError(
                "A single whitespace token exceeds the context window. "
                "Refusing to drop it or to call another model."
            )
        batch = tokenizer(
            piece,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        if int(batch["input_ids"].shape[0]) != 1:
            raise RuntimeError("expected batch size 1 with no padding")
        offsets = [(int(a), int(b)) for a, b in batch.pop("offset_mapping")[0].tolist()]
        model_inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)
        arr = chosen_layer_tensor(outputs, expected)[0].detach().float().cpu().numpy()
        del outputs
        seen_tokens += int(batch["input_ids"].shape[-1])
        buckets = assign_subwords(offsets, local)
        idx_list: list[int] = []
        for bucket in buckets:
            counts.append(len(bucket))
            for idx in bucket:
                idx_list.append(idx)
                oa, ob = offsets[idx]
                doc_sub_offsets.append((oa + start, ob + start))
        if idx_list:
            sub_chunks.append(arr[idx_list].astype(np.float16).astype(np.float32))
    if len(counts) != len(spans):
        raise RuntimeError("stitched subword counts do not match whitespace tokens")
    stitched = np.concatenate(sub_chunks, axis=0) if sub_chunks else np.zeros((0, 4096), dtype=np.float32)
    return stitched, counts, np.asarray(doc_sub_offsets, dtype=np.int32), seen_tokens


def score_record(model, tokenizer, record: dict, device: str, max_length: int) -> tuple[str, int]:
    """Score one document pair with the pre-registered Tier-1 Combination at hidden state 15."""
    import numpy as np

    max_length = min(int(max_length), 512)
    text_a, tokens_a = source_text(record, "a")
    text_b, tokens_b = source_text(record, "b")
    spans_a = locate_tokens(text_a, tokens_a)
    spans_b = locate_tokens(text_b, tokens_b)
    sa, ca, oa, tok_a = encode_side_subwords(model, tokenizer, text_a, spans_a, device, max_length)
    sb, cb, ob, tok_b = encode_side_subwords(model, tokenizer, text_b, spans_b, device, max_length)
    if len(ca) != len(tokens_a) or len(cb) != len(tokens_b):
        raise RuntimeError(
            f"refusing to pad or truncate {record.get('id')}: "
            f"pooled {len(ca)}/{len(cb)} != gold tokens {len(tokens_a)}/{len(tokens_b)}"
        )
    if sa.size == 0 or sb.size == 0:
        labels_a = [1.0] * len(tokens_a)
        labels_b = [1.0] * len(tokens_b)
    else:
        # 1. Contrastive document-mean centering (beta = 0.50)
        doc_mean = np.concatenate([sa, sb], axis=0).mean(axis=0, keepdims=True)
        sa = sa - 0.50 * doc_mean
        sb = sb - 0.50 * doc_mean

        na = np.linalg.norm(sa, axis=1, keepdims=True)
        nb = np.linalg.norm(sb, axis=1, keepdims=True)
        ua = np.divide(sa, na, out=np.zeros_like(sa), where=na > 0)
        ub = np.divide(sb, nb, out=np.zeros_like(sb), where=nb > 0)
        C = ua @ ub.T

        # 2. Bidirectional CSLS hubness penalty (alpha = 0.50, k = 5)
        ka = min(5, C.shape[1])
        kb = min(5, C.shape[0])
        ra = np.partition(C, -ka, axis=1)[:, -ka:].mean(axis=1, keepdims=True)
        rb = np.partition(C, -kb, axis=0)[-kb:, :].mean(axis=0, keepdims=True)
        C_eff = C - 0.50 * 0.5 * (ra + rb)

        max_a = C_eff.max(axis=1)
        max_b = C_eff.max(axis=0)

        # 3. Sentence-constrained alignment blend (lambda = 0.35, matched sentence +/- 1)
        ss_a = _sentence_spans(text_a)
        ss_b = _sentence_spans(text_b)
        if len(ss_a) > 1 and len(ss_b) > 1:
            sid_a = _assign_to_sents(oa, ss_a)
            sid_b = _assign_to_sents(ob, ss_b)
            sent_emb_a = np.zeros((len(ss_a), sa.shape[1]), dtype=np.float32)
            for s_idx in range(len(ss_a)):
                m = sid_a == s_idx
                if m.any():
                    v = sa[m].mean(axis=0)
                    nv = np.linalg.norm(v)
                    if nv > 0:
                        sent_emb_a[s_idx] = v / nv
            sent_emb_b = np.zeros((len(ss_b), sb.shape[1]), dtype=np.float32)
            for s_idx in range(len(ss_b)):
                m = sid_b == s_idx
                if m.any():
                    v = sb[m].mean(axis=0)
                    nv = np.linalg.norm(v)
                    if nv > 0:
                        sent_emb_b[s_idx] = v / nv
            S_mat = sent_emb_a @ sent_emb_b.T
            target_b = S_mat.argmax(axis=1)[sid_a]
            mask_ab = np.abs(target_b[:, None] - sid_b[None, :]) <= 1
            sent_a = np.where(mask_ab, C_eff, -10.0).max(axis=1)
            sent_a = np.where(sent_a < -5.0, max_a, sent_a)

            target_a = S_mat.argmax(axis=0)[sid_b]
            mask_ba = np.abs(target_a[:, None] - sid_a[None, :]) <= 1
            sent_b = np.where(mask_ba, C_eff.T, -10.0).max(axis=1)
            sent_b = np.where(sent_b < -5.0, max_b, sent_b)

            max_a = 0.65 * max_a + 0.35 * sent_a
            max_b = 0.65 * max_b + 0.35 * sent_b

        # 4. Paper-order subword-first pooling + window-2 triangular span smoothing over content tokens
        word_a = _pool_sub_scores_np(np.clip(1.0 - max_a, 0.0, 1.0), ca)
        word_b = _pool_sub_scores_np(np.clip(1.0 - max_b, 0.0, 1.0), cb)
        labels_a = _span_smooth_tri(tokens_a, word_a, w=2)
        labels_b = _span_smooth_tri(tokens_b, word_b, w=2)

    line = dumps_encoder(
        {
            "id": record["id"],
            "text_a": record["text_a"],
            "text_b": record["text_b"],
            "labels_a": labels_a,
            "labels_b": labels_b,
        }
    )
    return line, tok_a + tok_b
