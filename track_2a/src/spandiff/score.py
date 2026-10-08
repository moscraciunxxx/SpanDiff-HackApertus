"""Forward pass of Apertus v1.5 8B and the encoder line it produces."""

from __future__ import annotations

from spandiff import MODEL_ID
from spandiff.align import (
    diff_against,
    gold_tokens,
    locate_tokens,
    mean_pool,
    pack_spans,
    source_text,
    stitch_chunks,
)
from spandiff.schema import dumps_encoder


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
        truncation=False,
    )
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


def load_apertus(model_id: str = MODEL_ID):
    """Load the submitted 8B. Any other id is refused."""
    if model_id != MODEL_ID:
        raise SystemExit(
            f"refusing model {model_id}. The submitted scorer is only {MODEL_ID}."
        )
    import torch
    from transformers import AutoTokenizer

    try:
        from transformers import AutoModelForMultimodalLM
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


def score_record(model, tokenizer, record: dict, device: str, max_length: int) -> tuple[str, int]:
    # A full-document forward of a long page was killed with no traceback.
    # Every caller, including a judge re-run, uses the same 512-token windows.
    max_length = min(int(max_length), 512)
    text_a, tokens_a = source_text(record, "a")
    text_b, tokens_b = source_text(record, "b")
    spans_a = locate_tokens(text_a, tokens_a)
    spans_b = locate_tokens(text_b, tokens_b)
    vectors_a, tokens_in_a = encode_side(model, tokenizer, text_a, spans_a, device, max_length)
    vectors_b, tokens_in_b = encode_side(model, tokenizer, text_b, spans_b, device, max_length)
    return dumps_line(record, vectors_a, vectors_b), tokens_in_a + tokens_in_b
