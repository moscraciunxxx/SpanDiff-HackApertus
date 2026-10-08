"""Whitespace alignment and the DiffAlign difference on our own vectors."""

from __future__ import annotations


def locate_tokens(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    """Character spans of whitespace tokens, in order, inside `text`."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        idx = text.find(token, cursor)
        if idx < 0:
            raise ValueError(f"whitespace token not found in source text: {token!r}")
        spans.append((idx, idx + len(token)))
        cursor = idx + len(token)
    return spans


def gold_tokens(record: dict, side: str) -> list[str]:
    """Tokens the official script compares: ``str.split`` on ``text_a`` or ``text_b``.

    On the development files this length equals ``len(labels_a)`` or ``len(labels_b)``.
    A mismatch is refused. The scorer does not pad or truncate to hide it.
    """
    text_key = f"text_{side}"
    label_key = f"labels_{side}"
    tokens = record[text_key].split()
    labels = record[label_key]
    if len(tokens) != len(labels):
        raise ValueError(
            f"{record.get('id')}: {text_key}.split() is {len(tokens)} "
            f"but {label_key} has {len(labels)}. Refusing to pad or truncate."
        )
    return tokens


def source_text(record: dict, side: str) -> tuple[str, list[str]]:
    """Return the string to encode and the gold whitespace tokens for that side."""
    tokens = gold_tokens(record, side)
    untokenized = record.get(f"text_{side}_untokenized") or ""
    if untokenized:
        try:
            locate_tokens(untokenized, tokens)
            return untokenized, tokens
        except ValueError:
            pass
    spaced = " ".join(tokens)
    return spaced, tokens


def pack_spans(
    text: str,
    spans: list[tuple[int, int]],
    fits,
) -> list[list[tuple[int, int]]]:
    """Greedy groups of spans whose covered substring still fits the context."""
    groups: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    for span in spans:
        trial = current + [span]
        piece = text[trial[0][0] : trial[-1][1]]
        if current and not fits(piece):
            groups.append(current)
            current = [span]
        else:
            current = trial
    if current:
        groups.append(current)
    return groups


def assign_subwords(
    offsets: list[tuple[int, int]],
    spans: list[tuple[int, int]],
) -> list[list[int]]:
    """Put each overlapping subword on the whitespace span it overlaps most.

    An equal overlap stays on the earlier span.
    """
    buckets: list[list[int]] = [[] for _ in spans]
    for index, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        best = None
        best_overlap = 0
        for span_index, (span_start, span_end) in enumerate(spans):
            overlap = max(0, min(end, span_end) - max(start, span_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best = span_index
        if best is not None:
            buckets[best].append(index)
    return buckets


def mean_pool(
    hidden: list[list[float]],
    offsets: list[tuple[int, int]],
    spans: list[tuple[int, int]],
) -> list[list[float]]:
    if not hidden:
        raise ValueError("hidden states are empty")
    width = len(hidden[0])
    if any(len(row) != width for row in hidden):
        raise ValueError(
            "hidden rows do not share one width. Refusing to pad a short row with zeros."
        )
    buckets = assign_subwords(offsets, spans)
    pooled: list[list[float]] = []
    for bucket in buckets:
        if not bucket:
            pooled.append([0.0] * width)
            continue
        acc = [0.0] * width
        for index in bucket:
            row = hidden[index]
            for dim, value in enumerate(row):
                acc[dim] += float(value)
        pooled.append([value / len(bucket) for value in acc])
    return pooled


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(
            "cosine vectors do not share one width. Refusing a truncated dot product."
        )
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / ((left_norm ** 0.5) * (right_norm ** 0.5))


def difference(max_cosine: float) -> float:
    """clamp(1 - max cosine, 0, 1). Cosine may be negative. NaN is refused."""
    if max_cosine != max_cosine or max_cosine == float("inf") or max_cosine == float("-inf"):
        raise ValueError("rejected NaN")
    return min(1.0, max(0.0, 1.0 - max_cosine))


def stitch_chunks(chunks: list[list[list[float]]]) -> list[list[float]]:
    """Concatenate window vectors in chunk order. Do not reorder or drop a window."""
    stitched: list[list[float]] = []
    for chunk in chunks:
        stitched.extend(chunk)
    return stitched


def scores_after_stitch(
    source_chunks: list[list[list[float]]],
    other_chunks: list[list[list[float]]],
) -> list[float]:
    """Score each source token against the other side's full stitched sequence."""
    return diff_against(stitch_chunks(source_chunks), stitch_chunks(other_chunks))


def diff_against(source: list[list[float]], other: list[list[float]]) -> list[float]:
    """One score per source vector: 1 minus its best cosine on the other side.

    Same result as a per-pair loop. The matrix form is what makes a long page finish.
    A zero vector still scores 1, because its cosine is defined as 0.
    """
    if not source:
        return []
    if not other:
        return [1.0] * len(source)
    source_width = len(source[0])
    other_width = len(other[0])
    if any(len(row) != source_width for row in source) or any(len(row) != other_width for row in other):
        raise ValueError("vectors do not share one width. Refusing a truncated dot product.")
    if source_width != other_width:
        raise ValueError("source and other widths differ. Refusing a truncated dot product.")
    try:
        import numpy as np
    except ImportError:
        return [difference(max(cosine(vector, row) for row in other)) for vector in source]

    left = np.asarray(source, dtype=np.float64)
    right = np.asarray(other, dtype=np.float64)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("rejected NaN")
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denom = np.outer(left_norm, right_norm)
    cosines = np.zeros((left.shape[0], right.shape[0]), dtype=np.float64)
    np.divide(left @ right.T, denom, out=cosines, where=denom > 0.0)
    best = cosines.max(axis=1)
    scores = np.clip(1.0 - best, 0.0, 1.0)
    return [float(value) for value in scores]


def score_sides(
    vectors_a: list[list[float]],
    vectors_b: list[list[float]],
) -> tuple[list[float], list[float]]:
    if len(vectors_a) == 0 and len(vectors_b) == 0:
        return [], []
    return diff_against(vectors_a, vectors_b), diff_against(vectors_b, vectors_a)
