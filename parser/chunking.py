from __future__ import annotations

import math
import re

from utils.common import tokenize


def token_count(text: str) -> int:
    """Return token count using the internal lightweight tokenizer."""
    return len(tokenize(text))


def chunk_text(
    text: str,
    min_tokens: int = 300,
    max_tokens: int = 800,
    overlap_ratio: float = 0.15,
    mode: str = "token",
) -> list[str]:
    """Create chunks within configured size bounds."""
    if min_tokens <= 0 or max_tokens <= 0 or min_tokens > max_tokens:
        raise ValueError("Invalid chunk bounds.")
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio must be in [0, 1).")
    if mode not in {"token", "semantic_dynamic"}:
        raise ValueError("mode must be one of: token, semantic_dynamic")

    tokens = tokenize(text)
    if not tokens:
        return []
    if len(tokens) <= max_tokens:
        return [" ".join(tokens)]
    if mode == "semantic_dynamic":
        return _chunk_text_semantic_dynamic(text, min_tokens=min_tokens, max_tokens=max_tokens, overlap_ratio=overlap_ratio)

    return _chunk_text_token(tokens, min_tokens=min_tokens, max_tokens=max_tokens, overlap_ratio=overlap_ratio)


def _chunk_text_token(tokens: list[str], min_tokens: int, max_tokens: int, overlap_ratio: float) -> list[str]:
    step = max(1, int(max_tokens * (1 - overlap_ratio)))
    overlap = max_tokens - step
    if overlap < int(max_tokens * 0.1):
        overlap = int(max_tokens * 0.1)
        step = max_tokens - overlap
    if overlap > int(max_tokens * 0.2):
        overlap = int(max_tokens * 0.2)
        step = max_tokens - overlap

    chunks: list[str] = []
    i = 0
    while i < len(tokens):
        end = min(len(tokens), i + max_tokens)
        chunk_tokens = tokens[i:end]
        if len(chunk_tokens) < min_tokens and i != 0:
            previous = chunks.pop()
            merged = previous + " " + " ".join(chunk_tokens)
            chunks.append(merged)
            break
        chunks.append(" ".join(chunk_tokens))
        if end == len(tokens):
            break
        i += step

    return chunks


def _chunk_text_semantic_dynamic(
    text: str,
    *,
    min_tokens: int,
    max_tokens: int,
    overlap_ratio: float,
) -> list[str]:
    units = _semantic_units(text)
    if not units:
        return _chunk_text_token(tokenize(text), min_tokens=min_tokens, max_tokens=max_tokens, overlap_ratio=overlap_ratio)

    chunks: list[str] = []
    buffer: list[str] = []
    buf_tokens = 0
    overlap_target = overlap_tokens(max_tokens=max_tokens, overlap_ratio=overlap_ratio)

    for unit in units:
        unit_tokens = tokenize(unit)
        if not unit_tokens:
            continue
        unit_len = len(unit_tokens)

        # Split very long units by token window to avoid giant fragments.
        if unit_len > max_tokens:
            if buffer:
                chunks.append(" ".join(buffer))
                buffer, buf_tokens = [], 0
            sub_chunks = _chunk_text_token(unit_tokens, min_tokens=min_tokens, max_tokens=max_tokens, overlap_ratio=overlap_ratio)
            chunks.extend(sub_chunks)
            continue

        target_max = _dynamic_max_tokens(unit)
        target_max = min(max_tokens, max(min_tokens, target_max))
        if buf_tokens + unit_len <= target_max:
            buffer.append(unit)
            buf_tokens += unit_len
            continue

        if buf_tokens >= min_tokens:
            chunks.append(" ".join(buffer))
            # Keep small overlap from the tail of previous chunk.
            buffer = _tail_overlap_units(buffer, overlap_target)
            buf_tokens = token_count(" ".join(buffer)) if buffer else 0
        buffer.append(unit)
        buf_tokens += unit_len

    if buffer:
        if chunks and buf_tokens < min_tokens:
            chunks[-1] = chunks[-1] + " " + " ".join(buffer)
        else:
            chunks.append(" ".join(buffer))

    return [chunk for chunk in chunks if chunk.strip()]


def _semantic_units(text: str) -> list[str]:
    compact = re.sub(r"\r\n?", "\n", text)
    blocks = [b.strip() for b in re.split(r"\n{2,}", compact) if b and b.strip()]
    units: list[str] = []
    for block in blocks:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|(?<=:)\s+|(?=\s*[-*•]\s)", block) if s and s.strip()]
        if sentences:
            units.extend(sentences)
        else:
            units.append(block)
    return units


def _dynamic_max_tokens(unit: str) -> int:
    # Short procedural/list fragments should remain compact.
    if re.search(r"(^|\s)(step|steps|example|checklist|api|parameter|option)s?($|\s)", unit.lower()):
        return 220
    if re.search(r"(^|\s)[-*•]\s", unit):
        return 200
    # Explanatory content can be larger.
    return 320


def _tail_overlap_units(units: list[str], overlap_target: int) -> list[str]:
    if overlap_target <= 0 or not units:
        return []
    kept: list[str] = []
    total = 0
    for unit in reversed(units):
        size = token_count(unit)
        if kept and total + size > overlap_target:
            break
        kept.append(unit)
        total += size
    return list(reversed(kept))


def overlap_tokens(max_tokens: int, overlap_ratio: float) -> int:
    """Compute overlap size in tokens for a chunk window."""
    return int(math.floor(max_tokens * overlap_ratio))


def jaccard_similarity_tokens(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity across unique lowercase tokens."""
    ta = set(tokenize(text_a.lower()))
    tb = set(tokenize(text_b.lower()))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

