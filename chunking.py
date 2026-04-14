from __future__ import annotations

import math
import re


def simple_tokenize(text: str) -> list[str]:
    """Split text into lightweight word/punctuation tokens."""
    # Lightweight tokenization fallback without external tokenizer dependency.
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def token_count(text: str) -> int:
    """Return token count using the internal lightweight tokenizer."""
    return len(simple_tokenize(text))


def chunk_text(
    text: str,
    min_tokens: int = 300,
    max_tokens: int = 800,
    overlap_ratio: float = 0.15,
) -> list[str]:
    """Create overlapping token-based chunks within configured size bounds."""
    if min_tokens <= 0 or max_tokens <= 0 or min_tokens > max_tokens:
        raise ValueError("Invalid chunk bounds.")
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio must be in [0, 1).")

    tokens = simple_tokenize(text)
    if not tokens:
        return []
    if len(tokens) <= max_tokens:
        return [" ".join(tokens)]

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


def overlap_tokens(max_tokens: int, overlap_ratio: float) -> int:
    """Compute overlap size in tokens for a chunk window."""
    return int(math.floor(max_tokens * overlap_ratio))

