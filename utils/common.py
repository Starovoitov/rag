from __future__ import annotations

import re

# BM25: lowercase word tokens only (lexical matching; ignores standalone punctuation).
_BM25_WORD = re.compile(r"\w+")


def tokenize(text: str, *, for_bm25: bool = False) -> list[str]:
    """
    Tokenize text in one of two modes.

    - ``for_bm25=False`` (parser / data collection): words and punctuation as separate
      tokens, Unicode-aware. Matches chunking and ``token_count`` behavior.
    - ``for_bm25=True`` (BM25 indexer): lowercase word tokens only, suitable for
      term frequency and IDF statistics.
    """
    if for_bm25:
        return _BM25_WORD.findall(text.lower())
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def min_max_normalize(values: dict[str, float], *, epsilon: float = 1e-9) -> dict[str, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if (high - low) < epsilon:
        return {key: 0.5 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def rank_weight(rank: int) -> float:
    # Rank-aware contrastive weighting:
    # - 1..5: highest pressure to fix top-rank confusions
    # - 6..15: medium pressure
    # - 16..50: lower pressure
    if rank <= 5:
        return 1.0
    if rank <= 15:
        return 0.7
    return 0.4
