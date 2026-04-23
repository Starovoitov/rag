from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from retrieval.bm25 import BM25Result
from retrieval.semantic import SemanticResult


@dataclass
class HybridResult:
    """Merged result from semantic + BM25 search."""

    doc_id: str
    text: str
    score: float
    semantic_score: float
    bm25_score: float
    metadata: dict


def hybrid_search(
    semantic_results: list[SemanticResult],
    bm25_results: list[BM25Result],
    alpha: float = 0.7,
    top_k: int = 5,
    max_per_group: int | None = None,
    rrf_k: float = 60.0,
) -> list[HybridResult]:
    """
    Combine semantic and BM25 rankings into one ranking.

    Uses Reciprocal Rank Fusion (RRF), which is more robust than
    direct score interpolation when semantic and BM25 score ranges
    are not directly comparable.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in range [0.0, 1.0]")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be > 0")
    semantic_map = {item.doc_id: item for item in semantic_results}
    bm25_map = {item.doc_id: item for item in bm25_results}
    semantic_rank = {item.doc_id: rank for rank, item in enumerate(semantic_results, start=1)}
    bm25_rank = {item.doc_id: rank for rank, item in enumerate(bm25_results, start=1)}

    all_doc_ids = set(semantic_map) | set(bm25_map)
    merged: list[HybridResult] = []
    for doc_id in all_doc_ids:
        semantic_item = semantic_map.get(doc_id)
        bm25_item = bm25_map.get(doc_id)

        semantic_score = semantic_item.score if semantic_item else 0.0
        bm25_score = bm25_item.score if bm25_item else 0.0
        semantic_rrf = 1.0 / (rrf_k + semantic_rank[doc_id]) if doc_id in semantic_rank else 0.0
        bm25_rrf = 1.0 / (rrf_k + bm25_rank[doc_id]) if doc_id in bm25_rank else 0.0
        combined = alpha * semantic_rrf + (1.0 - alpha) * bm25_rrf

        source_item = semantic_item or bm25_item
        if source_item is None:
            continue

        merged.append(
            HybridResult(
                doc_id=doc_id,
                text=source_item.text,
                score=combined,
                semantic_score=semantic_score,
                bm25_score=bm25_score,
                metadata=source_item.metadata,
            )
        )

    merged.sort(key=lambda item: item.score, reverse=True)

    if max_per_group is None or max_per_group <= 0:
        return merged[: max(top_k, 0)]

    def group_key(metadata: dict[str, Any]) -> str | None:
        # Prefer stable source-level metadata keys to avoid top-k collapse
        # into near-duplicate chunks from the same source.
        for key in ("source_id", "source", "section", "title", "file_path", "path"):
            value = metadata.get(key)
            if value:
                return str(value)
        return None

    selected: list[HybridResult] = []
    per_group_counts: dict[str, int] = {}
    overflow: list[HybridResult] = []
    target_k = max(top_k, 0)
    for item in merged:
        if len(selected) >= target_k:
            break
        key = group_key(item.metadata or {})
        if key is None:
            selected.append(item)
            continue
        current = per_group_counts.get(key, 0)
        if current < max_per_group:
            selected.append(item)
            per_group_counts[key] = current + 1
        else:
            overflow.append(item)

    # Backfill if diversity constraint was too strict to reach target_k.
    if len(selected) < target_k:
        selected.extend(overflow[: target_k - len(selected)])

    return selected[:target_k]
