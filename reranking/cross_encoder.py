from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sentence_transformers import CrossEncoder


DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class RerankCandidate:
    """Candidate passage before reranking."""

    doc_id: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankedResult:
    """Reranked passage with cross-encoder score."""

    doc_id: str
    text: str
    score: float
    base_score: float
    metadata: dict[str, Any]


class CrossEncoderReranker:
    """
    Cross-encoder reranker wrapper.

    Usage:
    - retrieve initial candidates with BM25/semantic/hybrid,
    - call `rerank(query, candidates, top_k=...)` to reorder by CE relevance.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        max_length: int = 512,
    ) -> None:
        self.model = CrossEncoder(model_name, max_length=max_length)

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_k: int = 5,
        batch_size: int = 32,
        alpha: float = 0.75,
    ) -> list[RerankedResult]:
        def normalize(scores: list[float]) -> list[float]:
            min_s, max_s = min(scores), max(scores)
            if max_s - min_s < 1e-6:
                return [0.5] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]

        if not candidates or top_k <= 0:
            return []

        filtered_candidates = [candidate for candidate in candidates if candidate.text.strip()]
        pairs = [(query, candidate.text) for candidate in filtered_candidates]
        scores = self.model.predict(pairs, batch_size=batch_size)

        ce_scores = [float(score) for score in scores]
        base_scores = [float(candidate.score) for candidate in filtered_candidates]
        ce_norm = normalize(ce_scores)
        base_norm = normalize(base_scores)

        reranked = []
        for candidate, ce_score, base_score, ce_score_norm, base_score_norm in zip(
            filtered_candidates,
            ce_scores,
            base_scores,
            ce_norm,
            base_norm,
        ):
            combined = (alpha * ce_score_norm) + ((1.0 - alpha) * base_score_norm)

            reranked.append(
                RerankedResult(
                    doc_id=candidate.doc_id,
                    text=candidate.text,
                    score=combined,
                    base_score=base_score,
                    metadata=dict(candidate.metadata),
                )
            )

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]
