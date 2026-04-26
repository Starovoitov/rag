from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
from typing import Any

from sentence_transformers import CrossEncoder
from utils.common import min_max_normalize


DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CEScoreCalibrationMode(StrEnum):
    MINMAX = "minmax"
    SOFTMAX = "softmax"
    ZSCORE = "zscore"


def calibrate_ce_scores(
    scores: list[float],
    mode: CEScoreCalibrationMode,
    temperature: float,
) -> list[float]:
    t = max(temperature, 1e-6)
    if mode == CEScoreCalibrationMode.MINMAX:
        normalized = min_max_normalize({str(i): score for i, score in enumerate(scores)}, epsilon=1e-6)
        return [normalized[str(i)] for i in range(len(scores))]
    if mode == CEScoreCalibrationMode.SOFTMAX:
        shifted = [score / t for score in scores]
        max_s = max(shifted)
        exps = [math.exp(score - max_s) for score in shifted]
        total = sum(exps)
        if total <= 0:
            return [0.5] * len(scores)
        return [value / total for value in exps]
    if mode == CEScoreCalibrationMode.ZSCORE:
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        std = math.sqrt(variance)
        if std < 1e-6:
            return [0.5] * len(scores)
        z_scores = [(score - mean) / std for score in scores]
        # Sigmoid to keep output in [0,1] for stable fusion.
        return [1.0 / (1.0 + math.exp(-(z / t))) for z in z_scores]
    raise ValueError(f"Unsupported ce_calibration: {mode}")


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
    ce_score: float
    ce_score_norm: float
    base_score: float
    base_score_norm: float
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
        ce_calibration: CEScoreCalibrationMode | str = CEScoreCalibrationMode.MINMAX,
        ce_temperature: float = 1.0,
    ) -> list[RerankedResult]:
        if not candidates or top_k <= 0:
            return []

        filtered_candidates = [candidate for candidate in candidates if candidate.text.strip()]
        pairs = [(query, candidate.text) for candidate in filtered_candidates]
        scores = self.model.predict(pairs, batch_size=batch_size)

        ce_scores = [float(score) for score in scores]
        base_scores = [float(candidate.score) for candidate in filtered_candidates]
        calibration_mode = (
            ce_calibration
            if isinstance(ce_calibration, CEScoreCalibrationMode)
            else CEScoreCalibrationMode(ce_calibration)
        )
        ce_norm = calibrate_ce_scores(ce_scores, calibration_mode, ce_temperature)
        base_norm_map = min_max_normalize({str(i): score for i, score in enumerate(base_scores)}, epsilon=1e-6)
        base_norm = [base_norm_map[str(i)] for i in range(len(base_scores))]

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
                    ce_score=ce_score,
                    ce_score_norm=ce_score_norm,
                    base_score=base_score,
                    base_score_norm=base_score_norm,
                    metadata=dict(candidate.metadata),
                )
            )

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]
