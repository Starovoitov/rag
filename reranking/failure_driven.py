from __future__ import annotations

import json
import random
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from utils.common import rank_weight, tokenize

if TYPE_CHECKING:
    from reranking.cross_encoder import RerankCandidate


FAILURE_ANALYSIS_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


class FailureClassificationResult(BaseModel):
    """Structured failure classification output."""

    bucket: str
    reasons: dict[str, object] = Field(default_factory=dict)


class RerankerContext(BaseModel):
    """One training context row for failure-driven reranker training."""

    schema_version: str = "reranker_context_v1"
    query: str
    positives: list[str]
    negatives: list[str]
    weights: dict[str, float]
    failure_bucket: str
    source_miss_type: str


class RerankerContextBuildStats(BaseModel):
    """Counters collected while building reranker contexts."""

    samples_seen: int = 0
    samples_used: int = 0
    contexts_written: int = 0
    contexts_ranking_cutoff_failure: int = 0
    contexts_true_recall_failure: int = 0
    contexts_other: int = 0
    missing_positive_text: int = 0
    missing_negative_text: int = 0


class RerankerContextsBuildResult(BaseModel):
    """Contexts with associated build statistics."""

    contexts: list[RerankerContext]
    stats: RerankerContextBuildStats


class RerankerTrainingResult(BaseModel):
    """Result metadata for cross-encoder reranker training run."""

    out_dir: str
    model: str
    train_examples: int
    val_examples: int
    epochs: int
    batch_size: int


def rrf_fuse_doc_ids(ranked_lists: list[list[str]], top_k: int, rrf_k: int = 60) -> list[str]:
    """Fuse several ranked lists with reciprocal-rank fusion (RRF)."""
    if top_k <= 0:
        return []
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, _ in ordered[:top_k]]


def prefilter_rerank_candidates(
    query: str,
    candidates: list[RerankCandidate],
    keep_top_n: int,
) -> list[RerankCandidate]:
    """Keep top-N candidates by lexical query overlap before cross-encoder rerank."""
    if keep_top_n <= 0 or len(candidates) <= keep_top_n:
        return candidates

    query_terms = set(tokenize(query, for_bm25=True))
    if not query_terms:
        return candidates[:keep_top_n]

    ranked: list[tuple[float, int, RerankCandidate]] = []
    for idx, candidate in enumerate(candidates):
        doc_terms = set(tokenize(candidate.text, for_bm25=True))
        overlap = len(query_terms & doc_terms)
        score = float(overlap) + (1.0 / (idx + 1000.0))
        ranked.append((score, idx, candidate))

    ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [candidate for _, _, candidate in ranked[:keep_top_n]]


def content_tokens(text: str) -> set[str]:
    """Tokenize text and keep only meaningful non-stopword content tokens."""
    return {
        token
        for token in tokenize(text, for_bm25=True)
        if len(token) >= 3 and token not in FAILURE_ANALYSIS_STOPWORDS
    }


def text_similarity(left: str, right: str) -> float:
    """Estimate text similarity using max of lexical Jaccard and sequence ratio."""
    left_tokens = content_tokens(left)
    right_tokens = content_tokens(right)
    lexical = 0.0
    if left_tokens and right_tokens:
        lexical = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    seq = SequenceMatcher(None, left.lower(), right.lower()).ratio()
    return max(lexical, seq)


def single_chunk_overlap_ratio(gt_text: str, retrieved_text: str) -> float:
    """Compute share of GT content tokens covered by one retrieved chunk."""
    gt_tokens = content_tokens(gt_text)
    if not gt_tokens:
        return 0.0
    overlap = len(gt_tokens & content_tokens(retrieved_text))
    return overlap / len(gt_tokens)


def classify_failure(
    *,
    query: str,
    gt_doc_ids: list[str],
    top_k_doc_ids: list[str],
    all_ranked_doc_ids: list[str],
    doc_text_map: dict[str, str],
    near_miss_threshold: float,
    top_k: int,
) -> FailureClassificationResult:
    """Classify retrieval miss into near-miss/fragmentation/cutoff/recall buckets."""
    gt_texts = _extract_present_texts(gt_doc_ids, doc_text_map)
    retrieved_texts = _extract_present_texts(top_k_doc_ids, doc_text_map)

    near_miss_score = _compute_near_miss_score(gt_texts, retrieved_texts)
    if near_miss_score >= near_miss_threshold:
        return FailureClassificationResult(
            bucket="near_miss", reasons={"near_miss_score": near_miss_score}
        )

    best_single_overlap, combined_overlap = _compute_fragmentation_metrics(
        gt_texts, retrieved_texts
    )
    if combined_overlap >= 0.75 and best_single_overlap < 0.55:
        return FailureClassificationResult(
            bucket="fragmentation",
            reasons={
                "combined_overlap": combined_overlap,
                "best_single_overlap": best_single_overlap,
            },
        )

    gt_rank = _find_first_relevant_rank(all_ranked_doc_ids, gt_doc_ids)
    if gt_rank is not None and gt_rank > top_k:
        return FailureClassificationResult(
            bucket="ranking_cutoff_failure", reasons={"gt_first_rank": gt_rank, "top_k": top_k}
        )

    best_query_overlap = _compute_best_query_to_gt_overlap(query, gt_texts)
    return FailureClassificationResult(
        bucket="true_recall_failure", reasons={"best_query_to_gt_overlap": best_query_overlap}
    )


def inject_bm25_tail_candidates(
    *,
    query: str,
    merged_doc_ids: list[str],
    retriever: object,
    bm25_search_depth: int,
    rescue_tail_k: int,
) -> list[str]:
    """Append BM25-only tail candidates that are missing in merged ranking."""
    if rescue_tail_k <= 0 or bm25_search_depth <= 0:
        return merged_doc_ids
    bm25 = getattr(retriever, "bm25", None)
    index = getattr(bm25, "index", None)
    if index is None:
        return merged_doc_ids

    bm25_hits = index.search(query, top_k=bm25_search_depth)
    merged_set = set(merged_doc_ids)
    rescued = [item.doc_id for item in bm25_hits if item.doc_id not in merged_set][:rescue_tail_k]
    if not rescued:
        return merged_doc_ids
    return merged_doc_ids + rescued


def interleave_doc_ids(primary: list[str], secondary: list[str], limit: int) -> list[str]:
    """Interleave two ranked doc-id lists while preserving order and uniqueness."""
    merged: list[str] = []
    seen: set[str] = set()
    max_len = max(len(primary), len(secondary))
    for idx in range(max_len):
        if idx < len(primary):
            doc_id = primary[idx]
            if doc_id not in seen:
                merged.append(doc_id)
                seen.add(doc_id)
                if len(merged) >= limit:
                    return merged
        if idx < len(secondary):
            doc_id = secondary[idx]
            if doc_id not in seen:
                merged.append(doc_id)
                seen.add(doc_id)
                if len(merged) >= limit:
                    return merged
    return merged


def build_stratified_rerank_pool(
    *,
    hybrid_doc_ids: list[str],
    semantic_doc_ids: list[str] | None,
    bm25_doc_ids: list[str] | None,
    limit: int,
) -> list[str]:
    """Build balanced rerank pool from semantic, BM25, then hybrid rankings."""
    sem = semantic_doc_ids or []
    bm = bm25_doc_ids or []
    stratified = interleave_doc_ids(sem, bm, limit=max(limit * 2, 1))
    merged: list[str] = []
    seen: set[str] = set()
    for doc_id in stratified + hybrid_doc_ids + sem + bm:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        merged.append(doc_id)
        if len(merged) >= limit:
            break
    return merged


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity with safe guards for empty/degenerate vectors."""
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def mmr_select_candidates(
    *,
    candidate_doc_ids: list[str],
    query_embedding: list[float],
    doc_embeddings: dict[str, list[float]],
    lambda_: float,
    max_k: int,
    diversity_threshold: float | None,
) -> list[str]:
    """Select diverse top candidates via maximal marginal relevance (MMR)."""
    if max_k <= 0:
        return []
    seen: set[str] = set()
    candidates = [
        doc_id for doc_id in candidate_doc_ids if doc_id not in seen and not seen.add(doc_id)
    ]
    if not candidates:
        return []

    candidates_with_vec = [doc_id for doc_id in candidates if doc_id in doc_embeddings]
    candidates_without_vec = [doc_id for doc_id in candidates if doc_id not in doc_embeddings]
    if not candidates_with_vec:
        return candidates[:max_k]

    q_scores = {
        doc_id: cosine_similarity(query_embedding, doc_embeddings[doc_id])
        for doc_id in candidates_with_vec
    }
    selected: list[str] = []
    remaining = list(candidates_with_vec)

    while remaining and len(selected) < max_k:
        best_doc_id = None
        best_score = float("-inf")
        for doc_id in remaining:
            relevance = q_scores.get(doc_id, 0.0)
            if not selected:
                mmr_score = relevance
                max_sim_to_selected = 0.0
            else:
                max_sim_to_selected = max(
                    cosine_similarity(doc_embeddings[doc_id], doc_embeddings[selected_doc])
                    for selected_doc in selected
                )
                mmr_score = (lambda_ * relevance) - ((1.0 - lambda_) * max_sim_to_selected)

            if (
                diversity_threshold is not None
                and selected
                and max_sim_to_selected >= diversity_threshold
            ):
                continue
            if mmr_score > best_score:
                best_score = mmr_score
                best_doc_id = doc_id

        if best_doc_id is None:
            break
        selected.append(best_doc_id)
        remaining.remove(best_doc_id)

    if len(selected) < max_k:
        for doc_id in candidates_without_vec + remaining:
            if doc_id not in selected:
                selected.append(doc_id)
                if len(selected) >= max_k:
                    break
    return selected[:max_k]


def source_miss_type(
    *,
    relevant_doc_ids: list[str],
    semantic_doc_ids: list[str] | None,
    bm25_doc_ids: list[str] | None,
) -> str:
    """Classify which branch (semantic/BM25) missed the relevant documents."""
    relevant = set(relevant_doc_ids)
    if not relevant:
        return "no_gt"
    semantic_hits = bool(set(semantic_doc_ids or []).intersection(relevant))
    bm25_hits = bool(set(bm25_doc_ids or []).intersection(relevant))
    if not semantic_hits and not bm25_hits:
        return "both_miss"
    if not semantic_hits and bm25_hits:
        return "embedding_miss"
    if semantic_hits and not bm25_hits:
        return "bm25_miss"
    return "both_hit"


def build_reranker_training_contexts_from_failures(
    *,
    failure_records: list[dict[str, object]],
    doc_text_map: dict[str, str],
    max_negative_rank: int,
    max_negatives: int,
    ranking_cutoff_weight: float,
    true_recall_weight: float,
    default_weight: float,
) -> RerankerContextsBuildResult:
    """Convert failure records into weighted reranker context training rows."""
    contexts: list[RerankerContext] = []
    stats = RerankerContextBuildStats()
    for sample in failure_records:
        stats.samples_seen += 1
        query = str(sample.get("query", "")).strip()
        bucket = str(sample.get("bucket", ""))
        miss_type = str(sample.get("source_miss_type", ""))
        positives = [str(doc_id) for doc_id in sample.get("relevant_doc_ids", [])]
        retrieved = [str(doc_id) for doc_id in sample.get("retrieved_top_k_doc_ids", [])]
        retrieved_full = [str(doc_id) for doc_id in sample.get("retrieved_full_doc_ids", [])]
        bm25_branch = [str(doc_id) for doc_id in sample.get("bm25_branch_doc_ids", [])]
        if not query or not positives or not retrieved:
            continue
        stats.samples_used += 1

        sample_weight = _resolve_sample_weight(
            bucket, ranking_cutoff_weight, true_recall_weight, default_weight
        )
        positive_ids = _collect_positive_ids(positives, doc_text_map, stats)
        negative_pool = _build_negative_pool(
            bucket, retrieved, retrieved_full, bm25_branch, max_negative_rank
        )
        positive_set = set(positive_ids)
        negative_ids, negative_weights = _collect_negative_ids(
            negative_pool=negative_pool,
            positive_set=positive_set,
            doc_text_map=doc_text_map,
            stats=stats,
            sample_weight=sample_weight,
            max_negatives=max_negatives,
        )
        _append_source_miss_negatives(
            miss_type=miss_type,
            retrieved_full=retrieved_full,
            positive_set=positive_set,
            doc_text_map=doc_text_map,
            sample_weight=sample_weight,
            max_negatives=max_negatives,
            negative_ids=negative_ids,
            negative_weights=negative_weights,
        )

        if not positive_ids or not negative_ids:
            continue
        contexts.append(
            RerankerContext(
                query=query,
                positives=positive_ids,
                negatives=negative_ids,
                weights=negative_weights,
                failure_bucket=bucket,
                source_miss_type=miss_type,
            )
        )
        stats.contexts_written += 1
        _increment_context_bucket_stats(stats, bucket)
    return RerankerContextsBuildResult(contexts=contexts, stats=stats)


def train_reranker_from_contexts_jsonl(
    *,
    train_jsonl: Path,
    doc_text_map: dict[str, str],
    model_name: str,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    warmup_steps: int,
    val_ratio: float,
    seed: int,
) -> RerankerTrainingResult:
    """Train cross-encoder reranker from exported failure-driven JSONL contexts."""
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    from torch.utils.data import DataLoader

    rows = _load_training_rows(train_jsonl, seed)
    examples = _build_input_examples_from_rows(rows, doc_text_map)

    if not examples:
        raise ValueError("No train examples built from reranker JSONL.")

    train_examples, val_examples = _split_train_validation_examples(examples, val_ratio)
    if not train_examples:
        raise ValueError("No train split left after validation split.")

    model = CrossEncoder(model_name, num_labels=1, max_length=512)
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    evaluator = None
    if val_examples:
        evaluator = CEBinaryClassificationEvaluator.from_input_examples(
            val_examples, name="failure-driven-val"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_dataloader=train_loader,
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(out_dir),
        show_progress_bar=True,
    )
    model.save(str(out_dir))
    return RerankerTrainingResult(
        out_dir=str(out_dir),
        model=model_name,
        train_examples=len(train_examples),
        val_examples=len(val_examples),
        epochs=epochs,
        batch_size=batch_size,
    )


def _extract_present_texts(doc_ids: list[str], doc_text_map: dict[str, str]) -> list[str]:
    return [doc_text_map.get(doc_id, "") for doc_id in doc_ids if doc_text_map.get(doc_id, "")]


def _compute_near_miss_score(gt_texts: list[str], retrieved_texts: list[str]) -> float:
    near_miss_score = 0.0
    for gt_text in gt_texts:
        for hit_text in retrieved_texts:
            near_miss_score = max(near_miss_score, text_similarity(gt_text, hit_text))
    return near_miss_score


def _compute_fragmentation_metrics(
    gt_texts: list[str], retrieved_texts: list[str]
) -> tuple[float, float]:
    best_single_overlap = 0.0
    combined_overlap = 0.0
    if not gt_texts or not retrieved_texts:
        return best_single_overlap, combined_overlap
    top_window = retrieved_texts[: min(3, len(retrieved_texts))]
    combined_tokens = content_tokens(" ".join(top_window))
    for gt_text in gt_texts:
        gt_tokens = content_tokens(gt_text)
        if not gt_tokens:
            continue
        for hit_text in retrieved_texts:
            best_single_overlap = max(
                best_single_overlap, single_chunk_overlap_ratio(gt_text, hit_text)
            )
        combined_overlap = max(combined_overlap, len(gt_tokens & combined_tokens) / len(gt_tokens))
    return best_single_overlap, combined_overlap


def _find_first_relevant_rank(all_ranked_doc_ids: list[str], gt_doc_ids: list[str]) -> int | None:
    for rank, doc_id in enumerate(all_ranked_doc_ids, start=1):
        if doc_id in gt_doc_ids:
            return rank
    return None


def _compute_best_query_to_gt_overlap(query: str, gt_texts: list[str]) -> float:
    query_tokens = content_tokens(query)
    if not query_tokens:
        return 0.0
    best_query_overlap = 0.0
    for gt_text in gt_texts:
        gt_tokens = content_tokens(gt_text)
        if not gt_tokens:
            continue
        best_query_overlap = max(
            best_query_overlap, len(query_tokens & gt_tokens) / len(query_tokens)
        )
    return best_query_overlap


def _resolve_sample_weight(
    bucket: str, ranking_cutoff_weight: float, true_recall_weight: float, default_weight: float
) -> float:
    if bucket == "ranking_cutoff_failure":
        return ranking_cutoff_weight
    if bucket == "true_recall_failure":
        return true_recall_weight
    return default_weight


def _collect_positive_ids(
    positives: list[str], doc_text_map: dict[str, str], stats: RerankerContextBuildStats
) -> list[str]:
    positive_ids: list[str] = []
    for positive_id in positives:
        if positive_id not in doc_text_map:
            stats.missing_positive_text += 1
            continue
        positive_ids.append(positive_id)
    return positive_ids


def _build_negative_pool(
    bucket: str,
    retrieved: list[str],
    retrieved_full: list[str],
    bm25_branch: list[str],
    max_negative_rank: int,
) -> list[str]:
    if bucket == "ranking_cutoff_failure":
        return retrieved[:max_negative_rank]
    if bucket == "true_recall_failure":
        return (bm25_branch or retrieved_full or retrieved)[:max_negative_rank]
    return retrieved[:max_negative_rank]


def _collect_negative_ids(
    *,
    negative_pool: list[str],
    positive_set: set[str],
    doc_text_map: dict[str, str],
    stats: RerankerContextBuildStats,
    sample_weight: float,
    max_negatives: int,
) -> tuple[list[str], dict[str, float]]:
    negative_ids: list[str] = []
    negative_weights: dict[str, float] = {}
    for rank, negative_id in enumerate(negative_pool, start=1):
        if len(negative_ids) >= max_negatives:
            break
        if negative_id in positive_set:
            continue
        if negative_id not in doc_text_map:
            stats.missing_negative_text += 1
            continue
        if negative_id in negative_weights:
            continue
        negative_ids.append(negative_id)
        negative_weights[negative_id] = sample_weight * rank_weight(rank)
    return negative_ids, negative_weights


def _append_source_miss_negatives(
    *,
    miss_type: str,
    retrieved_full: list[str],
    positive_set: set[str],
    doc_text_map: dict[str, str],
    sample_weight: float,
    max_negatives: int,
    negative_ids: list[str],
    negative_weights: dict[str, float],
) -> None:
    if miss_type not in {"embedding_miss", "bm25_miss"} or len(negative_ids) >= max_negatives:
        return
    for rank, negative_id in enumerate(retrieved_full, start=len(negative_ids) + 1):
        if len(negative_ids) >= max_negatives:
            break
        if negative_id in positive_set or negative_id in negative_weights:
            continue
        if negative_id not in doc_text_map:
            continue
        negative_ids.append(negative_id)
        negative_weights[negative_id] = sample_weight * rank_weight(rank)


def _increment_context_bucket_stats(stats: RerankerContextBuildStats, bucket: str) -> None:
    if bucket == "ranking_cutoff_failure":
        stats.contexts_ranking_cutoff_failure += 1
    elif bucket == "true_recall_failure":
        stats.contexts_true_recall_failure += 1
    else:
        stats.contexts_other += 1


def _load_training_rows(train_jsonl: Path, seed: int) -> list[dict[str, object]]:
    rows = [
        json.loads(line)
        for line in train_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    random.Random(seed).shuffle(rows)
    return rows


def _build_input_examples_from_rows(
    rows: list[dict[str, object]], doc_text_map: dict[str, str]
) -> list[object]:
    from sentence_transformers import InputExample

    examples: list[object] = []
    for row in rows:
        query = str(row.get("query", "")).strip()
        if row.get("schema_version") == "reranker_context_v1":
            positives = [str(doc_id) for doc_id in row.get("positives", [])]
            negatives = [str(doc_id) for doc_id in row.get("negatives", [])]
            weights = row.get("weights", {}) or {}
            for positive_id in positives:
                positive_text = doc_text_map.get(positive_id, "").strip()
                if not query or not positive_text:
                    continue
                for negative_id in negatives:
                    negative_text = doc_text_map.get(negative_id, "").strip()
                    if not negative_text:
                        continue
                    sample_weight = float(weights.get(negative_id, 1.0))
                    repeats = max(1, int(round(sample_weight)))
                    for _ in range(repeats):
                        examples.append(InputExample(texts=[query, positive_text], label=1.0))
                        examples.append(InputExample(texts=[query, negative_text], label=0.0))
            continue
        if "positive_text" in row and "negative_text" in row:
            positive_text = str(row.get("positive_text", "")).strip()
            negative_text = str(row.get("negative_text", "")).strip()
            sample_weight = float(row.get("sample_weight", 1.0))
            if not query or not positive_text or not negative_text:
                continue
            repeats = max(1, int(round(sample_weight)))
            for _ in range(repeats):
                examples.append(InputExample(texts=[query, positive_text], label=1.0))
                examples.append(InputExample(texts=[query, negative_text], label=0.0))
    return examples


def _split_train_validation_examples(
    examples: list[object], val_ratio: float
) -> tuple[list[object], list[object]]:
    val_size = int(len(examples) * max(0.0, min(val_ratio, 0.5)))
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]
    return train_examples, val_examples
