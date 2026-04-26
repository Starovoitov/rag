#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING
from utils.common import tokenize

if TYPE_CHECKING:
    from reranking.cross_encoder import RerankCandidate


DEFAULT_EMBEDDING_MODEL = "intfloat/e5-base-v2"


def _dedupe_query_variants(variants: list[str]) -> list[str]:
    deduped: list[str] = []
    seen_keys: set[str] = set()
    for variant in variants:
        key = variant.strip().lower().rstrip("?.!")
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(variant.strip())
    return deduped


def _paraphrase_variants(base: str) -> list[str]:
    lowered = base.lower()
    variants: list[str] = []
    if "?" in base:
        variants.append(base.replace("?", "").strip())
    replacements = (
        ("rag", "retrieval augmented generation"),
        ("retrieval augmented generation", "rag"),
        ("reranker", "cross encoder reranker"),
        ("reranking", "cross encoder reranking"),
        ("hallucination", "factual consistency"),
        ("recall", "retrieval coverage"),
    )
    for src, dst in replacements:
        if src in lowered:
            variants.append(lowered.replace(src, dst))
    return _dedupe_query_variants(variants)


def _decomposition_variants(base: str) -> list[str]:
    lowered = base.lower().strip().rstrip("?")
    variants: list[str] = []
    if "why" in lowered and "recall" in lowered:
        variants.extend(
            [
                "how does recall work in retrieval systems",
                "what causes low recall in bm25 retrieval",
                "what causes embedding retrieval misses",
                "how does hybrid retrieval merge bm25 and dense results",
            ]
        )
    if "hybrid" in lowered:
        variants.append("how hybrid retrieval combines bm25 and semantic rankings")
    if "rerank" in lowered:
        variants.append("what features improve cross encoder reranking quality")
    if "evaluation" in lowered or "metrics" in lowered:
        variants.extend(
            [
                "how to evaluate retrieval quality in rag systems",
                "how mrr and ndcg diagnose reranking failures",
            ]
        )
    if not variants:
        variants.extend(
            [
                f"what are the key components of {lowered}",
                f"what common failure modes appear in {lowered}",
                f"how to improve {lowered}",
            ]
        )
    return _dedupe_query_variants(variants)


def _entity_concept_variants(base: str) -> list[str]:
    lowered = base.lower()
    variants: list[str] = []
    concept_map = (
        ("evaluation", ["faithfulness", "groundedness", "answer relevance", "retrieval quality metrics"]),
        ("metrics", ["mrr ndcg recall at k", "ranking quality diagnostics"]),
        ("recall", ["retrieval coverage", "candidate pool expansion", "bm25 dense recall tradeoff"]),
        ("hybrid", ["bm25 dense fusion", "rrf hybrid retrieval", "branch-specific retrieval misses"]),
        ("rerank", ["cross encoder reranking optimization", "hard negative reranker training"]),
        ("rag", ["retriever generator grounding", "evidence grounded answering"]),
    )
    for trigger, expansions in concept_map:
        if trigger in lowered:
            variants.extend(expansions)
    if not variants:
        variants.extend(["retrieval system failure analysis", "document ranking optimization"])
    return _dedupe_query_variants(variants)


def _parse_llm_expansion_payload(raw_text: str) -> tuple[list[str], list[str], list[str]]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [], [], []
    paraphrases = [str(item).strip() for item in payload.get("paraphrases", [])]
    decompositions = [str(item).strip() for item in payload.get("decompositions", [])]
    concept_expansions = [str(item).strip() for item in payload.get("concept_expansions", [])]
    return _dedupe_query_variants(paraphrases), _dedupe_query_variants(decompositions), _dedupe_query_variants(
        concept_expansions
    )


def _llm_structured_query_expansion(
    query: str,
    *,
    provider: str,
    model: str,
    api_base: str | None,
    api_key: str | None,
    timeout_seconds: int,
    retries: int,
    cache_enabled: bool = False,
    cache_capacity: int = 512,
    cache_ttl_seconds: float = 300.0,
) -> tuple[list[str], list[str], list[str]]:
    from generation.llm import call_llm
    from generation.run_rag import get_llm_config

    conf = get_llm_config(provider=provider, model=model or None)
    if api_base:
        conf.api_base = api_base
    if api_key:
        conf.api_key = api_key
    conf.max_tokens = 400
    conf.temperature = 0.2
    conf.top_p = 1.0
    conf.timeout_seconds = max(1, timeout_seconds)
    conf.retries = max(0, retries)
    conf.cache_enabled = cache_enabled
    conf.cache_capacity = max(1, cache_capacity)
    conf.cache_ttl_seconds = max(0.1, cache_ttl_seconds)
    system_prompt = (
        "Generate retrieval queries that maximize document coverage. "
        "Return strict JSON only with keys: paraphrases, decompositions, concept_expansions. "
        "Each value must be a list of strings."
    )
    user_prompt = (
        "Generate retrieval queries that maximize document coverage.\n\n"
        "Return:\n"
        "1 paraphrases\n"
        "3 decompositions\n"
        "3 concept-expansion queries\n\n"
        f"Query: {query}"
    )
    try:
        response = call_llm(system_prompt=system_prompt, user_prompt=user_prompt, config=conf)
        return _parse_llm_expansion_payload(response)
    except Exception:
        return [], [], []


def _llm_structured_query_expansion_batch(
    queries: list[str],
    *,
    provider: str,
    model: str,
    api_base: str | None,
    api_key: str | None,
    timeout_seconds: int,
    retries: int,
    cache_enabled: bool = False,
    cache_capacity: int = 512,
    cache_ttl_seconds: float = 300.0,
) -> dict[str, tuple[list[str], list[str], list[str]]]:
    from generation.llm import call_llm
    from generation.run_rag import get_llm_config

    unique_queries = [q.strip() for q in queries if q.strip()]
    if not unique_queries:
        return {}

    conf = get_llm_config(provider=provider, model=model or None)
    if api_base:
        conf.api_base = api_base
    if api_key:
        conf.api_key = api_key
    conf.max_tokens = 2000
    conf.temperature = 0.2
    conf.top_p = 1.0
    conf.timeout_seconds = max(1, timeout_seconds)
    conf.retries = max(0, retries)
    conf.cache_enabled = cache_enabled
    conf.cache_capacity = max(1, cache_capacity)
    conf.cache_ttl_seconds = max(0.1, cache_ttl_seconds)

    system_prompt = (
        "Generate retrieval queries that maximize document coverage. "
        "Return strict JSON only. "
        "Top-level key: expansions, value is an array. "
        "Each item must be {query, paraphrases, decompositions, concept_expansions}."
    )
    joined_queries = "\n".join(f"- {q}" for q in unique_queries)
    user_prompt = (
        "Generate retrieval queries that maximize document coverage.\n\n"
        "For EACH query below return:\n"
        "1 paraphrases\n"
        "3 decompositions\n"
        "3 concept-expansion queries\n\n"
        "Queries:\n"
        f"{joined_queries}"
    )
    try:
        response = call_llm(system_prompt=system_prompt, user_prompt=user_prompt, config=conf)
        text = response.strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()
        payload = json.loads(text)
    except Exception:
        return {}

    result: dict[str, tuple[list[str], list[str], list[str]]] = {}
    for item in payload.get("expansions", []):
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        paraphrases = _dedupe_query_variants([str(x).strip() for x in item.get("paraphrases", [])])
        decompositions = _dedupe_query_variants([str(x).strip() for x in item.get("decompositions", [])])
        concepts = _dedupe_query_variants([str(x).strip() for x in item.get("concept_expansions", [])])
        result[query] = (paraphrases, decompositions, concepts)
    return result


def _build_query_variants(
    query: str,
    max_variants: int,
    *,
    use_llm_structured_expansion: bool = False,
    llm_provider: str = "qwen",
    llm_model: str = "qwen-plus",
    llm_api_base: str | None = None,
    llm_api_key: str | None = None,
    llm_timeout_seconds: int = 8,
    llm_retries: int = 0,
    llm_cache_enabled: bool = False,
    llm_cache_capacity: int = 512,
    llm_cache_ttl_seconds: float = 300.0,
    llm_precomputed: tuple[list[str], list[str], list[str]] | None = None,
) -> list[str]:
    variants, _ = _build_query_variants_with_debug(
        query=query,
        max_variants=max_variants,
        use_llm_structured_expansion=use_llm_structured_expansion,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_base=llm_api_base,
        llm_api_key=llm_api_key,
        llm_timeout_seconds=llm_timeout_seconds,
        llm_retries=llm_retries,
        llm_cache_enabled=llm_cache_enabled,
        llm_cache_capacity=llm_cache_capacity,
        llm_cache_ttl_seconds=llm_cache_ttl_seconds,
        llm_precomputed=llm_precomputed,
    )
    return variants


def _build_query_variants_with_debug(
    *,
    query: str,
    max_variants: int,
    use_llm_structured_expansion: bool = False,
    llm_provider: str = "qwen",
    llm_model: str = "qwen-plus",
    llm_api_base: str | None = None,
    llm_api_key: str | None = None,
    llm_timeout_seconds: int = 8,
    llm_retries: int = 0,
    llm_cache_enabled: bool = False,
    llm_cache_capacity: int = 512,
    llm_cache_ttl_seconds: float = 300.0,
    llm_precomputed: tuple[list[str], list[str], list[str]] | None = None,
) -> tuple[list[str], dict[str, object]]:
    base = query.strip()
    if not base or max_variants <= 1:
        variants = [base] if base else []
        return variants, {
            "llm_requested": False,
            "llm_generated_count": 0,
            "llm_generated_preview": [],
            "fallback_used": False,
        }

    paraphrase = _paraphrase_variants(base)
    decomposition = _decomposition_variants(base)
    entity_concepts = _entity_concept_variants(base)
    llm_generated_preview: list[str] = []
    llm_generated_count = 0
    fallback_used = False
    if use_llm_structured_expansion:
        if llm_precomputed is not None:
            llm_paraphrase, llm_decomposition, llm_concepts = llm_precomputed
        else:
            llm_paraphrase, llm_decomposition, llm_concepts = _llm_structured_query_expansion(
                base,
                provider=llm_provider,
                model=llm_model,
                api_base=llm_api_base,
                api_key=llm_api_key,
                timeout_seconds=llm_timeout_seconds,
                retries=llm_retries,
                cache_enabled=llm_cache_enabled,
                cache_capacity=llm_cache_capacity,
                cache_ttl_seconds=llm_cache_ttl_seconds,
            )
        llm_generated = _dedupe_query_variants(llm_paraphrase + llm_decomposition + llm_concepts)
        llm_generated_preview = llm_generated[:3]
        llm_generated_count = len(llm_generated)
        fallback_used = llm_generated_count == 0
        paraphrase = _dedupe_query_variants(llm_paraphrase + paraphrase)
        decomposition = _dedupe_query_variants(llm_decomposition + decomposition)
        entity_concepts = _dedupe_query_variants(llm_concepts + entity_concepts)

    variants: list[str] = [base]
    layer_lists = [paraphrase, decomposition, entity_concepts]
    layer_index = 0
    while len(variants) < max_variants and any(layer_lists):
        layer = layer_lists[layer_index % len(layer_lists)]
        if layer:
            variants.append(layer.pop(0))
        layer_index += 1
        if layer_index > (max_variants * 6):
            break
    final_variants = _dedupe_query_variants(variants)[:max_variants]
    return final_variants, {
        "llm_requested": use_llm_structured_expansion,
        "llm_generated_count": llm_generated_count,
        "llm_generated_preview": llm_generated_preview,
        "fallback_used": fallback_used,
    }


def _rrf_fuse_doc_ids(ranked_lists: list[list[str]], top_k: int, rrf_k: int = 60) -> list[str]:
    if top_k <= 0:
        return []
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, _ in ordered[:top_k]]


def _prefilter_rerank_candidates(
    query: str,
    candidates: list["RerankCandidate"],
    keep_top_n: int,
) -> list["RerankCandidate"]:
    if keep_top_n <= 0 or len(candidates) <= keep_top_n:
        return candidates

    query_terms = set(tokenize(query, for_bm25=True))
    if not query_terms:
        return candidates[:keep_top_n]

    ranked: list[tuple[float, int, "RerankCandidate"]] = []
    for idx, candidate in enumerate(candidates):
        doc_terms = set(tokenize(candidate.text, for_bm25=True))
        overlap = len(query_terms & doc_terms)
        # Strongly prioritize lexical overlap, with a tiny bias to earlier base rank.
        score = float(overlap) + (1.0 / (idx + 1000.0))
        ranked.append((score, idx, candidate))

    ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [candidate for _, _, candidate in ranked[:keep_top_n]]


def _candidate_base_score(query: str, text: str, rank: int) -> float:
    query_terms = set(tokenize(query, for_bm25=True))
    doc_terms = set(tokenize(text, for_bm25=True))
    overlap = len(query_terms & doc_terms)
    return (1.0 / (60.0 + rank)) + (0.02 * overlap)


_FAILURE_ANALYSIS_STOPWORDS = {
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


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in tokenize(text, for_bm25=True)
        if len(token) >= 3 and token not in _FAILURE_ANALYSIS_STOPWORDS
    }


def _text_similarity(left: str, right: str) -> float:
    left_tokens = _content_tokens(left)
    right_tokens = _content_tokens(right)
    lexical = 0.0
    if left_tokens and right_tokens:
        lexical = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    seq = SequenceMatcher(None, left.lower(), right.lower()).ratio()
    return max(lexical, seq)


def _single_chunk_overlap_ratio(gt_text: str, retrieved_text: str) -> float:
    gt_tokens = _content_tokens(gt_text)
    if not gt_tokens:
        return 0.0
    overlap = len(gt_tokens & _content_tokens(retrieved_text))
    return overlap / len(gt_tokens)


def _classify_failure(
    *,
    query: str,
    gt_doc_ids: list[str],
    top_k_doc_ids: list[str],
    all_ranked_doc_ids: list[str],
    doc_text_map: dict[str, str],
    near_miss_threshold: float,
    top_k: int,
) -> tuple[str, dict[str, object]]:
    gt_texts = [doc_text_map.get(doc_id, "") for doc_id in gt_doc_ids if doc_text_map.get(doc_id, "")]
    retrieved_texts = [doc_text_map.get(doc_id, "") for doc_id in top_k_doc_ids if doc_text_map.get(doc_id, "")]

    near_miss_score = 0.0
    for gt_text in gt_texts:
        for hit_text in retrieved_texts:
            near_miss_score = max(near_miss_score, _text_similarity(gt_text, hit_text))
    if near_miss_score >= near_miss_threshold:
        return "near_miss", {"near_miss_score": near_miss_score}

    best_single_overlap = 0.0
    combined_overlap = 0.0
    if gt_texts and retrieved_texts:
        top_window = retrieved_texts[: min(3, len(retrieved_texts))]
        combined_tokens = _content_tokens(" ".join(top_window))
        for gt_text in gt_texts:
            gt_tokens = _content_tokens(gt_text)
            if not gt_tokens:
                continue
            for hit_text in retrieved_texts:
                best_single_overlap = max(best_single_overlap, _single_chunk_overlap_ratio(gt_text, hit_text))
            combined_overlap = max(combined_overlap, len(gt_tokens & combined_tokens) / len(gt_tokens))
    if combined_overlap >= 0.75 and best_single_overlap < 0.55:
        return "fragmentation", {
            "combined_overlap": combined_overlap,
            "best_single_overlap": best_single_overlap,
        }

    gt_rank = None
    for rank, doc_id in enumerate(all_ranked_doc_ids, start=1):
        if doc_id in gt_doc_ids:
            gt_rank = rank
            break
    if gt_rank is not None and gt_rank > top_k:
        return "ranking_cutoff_failure", {"gt_first_rank": gt_rank, "top_k": top_k}

    query_tokens = _content_tokens(query)
    best_query_overlap = 0.0
    for gt_text in gt_texts:
        gt_tokens = _content_tokens(gt_text)
        if not gt_tokens or not query_tokens:
            continue
        best_query_overlap = max(best_query_overlap, len(query_tokens & gt_tokens) / len(query_tokens))

    return "true_recall_failure", {"best_query_to_gt_overlap": best_query_overlap}


def _inject_bm25_tail_candidates(
    *,
    query: str,
    merged_doc_ids: list[str],
    retriever: object,
    bm25_search_depth: int,
    rescue_tail_k: int,
) -> list[str]:
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


def _minmax_normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if (high - low) < 1e-9:
        return {key: 0.5 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def _interleave_doc_ids(primary: list[str], secondary: list[str], limit: int) -> list[str]:
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


def _build_stratified_rerank_pool(
    *,
    hybrid_doc_ids: list[str],
    semantic_doc_ids: list[str] | None,
    bm25_doc_ids: list[str] | None,
    limit: int,
) -> list[str]:
    sem = semantic_doc_ids or []
    bm = bm25_doc_ids or []
    stratified = _interleave_doc_ids(sem, bm, limit=max(limit * 2, 1))
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


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def _mmr_select_candidates(
    *,
    candidate_doc_ids: list[str],
    query_embedding: list[float],
    doc_embeddings: dict[str, list[float]],
    lambda_: float,
    max_k: int,
    diversity_threshold: float | None,
) -> list[str]:
    if max_k <= 0:
        return []
    seen: set[str] = set()
    candidates = [doc_id for doc_id in candidate_doc_ids if doc_id not in seen and not seen.add(doc_id)]
    if not candidates:
        return []

    # Keep candidate order for docs without embeddings.
    candidates_with_vec = [doc_id for doc_id in candidates if doc_id in doc_embeddings]
    candidates_without_vec = [doc_id for doc_id in candidates if doc_id not in doc_embeddings]
    if not candidates_with_vec:
        return candidates[:max_k]

    q_scores = {
        doc_id: _cosine_similarity(query_embedding, doc_embeddings[doc_id]) for doc_id in candidates_with_vec
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
                    _cosine_similarity(doc_embeddings[doc_id], doc_embeddings[selected_doc])
                    for selected_doc in selected
                )
                mmr_score = (lambda_ * relevance) - ((1.0 - lambda_) * max_sim_to_selected)

            if diversity_threshold is not None and selected and max_sim_to_selected >= diversity_threshold:
                continue
            if mmr_score > best_score:
                best_score = mmr_score
                best_doc_id = doc_id

        if best_doc_id is None:
            break
        selected.append(best_doc_id)
        remaining.remove(best_doc_id)

    # Backfill with non-embedded docs and skipped docs to keep pool size stable.
    if len(selected) < max_k:
        for doc_id in candidates_without_vec + remaining:
            if doc_id not in selected:
                selected.append(doc_id)
                if len(selected) >= max_k:
                    break
    return selected[:max_k]


def _source_miss_type(
    *,
    relevant_doc_ids: list[str],
    semantic_doc_ids: list[str] | None,
    bm25_doc_ids: list[str] | None,
) -> str:
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


def _rank_weight(rank: int) -> float:
    # Rank-aware contrastive weighting:
    # - 1..5: highest pressure to fix top-rank confusions
    # - 6..15: medium pressure
    # - 16..50: lower pressure
    if rank <= 5:
        return 1.0
    if rank <= 15:
        return 0.7
    return 0.4


def _build_reranker_training_contexts_from_failures(
    *,
    failure_records: list[dict[str, object]],
    doc_text_map: dict[str, str],
    max_negative_rank: int,
    max_negatives: int,
    ranking_cutoff_weight: float,
    true_recall_weight: float,
    default_weight: float,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    contexts: list[dict[str, object]] = []
    stats = {
        "samples_seen": 0,
        "samples_used": 0,
        "contexts_written": 0,
        "contexts_ranking_cutoff_failure": 0,
        "contexts_true_recall_failure": 0,
        "contexts_other": 0,
        "missing_positive_text": 0,
        "missing_negative_text": 0,
    }
    for sample in failure_records:
        stats["samples_seen"] += 1
        query = str(sample.get("query", "")).strip()
        bucket = str(sample.get("bucket", ""))
        source_miss_type = str(sample.get("source_miss_type", ""))
        positives = [str(doc_id) for doc_id in sample.get("relevant_doc_ids", [])]
        retrieved = [str(doc_id) for doc_id in sample.get("retrieved_top_k_doc_ids", [])]
        retrieved_full = [str(doc_id) for doc_id in sample.get("retrieved_full_doc_ids", [])]
        bm25_branch = [str(doc_id) for doc_id in sample.get("bm25_branch_doc_ids", [])]
        if not query or not positives or not retrieved:
            continue
        stats["samples_used"] += 1

        if bucket == "ranking_cutoff_failure":
            sample_weight = ranking_cutoff_weight
        elif bucket == "true_recall_failure":
            sample_weight = true_recall_weight
        else:
            sample_weight = default_weight

        positive_ids: list[str] = []
        for positive_id in positives:
            if positive_id not in doc_text_map:
                stats["missing_positive_text"] += 1
                continue
            positive_ids.append(positive_id)

        if bucket == "ranking_cutoff_failure":
            negative_pool = retrieved[:max_negative_rank]
        elif bucket == "true_recall_failure":
            # True recall failures are mostly retrieval-side; prefer BM25 expansion as auxiliary signal.
            negative_pool = (bm25_branch or retrieved_full or retrieved)[:max_negative_rank]
        else:
            negative_pool = retrieved[:max_negative_rank]

        negative_ids: list[str] = []
        negative_weights: dict[str, float] = {}
        positive_set = set(positive_ids)
        for rank, negative_id in enumerate(negative_pool, start=1):
            if len(negative_ids) >= max_negatives:
                break
            if negative_id in positive_set:
                continue
            if negative_id not in doc_text_map:
                stats["missing_negative_text"] += 1
                continue
            if negative_id in negative_weights:
                continue
            negative_ids.append(negative_id)
            negative_weights[negative_id] = sample_weight * _rank_weight(rank)

        # Source-miss signal is useful for hard-negative enrichment only.
        if source_miss_type in {"embedding_miss", "bm25_miss"} and len(negative_ids) < max_negatives:
            for rank, negative_id in enumerate(retrieved_full, start=len(negative_ids) + 1):
                if len(negative_ids) >= max_negatives:
                    break
                if negative_id in positive_set or negative_id in negative_weights:
                    continue
                if negative_id not in doc_text_map:
                    continue
                negative_ids.append(negative_id)
                negative_weights[negative_id] = sample_weight * _rank_weight(rank)

        if not positive_ids or not negative_ids:
            continue
        contexts.append(
            {
                "schema_version": "reranker_context_v1",
                "query": query,
                "positives": positive_ids,
                "negatives": negative_ids,
                "weights": negative_weights,
                "failure_bucket": bucket,
                "source_miss_type": source_miss_type,
            }
        )
        stats["contexts_written"] += 1
        if bucket == "ranking_cutoff_failure":
            stats["contexts_ranking_cutoff_failure"] += 1
        elif bucket == "true_recall_failure":
            stats["contexts_true_recall_failure"] += 1
        else:
            stats["contexts_other"] += 1
    return contexts, stats


def _train_reranker_from_contexts_jsonl(
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
) -> dict[str, object]:
    from sentence_transformers import InputExample
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    from torch.utils.data import DataLoader

    rows = [json.loads(line) for line in train_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    random.Random(seed).shuffle(rows)

    examples: list[InputExample] = []
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

    if not examples:
        raise ValueError("No train examples built from reranker JSONL.")

    val_size = int(len(examples) * max(0.0, min(val_ratio, 0.5)))
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]
    if not train_examples:
        raise ValueError("No train split left after validation split.")

    model = CrossEncoder(model_name, num_labels=1, max_length=512)
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    evaluator = None
    if val_examples:
        evaluator = CEBinaryClassificationEvaluator.from_input_examples(val_examples, name="failure-driven-val")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_dataloader=train_loader,
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(out_dir),
        show_progress_bar=True,
    )
    # Ensure a loadable model exists at output path for subsequent evaluation.
    model.save(str(out_dir))
    return {
        "out_dir": str(out_dir),
        "model": model_name,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "epochs": epochs,
        "batch_size": batch_size,
    }


def cmd_build_parser(args: argparse.Namespace) -> None:
    from parser.pipeline import run_pipeline

    stats = run_pipeline(
        output_path=args.output,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        overlap_ratio=args.overlap_ratio,
        min_output_chunk_tokens=args.min_output_chunk_tokens,
        max_output_chunk_tokens=args.max_output_chunk_tokens,
        max_chunks_per_url=args.max_chunks_per_url,
        max_chunks_per_category=args.max_chunks_per_category,
        chunker_mode=args.chunker_mode,
        near_duplicate_jaccard=args.near_duplicate_jaccard,
    )
    stats["embedding_model"] = args.embedding_model
    print(json.dumps(stats, indent=2))


def cmd_demo_retrieval(args: argparse.Namespace) -> None:
    from demo_retrieval import run_demo

    run_demo(
        query=args.query,
        top_k=args.top_k,
        model_name=args.model,
        dataset_path=args.dataset,
        faiss_path=args.faiss_path,
        index_name=args.index,
        rerank=args.rerank,
        reranker_model=args.reranker_model,
        rerank_candidates=args.rerank_candidates,
    )


def cmd_evaluation_runner(args: argparse.Namespace) -> None:
    from evaluation.dataset import load_eval_samples
    from evaluation.metrics import RetrievalResult, evaluate_retrieval
    from evaluation.runner import QueryRun, build_retriever, parse_k_values
    from ingestion.loaders import load_bm25_documents_from_dataset

    samples = load_eval_samples(Path(args.dataset))
    if not samples:
        raise ValueError(f"No samples found in dataset: {args.dataset}")
    total_samples_before_filter = len(samples)
    if args.require_evidence:
        samples = [sample for sample in samples if sample.relevant_docs]
    filtered_out_samples = total_samples_before_filter - len(samples)
    if not samples:
        raise ValueError(
            "No samples left after filtering. "
            "Try running without --require-evidence or regenerate dataset with more evidence links."
        )

    k_values = parse_k_values(args.k_values)
    max_k = max(k_values)
    retriever = build_retriever(
        args.retriever,
        rag_dataset_path=args.rag_dataset,
        faiss_path=args.faiss_path,
        index_name=args.index,
        embedding_model=args.embedding_model,
        alpha=args.alpha,
        hybrid_candidate_multiplier=args.hybrid_candidate_multiplier,
        hybrid_max_per_group=args.hybrid_max_per_group,
        hybrid_rrf_k=args.hybrid_rrf_k,
        cache_enabled=args.retrieval_cache_enabled,
        cache_capacity=args.retrieval_cache_capacity,
        cache_ttl_seconds=args.retrieval_cache_ttl_seconds,
    )
    semantic_embedding_map: dict[str, list[float]] = {}
    if args.retriever == "hybrid":
        semantic_obj = getattr(retriever, "semantic", None)
        documents = getattr(semantic_obj, "documents", None)
        if documents:
            semantic_embedding_map = {doc.doc_id: doc.embedding for doc in documents}
    doc_text_map = {
        item["id"]: item["text"] for item in load_bm25_documents_from_dataset(args.rag_dataset)
    }
    reranker = None
    if args.rerank:
        from reranking.cross_encoder import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name=args.reranker_model)

    query_runs: list[QueryRun] = []
    metric_inputs: list[RetrievalResult] = []
    llm_expansion_stats = {
        "requested_queries": 0,
        "generated_queries": 0,
        "fallback_queries": 0,
    }
    llm_expansion_examples: list[dict[str, object]] = []
    failure_records: list[dict[str, object]] = []
    miss_type_counts: Counter[str] = Counter()
    failure_bucket_source_counts: dict[str, Counter[str]] = {}
    llm_expansion_cache: dict[str, tuple[list[str], list[str], list[str]]] = {}
    if args.multi_query and args.multi_query_llm_expansion:
        llm_expansion_cache = _llm_structured_query_expansion_batch(
            [sample.query for sample in samples],
            provider=args.multi_query_llm_provider,
            model=args.multi_query_llm_model,
            api_base=args.multi_query_llm_api_base,
            api_key=args.multi_query_llm_api_key,
            timeout_seconds=args.multi_query_llm_timeout_seconds,
            retries=args.multi_query_llm_retries,
            cache_enabled=args.llm_cache_enabled,
            cache_capacity=args.llm_cache_capacity,
            cache_ttl_seconds=args.llm_cache_ttl_seconds,
        )
    for sample in samples:
        retrieve_k = max(max_k, args.rerank_candidates) if args.rerank else max_k
        semantic_branch_doc_ids: list[str] | None = None
        bm25_branch_doc_ids: list[str] | None = None
        semantic_score_map: dict[str, float] = {}
        bm25_score_map: dict[str, float] = {}
        query_embedding_for_mmr: list[float] | None = None
        if args.retriever == "hybrid":
            source_probe_k = max(retrieve_k, args.soft_recall_rescue_bm25_depth)
            semantic_obj = getattr(retriever, "semantic", None)
            bm25_obj = getattr(retriever, "bm25", None)
            if semantic_obj is not None and hasattr(semantic_obj, "search") and hasattr(semantic_obj, "embedder"):
                from retrieval.semantic import search_semantic
                from utils.embedding_format import format_query_for_embedding

                query_embedding = semantic_obj.embedder.encode(
                    [format_query_for_embedding(sample.query, semantic_obj.embedding_model)],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )[0].tolist()
                query_embedding_for_mmr = query_embedding
                semantic_hits = search_semantic(query_embedding, semantic_obj.documents, top_k=source_probe_k)
                semantic_branch_doc_ids = [item.doc_id for item in semantic_hits]
                semantic_score_map = {item.doc_id: float(item.score) for item in semantic_hits}
            if bm25_obj is not None and hasattr(bm25_obj, "index"):
                bm25_hits = bm25_obj.index.search(sample.query, top_k=source_probe_k)
                bm25_branch_doc_ids = [item.doc_id for item in bm25_hits]
                bm25_score_map = {item.doc_id: float(item.score) for item in bm25_hits}
        if args.multi_query:
            query_variants, llm_debug = _build_query_variants_with_debug(
                query=sample.query,
                max_variants=args.multi_query_variants,
                use_llm_structured_expansion=args.multi_query_llm_expansion,
                llm_provider=args.multi_query_llm_provider,
                llm_model=args.multi_query_llm_model,
                llm_api_base=args.multi_query_llm_api_base,
                llm_api_key=args.multi_query_llm_api_key,
                llm_timeout_seconds=args.multi_query_llm_timeout_seconds,
                llm_retries=args.multi_query_llm_retries,
                llm_cache_enabled=args.llm_cache_enabled,
                llm_cache_capacity=args.llm_cache_capacity,
                llm_cache_ttl_seconds=args.llm_cache_ttl_seconds,
                llm_precomputed=llm_expansion_cache.get(sample.query),
            )
            if args.multi_query_llm_debug and llm_debug.get("llm_requested"):
                llm_expansion_stats["requested_queries"] += 1
                if int(llm_debug.get("llm_generated_count", 0)) > 0:
                    llm_expansion_stats["generated_queries"] += 1
                    if len(llm_expansion_examples) < 5:
                        llm_expansion_examples.append(
                            {
                                "query": sample.query,
                                "generated_preview": llm_debug.get("llm_generated_preview", []),
                            }
                        )
                if bool(llm_debug.get("fallback_used", False)):
                    llm_expansion_stats["fallback_queries"] += 1
            per_query_results = [retriever.search(query, top_k=retrieve_k) for query in query_variants if query]
            retrieved = _rrf_fuse_doc_ids(per_query_results, top_k=retrieve_k, rrf_k=args.multi_query_rrf_k)
        else:
            retrieved = retriever.search(sample.query, top_k=retrieve_k)
        if reranker is not None:
            if args.stratified_rerank_pool and args.retriever == "hybrid":
                retrieved = _build_stratified_rerank_pool(
                    hybrid_doc_ids=retrieved,
                    semantic_doc_ids=semantic_branch_doc_ids,
                    bm25_doc_ids=bm25_branch_doc_ids,
                    limit=retrieve_k,
                )
            if args.soft_recall_rescue and args.retriever == "hybrid":
                retrieved = _inject_bm25_tail_candidates(
                    query=sample.query,
                    merged_doc_ids=retrieved,
                    retriever=retriever,
                    bm25_search_depth=args.soft_recall_rescue_bm25_depth,
                    rescue_tail_k=args.soft_recall_rescue_tail_k,
                )
            if (
                args.mmr_before_rerank
                and query_embedding_for_mmr is not None
                and semantic_embedding_map
                and args.retriever == "hybrid"
            ):
                mmr_k = args.mmr_k if args.mmr_k > 0 else retrieve_k
                mmr_k = min(mmr_k, retrieve_k)
                diversity_threshold = (
                    args.mmr_diversity_threshold if args.mmr_diversity_threshold > 0 else None
                )
                retrieved = _mmr_select_candidates(
                    candidate_doc_ids=retrieved,
                    query_embedding=query_embedding_for_mmr,
                    doc_embeddings=semantic_embedding_map,
                    lambda_=args.mmr_lambda,
                    max_k=mmr_k,
                    diversity_threshold=diversity_threshold,
                )
            from reranking.cross_encoder import RerankCandidate

            sem_norm = _minmax_normalize(semantic_score_map)
            bm25_norm = _minmax_normalize(bm25_score_map)
            rerank_input = [
                RerankCandidate(
                    doc_id=doc_id,
                    text=doc_text_map.get(doc_id, ""),
                    score=(
                        (args.rerank_semantic_weight * sem_norm.get(doc_id, 0.0))
                        + (args.rerank_bm25_weight * bm25_norm.get(doc_id, 0.0))
                    ),
                    metadata={
                        "semantic_norm": sem_norm.get(doc_id, 0.0),
                        "bm25_norm": bm25_norm.get(doc_id, 0.0),
                    },
                )
                for rank, doc_id in enumerate(retrieved, start=1)
                if doc_text_map.get(doc_id, "")
            ]
            if args.hard_negative_semantic_floor > 0.0:
                rerank_input = [
                    candidate
                    for candidate in rerank_input
                    if candidate.metadata.get("semantic_norm", 0.0) >= args.hard_negative_semantic_floor
                ]
            if args.two_stage_rerank:
                rerank_input = _prefilter_rerank_candidates(
                    sample.query,
                    rerank_input,
                    keep_top_n=args.prefilter_candidates,
                )
            reranked = reranker.rerank(
                sample.query,
                rerank_input,
                top_k=retrieve_k,
                alpha=args.rerank_alpha,
                ce_calibration=args.ce_calibration,
                ce_temperature=args.ce_temperature,
            )
            retrieved = [item.doc_id for item in reranked]
        else:
            retrieved = retrieved[:retrieve_k]
        retrieved_for_metrics = retrieved[:max_k]
        query_runs.append(
            QueryRun(
                query=sample.query,
                relevant_doc_ids=sample.relevant_docs,
                retrieved_doc_ids=retrieved_for_metrics,
            )
        )
        metric_inputs.append(
            RetrievalResult(
                query=sample.query,
                retrieved_doc_ids=retrieved_for_metrics,
                relevant_doc_ids=sample.relevant_docs,
            )
        )
        if sample.relevant_docs and not set(retrieved_for_metrics).intersection(sample.relevant_docs):
            miss_type = _source_miss_type(
                relevant_doc_ids=sample.relevant_docs,
                semantic_doc_ids=semantic_branch_doc_ids,
                bm25_doc_ids=bm25_branch_doc_ids,
            )
            miss_type_counts[miss_type] += 1
            bucket, reasons = _classify_failure(
                query=sample.query,
                gt_doc_ids=sample.relevant_docs,
                top_k_doc_ids=retrieved_for_metrics,
                all_ranked_doc_ids=retrieved,
                doc_text_map=doc_text_map,
                near_miss_threshold=args.failure_near_miss_threshold,
                top_k=max_k,
            )
            if bucket not in failure_bucket_source_counts:
                failure_bucket_source_counts[bucket] = Counter()
            failure_bucket_source_counts[bucket][miss_type] += 1
            failure_records.append(
                {
                    "query": sample.query,
                    "bucket": bucket,
                    "relevant_doc_ids": sample.relevant_docs,
                    "retrieved_top_k_doc_ids": retrieved_for_metrics,
                    "retrieved_full_doc_ids": retrieved,
                    "bm25_branch_doc_ids": (bm25_branch_doc_ids or [])[: args.soft_recall_rescue_bm25_depth],
                    "source_miss_type": miss_type,
                    "reasons": reasons,
                }
            )

    metrics = evaluate_retrieval(metric_inputs, k_values)
    top1_doc_ids = [run.retrieved_doc_ids[0] for run in query_runs if run.retrieved_doc_ids]
    top1_counts = Counter(top1_doc_ids)
    top1_total = sum(top1_counts.values())
    top1_distribution = (
        {doc_id: count / top1_total for doc_id, count in top1_counts.most_common()}
        if top1_total
        else {}
    )
    max_top1_doc, max_top1_count = top1_counts.most_common(1)[0] if top1_counts else (None, 0)
    failure_bucket_counts = Counter(item["bucket"] for item in failure_records)
    failure_bucket_source_counts_json = {
        bucket: dict(counter) for bucket, counter in failure_bucket_source_counts.items()
    }
    manual_inspection_failed_queries = failure_records[: args.failure_sample_size]
    report = {
        "dataset": args.dataset,
        "retriever": args.retriever,
        "rerank_enabled": args.rerank,
        "rerank_alpha": args.rerank_alpha if args.rerank else None,
        "two_stage_rerank_enabled": args.two_stage_rerank,
        "reranker_model": args.reranker_model if args.rerank else None,
        "stratified_rerank_pool_enabled": args.stratified_rerank_pool,
        "hard_negative_semantic_floor": args.hard_negative_semantic_floor,
        "rerank_prior_weights": {
            "semantic": args.rerank_semantic_weight,
            "bm25": args.rerank_bm25_weight,
        },
        "ce_calibration": args.ce_calibration if args.rerank else None,
        "ce_temperature": args.ce_temperature if args.rerank else None,
        "soft_recall_rescue_enabled": args.soft_recall_rescue,
        "soft_recall_rescue_tail_k": args.soft_recall_rescue_tail_k if args.soft_recall_rescue else 0,
        "soft_recall_rescue_bm25_depth": args.soft_recall_rescue_bm25_depth if args.soft_recall_rescue else 0,
        "multi_query_enabled": args.multi_query,
        "cache": {
            "retrieval_enabled": args.retrieval_cache_enabled,
            "retrieval_capacity": args.retrieval_cache_capacity,
            "retrieval_ttl_seconds": args.retrieval_cache_ttl_seconds,
            "llm_enabled": args.llm_cache_enabled,
            "llm_capacity": args.llm_cache_capacity,
            "llm_ttl_seconds": args.llm_cache_ttl_seconds,
        },
        "mmr_before_rerank": args.mmr_before_rerank,
        "mmr_lambda": args.mmr_lambda if args.mmr_before_rerank else None,
        "mmr_k": args.mmr_k if args.mmr_before_rerank else None,
        "mmr_diversity_threshold": (
            args.mmr_diversity_threshold if (args.mmr_before_rerank and args.mmr_diversity_threshold > 0) else None
        ),
        "require_evidence": args.require_evidence,
        "k_values": k_values,
        "samples_total": len(samples),
        "samples_total_before_filter": total_samples_before_filter,
        "samples_filtered_out": filtered_out_samples,
        "samples_with_ground_truth": sum(1 for s in samples if s.relevant_docs),
        "metrics": metrics,
        "evaluation": {
            "failed_queries_for_manual_inspection": manual_inspection_failed_queries,
            "failed_queries_for_manual_inspection_count": len(manual_inspection_failed_queries),
        },
        "diagnostics": {
            "top1_unique_docs": len(top1_counts),
            "top1_total_queries": top1_total,
            "top1_most_common_doc_id": max_top1_doc,
            "top1_most_common_doc_fraction": (max_top1_count / top1_total) if top1_total else 0.0,
            "top1_distribution": top1_distribution,
            "failure_analysis": {
                "enabled": True,
                "failure_count": len(failure_records),
                "failure_bucket_counts": dict(failure_bucket_counts),
                "failure_source_miss_counts": dict(miss_type_counts),
                "failure_bucket_source_miss_counts": failure_bucket_source_counts_json,
                "near_miss_threshold": args.failure_near_miss_threshold,
                "manual_inspection_samples": manual_inspection_failed_queries,
            },
        },
        "runs": [run.__dict__ for run in query_runs],
    }

    reranker_dataset_export: dict[str, object] | None = None
    reranker_training_result: dict[str, object] | None = None
    if args.export_reranker_train_jsonl or args.train_reranker:
        out_jsonl = Path(args.export_reranker_train_jsonl or "artifacts/datasets/reranker_train.jsonl")
        contexts, pair_stats = _build_reranker_training_contexts_from_failures(
            failure_records=failure_records,
            doc_text_map=doc_text_map,
            max_negative_rank=max(1, args.reranker_train_max_negative_rank),
            max_negatives=max(1, args.reranker_train_max_negatives),
            ranking_cutoff_weight=max(0.1, args.reranker_train_weight_ranking_cutoff),
            true_recall_weight=max(0.1, args.reranker_train_weight_true_recall),
            default_weight=max(0.1, args.reranker_train_weight_default),
        )
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as fp:
            for row in contexts:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        reranker_dataset_export = {
            "path": str(out_jsonl),
            "contexts": len(contexts),
            "stats": pair_stats,
            "schema_version": "reranker_context_v1",
        }
        report["reranker_dataset_export"] = reranker_dataset_export
        print(f"- reranker_dataset_export: {out_jsonl} ({len(contexts)} contexts)")
        if args.train_reranker:
            reranker_training_result = _train_reranker_from_contexts_jsonl(
                train_jsonl=out_jsonl,
                doc_text_map=doc_text_map,
                model_name=args.train_reranker_model,
                out_dir=Path(args.train_reranker_out_dir),
                epochs=max(1, args.train_reranker_epochs),
                batch_size=max(1, args.train_reranker_batch_size),
                warmup_steps=max(0, args.train_reranker_warmup_steps),
                val_ratio=args.train_reranker_val_ratio,
                seed=args.train_reranker_seed,
            )
            report["reranker_training"] = reranker_training_result
            print(f"- reranker_training_out: {reranker_training_result['out_dir']}")

    print("Retrieval benchmark report")
    print(f"- dataset: {args.dataset}")
    print(f"- retriever: {args.retriever}")
    print(f"- samples: {len(samples)}")
    if args.require_evidence:
        print(f"- require_evidence: true (filtered_out={filtered_out_samples})")
    for key in sorted(metrics):
        print(f"- {key}: {metrics[key]:.4f}")
    if failure_records:
        print("- failure_buckets:")
        for bucket in ("near_miss", "fragmentation", "ranking_cutoff_failure", "true_recall_failure"):
            print(f"  - {bucket}: {failure_bucket_counts.get(bucket, 0)}")
        print("- failure_source_miss:")
        for key in ("embedding_miss", "bm25_miss", "both_miss", "both_hit"):
            print(f"  - {key}: {miss_type_counts.get(key, 0)}")
        print("- failure_bucket_source_miss:")
        for bucket in ("near_miss", "fragmentation", "ranking_cutoff_failure", "true_recall_failure"):
            per_bucket = failure_bucket_source_counts.get(bucket, Counter())
            print(
                "  - "
                f"{bucket}: embedding_miss={per_bucket.get('embedding_miss', 0)}, "
                f"bm25_miss={per_bucket.get('bm25_miss', 0)}, "
                f"both_miss={per_bucket.get('both_miss', 0)}, "
                f"both_hit={per_bucket.get('both_hit', 0)}"
            )
        print(f"- failed_queries_for_manual_inspection: {min(len(failure_records), args.failure_sample_size)}")
    if args.multi_query_llm_debug and args.multi_query_llm_expansion:
        print(
            "- multi_query_llm_debug: "
            f"requested={llm_expansion_stats['requested_queries']}, "
            f"generated={llm_expansion_stats['generated_queries']}, "
            f"fallback={llm_expansion_stats['fallback_queries']}"
        )
        for item in llm_expansion_examples:
            print(f"  - llm_example_query: {item['query']}")
            print(f"    generated_preview: {item['generated_preview']}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved JSON report to {out_path}")


def cmd_run_rag(args: argparse.Namespace) -> None:
    from generation.run_rag import run_rag

    run_rag(
        question=args.question,
        provider=args.provider,
        model=args.model,
        top_k=args.top_k,
        max_context_tokens=args.max_context_tokens,
        faiss_path=args.faiss_path,
        index_name=args.index,
        embedding_model=args.embedding_model,
        stream=args.stream,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        rerank=args.rerank,
        reranker_model=args.reranker_model,
        rerank_candidates=args.rerank_candidates,
        llm_cache_enabled=args.llm_cache_enabled,
        llm_cache_capacity=args.llm_cache_capacity,
        llm_cache_ttl_seconds=args.llm_cache_ttl_seconds,
    )


def cmd_cleanup_faiss(args: argparse.Namespace) -> None:
    from ingestion.cleaner import cleanup_faiss_db

    result = cleanup_faiss_db(
        persist_directory=args.faiss_path,
        index_name=args.index,
        drop_persist_directory=args.drop_persist_directory,
    )
    print(json.dumps(result, indent=2))


def cmd_reranker_pipeline(args: argparse.Namespace) -> None:
    eval_args = argparse.Namespace(
        # Core IO
        dataset=args.dataset,
        rag_dataset=args.rag_dataset,
        faiss_path=args.faiss_path,
        index=args.index,
        out_json=args.out_json,
        # Retrieval stack defaults (current tuned setup)
        retriever="hybrid",
        k_values=args.k_values,
        embedding_model=args.embedding_model,
        alpha=0.65,
        hybrid_candidate_multiplier=80,
        hybrid_max_per_group=1,
        hybrid_rrf_k=80.0,
        # Reranker stack defaults
        rerank=True,
        reranker_model=args.reranker_model,
        rerank_candidates=40,
        rerank_alpha=0.45,
        ce_calibration="zscore",
        ce_temperature=1.0,
        stratified_rerank_pool=True,
        hard_negative_semantic_floor=0.12,
        rerank_semantic_weight=0.55,
        rerank_bm25_weight=0.45,
        two_stage_rerank=True,
        prefilter_candidates=40,
        # Multi-query + rescue + MMR defaults
        multi_query=True,
        multi_query_variants=3,
        multi_query_rrf_k=60,
        multi_query_llm_expansion=args.multi_query_llm_expansion,
        multi_query_llm_provider=args.multi_query_llm_provider,
        multi_query_llm_model=args.multi_query_llm_model,
        multi_query_llm_api_base=args.multi_query_llm_api_base,
        multi_query_llm_api_key=args.multi_query_llm_api_key,
        multi_query_llm_timeout_seconds=args.multi_query_llm_timeout_seconds,
        multi_query_llm_retries=args.multi_query_llm_retries,
        multi_query_llm_debug=args.multi_query_llm_debug,
        retrieval_cache_enabled=args.retrieval_cache_enabled,
        retrieval_cache_capacity=args.retrieval_cache_capacity,
        retrieval_cache_ttl_seconds=args.retrieval_cache_ttl_seconds,
        llm_cache_enabled=args.llm_cache_enabled,
        llm_cache_capacity=args.llm_cache_capacity,
        llm_cache_ttl_seconds=args.llm_cache_ttl_seconds,
        soft_recall_rescue=True,
        soft_recall_rescue_tail_k=20,
        soft_recall_rescue_bm25_depth=200,
        mmr_before_rerank=True,
        mmr_lambda=0.82,
        mmr_k=30,
        mmr_diversity_threshold=0.0,
        # Failure analysis defaults
        require_evidence=True,
        failure_near_miss_threshold=0.80,
        failure_sample_size=20,
        # Integrated reranker dataset + train toggles
        export_reranker_train_jsonl=args.export_reranker_train_jsonl,
        reranker_train_max_negative_rank=20,
        reranker_train_max_negatives=16,
        reranker_train_weight_ranking_cutoff=2.0,
        reranker_train_weight_true_recall=0.3,
        reranker_train_weight_default=1.0,
        train_reranker=args.train_reranker,
        train_reranker_model=args.train_reranker_model,
        train_reranker_out_dir=args.train_reranker_out_dir,
        train_reranker_epochs=args.train_reranker_epochs,
        train_reranker_batch_size=args.train_reranker_batch_size,
        train_reranker_warmup_steps=args.train_reranker_warmup_steps,
        train_reranker_val_ratio=args.train_reranker_val_ratio,
        train_reranker_seed=args.train_reranker_seed,
    )
    cmd_evaluation_runner(eval_args)


def cmd_build_evaluation_dataset(args: argparse.Namespace) -> None:
    from evaluation.dataset import build_evaluation_dataset

    count, stats = build_evaluation_dataset(
        rag_path=Path(args.rag),
        eval_json_path=Path(args.eval),
        out_path=Path(args.out),
        fuzzy_ratio=args.fuzzy_ratio,
        lexical_min_hits=args.lexical_min_hits,
        max_chunk_ids=args.max_chunk_ids,
        semantic_fallback=not args.no_semantic_fallback,
        semantic_model=args.semantic_model,
        semantic_min_score=args.semantic_min_score,
        max_gt_url_share=args.max_gt_url_share,
        target_multi_gt_share=args.target_multi_gt_share,
        keep_max_ids_for_multi=args.keep_max_ids_for_multi,
        excerpt_max=args.excerpt_max,
    )
    print(f"Wrote {args.out} ({count} records).")
    print("Stats:", json.dumps(stats, indent=2))


def cmd_dataset_audit(args: argparse.Namespace) -> None:
    from commands.dataset_audit import audit

    report = audit(Path(args.rag), Path(args.eval))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_build_reranker_dataset(args: argparse.Namespace) -> None:
    from commands.build_reranker_dataset import build_contexts, load_chunk_texts

    report = json.loads(Path(args.eval_report).read_text(encoding="utf-8"))
    chunk_texts = load_chunk_texts(Path(args.rag_dataset))
    contexts, stats = build_contexts(
        report=report,
        chunk_texts=chunk_texts,
        max_negative_rank=max(1, args.max_negative_rank),
        max_negatives=max(1, args.max_negatives),
        ranking_cutoff_weight=max(0.1, args.ranking_cutoff_weight),
        true_recall_weight=max(0.1, args.true_recall_weight),
        default_weight=max(0.1, args.default_weight),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for row in contexts:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path} ({len(contexts)} contexts).")
    print("Stats:", json.dumps(stats, indent=2))


def cmd_train_reranker(args: argparse.Namespace) -> None:
    from commands.train_reranker import load_chunk_texts, load_pairwise_samples
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    from torch.utils.data import DataLoader

    chunk_texts = load_chunk_texts(Path(args.rag_dataset))
    train_examples, val_examples = load_pairwise_samples(
        Path(args.train_jsonl),
        seed=args.seed,
        val_ratio=args.val_ratio,
        chunk_texts=chunk_texts,
    )
    if not train_examples:
        raise ValueError("No training examples loaded. Check --train-jsonl contents.")

    model = CrossEncoder(args.model, num_labels=1, max_length=512)
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    evaluator = None
    if val_examples:
        evaluator = CEBinaryClassificationEvaluator.from_input_examples(
            val_examples,
            name="failure-driven-val",
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_dataloader=train_loader,
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        output_path=str(out_dir),
        show_progress_bar=True,
    )
    model.save(str(out_dir))
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "train_examples": len(train_examples),
                "val_examples": len(val_examples),
                "model": args.model,
            },
            indent=2,
        )
    )


def cmd_run_experiments(args: argparse.Namespace) -> None:
    from experiments.run_experiments import run_experiments

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    run_experiments(
        question=args.question,
        models=models,
        top_k=args.top_k,
        max_context_tokens=args.max_context_tokens,
        faiss_path=args.faiss_path,
        index_name=args.index,
        embedding_model=args.embedding_model,
        log_path=args.log_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single entrypoint for project workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser_cmd = subparsers.add_parser(
        "build_parser",
        aliases=["build-parser"],
        help="Run parser pipeline and build rag_dataset.jsonl",
    )
    build_parser_cmd.add_argument("--output", default="data/rag_dataset.jsonl")
    build_parser_cmd.add_argument("--min-tokens", type=int, default=300)
    build_parser_cmd.add_argument("--max-tokens", type=int, default=800)
    build_parser_cmd.add_argument("--overlap-ratio", type=float, default=0.15)
    build_parser_cmd.add_argument("--min-output-chunk-tokens", type=int, default=120)
    build_parser_cmd.add_argument("--max-output-chunk-tokens", type=int, default=650)
    build_parser_cmd.add_argument("--max-chunks-per-url", type=int, default=12)
    build_parser_cmd.add_argument("--max-chunks-per-category", type=int, default=45)
    build_parser_cmd.add_argument("--chunker-mode", choices=("token", "semantic_dynamic"), default="token")
    build_parser_cmd.add_argument(
        "--near-duplicate-jaccard",
        type=float,
        default=0.0,
        help="Skip near-duplicate chunks from the same URL when similarity >= threshold (0 disables).",
    )
    build_parser_cmd.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    build_parser_cmd.set_defaults(handler=cmd_build_parser)

    demo_cmd = subparsers.add_parser("demo_retrieval", help="Run BM25/semantic/hybrid retrieval demo.")
    demo_cmd.add_argument("--query", "-q", default="database caching performance")
    demo_cmd.add_argument("--top-k", "-k", type=int, default=4)
    demo_cmd.add_argument("--model", "-m", default=DEFAULT_EMBEDDING_MODEL)
    demo_cmd.add_argument("--dataset", default="data/rag_dataset.jsonl")
    demo_cmd.add_argument("--faiss-path", default="artifacts/faiss")
    demo_cmd.add_argument("--index", default="rag_chunks")
    demo_cmd.add_argument("--rerank", action="store_true")
    demo_cmd.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    demo_cmd.add_argument("--rerank-candidates", type=int, default=20)
    demo_cmd.set_defaults(handler=cmd_demo_retrieval)

    eval_cmd = subparsers.add_parser(
        "evaluation_runner",
        aliases=["evaluate"],
        help="Run retrieval benchmark over eval dataset.",
    )
    eval_cmd.add_argument("--dataset", default="data/evaluation_with_evidence.jsonl")
    eval_cmd.add_argument("--retriever", choices=("semantic", "bm25", "hybrid"), default="semantic")
    eval_cmd.add_argument("--k-values", default="1,3,5")
    eval_cmd.add_argument("--rag-dataset", default="data/rag_dataset.jsonl")
    eval_cmd.add_argument("--faiss-path", default="artifacts/faiss")
    eval_cmd.add_argument("--index", default="rag_chunks")
    eval_cmd.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    eval_cmd.add_argument("--alpha", type=float, default=0.7)
    eval_cmd.add_argument(
        "--hybrid-candidate-multiplier",
        type=int,
        default=2,
        help="Hybrid per-branch candidate pool multiplier before merge/rerank.",
    )
    eval_cmd.add_argument(
        "--hybrid-max-per-group",
        type=int,
        default=1,
        help="Max documents per source/section group in hybrid top-k (<=0 disables).",
    )
    eval_cmd.add_argument(
        "--hybrid-rrf-k",
        type=float,
        default=60.0,
        help="RRF k parameter used to fuse semantic and BM25 ranks.",
    )
    eval_cmd.add_argument("--rerank", action="store_true")
    eval_cmd.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    eval_cmd.add_argument("--rerank-candidates", type=int, default=20)
    eval_cmd.add_argument("--rerank-alpha", type=float, default=0.75)
    eval_cmd.add_argument(
        "--ce-calibration",
        choices=("minmax", "softmax", "zscore"),
        default="minmax",
        help="How to normalize CE scores per query before fusion with prior scores.",
    )
    eval_cmd.add_argument(
        "--ce-temperature",
        type=float,
        default=1.0,
        help="Temperature for CE calibration modes (softmax/zscore). >1 flattens scores.",
    )
    eval_cmd.add_argument(
        "--stratified-rerank-pool",
        action="store_true",
        help="Interleave semantic/BM25 candidates before reranking.",
    )
    eval_cmd.add_argument(
        "--hard-negative-semantic-floor",
        type=float,
        default=0.0,
        help="Drop rerank candidates with normalized semantic score below threshold.",
    )
    eval_cmd.add_argument("--rerank-semantic-weight", type=float, default=0.55)
    eval_cmd.add_argument("--rerank-bm25-weight", type=float, default=0.45)
    eval_cmd.add_argument(
        "--two-stage-rerank",
        action="store_true",
        help="Apply lexical prefilter before cross-encoder reranking.",
    )
    eval_cmd.add_argument(
        "--prefilter-candidates",
        type=int,
        default=40,
        help="How many candidates to keep for stage-2 cross-encoder reranking.",
    )
    eval_cmd.add_argument("--multi-query", action="store_true")
    eval_cmd.add_argument("--multi-query-variants", type=int, default=3)
    eval_cmd.add_argument("--multi-query-rrf-k", type=int, default=60)
    eval_cmd.add_argument(
        "--multi-query-llm-expansion",
        action="store_true",
        help="Use LLM-based structured expansion (paraphrase + decomposition + concept queries).",
    )
    eval_cmd.add_argument("--multi-query-llm-provider", default="qwen")
    eval_cmd.add_argument("--multi-query-llm-model", default="qwen-plus")
    eval_cmd.add_argument("--multi-query-llm-api-base", default=None)
    eval_cmd.add_argument("--multi-query-llm-api-key", default=None)
    eval_cmd.add_argument("--multi-query-llm-timeout-seconds", type=int, default=8)
    eval_cmd.add_argument("--multi-query-llm-retries", type=int, default=0)
    eval_cmd.add_argument("--multi-query-llm-debug", action="store_true")
    eval_cmd.add_argument(
        "--retrieval-cache-enabled",
        action="store_true",
        help="Enable in-memory cache for retrieval query results.",
    )
    eval_cmd.add_argument("--retrieval-cache-capacity", type=int, default=10000)
    eval_cmd.add_argument("--retrieval-cache-ttl-seconds", type=float, default=300.0)
    eval_cmd.add_argument(
        "--llm-cache-enabled",
        action="store_true",
        help="Enable in-memory cache for LLM calls (query expansion / generation).",
    )
    eval_cmd.add_argument("--llm-cache-capacity", type=int, default=512)
    eval_cmd.add_argument("--llm-cache-ttl-seconds", type=float, default=300.0)
    eval_cmd.add_argument(
        "--soft-recall-rescue",
        action="store_true",
        help="Inject BM25-only tail candidates into reranker pool after hybrid retrieval.",
    )
    eval_cmd.add_argument(
        "--soft-recall-rescue-tail-k",
        type=int,
        default=30,
        help="How many BM25-only candidates to inject into reranker pool.",
    )
    eval_cmd.add_argument(
        "--soft-recall-rescue-bm25-depth",
        type=int,
        default=200,
        help="How deep to search BM25 before extracting BM25-only tail candidates.",
    )
    eval_cmd.add_argument(
        "--mmr-before-rerank",
        action="store_true",
        help="Apply MMR diversity selection after fusion/rescue and before CE reranking.",
    )
    eval_cmd.add_argument("--mmr-lambda", type=float, default=0.75)
    eval_cmd.add_argument(
        "--mmr-k",
        type=int,
        default=40,
        help="Candidate pool size kept after MMR (<=0 uses rerank candidate size).",
    )
    eval_cmd.add_argument(
        "--mmr-diversity-threshold",
        type=float,
        default=0.0,
        help="Optional hard max cosine similarity to selected docs (<=0 disables).",
    )
    eval_cmd.add_argument(
        "--require-evidence",
        action="store_true",
        help="Evaluate only samples with non-empty expected_evidence.chunk_ids.",
    )
    eval_cmd.add_argument(
        "--failure-near-miss-threshold",
        type=float,
        default=0.80,
        help="Similarity threshold used to classify failures as near_miss.",
    )
    eval_cmd.add_argument(
        "--failure-sample-size",
        type=int,
        default=20,
        help="Number of failed queries to include for manual inspection.",
    )
    eval_cmd.add_argument(
        "--export-reranker-train-jsonl",
        default=None,
        help="Optional path to export pairwise hard-negative reranker training dataset.",
    )
    eval_cmd.add_argument("--reranker-train-max-negative-rank", type=int, default=20)
    eval_cmd.add_argument("--reranker-train-max-negatives", type=int, default=16)
    eval_cmd.add_argument("--reranker-train-weight-ranking-cutoff", type=float, default=2.0)
    eval_cmd.add_argument("--reranker-train-weight-true-recall", type=float, default=0.3)
    eval_cmd.add_argument("--reranker-train-weight-default", type=float, default=1.0)
    eval_cmd.add_argument(
        "--train-reranker",
        action="store_true",
        help="Train reranker in the same run after exporting hard-negative dataset.",
    )
    eval_cmd.add_argument("--train-reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    eval_cmd.add_argument("--train-reranker-out-dir", default="artifacts/models/reranker-failure-driven")
    eval_cmd.add_argument("--train-reranker-epochs", type=int, default=2)
    eval_cmd.add_argument("--train-reranker-batch-size", type=int, default=16)
    eval_cmd.add_argument("--train-reranker-warmup-steps", type=int, default=100)
    eval_cmd.add_argument("--train-reranker-val-ratio", type=float, default=0.1)
    eval_cmd.add_argument("--train-reranker-seed", type=int, default=42)
    eval_cmd.add_argument("--out-json", default=None)
    eval_cmd.set_defaults(handler=cmd_evaluation_runner)

    rerank_pipeline_cmd = subparsers.add_parser(
        "reranker_pipeline",
        help="One-shot eval + failure dataset export + optional reranker training.",
    )
    rerank_pipeline_cmd.add_argument("--dataset", default="data/evaluation_with_evidence.jsonl")
    rerank_pipeline_cmd.add_argument("--rag-dataset", default="data/rag_dataset.jsonl")
    rerank_pipeline_cmd.add_argument("--faiss-path", default="artifacts/faiss")
    rerank_pipeline_cmd.add_argument("--index", default="rag_chunks")
    rerank_pipeline_cmd.add_argument("--embedding-model", default="intfloat/e5-base-v2")
    rerank_pipeline_cmd.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_pipeline_cmd.add_argument("--k-values", default="1,10,20")
    rerank_pipeline_cmd.add_argument("--multi-query-llm-expansion", action="store_true")
    rerank_pipeline_cmd.add_argument("--multi-query-llm-provider", default="qwen")
    rerank_pipeline_cmd.add_argument("--multi-query-llm-model", default="qwen-plus")
    rerank_pipeline_cmd.add_argument("--multi-query-llm-api-base", default=None)
    rerank_pipeline_cmd.add_argument("--multi-query-llm-api-key", default=None)
    rerank_pipeline_cmd.add_argument("--multi-query-llm-timeout-seconds", type=int, default=8)
    rerank_pipeline_cmd.add_argument("--multi-query-llm-retries", type=int, default=0)
    rerank_pipeline_cmd.add_argument("--multi-query-llm-debug", action="store_true")
    rerank_pipeline_cmd.add_argument("--retrieval-cache-enabled", action="store_true")
    rerank_pipeline_cmd.add_argument("--retrieval-cache-capacity", type=int, default=10000)
    rerank_pipeline_cmd.add_argument("--retrieval-cache-ttl-seconds", type=float, default=300.0)
    rerank_pipeline_cmd.add_argument("--llm-cache-enabled", action="store_true")
    rerank_pipeline_cmd.add_argument("--llm-cache-capacity", type=int, default=512)
    rerank_pipeline_cmd.add_argument("--llm-cache-ttl-seconds", type=float, default=300.0)
    rerank_pipeline_cmd.add_argument("--out-json", default="experiments/results/retrieval_report_best.json")
    rerank_pipeline_cmd.add_argument(
        "--export-reranker-train-jsonl",
        default="artifacts/datasets/reranker_train.jsonl",
        help="Path for exported reranker pairwise training dataset.",
    )
    rerank_pipeline_cmd.add_argument("--train-reranker", action="store_true")
    rerank_pipeline_cmd.add_argument("--train-reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_pipeline_cmd.add_argument("--train-reranker-out-dir", default="artifacts/models/reranker-failure-driven")
    rerank_pipeline_cmd.add_argument("--train-reranker-epochs", type=int, default=2)
    rerank_pipeline_cmd.add_argument("--train-reranker-batch-size", type=int, default=16)
    rerank_pipeline_cmd.add_argument("--train-reranker-warmup-steps", type=int, default=100)
    rerank_pipeline_cmd.add_argument("--train-reranker-val-ratio", type=float, default=0.1)
    rerank_pipeline_cmd.add_argument("--train-reranker-seed", type=int, default=42)
    rerank_pipeline_cmd.set_defaults(handler=cmd_reranker_pipeline)

    rag_cmd = subparsers.add_parser(
        "run_rag",
        aliases=["run-rag"],
        help="Run full RAG query against selected LLM provider.",
    )
    rag_cmd.add_argument("--question", "-q", required=True)
    rag_cmd.add_argument("--provider", default="openai", choices=("openai", "gigachat", "ollama", "qwen"))
    rag_cmd.add_argument("--model", default=None)
    rag_cmd.add_argument("--top-k", type=int, default=5)
    rag_cmd.add_argument("--max-context-tokens", type=int, default=2500)
    rag_cmd.add_argument("--faiss-path", default="artifacts/faiss")
    rag_cmd.add_argument("--index", default="rag_chunks")
    rag_cmd.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    rag_cmd.add_argument("--stream", action="store_true")
    rag_cmd.add_argument("--max-tokens", type=int, default=512)
    rag_cmd.add_argument("--temperature", type=float, default=0.1)
    rag_cmd.add_argument("--top-p", type=float, default=0.95)
    rag_cmd.add_argument("--rerank", action="store_true")
    rag_cmd.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    rag_cmd.add_argument("--rerank-candidates", type=int, default=20)
    rag_cmd.add_argument("--llm-cache-enabled", action="store_true")
    rag_cmd.add_argument("--llm-cache-capacity", type=int, default=512)
    rag_cmd.add_argument("--llm-cache-ttl-seconds", type=float, default=300.0)
    rag_cmd.set_defaults(handler=cmd_run_rag)

    clean_cmd = subparsers.add_parser("cleanup_faiss", help="Delete FAISS index and optionally full directory.")
    clean_cmd.add_argument("--faiss-path", default="artifacts/faiss")
    clean_cmd.add_argument("--index", default="rag_chunks")
    clean_cmd.add_argument("--drop-persist-directory", action="store_true")
    clean_cmd.set_defaults(handler=cmd_cleanup_faiss)

    build_eval_cmd = subparsers.add_parser(
        "build_evaluation_dataset",
        aliases=["build-eval-dataset"],
        help="Build structured evaluation JSONL from eval JSON and rag dataset.",
    )
    build_eval_cmd.add_argument("--rag", default="data/rag_dataset.jsonl")
    build_eval_cmd.add_argument("--eval", default="evaluation/evaluation.json")
    build_eval_cmd.add_argument("--out", default="data/evaluation_with_evidence.jsonl")
    build_eval_cmd.add_argument("--fuzzy-ratio", type=float, default=0.86)
    build_eval_cmd.add_argument("--lexical-min-hits", type=int, default=2)
    build_eval_cmd.add_argument("--max-chunk-ids", type=int, default=2)
    build_eval_cmd.add_argument("--no-semantic-fallback", action="store_true")
    build_eval_cmd.add_argument("--semantic-model", default=DEFAULT_EMBEDDING_MODEL)
    build_eval_cmd.add_argument("--semantic-min-score", type=float, default=0.56)
    build_eval_cmd.add_argument("--max-gt-url-share", type=float, default=0.25)
    build_eval_cmd.add_argument("--target-multi-gt-share", type=float, default=0.4)
    build_eval_cmd.add_argument("--keep-max-ids-for-multi", type=int, default=1)
    build_eval_cmd.add_argument("--excerpt-max", type=int, default=320)
    build_eval_cmd.set_defaults(handler=cmd_build_evaluation_dataset)

    audit_cmd = subparsers.add_parser(
        "dataset_audit",
        aliases=["audit-dataset"],
        help="Audit rag/evaluation datasets and print quality report.",
    )
    audit_cmd.add_argument("--rag", default="data/rag_dataset.jsonl")
    audit_cmd.add_argument("--eval", default="data/evaluation_with_evidence.jsonl")
    audit_cmd.add_argument("--out", default=None)
    audit_cmd.set_defaults(handler=cmd_dataset_audit)

    build_reranker_ds_cmd = subparsers.add_parser(
        "build_reranker_dataset",
        aliases=["build-reranker-dataset"],
        help="Build context-aware reranker training JSONL from retrieval report.",
    )
    build_reranker_ds_cmd.add_argument("--eval-report", default="experiments/results/retrieval_report_best.json")
    build_reranker_ds_cmd.add_argument("--rag-dataset", default="data/rag_dataset.jsonl")
    build_reranker_ds_cmd.add_argument("--out", default="artifacts/datasets/reranker_train.jsonl")
    build_reranker_ds_cmd.add_argument("--max-negative-rank", type=int, default=20)
    build_reranker_ds_cmd.add_argument("--max-negatives", type=int, default=16)
    build_reranker_ds_cmd.add_argument("--ranking-cutoff-weight", type=float, default=2.0)
    build_reranker_ds_cmd.add_argument("--true-recall-weight", type=float, default=0.3)
    build_reranker_ds_cmd.add_argument("--default-weight", type=float, default=1.0)
    build_reranker_ds_cmd.set_defaults(handler=cmd_build_reranker_dataset)

    train_reranker_cmd = subparsers.add_parser(
        "train_reranker",
        aliases=["train-reranker"],
        help="Fine-tune cross-encoder reranker on context dataset.",
    )
    train_reranker_cmd.add_argument("--train-jsonl", default="artifacts/datasets/reranker_train.jsonl")
    train_reranker_cmd.add_argument("--rag-dataset", default="data/rag_dataset.jsonl")
    train_reranker_cmd.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    train_reranker_cmd.add_argument("--out-dir", default="artifacts/models/reranker-failure-driven")
    train_reranker_cmd.add_argument("--epochs", type=int, default=2)
    train_reranker_cmd.add_argument("--batch-size", type=int, default=16)
    train_reranker_cmd.add_argument("--warmup-steps", type=int, default=100)
    train_reranker_cmd.add_argument("--val-ratio", type=float, default=0.1)
    train_reranker_cmd.add_argument("--seed", type=int, default=42)
    train_reranker_cmd.set_defaults(handler=cmd_train_reranker)

    experiments_cmd = subparsers.add_parser(
        "run_experiments",
        aliases=["run-experiments"],
        help="Run RAG answer comparison across configured LLM providers.",
    )
    experiments_cmd.add_argument("--question", "-q", required=True)
    experiments_cmd.add_argument("--models", default="openai,gigachat,ollama,qwen")
    experiments_cmd.add_argument("--top-k", type=int, default=5)
    experiments_cmd.add_argument("--max-context-tokens", type=int, default=2500)
    experiments_cmd.add_argument("--faiss-path", default="artifacts/faiss")
    experiments_cmd.add_argument("--index", default="rag_chunks")
    experiments_cmd.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    experiments_cmd.add_argument("--log-path", default="experiments/logs/llm_experiment_results.jsonl")
    experiments_cmd.set_defaults(handler=cmd_run_experiments)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
