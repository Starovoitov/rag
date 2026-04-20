from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    retrieved_doc_ids: list[str]
    relevant_doc_ids: list[str]


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: list[str], k: int) -> float:
    if k <= 0 or not relevant_doc_ids:
        return 0.0
    retrieved_top_k = set(retrieved_doc_ids[:k])
    relevant = set(relevant_doc_ids)
    return len(retrieved_top_k & relevant) / len(relevant)


def precision_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    if not top_k:
        return 0.0
    relevant = set(relevant_doc_ids)
    return sum(1 for doc_id in top_k if doc_id in relevant) / len(top_k)


def hit_rate_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: list[str], k: int) -> float:
    if k <= 0 or not relevant_doc_ids:
        return 0.0
    top_k = set(retrieved_doc_ids[:k])
    return 1.0 if top_k.intersection(relevant_doc_ids) else 0.0


def reciprocal_rank(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    relevant = set(relevant_doc_ids)
    if not relevant:
        return 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mrr(results: list[RetrievalResult]) -> float:
    if not results:
        return 0.0
    return sum(reciprocal_rank(item.retrieved_doc_ids, item.relevant_doc_ids) for item in results) / len(results)


def dcg_at_k(retrieved_doc_ids: list[str], relevance_map: dict[str, float], k: int) -> float:
    if k <= 0:
        return 0.0
    dcg = 0.0
    for idx, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        rel = float(relevance_map.get(doc_id, 0.0))
        if rel <= 0:
            continue
        dcg += (2**rel - 1) / math.log2(idx + 1)
    return dcg


def ndcg_at_k(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: list[str],
    k: int,
    relevance_map: dict[str, float] | None = None,
) -> float:
    if k <= 0 or not relevant_doc_ids:
        return 0.0
    rel_map = relevance_map or {doc_id: 1.0 for doc_id in relevant_doc_ids}
    actual = dcg_at_k(retrieved_doc_ids, rel_map, k)
    ideal_order = sorted(rel_map.items(), key=lambda item: item[1], reverse=True)
    ideal_docs = [doc_id for doc_id, _ in ideal_order]
    ideal = dcg_at_k(ideal_docs, rel_map, k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def evaluate_retrieval(
    results: list[RetrievalResult],
    k_values: list[int],
) -> dict[str, float]:
    if not results:
        metrics: dict[str, float] = {"mrr": 0.0}
        for k in k_values:
            metrics[f"recall@{k}"] = 0.0
            metrics[f"ndcg@{k}"] = 0.0
        return metrics

    aggregations: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for k in k_values:
            aggregations[f"recall@{k}"].append(
                recall_at_k(result.retrieved_doc_ids, result.relevant_doc_ids, k)
            )
            aggregations[f"precision@{k}"].append(
                precision_at_k(result.retrieved_doc_ids, result.relevant_doc_ids, k)
            )
            aggregations[f"hit_rate@{k}"].append(
                hit_rate_at_k(result.retrieved_doc_ids, result.relevant_doc_ids, k)
            )
            aggregations[f"ndcg@{k}"].append(
                ndcg_at_k(result.retrieved_doc_ids, result.relevant_doc_ids, k)
            )

    output: dict[str, float] = {"mrr": mrr(results)}
    for name, values in aggregations.items():
        output[name] = sum(values) / len(values) if values else 0.0
    return output
