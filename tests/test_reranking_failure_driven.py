from __future__ import annotations

import unittest
from dataclasses import dataclass

from reranking.failure_driven import (
    build_reranker_training_contexts_from_failures,
    build_stratified_rerank_pool,
    classify_failure,
    cosine_similarity,
    interleave_doc_ids,
    mmr_select_candidates,
    prefilter_rerank_candidates,
    rrf_fuse_doc_ids,
    source_miss_type,
)


class TestFailureDrivenHelpers(unittest.TestCase):
    def test_rrf_fuse_doc_ids_merges_ranks(self) -> None:
        fused = rrf_fuse_doc_ids([["a", "b"], ["b", "c"]], top_k=3, rrf_k=1)
        self.assertEqual(fused[0], "b")
        self.assertEqual(set(fused), {"a", "b", "c"})

    def test_prefilter_rerank_candidates_uses_query_overlap(self) -> None:
        candidates = [
            _Candidate(doc_id="a", text="alpha beta"),
            _Candidate(doc_id="b", text="gamma"),
            _Candidate(doc_id="c", text="alpha alpha"),
        ]
        filtered = prefilter_rerank_candidates("alpha", candidates, keep_top_n=2)
        self.assertEqual([item.doc_id for item in filtered], ["a", "c"])

    def test_interleave_doc_ids_preserves_uniqueness(self) -> None:
        merged = interleave_doc_ids(["a", "b", "c"], ["b", "d"], limit=4)
        self.assertEqual(merged, ["a", "b", "d", "c"])

    def test_build_stratified_rerank_pool_balances_sources(self) -> None:
        pool = build_stratified_rerank_pool(
            hybrid_doc_ids=["h1", "h2", "s1"],
            semantic_doc_ids=["s1", "s2"],
            bm25_doc_ids=["b1", "b2"],
            limit=4,
        )
        self.assertEqual(pool, ["s1", "b1", "s2", "b2"])

    def test_cosine_similarity_handles_orthogonal_vectors(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, places=6)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0, places=6)

    def test_mmr_select_candidates_keeps_diversity(self) -> None:
        selected = mmr_select_candidates(
            candidate_doc_ids=["a", "b", "c"],
            query_embedding=[1.0, 0.0],
            doc_embeddings={
                "a": [1.0, 0.0],
                "b": [0.99, 0.01],
                "c": [0.0, 1.0],
            },
            lambda_=0.7,
            max_k=2,
            diversity_threshold=0.95,
        )
        self.assertEqual(selected, ["a", "c"])

    def test_source_miss_type_detects_embedding_miss(self) -> None:
        miss = source_miss_type(
            relevant_doc_ids=["g1"],
            semantic_doc_ids=["x1"],
            bm25_doc_ids=["g1", "x2"],
        )
        self.assertEqual(miss, "embedding_miss")

    def test_classify_failure_returns_ranking_cutoff(self) -> None:
        result = classify_failure(
            query="retrieval cache ttl",
            gt_doc_ids=["g1"],
            top_k_doc_ids=["d1", "d2"],
            all_ranked_doc_ids=["d1", "d2", "g1", "d3"],
            doc_text_map={
                "g1": "retrieval cache ttl configuration",
                "d1": "other content",
                "d2": "different content",
            },
            near_miss_threshold=0.95,
            top_k=2,
        )
        self.assertEqual(result.bucket, "ranking_cutoff_failure")
        self.assertEqual(result.reasons["gt_first_rank"], 3)

    def test_build_reranker_training_contexts_from_failures(self) -> None:
        result = build_reranker_training_contexts_from_failures(
            failure_records=[
                {
                    "query": "what is cache ttl",
                    "bucket": "ranking_cutoff_failure",
                    "source_miss_type": "both_hit",
                    "relevant_doc_ids": ["p1"],
                    "retrieved_top_k_doc_ids": ["n1", "n2"],
                    "retrieved_full_doc_ids": ["n1", "n2", "n3"],
                    "bm25_branch_doc_ids": ["n2", "n3"],
                }
            ],
            doc_text_map={"p1": "pos", "n1": "neg1", "n2": "neg2", "n3": "neg3"},
            max_negative_rank=3,
            max_negatives=2,
            ranking_cutoff_weight=2.0,
            true_recall_weight=1.5,
            default_weight=1.0,
        )
        contexts = result.contexts
        stats = result.stats
        self.assertEqual(len(contexts), 1)
        row = contexts[0]
        self.assertEqual(row.schema_version, "reranker_context_v1")
        self.assertEqual(row.positives, ["p1"])
        self.assertEqual(row.negatives, ["n1", "n2"])
        self.assertEqual(stats.contexts_written, 1)


@dataclass
class _Candidate:
    doc_id: str
    text: str
