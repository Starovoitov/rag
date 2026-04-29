from __future__ import annotations

import sys
import types
import unittest

from evaluation.dataset import EvalSample


class _DummySentenceTransformer:
    def __init__(self, *args, **kwargs) -> None:  # noqa: ARG002
        pass

    def encode(self, *args, **kwargs):  # noqa: ARG002
        return [[0.0, 1.0]]


class _FakeRetriever:
    def search(self, query: str, top_k: int) -> list[str]:
        return [f"{query}-doc-{i}" for i in range(top_k)]


class TestEvaluationRunnerHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._old_st = sys.modules.get("sentence_transformers")
        cls._old_ingestion = sys.modules.get("ingestion.loaders")
        cls._old_bm25 = sys.modules.get("retrieval.bm25")
        cls._old_hybrid = sys.modules.get("retrieval.hybrid")
        cls._old_semantic = sys.modules.get("retrieval.semantic")
        fake = types.ModuleType("sentence_transformers")
        fake.SentenceTransformer = _DummySentenceTransformer
        sys.modules["sentence_transformers"] = fake
        fake_ingestion = types.ModuleType("ingestion.loaders")
        fake_ingestion.load_bm25_documents_from_dataset = lambda *args, **kwargs: []  # noqa: ARG005
        fake_ingestion.load_semantic_documents_from_faiss = lambda *args, **kwargs: []  # noqa: ARG005
        sys.modules["ingestion.loaders"] = fake_ingestion

        fake_bm25 = types.ModuleType("retrieval.bm25")
        fake_bm25.BM25Document = object
        fake_bm25.BM25Index = object
        sys.modules["retrieval.bm25"] = fake_bm25

        fake_hybrid = types.ModuleType("retrieval.hybrid")
        fake_hybrid.hybrid_search = lambda *args, **kwargs: []  # noqa: ARG005
        sys.modules["retrieval.hybrid"] = fake_hybrid

        fake_semantic = types.ModuleType("retrieval.semantic")
        fake_semantic.SemanticDocument = object
        fake_semantic.search_semantic = lambda *args, **kwargs: []  # noqa: ARG005
        sys.modules["retrieval.semantic"] = fake_semantic
        from evaluation import runner as runner_mod

        cls.runner = runner_mod

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._old_st is None:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = cls._old_st
        if cls._old_ingestion is None:
            sys.modules.pop("ingestion.loaders", None)
        else:
            sys.modules["ingestion.loaders"] = cls._old_ingestion
        if cls._old_bm25 is None:
            sys.modules.pop("retrieval.bm25", None)
        else:
            sys.modules["retrieval.bm25"] = cls._old_bm25
        if cls._old_hybrid is None:
            sys.modules.pop("retrieval.hybrid", None)
        else:
            sys.modules["retrieval.hybrid"] = cls._old_hybrid
        if cls._old_semantic is None:
            sys.modules.pop("retrieval.semantic", None)
        else:
            sys.modules["retrieval.semantic"] = cls._old_semantic

    def test_retrieval_cache_key(self) -> None:
        self.assertEqual(self.runner._retrieval_cache_key("q", 3), "q||3")

    def test_parse_k_values_sorted_and_unique(self) -> None:
        self.assertEqual(self.runner.parse_k_values("5,1,5,3"), [1, 3, 5])

    def test_parse_k_values_rejects_non_positive(self) -> None:
        with self.assertRaises(ValueError):
            self.runner.parse_k_values("1,0,2")

    def test_run_benchmark_returns_metrics_and_details(self) -> None:
        samples = [
            EvalSample(query="q1", relevant_docs=["q1-doc-0"]),
            EvalSample(query="q2", relevant_docs=["q2-doc-9"]),
        ]
        metrics, details = self.runner.run_benchmark(samples, _FakeRetriever(), max_k=3)
        self.assertIn("mrr", metrics)
        self.assertIn("recall@3", metrics)
        self.assertEqual(len(details), 2)
        self.assertEqual(details[0].query, "q1")

    def test_run_benchmark_handles_empty_samples(self) -> None:
        metrics, details = self.runner.run_benchmark([], _FakeRetriever(), max_k=2)
        self.assertEqual(metrics["mrr"], 0.0)
        self.assertEqual(details, [])


if __name__ == "__main__":
    unittest.main()
