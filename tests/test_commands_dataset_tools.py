from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path

from commands.dataset_audit import _quality_score, audit, top_share

module_path = Path(__file__).resolve().parents[1] / "commands" / "build_reranker_dataset.py"
spec = importlib.util.spec_from_file_location("test_build_reranker_dataset_module", module_path)
assert spec is not None and spec.loader is not None
_mod = importlib.util.module_from_spec(spec)
fake_ingestion = types.ModuleType("ingestion")
fake_loaders = types.ModuleType("ingestion.loaders")
fake_loaders.load_chunk_texts = lambda *args, **kwargs: {}  # noqa: ARG005
sys.modules.setdefault("ingestion", fake_ingestion)
sys.modules["ingestion.loaders"] = fake_loaders
spec.loader.exec_module(_mod)
_build_negative_pool = _mod._build_negative_pool
_collect_negative_ids = _mod._collect_negative_ids
_sample_weight_for_bucket = _mod._sample_weight_for_bucket


class TestCommandsDatasetTools(unittest.TestCase):
    def test_sample_weight_for_bucket(self) -> None:
        self.assertEqual(
            _sample_weight_for_bucket(
                "ranking_cutoff_failure",
                ranking_cutoff_weight=3.0,
                true_recall_weight=2.0,
                default_weight=1.0,
            ),
            3.0,
        )
        self.assertEqual(
            _sample_weight_for_bucket(
                "true_recall_failure",
                ranking_cutoff_weight=3.0,
                true_recall_weight=2.0,
                default_weight=1.0,
            ),
            2.0,
        )
        self.assertEqual(
            _sample_weight_for_bucket(
                "other",
                ranking_cutoff_weight=3.0,
                true_recall_weight=2.0,
                default_weight=1.0,
            ),
            1.0,
        )

    def test_build_negative_pool_true_recall_prefers_bm25(self) -> None:
        pool = _build_negative_pool(
            bucket="true_recall_failure",
            retrieved=["a", "b"],
            retrieved_full=["c", "d"],
            bm25_branch=["x", "y", "z"],
            max_negative_rank=2,
        )
        self.assertEqual(pool, ["x", "y"])

    def test_collect_negative_ids_ignores_positives_and_unknowns(self) -> None:
        stats = {"missing_negative_text": 0}
        negative_ids, negative_weights = _collect_negative_ids(
            negative_pool=["p1", "n1", "n_missing", "n2"],
            retrieved_full=["n3"],
            positive_ids=["p1"],
            source_miss_type="none",
            chunk_texts={"p1": "pos", "n1": "neg-1", "n2": "neg-2", "n3": "neg-3"},
            sample_weight=2.0,
            max_negatives=2,
            stats=stats,
        )
        self.assertEqual(negative_ids, ["n1", "n2"])
        self.assertEqual(set(negative_weights.keys()), {"n1", "n2"})
        self.assertEqual(stats["missing_negative_text"], 1)

    def test_top_share_handles_empty_counter(self) -> None:
        self.assertEqual(top_share({}, 1), 0.0)

    def test_quality_score_is_clamped(self) -> None:
        score = _quality_score(
            rows_total=10,
            queries_with_no_gt=0,
            gt_url_counter=Counter({"u1": 10}),
            gt_chunk_counter=Counter({"c1": 10}),
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_audit_smoke_with_minimal_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rag = root / "rag.jsonl"
            evalf = root / "eval.jsonl"
            rag.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "record_type": "raw_chunk",
                                "chunk_id": "c1",
                                "text": "chunk text",
                                "token_count": 10,
                                "overlap_tokens": 2,
                                "metadata": {"category": "docs", "url": "https://x"},
                            }
                        ),
                        json.dumps(
                            {
                                "record_type": "raw_chunk",
                                "chunk_id": "c2",
                                "text": "chunk text 2",
                                "token_count": 12,
                                "overlap_tokens": 2,
                                "metadata": {"category": "docs", "url": "https://x"},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            evalf.write_text(
                json.dumps(
                    {
                        "expected_evidence": {"chunk_ids": ["c1"], "resolution_method": "manual"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = audit(rag, evalf)
            self.assertEqual(report["rag"]["raw_chunks"], 2)
            self.assertEqual(report["evaluation"]["rows_total"], 1)
            self.assertIn("quality_score", report)


if __name__ == "__main__":
    unittest.main()
