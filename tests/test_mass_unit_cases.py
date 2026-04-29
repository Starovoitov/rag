from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from utils.common import min_max_normalize, rank_weight, tokenize

module_path = Path(__file__).resolve().parents[1] / "generation" / "prompt.py"
spec = importlib.util.spec_from_file_location("test_mass_generation_prompt_module", module_path)
assert spec is not None and spec.loader is not None
_prompt_mod = importlib.util.module_from_spec(spec)
sys.modules["test_mass_generation_prompt_module"] = _prompt_mod
spec.loader.exec_module(_prompt_mod)
SourceChunk = _prompt_mod.SourceChunk
estimate_tokens = _prompt_mod.estimate_tokens
merge_top_k_documents = _prompt_mod.merge_top_k_documents
format_context_with_citations = _prompt_mod.format_context_with_citations
build_rag_messages = _prompt_mod.build_rag_messages


def _add_test(cls: type[unittest.TestCase], name: str, fn) -> None:
    setattr(cls, name, fn)


class TestUtilsCommonMass(unittest.TestCase):
    pass


class TestGenerationPromptMass(unittest.TestCase):
    pass


# 5 tests: rank_weight boundary coverage across broad integer range.
for rank in [-10, 0, 5, 10, 40]:
    expected = 1.0 if rank <= 5 else 0.7 if rank <= 15 else 0.4

    def _rank_test(self, rank=rank, expected=expected):
        self.assertEqual(rank_weight(rank), expected)

    _add_test(TestUtilsCommonMass, f"test_rank_weight_case_{rank + 10:03d}", _rank_test)


# 5 tests: token estimate behavior for incremental lengths.
for text_len in [1, 4, 9, 16, 40]:
    expected = max(1, text_len // 4)

    def _estimate_test(self, text_len=text_len, expected=expected):
        self.assertEqual(estimate_tokens("x" * text_len), expected)

    _add_test(TestGenerationPromptMass, f"test_estimate_tokens_len_{text_len:03d}", _estimate_test)


# 5 tests: min-max normalization keeps values in [0, 1] and stable ordering.
for offset in range(5):
    values = {
        "a": float(offset),
        "b": float(offset + 2),
        "c": float(offset + 4),
    }

    def _normalize_test(self, values=values):
        out = min_max_normalize(values)
        self.assertAlmostEqual(out["a"], 0.0)
        self.assertAlmostEqual(out["b"], 0.5)
        self.assertAlmostEqual(out["c"], 1.0)

    _add_test(TestUtilsCommonMass, f"test_min_max_normalize_case_{offset:03d}", _normalize_test)


# 20 tests: tokenize punctuation retention in default mode and lowercasing in BM25 mode.
for idx in range(20):
    text = f"Token-{idx}, VALUE {idx}!"
    expected_default = ["Token", "-", str(idx), ",", "VALUE", str(idx), "!"]
    expected_bm25 = ["token", str(idx), "value", str(idx)]

    def _tokenize_test(self, text=text, expected_default=expected_default, expected_bm25=expected_bm25):
        self.assertEqual(tokenize(text), expected_default)
        self.assertEqual(tokenize(text, for_bm25=True), expected_bm25)

    _add_test(TestUtilsCommonMass, f"test_tokenize_case_{idx:03d}", _tokenize_test)


# 20 tests: merge_top_k_documents score ordering and top-k trimming.
for top_k in range(1, 21):
    chunks = [
        SourceChunk(doc_id="low", text="L", score=0.1),
        SourceChunk(doc_id="high", text="H", score=0.9),
        SourceChunk(doc_id="mid", text="M", score=0.5),
    ]
    expected = ["high", "mid", "low"][:top_k]

    def _merge_test(self, top_k=top_k, chunks=chunks, expected=expected):
        out = merge_top_k_documents(chunks, top_k=top_k)
        self.assertEqual([chunk.doc_id for chunk in out], expected)

    _add_test(TestGenerationPromptMass, f"test_merge_top_k_case_{top_k:03d}", _merge_test)


# 5 tests: format_context_with_citations skips blank chunks.
for idx in range(5):
    chunks = [
        SourceChunk(doc_id="empty", text="   ", score=0.9),
        SourceChunk(doc_id=f"valid-{idx}", text=f"Valid text {idx}", score=0.5),
    ]

    def _context_skip_blank_test(self, chunks=chunks):
        context, used = format_context_with_citations(chunks, max_context_tokens=200)
        self.assertEqual(len(used), 1)
        self.assertIn("Valid text", context)
        self.assertNotIn("empty", context)

    _add_test(TestGenerationPromptMass, f"test_context_skips_blank_case_{idx:03d}", _context_skip_blank_test)


# 5 tests: build_rag_messages includes guardrail and uses requested question text.
for idx in range(5):
    question = f"What is item {idx}?"
    chunks = [SourceChunk(doc_id=f"d{idx}", text=f"Answer fragment {idx}", score=1.0)]

    def _build_messages_test(self, question=question, chunks=chunks):
        payload = build_rag_messages(question=question, chunks=chunks, top_k=1, max_context_tokens=200)
        self.assertIn("Additional guardrail", payload["system_prompt"])
        self.assertIn(question, payload["user_prompt"])
        self.assertEqual(len(payload["used_chunks"]), 1)

    _add_test(TestGenerationPromptMass, f"test_build_messages_case_{idx:03d}", _build_messages_test)


# 5 tests: min-max normalization equal-values fallback to 0.5.
for idx in range(5):
    base = float(idx * 3 + 1)
    values = {"x": base, "y": base, "z": base}

    def _normalize_equal_values_test(self, values=values):
        out = min_max_normalize(values)
        self.assertEqual(out, {"x": 0.5, "y": 0.5, "z": 0.5})

    _add_test(
        TestUtilsCommonMass,
        f"test_min_max_equal_values_case_{idx:03d}",
        _normalize_equal_values_test,
    )


if __name__ == "__main__":
    unittest.main()
