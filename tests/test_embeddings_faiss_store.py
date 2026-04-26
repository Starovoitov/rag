from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


class _FakeArray:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = values


class _FakeMatrix:
    def __init__(self, values: list[list[float]]) -> None:
        self._values = values

    def tolist(self) -> list[list[float]]:
        return self._values


class _FakeIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.vectors: list[list[float]] = []
        self.ntotal = 0

    def add(self, vectors: _FakeArray) -> None:
        self.vectors.extend(vectors.values)
        self.ntotal = len(self.vectors)

    def reconstruct_n(self, start: int, n: int) -> _FakeMatrix:
        return _FakeMatrix(self.vectors[start : start + n])


class TestFaissStore(unittest.TestCase):
    def setUp(self) -> None:
        self._old_np = sys.modules.get("numpy")
        self._old_faiss = sys.modules.get("faiss")

        np_mod = types.ModuleType("numpy")
        np_mod.float32 = float
        np_mod.array = lambda values, dtype=None: _FakeArray(values)  # noqa: ARG005
        sys.modules["numpy"] = np_mod

        self._saved_index: _FakeIndex | None = None
        faiss_mod = types.ModuleType("faiss")

        def _write_index(index: _FakeIndex, path: str) -> None:
            self._saved_index = index
            Path(path).write_text("fake-index", encoding="utf-8")

        def _read_index(path: str) -> _FakeIndex:  # noqa: ARG001
            return self._saved_index or _FakeIndex(0)

        faiss_mod.IndexFlatIP = _FakeIndex
        faiss_mod.write_index = _write_index
        faiss_mod.read_index = _read_index
        sys.modules["faiss"] = faiss_mod

        module_path = Path(__file__).resolve().parents[1] / "embeddings" / "faiss_store.py"
        spec = importlib.util.spec_from_file_location("test_faiss_store_module", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.faiss_store = module

    def tearDown(self) -> None:
        if self._old_np is None:
            sys.modules.pop("numpy", None)
        else:
            sys.modules["numpy"] = self._old_np
        if self._old_faiss is None:
            sys.modules.pop("faiss", None)
        else:
            sys.modules["faiss"] = self._old_faiss

    def test_save_empty_records(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            count = self.faiss_store.save_faiss_index([], persist_directory=td, index_name="idx")
            self.assertEqual(count, 0)
            store_path = Path(td) / "idx" / "store.json"
            payload = json.loads(store_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["ids"], [])

    def test_save_and_load_roundtrip(self) -> None:
        records = [
            {"id": "d1", "text": "doc1", "embedding": [0.1, 0.2], "metadata": {"src": "a"}},
            {"id": "d2", "text": "doc2", "embedding": [0.3, 0.4], "metadata": {"src": "b"}},
        ]
        with tempfile.TemporaryDirectory() as td:
            saved = self.faiss_store.save_faiss_index(records, persist_directory=td, index_name="idx")
            self.assertEqual(saved, 2)
            docs = self.faiss_store.load_semantic_documents_from_faiss(persist_directory=td, index_name="idx")
            self.assertEqual(len(docs), 2)
            self.assertEqual(docs[0].doc_id, "d1")
            self.assertEqual(docs[1].metadata["src"], "b")

    def test_load_returns_empty_when_store_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            docs = self.faiss_store.load_semantic_documents_from_faiss(persist_directory=td, index_name="missing")
            self.assertEqual(docs, [])

    def test_load_skips_docs_without_text_or_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "idx"
            root.mkdir(parents=True)
            (root / "store.json").write_text(
                json.dumps({"ids": ["a", "b"], "texts": ["", "doc"], "metadatas": [{}, {}]}),
                encoding="utf-8",
            )
            (root / "vectors.index").write_text("fake-index", encoding="utf-8")
            self._saved_index = _FakeIndex(2)
            self._saved_index.vectors = [[0.1, 0.2], [0.3, 0.4]]
            self._saved_index.ntotal = 2
            docs = self.faiss_store.load_semantic_documents_from_faiss(persist_directory=td, index_name="idx")
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].doc_id, "b")

    def test_load_migrates_legacy_root_index_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            old_cwd = Path.cwd()
            legacy_root = workspace / "rag_chunks"
            legacy_root.mkdir(parents=True)
            (legacy_root / "store.json").write_text(
                json.dumps({"ids": ["d1"], "texts": ["doc1"], "metadatas": [{}]}),
                encoding="utf-8",
            )
            (legacy_root / "vectors.index").write_text("fake-index", encoding="utf-8")
            self._saved_index = _FakeIndex(2)
            self._saved_index.vectors = [[0.1, 0.2]]
            self._saved_index.ntotal = 1

            try:
                # Migration checks relative legacy path ("./rag_chunks"), so run in temp workspace.
                os.chdir(workspace)
                docs = self.faiss_store.load_semantic_documents_from_faiss(
                    persist_directory=str(workspace / "data" / "faiss"),
                    index_name="rag_chunks",
                )
            finally:
                os.chdir(old_cwd)

            self.assertEqual(len(docs), 1)
            self.assertFalse((workspace / "rag_chunks").exists())
            self.assertTrue((workspace / "data" / "faiss" / "rag_chunks" / "store.json").exists())

    def test_save_does_not_attempt_migration_for_dot_index_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            old_cwd = Path.cwd()
            try:
                os.chdir(workspace)
                saved = self.faiss_store.save_faiss_index(
                    [
                        {
                            "id": "d1",
                            "text": "doc1",
                            "embedding": [0.1, 0.2],
                            "metadata": {},
                        }
                    ],
                    persist_directory="data/faiss",
                    index_name=".",
                )
            finally:
                os.chdir(old_cwd)

            self.assertEqual(saved, 1)
            self.assertTrue((workspace / "data" / "faiss" / "vectors.index").exists())
            self.assertTrue((workspace / "data" / "faiss" / "store.json").exists())


if __name__ == "__main__":
    unittest.main()

