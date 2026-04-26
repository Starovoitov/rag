from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from retrieval.semantic import SemanticDocument


INDEX_FILENAME = "vectors.index"
STORE_FILENAME = "store.json"


def _maybe_migrate_legacy_index_dir(persist_directory: str, index_name: str) -> Path:
    """
    Move legacy root-level index folder (e.g. ./rag_chunks) into persist_directory/index_name.

    Migration runs only when destination does not already exist.
    """
    target_root = Path(persist_directory) / index_name
    if target_root.exists():
        return target_root

    # Never treat cwd/parent/special values as a legacy index directory name.
    normalized_name = index_name.strip()
    if normalized_name in {"", ".", ".."}:
        return target_root
    if Path(normalized_name).name != normalized_name:
        return target_root

    legacy_root = Path(index_name)
    if not legacy_root.is_dir():
        return target_root

    # Avoid pathological self-moves when paths already point to the same location.
    try:
        if legacy_root.resolve() == target_root.resolve():
            return target_root
    except FileNotFoundError:
        return target_root

    target_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(legacy_root), str(target_root))
    return target_root


def _persist_paths(persist_directory: str, index_name: str) -> Path:
    root = _maybe_migrate_legacy_index_dir(persist_directory, index_name)
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_faiss_index(
    embedding_records: list[dict[str, Any]],
    persist_directory: str = "data/faiss",
    index_name: str = ".",
) -> int:
    """
    Build a FAISS IndexFlatIP index from normalized embedding vectors and persist.

    Vectors must be L2-normalized so inner product equals cosine similarity.
    Row i in FAISS matches store.json entries at index i.
    """
    root = _persist_paths(persist_directory, index_name)
    index_path = root / INDEX_FILENAME
    store_path = root / STORE_FILENAME

    if not embedding_records:
        store = {"ids": [], "texts": [], "metadatas": []}
        store_path.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
        if index_path.exists():
            index_path.unlink()
        return 0

    dim = len(embedding_records[0]["embedding"])
    vectors = np.array(
        [r["embedding"] for r in embedding_records],
        dtype=np.float32,
    )
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(index_path))

    store = {
        "ids": [r["id"] for r in embedding_records],
        "texts": [r["text"] for r in embedding_records],
        "metadatas": [r.get("metadata", {}) for r in embedding_records],
    }
    store_path.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
    return len(embedding_records)


def load_semantic_documents_from_faiss(
    persist_directory: str = "data/faiss",
    index_name: str = ".",
) -> list[SemanticDocument]:
    """Load vectors and parallel text/metadata from a persisted FAISS index."""
    root = _maybe_migrate_legacy_index_dir(persist_directory, index_name)
    index_path = root / INDEX_FILENAME
    store_path = root / STORE_FILENAME

    if not store_path.is_file():
        return []

    store = json.loads(store_path.read_text(encoding="utf-8"))
    ids: list[str] = store.get("ids", [])
    texts: list[str] = store.get("texts", [])
    metadatas: list[Any] = store.get("metadatas", [])

    embeddings_flat: list[list[float]] = []
    if index_path.is_file():
        index = faiss.read_index(str(index_path))
        n = int(index.ntotal)
        if n > 0:
            full = index.reconstruct_n(0, n)
            embeddings_flat = full.tolist()

    results: list[SemanticDocument] = []
    for i, doc_id in enumerate(ids):
        text = texts[i] if i < len(texts) else ""
        embedding = embeddings_flat[i] if i < len(embeddings_flat) else []
        metadata = metadatas[i] if i < len(metadatas) else {}
        if text == "" or len(embedding) == 0:
            continue
        md = metadata if isinstance(metadata, dict) else {}
        results.append(
            SemanticDocument(
                doc_id=doc_id,
                text=text,
                embedding=list(embedding),
                metadata=md,
            )
        )
    return results
