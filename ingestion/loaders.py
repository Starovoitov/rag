from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from embeddings.embedder import generate_embeddings, upsert_embeddings_to_faiss
from embeddings.faiss_store import load_semantic_documents_from_faiss as load_from_faiss_store
from parser.pipeline import run_pipeline
from retrieval.semantic import SemanticDocument


DEFAULT_EMBEDDING_MODEL = "intfloat/e5-base-v2"


def _read_raw_chunks(dataset_path: str) -> Iterator[dict[str, Any]]:
    """Stream raw chunk records from dataset JSONL."""
    with Path(dataset_path).open("r", encoding="utf-8") as dataset:
        for line in dataset:
            item = json.loads(line)
            if item.get("record_type") != "raw_chunk":
                continue
            yield item


def run_parser_and_upsert_to_faiss(
    dataset_path: str = "data/rag_dataset.jsonl",
    persist_directory: str = "artifacts/faiss",
    index_name: str = "rag_chunks",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    min_tokens: int = 300,
    max_tokens: int = 800,
    overlap_ratio: float = 0.15,
) -> dict[str, int]:
    """
    Run parser pipeline and persist chunk embeddings into FAISS.

    Returns parser stats with additional embedding/faiss counters.
    """
    stats = run_pipeline(
        output_path=dataset_path,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_ratio=overlap_ratio,
    )
    embedding_input_path = Path(dataset_path).with_name("embeddings_input.jsonl")
    raw_chunks_count = 0
    with embedding_input_path.open("w", encoding="utf-8") as out:
        for row in _read_raw_chunks(dataset_path):
            payload = {
                "id": row["chunk_id"],
                "text": row["text"],
                "metadata": row.get("metadata", {}),
            }
            out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            raw_chunks_count += 1

    records = generate_embeddings(
        input_jsonl=str(embedding_input_path),
        model_name=model_name,
    )
    upserted = upsert_embeddings_to_faiss(
        embedding_records=records,
        persist_directory=persist_directory,
        index_name=index_name,
    )
    stats["raw_chunks_for_embedding"] = raw_chunks_count
    stats["embeddings_upserted"] = upserted
    return stats


def load_bm25_documents_from_dataset(dataset_path: str = "data/rag_dataset.jsonl") -> list[dict[str, Any]]:
    """Load raw chunk documents from dataset for lexical retrieval."""
    return [
        {
            "id": item["chunk_id"],
            "text": item["text"],
            "metadata": item.get("metadata", {}),
        }
        for item in _read_raw_chunks(dataset_path)
    ]


def load_chunk_texts(rag_dataset_path: str | Path) -> dict[str, str]:
    """
    Load chunk_id -> text map from raw_chunk rows.

    Used by reranker dataset/training utilities.
    """
    dataset_path = str(rag_dataset_path)
    chunk_texts: dict[str, str] = {}
    for item in _read_raw_chunks(dataset_path):
        chunk_id = str(item.get("chunk_id", "")).strip()
        text = str(item.get("text", "")).strip()
        if chunk_id and text:
            chunk_texts[chunk_id] = text
    return chunk_texts


def load_semantic_documents_from_faiss(
    persist_directory: str = "artifacts/faiss",
    index_name: str = "rag_chunks",
) -> list[SemanticDocument]:
    """Load documents and precomputed embeddings from a persisted FAISS store."""
    return load_from_faiss_store(
        persist_directory=persist_directory,
        index_name=index_name,
    )
