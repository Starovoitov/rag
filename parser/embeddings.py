from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


def prepare_embedding_input(
    input_jsonl: str = "data/rag_dataset.jsonl",
    output_jsonl: str = "data/embeddings_input.jsonl",
) -> int:
    """Export raw chunk records into a compact embeddings input JSONL file."""
    src = Path(input_jsonl)
    dst = Path(output_jsonl)
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            item: dict[str, Any] = json.loads(line)
            if item.get("record_type") != "raw_chunk":
                continue
            payload = {
                "id": item["chunk_id"],
                "text": item["text"],
                "metadata": item.get("metadata", {}),
            }
            f_out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
    return written


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """
    Convert metadata values to Chroma-safe scalar types.

    Chroma metadata supports only string, number, and boolean values.
    """
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif value is None:
            sanitized[key] = "null"
        else:
            sanitized[key] = json.dumps(value, ensure_ascii=False)
    return sanitized


def generate_embeddings(
    input_jsonl: str = "data/embeddings_input.jsonl",
    model_name: str = "intfloat/e5-small-v2",
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    """
    Load embedding input JSONL and attach vectors from sentence-transformers.

    Returns a list of dicts containing id, text, metadata, and embedding.
    """
    src = Path(input_jsonl)
    model = SentenceTransformer(model_name)

    records: list[dict[str, Any]] = []
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []

    with src.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            item: dict[str, Any] = json.loads(line)
            ids.append(item["id"])
            texts.append(item["text"])
            metadatas.append(item.get("metadata", {}))

    # E5 models expect task prefixes; use "passage:" for chunk documents.
    model_inputs = [f"passage: {text}" for text in texts]
    vectors = model.encode(
        model_inputs,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for idx, vector in enumerate(vectors):
        records.append(
            {
                "id": ids[idx],
                "text": texts[idx],
                "metadata": metadatas[idx],
                "embedding": vector.tolist(),
            }
        )
    return records


def upsert_embeddings_to_chroma(
    embedding_records: list[dict[str, Any]],
    persist_directory: str = "data/chroma",
    collection_name: str = "rag_chunks",
    batch_size: int = 256,
) -> int:
    """Write embedding records into a local persistent Chroma collection."""
    dst = Path(persist_directory)
    dst.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(dst))
    collection = client.get_or_create_collection(name=collection_name)

    total = len(embedding_records)
    for start in range(0, total, batch_size):
        chunk = embedding_records[start : start + batch_size]
        collection.upsert(
            ids=[item["id"] for item in chunk],
            documents=[item["text"] for item in chunk],
            metadatas=[_sanitize_metadata(item.get("metadata", {})) for item in chunk],
            embeddings=[item["embedding"] for item in chunk],
        )
    return total


def build_chroma_collection(
    input_jsonl: str = "data/embeddings_input.jsonl",
    persist_directory: str = "data/chroma",
    collection_name: str = "rag_chunks",
    model_name: str = "intfloat/e5-small-v2",
) -> int:
    """
    End-to-end helper: read JSONL, generate embeddings, and upsert into Chroma.
    """
    records = generate_embeddings(
        input_jsonl=input_jsonl,
        model_name=model_name,
    )
    return upsert_embeddings_to_chroma(
        embedding_records=records,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

