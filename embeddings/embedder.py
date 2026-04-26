from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from embeddings.faiss_store import save_faiss_index
from utils.embedding_format import format_passage_for_embedding


DEFAULT_EMBEDDING_MODEL = "intfloat/e5-base-v2"


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


def generate_embeddings(
    input_jsonl: str = "data/embeddings_input.jsonl",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
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

    model_inputs = [format_passage_for_embedding(text, model_name) for text in texts]
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


def upsert_embeddings_to_faiss(
    embedding_records: list[dict[str, Any]],
    persist_directory: str = "data/faiss",
    index_name: str = ".",
) -> int:
    """Persist embedding records into a local FAISS index + sidecar store."""
    return save_faiss_index(
        embedding_records=embedding_records,
        persist_directory=persist_directory,
        index_name=index_name,
    )


def build_faiss_index(
    input_jsonl: str = "data/embeddings_input.jsonl",
    persist_directory: str = "data/faiss",
    index_name: str = ".",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> int:
    """
    End-to-end helper: read JSONL, generate embeddings, and persist into FAISS.
    """
    records = generate_embeddings(
        input_jsonl=input_jsonl,
        model_name=model_name,
    )
    return upsert_embeddings_to_faiss(
        embedding_records=records,
        persist_directory=persist_directory,
        index_name=index_name,
    )
