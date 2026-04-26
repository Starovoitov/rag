#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from typing import Any

from sentence_transformers import SentenceTransformer

from generation.config import DEFAULT_LLM_CONFIG_PATH
from generation.llm import call_llm
from generation.prompt import SourceChunk, build_rag_messages
from generation.run_rag import build_model_configs
from ingestion.loaders import load_semantic_documents_from_faiss
from retrieval.semantic import search_semantic
from utils.logger import get_json_logger, log_event

DEFAULT_EMBEDDING_MODEL = "intfloat/e5-base-v2"



def _to_source_chunks(items: list[Any]) -> list[SourceChunk]:
    chunks: list[SourceChunk] = []
    for it in items:
        chunks.append(
            SourceChunk(
                doc_id=it.doc_id,
                text=it.text,
                score=getattr(it, "score", None),
                metadata=getattr(it, "metadata", {}),
            )
        )
    return chunks


def run_experiments(
    question: str,
    models: list[str],
    top_k: int,
    max_context_tokens: int,
    faiss_path: str,
    index_name: str,
    embedding_model: str,
    log_path: str,
    llm_config_path: str = DEFAULT_LLM_CONFIG_PATH,
) -> None:
    logger = get_json_logger("experiments.run", log_path)
    semantic_docs = load_semantic_documents_from_faiss(
        persist_directory=faiss_path,
        index_name=index_name,
    )
    if not semantic_docs:
        raise ValueError(
            f"No semantic docs in FAISS index '{index_name}' at '{faiss_path}'. "
            "Run ingestion first."
        )

    embedder = SentenceTransformer(embedding_model)
    query_embedding = embedder.encode(
        [f"query: {question}"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()

    semantic_hits = search_semantic(query_embedding, semantic_docs, top_k=max(top_k * 2, 10))
    source_chunks = _to_source_chunks(semantic_hits)
    prompt_data = build_rag_messages(
        question=question,
        chunks=source_chunks,
        top_k=top_k,
        max_context_tokens=max_context_tokens,
    )

    configs = build_model_configs(config_path=llm_config_path)
    for model_key in models:
        conf = configs.get(model_key)
        if conf is None:
            print(f"[{model_key}] skipped: unknown model key")
            continue
        start = time.perf_counter()
        try:
            answer = call_llm(
                system_prompt=prompt_data["system_prompt"],
                user_prompt=prompt_data["user_prompt"],
                config=conf,
            )
            print(f"\n=== {model_key} ({conf.model}) ===")
            print(answer)
            log_event(
                logger,
                {
                    "question": question,
                    "model_key": model_key,
                    "provider_model": conf.model,
                    "answer": answer,
                    "context_tokens_estimate": prompt_data["context_tokens_estimate"],
                    "used_sources": [chunk.doc_id for chunk in prompt_data["used_chunks"]],
                    "elapsed_ms": int((time.perf_counter() - start) * 1000),
                },
            )
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            print(f"\n=== {model_key} ({conf.model}) FAILED ===")
            print(err)
            log_event(
                logger,
                {
                    "question": question,
                    "model_key": model_key,
                    "provider_model": conf.model,
                    "error": err,
                    "context_tokens_estimate": prompt_data["context_tokens_estimate"],
                    "elapsed_ms": int((time.perf_counter() - start) * 1000),
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG answer comparison across LLM providers.")
    parser.add_argument("--question", "-q", required=True, help="Question for the RAG system.")
    parser.add_argument(
        "--models",
        default="openai,gigachat,ollama,qwen",
        help="Comma-separated model keys to run: openai,gigachat,ollama",
    )
    parser.add_argument("--top-k", type=int, default=5, help="How many retrieved docs to include.")
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=2500,
        help="Approx context token budget before truncation.",
    )
    parser.add_argument("--faiss-path", default="data/faiss", help="FAISS persist directory.")
    parser.add_argument("--index", default=".", help="FAISS index name.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--log-path", default="experiments/logs/llm_experiment_results.jsonl")
    parser.add_argument("--llm-config-path", default=DEFAULT_LLM_CONFIG_PATH)

    args = parser.parse_args()
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    run_experiments(
        question=args.question,
        models=models,
        top_k=args.top_k,
        max_context_tokens=args.max_context_tokens,
        faiss_path=args.faiss_path,
        index_name=args.index,
        embedding_model=args.embedding_model,
        log_path=args.log_path,
        llm_config_path=args.llm_config_path,
    )

