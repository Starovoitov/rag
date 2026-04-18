#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from sentence_transformers import SentenceTransformer

from generation.llm import LLMConfig, call_llm, stream_llm
from generation.prompt import SourceChunk, build_rag_messages
from ingestion.loaders import load_semantic_documents_from_faiss
from retrieval.semantic import search_semantic

DEFAULT_EMBEDDING_MODEL = "intfloat/e5-small-v2"


def _config_for_provider(provider: str, model: str | None) -> LLMConfig:
    provider = provider.lower()
    if provider == "openai":
        return LLMConfig(
            provider="openai",
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/chat/completions"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    if provider == "gigachat":
        return LLMConfig(
            provider="gigachat",
            model=model or os.getenv("GIGACHAT_MODEL", "GigaChat-Pro"),
            api_base=os.getenv("GIGACHAT_API_BASE", "https://api.gigachat.ru/v1/chat/completions"),
            api_key=os.getenv("GIGACHAT_API_KEY"),
        )
    if provider == "ollama":
        return LLMConfig(
            provider="ollama",
            model=model or os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1/chat/completions"),
            api_key=os.getenv("OLLAMA_API_KEY"),
        )
    raise ValueError(f"Unsupported provider: {provider}")


def run_rag(
    question: str,
    provider: str,
    model: str | None,
    top_k: int,
    max_context_tokens: int,
    faiss_path: str,
    index_name: str,
    embedding_model: str,
    stream: bool,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
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

    hits = search_semantic(query_embedding, semantic_docs, top_k=max(top_k * 2, 10))
    chunks = [
        SourceChunk(
            doc_id=item.doc_id,
            text=item.text,
            score=item.score,
            metadata=item.metadata,
        )
        for item in hits
    ]
    prompt_data = build_rag_messages(
        question=question,
        chunks=chunks,
        top_k=top_k,
        max_context_tokens=max_context_tokens,
    )

    conf = _config_for_provider(provider=provider, model=model)
    conf.max_tokens = max_tokens
    conf.temperature = temperature
    conf.top_p = top_p
    conf.enable_streaming = stream

    print("Used sources:")
    for idx, chunk in enumerate(prompt_data["used_chunks"], start=1):
        print(f"[{idx}] {chunk.doc_id}")

    print("\nAnswer:")
    if stream:
        for token in stream_llm(
            system_prompt=prompt_data["system_prompt"],
            user_prompt=prompt_data["user_prompt"],
            config=conf,
        ):
            print(token, end="", flush=True)
        print()
        return

    answer = call_llm(
        system_prompt=prompt_data["system_prompt"],
        user_prompt=prompt_data["user_prompt"],
        config=conf,
    )
    print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single RAG query against one LLM provider.")
    parser.add_argument("--question", "-q", required=True, help="Question to ask.")
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "gigachat", "ollama"],
        help="LLM provider config to use.",
    )
    parser.add_argument("--model", default=None, help="Override provider default model.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-tokens", type=int, default=2500)
    parser.add_argument("--faiss-path", default="data/faiss")
    parser.add_argument("--index", default="rag_chunks")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--stream", action="store_true", help="Stream answer tokens.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    run_rag(
        question=args.question,
        provider=args.provider,
        model=args.model,
        top_k=args.top_k,
        max_context_tokens=args.max_context_tokens,
        faiss_path=args.faiss_path,
        index_name=args.index,
        embedding_model=args.embedding_model,
        stream=args.stream,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
