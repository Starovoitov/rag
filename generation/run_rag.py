#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from sentence_transformers import SentenceTransformer

from generation.llm import LLMConfig, call_llm, stream_llm
from generation.prompt import SourceChunk, build_rag_messages
from ingestion.loaders import load_semantic_documents_from_faiss
from retrieval.semantic import search_semantic
from utils.embedding_format import format_query_for_embedding

DEFAULT_EMBEDDING_MODEL = "intfloat/e5-base-v2"

KNOWN_LLM_PROVIDERS: tuple[str, ...] = ("openai", "gigachat", "ollama", "qwen")


def _guess_embedding_models_by_dim(dim: int) -> str:
    known_dims = {
        384: "intfloat/e5-small-v2",
        768: "intfloat/e5-base-v2",
        1024: "intfloat/e5-large-v2",
    }
    return known_dims.get(dim, "unknown")


def get_llm_config(provider: str, model: str | None = None) -> LLMConfig:
    """
    Single source of truth for OpenAI-compatible LLM endpoints (incl. GigaChat, Ollama gateway, Qwen).

    `model` overrides the env default for that provider when set.
    """
    key = provider.lower()
    if key == "openai":
        return LLMConfig(
            provider="openai",
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/chat/completions"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    if key == "gigachat":
        return LLMConfig(
            provider="gigachat",
            model=model or os.getenv("GIGACHAT_MODEL", "GigaChat-Pro"),
            api_base=os.getenv("GIGACHAT_API_BASE", "https://api.gigachat.ru/v1/chat/completions"),
            api_key=os.getenv("GIGACHAT_API_KEY"),
        )
    if key == "ollama":
        return LLMConfig(
            provider="openai",
            model=model or "Lexi-Llama-3-8B-Uncensored_Q4_K_M",
            api_base=os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:1337/v1/chat/completions"),
            api_key=f"{os.getenv("OLLAMA_API_KEY")}",
        )
    if key == "qwen":
        return LLMConfig(
            provider="openai",
            model=model or os.getenv("QWEN_MODEL", "qwen-plus"),
            api_base=os.getenv(
                "QWEN_API_BASE",
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
            ),
            api_key=f"{os.getenv("QWEN_API_KEY")}",
        )
    raise ValueError(f"Unsupported provider: {provider}")


def build_model_configs() -> dict[str, LLMConfig]:
    """Named provider configs for multi-model experiments (env-driven defaults)."""
    return {name: get_llm_config(name) for name in KNOWN_LLM_PROVIDERS}



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
    rerank: bool = False,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerank_candidates: int = 20,
    llm_cache_enabled: bool = False,
    llm_cache_capacity: int = 512,
    llm_cache_ttl_seconds: float = 300.0,
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
        [format_query_for_embedding(question, embedding_model)],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()
    query_dim = len(query_embedding)
    doc_dim = len(semantic_docs[0].embedding) if semantic_docs else 0
    if doc_dim and query_dim != doc_dim:
        suggested_doc_model = _guess_embedding_models_by_dim(doc_dim)
        suggested_query_model = _guess_embedding_models_by_dim(query_dim)
        raise ValueError(
            "Embedding dimension mismatch between query and indexed documents: "
            f"query_dim={query_dim} (model='{embedding_model}', likely '{suggested_query_model}'), "
            f"doc_dim={doc_dim} (likely '{suggested_doc_model}'). "
            "Use `--embedding-model` that matches the model used during index build, "
            "or rebuild the FAISS index with the selected embedding model."
        )

    candidate_k = max(top_k * 2, 10)
    if rerank:
        candidate_k = max(candidate_k, rerank_candidates)
    hits = search_semantic(query_embedding, semantic_docs, top_k=candidate_k)
    if rerank:
        from reranking.cross_encoder import CrossEncoderReranker, RerankCandidate

        reranker = CrossEncoderReranker(model_name=reranker_model)
        hits = reranker.rerank(
            query=question,
            candidates=[
                RerankCandidate(
                    doc_id=item.doc_id,
                    text=item.text,
                    score=item.score,
                    metadata=item.metadata,
                )
                for item in hits
            ],
            top_k=top_k,
        )
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

    conf = get_llm_config(provider=provider, model=model)
    conf.max_tokens = max_tokens
    conf.temperature = temperature
    conf.top_p = top_p
    conf.enable_streaming = stream
    conf.cache_enabled = llm_cache_enabled
    conf.cache_capacity = max(1, llm_cache_capacity)
    conf.cache_ttl_seconds = max(0.1, llm_cache_ttl_seconds)

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
        choices=KNOWN_LLM_PROVIDERS,
        help="LLM provider config to use.",
    )
    parser.add_argument("--model", default=None, help="Override provider default model.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-tokens", type=int, default=2500)
    parser.add_argument("--faiss-path", default="artifacts/faiss")
    parser.add_argument("--index", default="rag_chunks")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--stream", action="store_true", help="Stream answer tokens.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--rerank", action="store_true", help="Apply cross-encoder reranking.")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--rerank-candidates", type=int, default=20)
    parser.add_argument("--llm-cache-enabled", action="store_true", help="Enable in-memory LLM response cache.")
    parser.add_argument("--llm-cache-capacity", type=int, default=512)
    parser.add_argument("--llm-cache-ttl-seconds", type=float, default=300.0)
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
        rerank=args.rerank,
        reranker_model=args.reranker_model,
        rerank_candidates=args.rerank_candidates,
        llm_cache_enabled=args.llm_cache_enabled,
        llm_cache_capacity=args.llm_cache_capacity,
        llm_cache_ttl_seconds=args.llm_cache_ttl_seconds,
    )

