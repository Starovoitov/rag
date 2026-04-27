#!/usr/bin/env python3
from __future__ import annotations

import argparse

from sentence_transformers import SentenceTransformer

from generation.config import DEFAULT_LLM_CONFIG_PATH, load_llm_provider_configs
from generation.llm import LLMConfig, call_llm, stream_llm
from generation.prompt import SourceChunk, build_rag_messages
from ingestion.loaders import load_semantic_documents_from_faiss
from retrieval.semantic import search_semantic
from utils.embedding_format import format_query_for_embedding
from utils.logger import configure_runtime_logger

DEFAULT_EMBEDDING_MODEL = "intfloat/e5-base-v2"
FALLBACK_LLM_PROVIDERS: tuple[str, ...] = ("openai", "gigachat", "ollama", "qwen")

def _guess_embedding_models_by_dim(dim: int) -> str:
    known_dims = {
        384: "intfloat/e5-small-v2",
        768: "intfloat/e5-base-v2",
        1024: "intfloat/e5-large-v2",
    }
    return known_dims.get(dim, "unknown")


def get_llm_config(provider: str, model: str | None = None, *, config_path: str = DEFAULT_LLM_CONFIG_PATH) -> LLMConfig:
    """Load provider defaults from config and optionally override model."""
    configs = load_llm_provider_configs(config_path=config_path)
    key = provider.lower().strip()
    if key not in configs:
        raise ValueError(f"Unsupported provider: {provider}")
    config = configs[key]
    if model:
        config.model = model
    return config


def build_model_configs(config_path: str = DEFAULT_LLM_CONFIG_PATH) -> dict[str, LLMConfig]:
    """Named provider configs loaded from root config."""
    return load_llm_provider_configs(config_path=config_path)


def _load_known_providers_safe(config_path: str) -> tuple[str, ...]:
    """Load provider names for argparse without failing when config is missing."""
    try:
        provider_names = tuple(load_llm_provider_configs(config_path=config_path).keys())
        return provider_names or FALLBACK_LLM_PROVIDERS
    except Exception:
        return FALLBACK_LLM_PROVIDERS



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
    log_level: str = "INFO",
    log_path: str | None = None,
    log_json: bool = False,
    llm_config_path: str = DEFAULT_LLM_CONFIG_PATH,
    rerank_top1_margin_lambda: float = 0.0,
) -> None:
    logger = configure_runtime_logger(
        "rag.run_rag",
        level=log_level,
        log_path=log_path,
        json_logs=log_json,
    )
    logger.info("starting run_rag pipeline")
    try:
        semantic_docs = load_semantic_documents_from_faiss(
            persist_directory=faiss_path,
            index_name=index_name,
        )
        if not semantic_docs:
            logger.error("no semantic docs found in FAISS index")
            raise ValueError(
                f"No semantic docs in FAISS index '{index_name}' at '{faiss_path}'. "
                "Run ingestion first."
            )
        logger.info("loaded semantic documents: count=%s", len(semantic_docs))

        logger.info("loading embedding model: %s", embedding_model)
        embedder = SentenceTransformer(embedding_model)
        query_embedding = embedder.encode(
            [format_query_for_embedding(question, embedding_model)],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()
        query_dim = len(query_embedding)
        doc_dim = len(semantic_docs[0].embedding) if semantic_docs else 0
        logger.info("encoded query embedding: query_dim=%s doc_dim=%s", query_dim, doc_dim)
        if doc_dim and query_dim != doc_dim:
            suggested_doc_model = _guess_embedding_models_by_dim(doc_dim)
            suggested_query_model = _guess_embedding_models_by_dim(query_dim)
            logger.error("embedding dimension mismatch detected")
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
        logger.info("running semantic retrieval: top_k=%s candidate_k=%s", top_k, candidate_k)
        hits = search_semantic(query_embedding, semantic_docs, top_k=candidate_k)
        logger.info("retrieval completed: hits=%s", len(hits))
        if rerank:
            # Heavy reranker deps are loaded only when reranking is enabled.
            from reranking.cross_encoder import CrossEncoderReranker, RerankCandidate

            logger.info("running reranker: model=%s", reranker_model)
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
                top1_margin_lambda=rerank_top1_margin_lambda,
            )
            logger.info("reranking completed: hits=%s", len(hits))
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
        if not prompt_data["used_chunks"]:
            logger.warning("prompt built with zero chunks")
        logger.info("prompt built: used_chunks=%s", len(prompt_data["used_chunks"]))

        conf = get_llm_config(provider=provider, model=model, config_path=llm_config_path)
        conf.max_tokens = max_tokens
        conf.temperature = temperature
        conf.top_p = top_p
        conf.enable_streaming = stream
        conf.cache_enabled = llm_cache_enabled
        conf.cache_capacity = max(1, llm_cache_capacity)
        conf.cache_ttl_seconds = max(0.1, llm_cache_ttl_seconds)
        logger.info(
            "configured llm call: provider=%s model=%s stream=%s cache_enabled=%s",
            provider,
            conf.model,
            stream,
            conf.cache_enabled,
        )

        print("Used sources:")
        for idx, chunk in enumerate(prompt_data["used_chunks"], start=1):
            print(f"[{idx}] {chunk.doc_id}")

        print("\nAnswer:")
        if stream:
            logger.info("starting streaming answer")
            for token in stream_llm(
                system_prompt=prompt_data["system_prompt"],
                user_prompt=prompt_data["user_prompt"],
                config=conf,
            ):
                print(token, end="", flush=True)
            print()
            logger.info("streaming answer finished")
            return

        logger.info("starting non-stream llm call")
        answer = call_llm(
            system_prompt=prompt_data["system_prompt"],
            user_prompt=prompt_data["user_prompt"],
            config=conf,
        )
        print(answer)
        logger.info("run_rag completed successfully")
    except Exception:
        logger.exception("run_rag failed")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single RAG query against one LLM provider.")
    parser.add_argument("--question", "-q", required=True, help="Question to ask.")
    known_providers = _load_known_providers_safe(DEFAULT_LLM_CONFIG_PATH)
    parser.add_argument(
        "--provider",
        default=known_providers[0],
        choices=known_providers,
        help="LLM provider config to use.",
    )
    parser.add_argument("--model", default=None, help="Override provider default model.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-tokens", type=int, default=2500)
    parser.add_argument("--faiss-path", default="data/faiss")
    parser.add_argument("--index", default=".")
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
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--log-path", default=None, help="Optional runtime log file path.")
    parser.add_argument("--log-json", action="store_true", help="Emit runtime logs in JSON format.")
    parser.add_argument("--llm-config-path", default=DEFAULT_LLM_CONFIG_PATH)
    parser.add_argument("--rerank-top1-margin-lambda", type=float, default=0.0)
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
        log_level=args.log_level,
        log_path=args.log_path,
        log_json=args.log_json,
        llm_config_path=args.llm_config_path,
        rerank_top1_margin_lambda=args.rerank_top1_margin_lambda,
    )

