#!/usr/bin/env python3
"""
Run BM25, semantic (cosine), and hybrid retrieval on a tiny sample corpus.

Usage from project root:
  poetry run python demo_retrieval.py
  poetry run python demo_retrieval.py --query "how does caching work"
"""

from __future__ import annotations

import argparse

from sentence_transformers import SentenceTransformer

from ingestion.loaders import (
    load_bm25_documents_from_dataset,
    load_semantic_documents_from_faiss,
)
from retrieval.bm25 import BM25Document, BM25Index
from retrieval.hybrid import hybrid_search
from retrieval.semantic import search_semantic
from utils.embedding_format import format_query_for_embedding

# Same model as embeddings/embedder.py.
DEFAULT_MODEL = "intfloat/e5-base-v2"


def run_demo(
    query: str,
    top_k: int,
    model_name: str,
    dataset_path: str,
    faiss_path: str,
    index_name: str,
    rerank: bool = False,
    reranker_model: str = "BAAI/bge-reranker-large",
    rerank_candidates: int = 20,
) -> None:
    dataset_docs = load_bm25_documents_from_dataset(dataset_path=dataset_path)
    if not dataset_docs:
        raise ValueError(
            f"No raw_chunk records found in {dataset_path}. "
            "Run parser ingestion before demo retrieval."
        )

    bm25_docs = [
        BM25Document(doc_id=item["id"], text=item["text"], metadata=item["metadata"])
        for item in dataset_docs
    ]
    bm25_index = BM25Index(bm25_docs)
    bm25_results = bm25_index.search(query, top_k=top_k)

    model = SentenceTransformer(model_name)
    query_vec = model.encode(
        [format_query_for_embedding(query, model_name)],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()

    semantic_docs = load_semantic_documents_from_faiss(
        persist_directory=faiss_path,
        index_name=index_name,
    )
    if not semantic_docs:
        raise ValueError(
            f"No embeddings found in FAISS index '{index_name}' at {faiss_path}. "
            "Run parser + embedding ingestion first."
        )
    candidate_k = max(top_k, rerank_candidates) if rerank else top_k
    semantic_results = search_semantic(query_vec, semantic_docs, top_k=candidate_k)

    hybrid_results = hybrid_search(semantic_results, bm25_results, alpha=0.7, top_k=candidate_k)
    reranked_results = []
    if rerank:
        from reranking.cross_encoder import CrossEncoderReranker, RerankCandidate

        reranker = CrossEncoderReranker(model_name=reranker_model)
        reranked_results = reranker.rerank(
            query=query,
            candidates=[
                RerankCandidate(
                    doc_id=item.doc_id,
                    text=item.text,
                    score=item.score,
                    metadata=item.metadata,
                )
                for item in hybrid_results
            ],
            top_k=top_k,
        )

    def print_block(title: str, rows: list) -> None:
        print(f"\n--- {title} ---")
        if not rows:
            print("(no results)")
            return
        for r in rows:
            text_preview = (r.text[:120] + "…") if len(r.text) > 120 else r.text
            if hasattr(r, "semantic_score"):
                print(
                    f"  id={r.doc_id}  score={r.score:.4f}  "
                    f"sem={r.semantic_score:.4f}  bm25_raw={r.bm25_score:.4f}"
                )
            else:
                print(f"  id={r.doc_id}  score={r.score:.4f}")
            print(f"       {text_preview}")

    print_block("BM25 (lexical)", bm25_results[:top_k])
    print_block("Semantic (cosine)", semantic_results[:top_k])
    print_block("Hybrid (RRF fusion: alpha * semantic_rrf + (1-alpha) * bm25_rrf)", hybrid_results[:top_k])
    if rerank:
        print_block("Cross-encoder reranked (over hybrid candidates)", reranked_results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo BM25 + semantic + hybrid retrieval.")
    parser.add_argument(
        "--query",
        "-q",
        default="database caching performance",
        help="Search query text.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=10,
        help="Number of hits to show per method.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"Sentence-transformers model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--dataset",
        default="data/rag_dataset.jsonl",
        help="Dataset JSONL created by parser pipeline.",
    )
    parser.add_argument(
        "--faiss-path",
        default="data/faiss",
        help="Persistent FAISS directory containing chunk embeddings.",
    )
    parser.add_argument(
        "--index",
        default="rag_chunks",
        help="FAISS index name with precomputed embeddings.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Apply cross-encoder reranking over hybrid candidates.",
    )
    parser.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model name used when --rerank is enabled.",
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=20,
        help="How many hybrid candidates to rerank before trimming to top-k.",
    )
    args = parser.parse_args()
    run_demo(
        query=args.query,
        top_k=args.top_k,
        model_name=args.model,
        dataset_path=args.dataset,
        faiss_path=args.faiss_path,
        index_name=args.index,
        rerank=args.rerank,
        reranker_model=args.reranker_model,
        rerank_candidates=args.rerank_candidates,
    )


if __name__ == "__main__":
    main()
