from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from sentence_transformers import SentenceTransformer

from evaluation.dataset import EvalSample, load_eval_samples
from evaluation.metrics import RetrievalResult, evaluate_retrieval
from ingestion.loaders import load_bm25_documents_from_dataset, load_semantic_documents_from_faiss
from retrieval.bm25 import BM25Document, BM25Index
from retrieval.hybrid import hybrid_search
from retrieval.semantic import SemanticDocument, search_semantic
from utils.embedding_format import format_query_for_embedding

DEFAULT_EMBEDDING_MODEL = "intfloat/e5-small-v2"


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[str]:
        ...


@dataclass(frozen=True)
class QueryRun:
    query: str
    relevant_doc_ids: list[str]
    retrieved_doc_ids: list[str]


class SemanticRetriever:
    def __init__(self, documents: list[SemanticDocument], embedding_model: str) -> None:
        self.documents = documents
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_model = embedding_model

    def search(self, query: str, top_k: int) -> list[str]:
        query_embedding = self.embedder.encode(
            [format_query_for_embedding(query, self.embedding_model)],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()
        hits = search_semantic(query_embedding, self.documents, top_k=top_k)
        return [item.doc_id for item in hits]


class BM25Retriever:
    def __init__(self, index: BM25Index) -> None:
        self.index = index

    def search(self, query: str, top_k: int) -> list[str]:
        hits = self.index.search(query, top_k=top_k)
        return [item.doc_id for item in hits]


class HybridRetriever:
    def __init__(
        self,
        semantic: SemanticRetriever,
        bm25: BM25Retriever,
        alpha: float = 0.7,
        candidate_multiplier: int = 2,
        max_per_group: int = 1,
        rrf_k: float = 60.0,
    ) -> None:
        self.semantic = semantic
        self.bm25 = bm25
        self.alpha = alpha
        self.candidate_multiplier = max(1, candidate_multiplier)
        self.max_per_group = max_per_group
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int) -> list[str]:
        query_embedding = self.semantic.embedder.encode(
            [format_query_for_embedding(query, self.semantic.embedding_model)],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()
        branch_k = max(top_k * self.candidate_multiplier, 10)
        semantic_hits = search_semantic(query_embedding, self.semantic.documents, top_k=branch_k)
        bm25_hits = self.bm25.index.search(query, top_k=branch_k)
        merged = hybrid_search(
            semantic_hits,
            bm25_hits,
            alpha=self.alpha,
            top_k=top_k,
            max_per_group=self.max_per_group,
            rrf_k=self.rrf_k,
        )
        return [item.doc_id for item in merged]


def build_retriever(
    mode: str,
    *,
    rag_dataset_path: str,
    faiss_path: str,
    index_name: str,
    embedding_model: str,
    alpha: float,
    hybrid_candidate_multiplier: int = 2,
    hybrid_max_per_group: int = 1,
    hybrid_rrf_k: float = 60.0,
) -> Retriever:
    if mode == "semantic":
        docs = load_semantic_documents_from_faiss(persist_directory=faiss_path, index_name=index_name)
        if not docs:
            raise ValueError(f"No semantic docs in FAISS index '{index_name}' at '{faiss_path}'.")
        return SemanticRetriever(docs, embedding_model=embedding_model)

    bm25_docs = load_bm25_documents_from_dataset(rag_dataset_path)
    if not bm25_docs:
        raise ValueError(f"No raw chunks in dataset '{rag_dataset_path}'.")
    bm25_index = BM25Index(
        [
            BM25Document(doc_id=item["id"], text=item["text"], metadata=item.get("metadata", {}))
            for item in bm25_docs
        ]
    )
    if mode == "bm25":
        return BM25Retriever(index=bm25_index)
    if mode == "hybrid":
        semantic_docs = load_semantic_documents_from_faiss(persist_directory=faiss_path, index_name=index_name)
        if not semantic_docs:
            raise ValueError(f"No semantic docs in FAISS index '{index_name}' at '{faiss_path}'.")
        semantic = SemanticRetriever(semantic_docs, embedding_model=embedding_model)
        return HybridRetriever(
            semantic=semantic,
            bm25=BM25Retriever(index=bm25_index),
            alpha=alpha,
            candidate_multiplier=hybrid_candidate_multiplier,
            max_per_group=hybrid_max_per_group,
            rrf_k=hybrid_rrf_k,
        )
    raise ValueError(f"Unsupported retriever mode: {mode}")


def run_benchmark(
    samples: list[EvalSample],
    retriever: Retriever,
    *,
    max_k: int,
) -> tuple[dict[str, float], list[QueryRun]]:
    results: list[RetrievalResult] = []
    details: list[QueryRun] = []
    for sample in samples:
        retrieved_doc_ids = retriever.search(sample.query, top_k=max_k)
        results.append(
            RetrievalResult(
                query=sample.query,
                retrieved_doc_ids=retrieved_doc_ids,
                relevant_doc_ids=sample.relevant_docs,
            )
        )
        details.append(
            QueryRun(
                query=sample.query,
                relevant_doc_ids=sample.relevant_docs,
                retrieved_doc_ids=retrieved_doc_ids,
            )
        )
    return evaluate_retrieval(results, [max_k]), details


def parse_k_values(raw: str) -> list[int]:
    parsed = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not parsed or any(k <= 0 for k in parsed):
        raise ValueError("--k-values must contain positive integers, e.g. 1,3,5")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval benchmark on evaluation dataset.")
    parser.add_argument("--dataset", default="data/evaluation_with_evidence.jsonl")
    parser.add_argument("--retriever", choices=("semantic", "bm25", "hybrid"), default="semantic")
    parser.add_argument("--k-values", default="1,3,5")
    parser.add_argument("--rag-dataset", default="data/rag_dataset.jsonl")
    parser.add_argument("--faiss-path", default="data/faiss")
    parser.add_argument("--index", default="rag_chunks")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--alpha", type=float, default=0.7, help="Hybrid semantic/BM25 mix.")
    parser.add_argument("--rerank", action="store_true", help="Apply cross-encoder reranking.")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--rerank-candidates", type=int, default=20)
    parser.add_argument(
        "--require-evidence",
        action="store_true",
        help="Evaluate only samples with non-empty expected_evidence.chunk_ids.",
    )
    parser.add_argument("--out-json", default=None, help="Optional path to save JSON report.")
    args = parser.parse_args()

    samples = load_eval_samples(Path(args.dataset))
    if not samples:
        raise ValueError(f"No samples found in dataset: {args.dataset}")
    total_samples_before_filter = len(samples)
    if args.require_evidence:
        samples = [sample for sample in samples if sample.relevant_docs]
    filtered_out_samples = total_samples_before_filter - len(samples)
    if not samples:
        raise ValueError(
            "No samples left after filtering. "
            "Try running without --require-evidence or regenerate dataset with more evidence links."
        )

    k_values = parse_k_values(args.k_values)
    max_k = max(k_values)

    retriever = build_retriever(
        args.retriever,
        rag_dataset_path=args.rag_dataset,
        faiss_path=args.faiss_path,
        index_name=args.index,
        embedding_model=args.embedding_model,
        alpha=args.alpha,
    )
    doc_text_map = {
        item["id"]: item["text"] for item in load_bm25_documents_from_dataset(args.rag_dataset)
    }
    reranker = None
    if args.rerank:
        from reranking.cross_encoder import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name=args.reranker_model)
    query_runs: list[QueryRun] = []
    metric_inputs: list[RetrievalResult] = []
    for sample in samples:
        retrieve_k = max(max_k, args.rerank_candidates) if args.rerank else max_k
        retrieved = retriever.search(sample.query, top_k=retrieve_k)
        if reranker is not None:
            from reranking.cross_encoder import RerankCandidate

            rerank_input = [
                RerankCandidate(
                    doc_id=doc_id,
                    text=doc_text_map.get(doc_id, ""),
                )
                for doc_id in retrieved
                if doc_text_map.get(doc_id, "")
            ]
            reranked = reranker.rerank(sample.query, rerank_input, top_k=max_k)
            retrieved = [item.doc_id for item in reranked]
        else:
            retrieved = retrieved[:max_k]
        query_runs.append(
            QueryRun(query=sample.query, relevant_doc_ids=sample.relevant_docs, retrieved_doc_ids=retrieved)
        )
        metric_inputs.append(
            RetrievalResult(
                query=sample.query,
                retrieved_doc_ids=retrieved,
                relevant_doc_ids=sample.relevant_docs,
            )
        )

    metrics = evaluate_retrieval(metric_inputs, k_values)
    report = {
        "dataset": args.dataset,
        "retriever": args.retriever,
        "rerank_enabled": args.rerank,
        "reranker_model": args.reranker_model if args.rerank else None,
        "require_evidence": args.require_evidence,
        "k_values": k_values,
        "samples_total": len(samples),
        "samples_total_before_filter": total_samples_before_filter,
        "samples_filtered_out": filtered_out_samples,
        "samples_with_ground_truth": sum(1 for s in samples if s.relevant_docs),
        "metrics": metrics,
        "runs": [asdict(run) for run in query_runs],
    }

    print("Retrieval benchmark report")
    print(f"- dataset: {args.dataset}")
    print(f"- retriever: {args.retriever}")
    print(f"- samples: {len(samples)}")
    if args.require_evidence:
        print(f"- require_evidence: true (filtered_out={filtered_out_samples})")
    for key in sorted(metrics):
        print(f"- {key}: {metrics[key]:.4f}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved JSON report to {out_path}")


if __name__ == "__main__":
    main()
