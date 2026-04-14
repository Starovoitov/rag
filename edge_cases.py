from __future__ import annotations

import uuid

from parser.models import EdgeCaseRecord


def build_edge_cases(metadata: dict[str, str]) -> list[EdgeCaseRecord]:
    """Build standard edge-case samples for RAG failure modes."""
    return [
        EdgeCaseRecord(
            record_type="edge_case",
            edge_case_id=str(uuid.uuid4()),
            edge_case_type="bad_chunking",
            prompt="Answer from a chunk split in the middle of a definition.",
            flawed_example=(
                "Chunk A: 'FAISS is a library for effi...'\n"
                "Chunk B: '...cient similarity search and clustering.'"
            ),
            corrected_example=(
                "Single chunk with complete sentence and context about index type."
            ),
            why_it_fails="Definition is broken and retrieval returns incomplete meaning.",
            metadata=metadata,
        ),
        EdgeCaseRecord(
            record_type="edge_case",
            edge_case_id=str(uuid.uuid4()),
            edge_case_type="hallucination",
            prompt="Generate answer when retrieved context is weak.",
            flawed_example="Model invents FAISS benchmark numbers not in source.",
            corrected_example="Model says uncertainty and asks for more grounded context.",
            why_it_fails="No citation guardrails and weak relevance filtering.",
            metadata=metadata,
        ),
        EdgeCaseRecord(
            record_type="edge_case",
            edge_case_id=str(uuid.uuid4()),
            edge_case_type="wrong_retrieval",
            prompt="Question about reranking returns generic embeddings text.",
            flawed_example="Top-k uses only cosine similarity from unrelated chunks.",
            corrected_example="Hybrid retrieval + cross-encoder reranker before answer.",
            why_it_fails="Retriever misses intent; no multi-query or reranking step.",
            metadata=metadata,
        ),
    ]

