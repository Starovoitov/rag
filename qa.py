from __future__ import annotations

import re
import uuid

from parser.models import QAPairRecord


QUESTION_TEMPLATES = [
    "What is {topic} in RAG?",
    "Why is {topic} important for retrieval quality?",
    "How does {topic} work in a practical pipeline?",
]


def build_qa_pairs(
    chunk_text: str, metadata: dict[str, str], priority_topics: list[str]
) -> list[QAPairRecord]:
    """Generate simple topic-focused Q/A pairs from a chunk summary."""
    summary = summarize_for_answer(chunk_text)
    records: list[QAPairRecord] = []
    for topic in priority_topics[:3]:
        for template in QUESTION_TEMPLATES[:1]:
            question = template.format(topic=topic)
            answer = (
                f"{topic.capitalize()} helps the system return better context. "
                f"From this source: {summary}"
            )
            records.append(
                QAPairRecord(
                    record_type="qa_pair",
                    qa_id=str(uuid.uuid4()),
                    question=question,
                    answer=answer,
                    metadata=metadata,
                )
            )
    return records


def summarize_for_answer(text: str, max_chars: int = 240) -> str:
    """Compress chunk text into a short answer-friendly snippet."""
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."

