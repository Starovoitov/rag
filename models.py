from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class SourceSpec:
    category: str
    subtopic: str
    url: str
    source_type: str
    priority_topics: list[str]


@dataclass
class ParsedDocument:
    source: SourceSpec
    title: str
    text: str
    scraped_at: str = field(default_factory=utc_now_iso)


@dataclass
class RawChunkRecord:
    record_type: str
    chunk_id: str
    text: str
    token_count: int
    overlap_tokens: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QAPairRecord:
    record_type: str
    qa_id: str
    question: str
    answer: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EdgeCaseRecord:
    record_type: str
    edge_case_id: str
    edge_case_type: str
    prompt: str
    flawed_example: str
    corrected_example: str
    why_it_fails: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

