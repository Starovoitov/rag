from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from parser.chunking import chunk_text, jaccard_similarity_tokens, overlap_tokens, token_count
from parser.edge_cases import build_edge_cases
from parser.models import RawChunkRecord
from parser.normalize import normalize_text
from parser.qa import build_qa_pairs
from parser.scraper import scrape_source
from parser.sources import build_sources


def run_pipeline(
    output_path: str = "data/rag_dataset.jsonl",
    min_tokens: int = 300,
    max_tokens: int = 800,
    overlap_ratio: float = 0.15,
    min_output_chunk_tokens: int = 120,
    max_output_chunk_tokens: int = 650,
    max_chunks_per_url: int = 12,
    max_chunks_per_category: int = 45,
    chunker_mode: str = "token",
    near_duplicate_jaccard: float = 0.0,
) -> dict[str, int]:
    """Run scraping-to-JSONL flow and return aggregate output counters."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    counters = {
        "raw_chunks": 0,
        "qa_pairs": 0,
        "edge_cases": 0,
        "sources_ok": 0,
        "skipped_by_token_filter": 0,
        "skipped_by_url_cap": 0,
        "skipped_by_category_cap": 0,
        "skipped_by_near_duplicate": 0,
    }
    sources = build_sources()
    chunks_per_url: dict[str, int] = {}
    chunks_per_category: dict[str, int] = {}
    accepted_chunks_by_url: dict[str, list[str]] = {}

    with out.open("w", encoding="utf-8") as f:
        for source in sources:
            try:
                parsed = scrape_source(source)
            except Exception as exc:  # noqa: BLE001
                err = {
                    "record_type": "source_error",
                    "url": source.url,
                    "category": source.category,
                    "error": str(exc),
                }
                f.write(json.dumps(err, ensure_ascii=False) + "\n")
                continue

            counters["sources_ok"] += 1
            normalized = normalize_text(parsed.text)
            chunks = chunk_text(
                normalized,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                overlap_ratio=overlap_ratio,
                mode=chunker_mode,
            )
            overlap = overlap_tokens(max_tokens=max_tokens, overlap_ratio=overlap_ratio)

            for idx, chunk in enumerate(chunks):
                chunk_tokens = token_count(chunk)
                if chunk_tokens < min_output_chunk_tokens or chunk_tokens > max_output_chunk_tokens:
                    counters["skipped_by_token_filter"] += 1
                    continue

                url_count = chunks_per_url.get(source.url, 0)
                if max_chunks_per_url > 0 and url_count >= max_chunks_per_url:
                    counters["skipped_by_url_cap"] += 1
                    continue
                category_count = chunks_per_category.get(source.category, 0)
                if max_chunks_per_category > 0 and category_count >= max_chunks_per_category:
                    counters["skipped_by_category_cap"] += 1
                    continue
                if near_duplicate_jaccard > 0:
                    previous_chunks = accepted_chunks_by_url.get(source.url, [])
                    if any(jaccard_similarity_tokens(chunk, prev) >= near_duplicate_jaccard for prev in previous_chunks):
                        counters["skipped_by_near_duplicate"] += 1
                        continue

                metadata = enrich_metadata(
                    source=source,
                    title=parsed.title,
                    chunk_index=idx,
                    chunk_text=chunk,
                    scraped_at=parsed.scraped_at,
                )
                raw = RawChunkRecord(
                    record_type="raw_chunk",
                    chunk_id=metadata["chunk_id"],
                    text=chunk,
                    token_count=chunk_tokens,
                    overlap_tokens=overlap,
                    metadata=metadata,
                )
                f.write(json.dumps(raw.to_dict(), ensure_ascii=False) + "\n")
                counters["raw_chunks"] += 1
                chunks_per_url[source.url] = url_count + 1
                chunks_per_category[source.category] = category_count + 1
                accepted_chunks_by_url.setdefault(source.url, []).append(chunk)

                for qa in build_qa_pairs(
                    chunk_text=chunk,
                    metadata=metadata,
                    priority_topics=source.priority_topics,
                ):
                    f.write(json.dumps(qa.to_dict(), ensure_ascii=False) + "\n")
                    counters["qa_pairs"] += 1

            sample_metadata: dict[str, Any] = {
                "url": source.url,
                "category": source.category,
                "subtopic": source.subtopic,
                "title": parsed.title,
            }
            for edge_case in build_edge_cases(sample_metadata):
                f.write(json.dumps(edge_case.to_dict(), ensure_ascii=False) + "\n")
                counters["edge_cases"] += 1

    return counters


def enrich_metadata(
    source: Any,
    title: str,
    chunk_index: int,
    chunk_text: str,
    scraped_at: str,
) -> dict[str, Any]:
    """Build normalized metadata for each chunk with stable short chunk id."""
    digest = hashlib.sha256(f"{source.url}:{chunk_index}:{chunk_text}".encode()).hexdigest()
    return {
        "chunk_id": digest[:16],
        "url": source.url,
        "title": title,
        "category": source.category,
        "subtopic": source.subtopic,
        "source_type": source.source_type,
        "priority_topics": source.priority_topics,
        "chunk_index": chunk_index,
        "language": "en",
        "scraped_at": scraped_at,
    }

