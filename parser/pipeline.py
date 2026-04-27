from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from parser.chunking import chunk_text, jaccard_similarity_tokens, overlap_tokens, token_count
from parser.edge_cases import build_edge_cases
from parser.models import RawChunkRecord
from parser.normalize import normalize_text
from parser.qa import build_qa_pairs
from parser.scraper import scrape_source
from parser.sources import (
    DEFAULT_SOURCES_CONFIG_PATH,
    build_alias_groups,
    build_seed_chunks,
    build_sources,
)
from utils.logger import configure_runtime_logger


def _contains_phrase(text_lower: str, phrase: str) -> bool:
    return phrase.lower() in text_lower


def enrich_chunk_aliases(chunk_text: str, alias_groups: tuple[tuple[str, tuple[str, ...]], ...]) -> str:
    """
    Add compact alias bridges to improve lexical + semantic retrievability.

    This helps when a chunk explains a concept but doesn't include common query terms.
    """
    chunk = chunk_text.strip()
    if not chunk:
        return chunk_text

    lowered = chunk.lower()
    bridge_clauses: list[str] = []
    for primary, aliases in alias_groups:
        terms = (primary, *aliases)
        present_terms = [term for term in terms if _contains_phrase(lowered, term)]
        if not present_terms:
            continue
        missing_aliases = [term for term in aliases if term not in present_terms]
        if not missing_aliases:
            continue
        bridge_clauses.append(f"{primary} (also known as {', '.join(missing_aliases)})")

    if not bridge_clauses:
        return chunk_text
    return f"{chunk} Alias notes: {'; '.join(bridge_clauses)}."


def _add_multihop_seed_chunks(
    *,
    f: Any,
    counters: dict[str, int],
    min_output_chunk_tokens: int,
    max_output_chunk_tokens: int,
    alias_groups: tuple[tuple[str, tuple[str, ...]], ...],
    seed_chunks: tuple[dict[str, str], ...],
) -> None:
    pseudo_source = SimpleNamespace(
        url="seed://multi-hop-retrieval",
        category="advanced_rag_ideas",
        subtopic="multi_hop_retrieval",
        source_type="seed",
        priority_topics=[
            "multi-hop retrieval",
            "multi-step retrieval",
            "iterative retrieval",
            "compositional retrieval",
            "graph rag",
            "agentic rag",
        ],
    )

    for idx, seed in enumerate(seed_chunks):
        text = f"Title: {seed['title']}\n\nContent:\n{seed['content']}".strip()
        enriched = enrich_chunk_aliases(text, alias_groups=alias_groups)
        chunk_tokens = token_count(enriched)
        if chunk_tokens < min_output_chunk_tokens or chunk_tokens > max_output_chunk_tokens:
            counters["skipped_by_token_filter"] += 1
            continue

        metadata = enrich_metadata(
            source=pseudo_source,
            title=seed["title"],
            chunk_index=idx,
            chunk_text=enriched,
            scraped_at="1970-01-01T00:00:00+00:00",
        )
        raw = RawChunkRecord(
            record_type="raw_chunk",
            chunk_id=metadata["chunk_id"],
            text=enriched,
            token_count=chunk_tokens,
            overlap_tokens=0,
            metadata=metadata,
        )
        f.write(json.dumps(raw.to_dict(), ensure_ascii=False) + "\n")
        counters["raw_chunks"] += 1

        for qa in build_qa_pairs(
            chunk_text=enriched,
            metadata=metadata,
            priority_topics=pseudo_source.priority_topics,
        ):
            f.write(json.dumps(qa.to_dict(), ensure_ascii=False) + "\n")
            counters["qa_pairs"] += 1


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
    sources_config: str = DEFAULT_SOURCES_CONFIG_PATH,
    log_level: str = "INFO",
    log_path: str | None = None,
    log_json: bool = False,
) -> dict[str, int]:
    """Run scraping-to-JSONL flow and return aggregate output counters."""
    logger = configure_runtime_logger(
        "rag.build_parser",
        level=log_level,
        log_path=log_path,
        json_logs=log_json,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    counters = {
        "raw_chunks": 0,
        "qa_pairs": 0,
        "edge_cases": 0,
        "sources_ok": 0,
        "skipped_duplicate_urls": 0,
        "skipped_by_token_filter": 0,
        "skipped_by_url_cap": 0,
        "skipped_by_category_cap": 0,
        "skipped_by_near_duplicate": 0,
    }
    sources = build_sources(config_path=sources_config)
    alias_groups = build_alias_groups(config_path=sources_config)
    seed_chunks = build_seed_chunks(config_path=sources_config)
    total_sources = len(sources)
    chunks_per_url: dict[str, int] = {}
    chunks_per_category: dict[str, int] = {}
    accepted_chunks_by_url: dict[str, list[str]] = {}
    seen_urls: set[str] = set()
    logger.info("starting parser pipeline: output=%s sources=%s", output_path, total_sources)

    with out.open("w", encoding="utf-8") as f:
        for source_idx, source in enumerate(sources, start=1):
            if source.url in seen_urls:
                counters["skipped_duplicate_urls"] += 1
                logger.warning("skipping duplicate URL in sources config: %s", source.url)
                continue
            seen_urls.add(source.url)
            logger.info(
                "processing source %s/%s: category=%s subtopic=%s url=%s",
                source_idx,
                total_sources,
                source.category,
                source.subtopic,
                source.url,
            )
            source_raw_before = counters["raw_chunks"]
            source_qa_before = counters["qa_pairs"]
            source_skipped_token_before = counters["skipped_by_token_filter"]
            source_skipped_url_before = counters["skipped_by_url_cap"]
            source_skipped_category_before = counters["skipped_by_category_cap"]
            source_skipped_near_dup_before = counters["skipped_by_near_duplicate"]
            try:
                parsed = scrape_source(source)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "source failed: category=%s subtopic=%s url=%s error=%s",
                    source.category,
                    source.subtopic,
                    source.url,
                    exc,
                )
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
                enriched_chunk = enrich_chunk_aliases(chunk, alias_groups=alias_groups)
                chunk_tokens = token_count(enriched_chunk)
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
                    chunk_text=enriched_chunk,
                    scraped_at=parsed.scraped_at,
                )
                raw = RawChunkRecord(
                    record_type="raw_chunk",
                    chunk_id=metadata["chunk_id"],
                    text=enriched_chunk,
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
                    chunk_text=enriched_chunk,
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
            logger.info(
                "source complete %s/%s: raw_chunks=%s qa_pairs=%s skipped(token/url/category/near_dup)=%s/%s/%s/%s",
                source_idx,
                total_sources,
                counters["raw_chunks"] - source_raw_before,
                counters["qa_pairs"] - source_qa_before,
                counters["skipped_by_token_filter"] - source_skipped_token_before,
                counters["skipped_by_url_cap"] - source_skipped_url_before,
                counters["skipped_by_category_cap"] - source_skipped_category_before,
                counters["skipped_by_near_duplicate"] - source_skipped_near_dup_before,
            )
        seed_raw_before = counters["raw_chunks"]
        seed_qa_before = counters["qa_pairs"]
        _add_multihop_seed_chunks(
            f=f,
            counters=counters,
            min_output_chunk_tokens=min_output_chunk_tokens,
            max_output_chunk_tokens=max_output_chunk_tokens,
            alias_groups=alias_groups,
            seed_chunks=seed_chunks,
        )
        logger.info(
            "multihop seed chunks added: raw_chunks=%s qa_pairs=%s",
            counters["raw_chunks"] - seed_raw_before,
            counters["qa_pairs"] - seed_qa_before,
        )

    logger.info("parser pipeline completed: %s", json.dumps(counters, ensure_ascii=False))
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

