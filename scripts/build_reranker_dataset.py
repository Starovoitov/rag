#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_chunk_texts(rag_dataset_path: Path) -> dict[str, str]:
    chunk_texts: dict[str, str] = {}
    with rag_dataset_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            row = json.loads(line)
            if row.get("record_type") != "raw_chunk":
                continue
            chunk_id = str(row.get("chunk_id", "")).strip()
            text = str(row.get("text", "")).strip()
            if chunk_id and text:
                chunk_texts[chunk_id] = text
    return chunk_texts


def build_pairs(
    *,
    report: dict,
    chunk_texts: dict[str, str],
    max_negative_rank: int,
    max_negatives_per_positive: int,
    ranking_cutoff_weight: float,
    true_recall_weight: float,
    default_weight: float,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    diagnostics = report.get("diagnostics", {})
    failure_analysis = diagnostics.get("failure_analysis", {})
    samples = failure_analysis.get("manual_inspection_samples", [])

    pairs: list[dict[str, object]] = []
    stats = {
        "samples_seen": 0,
        "samples_used": 0,
        "pairs_written": 0,
        "missing_positive_text": 0,
        "missing_negative_text": 0,
        "pairs_ranking_cutoff_failure": 0,
        "pairs_true_recall_failure": 0,
        "pairs_other": 0,
    }

    for sample in samples:
        stats["samples_seen"] += 1
        query = str(sample.get("query", "")).strip()
        positives = [str(doc_id) for doc_id in sample.get("relevant_doc_ids", [])]
        retrieved = [str(doc_id) for doc_id in sample.get("retrieved_top_k_doc_ids", [])]
        bucket = str(sample.get("bucket", ""))
        source_miss_type = str(sample.get("source_miss_type", ""))
        if bucket == "ranking_cutoff_failure":
            sample_weight = ranking_cutoff_weight
        elif bucket == "true_recall_failure":
            sample_weight = true_recall_weight
        else:
            sample_weight = default_weight

        if not query or not positives or not retrieved:
            continue

        stats["samples_used"] += 1
        top_ranked_negatives = [doc_id for doc_id in retrieved[:max_negative_rank] if doc_id not in set(positives)]

        for positive_id in positives:
            positive_text = chunk_texts.get(positive_id, "")
            if not positive_text:
                stats["missing_positive_text"] += 1
                continue

            negatives_added = 0
            for rank, negative_id in enumerate(top_ranked_negatives, start=1):
                if negatives_added >= max_negatives_per_positive:
                    break
                negative_text = chunk_texts.get(negative_id, "")
                if not negative_text:
                    stats["missing_negative_text"] += 1
                    continue
                pairs.append(
                    {
                        "schema_version": "reranker_pairwise_v1",
                        "query": query,
                        "positive": {"doc_id": positive_id, "text": positive_text},
                        "negative": {"doc_id": negative_id, "text": negative_text, "rank": rank},
                        "failure_bucket": bucket,
                        "source_miss_type": source_miss_type,
                        "sample_weight": sample_weight,
                    }
                )
                negatives_added += 1
                stats["pairs_written"] += 1
                if bucket == "ranking_cutoff_failure":
                    stats["pairs_ranking_cutoff_failure"] += 1
                elif bucket == "true_recall_failure":
                    stats["pairs_true_recall_failure"] += 1
                else:
                    stats["pairs_other"] += 1

    return pairs, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pairwise reranker training data from retrieval report.")
    parser.add_argument("--eval-report", type=Path, default=Path("data/retrieval_report_best.json"))
    parser.add_argument("--rag-dataset", type=Path, default=Path("data/rag_dataset.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/reranker_train.jsonl"))
    parser.add_argument("--max-negative-rank", type=int, default=20)
    parser.add_argument("--max-negatives-per-positive", type=int, default=8)
    parser.add_argument("--ranking-cutoff-weight", type=float, default=2.0)
    parser.add_argument("--true-recall-weight", type=float, default=1.5)
    parser.add_argument("--default-weight", type=float, default=1.0)
    args = parser.parse_args()

    report = json.loads(args.eval_report.read_text(encoding="utf-8"))
    chunk_texts = load_chunk_texts(args.rag_dataset)
    pairs, stats = build_pairs(
        report=report,
        chunk_texts=chunk_texts,
        max_negative_rank=max(1, args.max_negative_rank),
        max_negatives_per_positive=max(1, args.max_negatives_per_positive),
        ranking_cutoff_weight=max(0.1, args.ranking_cutoff_weight),
        true_recall_weight=max(0.1, args.true_recall_weight),
        default_weight=max(0.1, args.default_weight),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fp:
        for row in pairs:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {args.out} ({len(pairs)} pairs).")
    print("Stats:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
