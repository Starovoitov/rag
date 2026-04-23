from __future__ import annotations

import argparse
import json

from parser.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for running the data pipeline."""
    parser = argparse.ArgumentParser(
        description="Parse RAG sources into JSONL dataset (chunks, Q/A, edge cases)."
    )
    parser.add_argument(
        "--output",
        default="data/rag_dataset.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--min-tokens", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--overlap-ratio", type=float, default=0.15)
    parser.add_argument("--min-output-chunk-tokens", type=int, default=120)
    parser.add_argument("--max-output-chunk-tokens", type=int, default=650)
    parser.add_argument("--max-chunks-per-url", type=int, default=12)
    parser.add_argument("--max-chunks-per-category", type=int, default=45)
    parser.add_argument("--chunker-mode", choices=("token", "semantic_dynamic"), default="token")
    parser.add_argument("--near-duplicate-jaccard", type=float, default=0.0)
    return parser


def main() -> None:
    """Parse CLI arguments, run pipeline, and print run statistics."""
    args = build_parser().parse_args()
    stats = run_pipeline(
        output_path=args.output,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        overlap_ratio=args.overlap_ratio,
        min_output_chunk_tokens=args.min_output_chunk_tokens,
        max_output_chunk_tokens=args.max_output_chunk_tokens,
        max_chunks_per_url=args.max_chunks_per_url,
        max_chunks_per_category=args.max_chunks_per_category,
        chunker_mode=args.chunker_mode,
        near_duplicate_jaccard=args.near_duplicate_jaccard,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

