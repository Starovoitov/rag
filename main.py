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
        default="parser/output/rag_dataset.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--min-tokens", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--overlap-ratio", type=float, default=0.15)
    return parser


def main() -> None:
    """Parse CLI arguments, run pipeline, and print run statistics."""
    args = build_parser().parse_args()
    stats = run_pipeline(
        output_path=args.output,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        overlap_ratio=args.overlap_ratio,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

