from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def prepare_embedding_input(
    input_jsonl: str = "parser/output/rag_dataset.jsonl",
    output_jsonl: str = "parser/output/embeddings_input.jsonl",
) -> int:
    """Export raw chunk records into a compact embeddings input JSONL file."""
    src = Path(input_jsonl)
    dst = Path(output_jsonl)
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            item: dict[str, Any] = json.loads(line)
            if item.get("record_type") != "raw_chunk":
                continue
            payload = {
                "id": item["chunk_id"],
                "text": item["text"],
                "metadata": item.get("metadata", {}),
            }
            f_out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
    return written

