from __future__ import annotations

import json
import random
from pathlib import Path

from sentence_transformers import InputExample
from ingestion.loaders import load_chunk_texts

def load_pairwise_samples(
    path: Path,
    *,
    seed: int,
    val_ratio: float,
    chunk_texts: dict[str, str],
) -> tuple[list[InputExample], list[InputExample]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    random.Random(seed).shuffle(rows)

    examples: list[InputExample] = []
    for row in rows:
        query = str(row.get("query", "")).strip()
        if row.get("schema_version") == "reranker_context_v1":
            positives = [str(doc_id) for doc_id in row.get("positives", [])]
            negatives = [str(doc_id) for doc_id in row.get("negatives", [])]
            weights = row.get("weights", {}) or {}
            for positive_id in positives:
                positive_text = chunk_texts.get(positive_id, "").strip()
                if not positive_text:
                    continue
                for negative_id in negatives:
                    negative_text = chunk_texts.get(negative_id, "").strip()
                    if not negative_text:
                        continue
                    sample_weight = float(weights.get(negative_id, 1.0))
                    repeats = max(1, int(round(sample_weight)))
                    for _ in range(repeats):
                        examples.append(InputExample(texts=[query, positive_text], label=1.0))
                        examples.append(InputExample(texts=[query, negative_text], label=0.0))
            continue
        if "positive_text" in row and "negative_text" in row:
            positive_text = str(row.get("positive_text", "")).strip()
            negative_text = str(row.get("negative_text", "")).strip()
        else:
            positive = row.get("positive", {})
            negative = row.get("negative", {})
            positive_text = str((positive or {}).get("text", "")).strip()
            negative_text = str((negative or {}).get("text", "")).strip()
        sample_weight = float(row.get("sample_weight", 1.0))
        if not query or not positive_text or not negative_text:
            continue
        repeats = max(1, int(round(sample_weight)))
        # Binary objective with hard negatives is a strong practical proxy for pairwise ranking.
        for _ in range(repeats):
            examples.append(InputExample(texts=[query, positive_text], label=1.0))
            examples.append(InputExample(texts=[query, negative_text], label=0.0))

    val_size = int(len(examples) * max(0.0, min(val_ratio, 0.5)))
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]
    return train_examples, val_examples

