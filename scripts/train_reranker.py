#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader


def load_pairwise_samples(path: Path, *, seed: int, val_ratio: float) -> tuple[list[InputExample], list[InputExample]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    random.Random(seed).shuffle(rows)

    examples: list[InputExample] = []
    for row in rows:
        query = str(row.get("query", "")).strip()
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune cross-encoder reranker on hard negatives.")
    parser.add_argument("--train-jsonl", type=Path, default=Path("data/reranker_train.jsonl"))
    parser.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--out-dir", type=Path, default=Path("models/reranker-failure-driven"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_examples, val_examples = load_pairwise_samples(
        args.train_jsonl,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )
    if not train_examples:
        raise ValueError("No training examples loaded. Check --train-jsonl contents.")

    model = CrossEncoder(args.model, num_labels=1, max_length=512)
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)

    evaluator = None
    if val_examples:
        evaluator = CEBinaryClassificationEvaluator.from_input_examples(
            val_examples,
            name="failure-driven-val",
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_dataloader=train_loader,
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        output_path=str(args.out_dir),
        show_progress_bar=True,
    )
    # Ensure a loadable model is persisted even if trainer callbacks only emit eval artifacts.
    model.save(str(args.out_dir))

    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "train_examples": len(train_examples),
                "val_examples": len(val_examples),
                "model": args.model,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
