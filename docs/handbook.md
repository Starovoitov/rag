# Handbook

Primary command reference with practical variants.

## Setup

Install dependencies:

```bash
make install
```

Run tests:

```bash
poetry run python -m unittest discover -s tests -v
```

## Core Commands

Show CLI help:

```bash
python main.py --help
```

Build parser dataset:

```bash
python main.py build_parser --output data/rag_dataset.jsonl
```

Build FAISS index:

```bash
python main.py build_faiss \
  --prepare-input \
  --rag-dataset data/rag_dataset.jsonl \
  --input-jsonl data/embeddings_input.jsonl \
  --faiss-path data/faiss \
  --index .
```

Build evaluation dataset:

```bash
python main.py build_evaluation_dataset \
  --rag data/rag_dataset.jsonl \
  --eval evaluation/evaluation.json \
  --out data/evaluation_with_evidence.jsonl
```

Run RAG:

```bash
python main.py run_rag --question "What is RAG?" --provider openai
```

Cleanup FAISS index:

```bash
python main.py cleanup_faiss --faiss-path data/faiss --index .
```

## Evaluation Runner Variants

Dataset used below:

```bash
DATASET=data/evaluation_with_evidence.jsonl
```

### 1) Failure-driven reranker

```bash
python main.py evaluation_runner \
  --dataset "$DATASET" \
  --retriever hybrid \
  --k-values 1,3,5,10,20,30 \
  --rerank \
  --reranker-model models/reranker-failure-driven \
  --out-json experiments/results/retrieval_report_failure_driven.json
```

### 2) Reranker (not failure-driven)

```bash
python main.py evaluation_runner \
  --dataset "$DATASET" \
  --retriever hybrid \
  --k-values 1,3,5,10,20,30 \
  --rerank \
  --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --out-json experiments/results/retrieval_report_reranker_baseline.json
```

### 3) Without reranker

```bash
python main.py evaluation_runner \
  --dataset "$DATASET" \
  --retriever hybrid \
  --k-values 1,3,5,10,20,30 \
  --out-json experiments/results/retrieval_report_no_reranker.json
```

## Pipeline Shortcut

Run tuned full pipeline with in-loop reranker training:

```bash
python main.py reranker_pipeline --train-reranker
```
