## Project Features Catalog

This document lists core project features by category with short practical descriptions.

## Unified CLI and Pipeline Entry Points

- `main.py` is the single CLI entry point for parser, retrieval, evaluation, RAG generation, FAISS maintenance, and reranker workflows.
- `scripts/run_best_eval.sh` is the stable end-to-end reproducible benchmark pipeline (cleanup -> parser -> FAISS -> eval dataset -> evaluation runner -> dataset audit).
- `reranker_pipeline` runs tuned retrieval evaluation, exports failure-driven training data, and can train a reranker in one pass.

## Data Ingestion and Parser Features

- Source loading from configured registries (for URLs/repos/docs feeds).
- Text normalization and chunking with token-window controls.
- Configurable chunk constraints:
  - `--min-tokens`, `--max-tokens`, `--overlap-ratio`
  - `--min-output-chunk-tokens`, `--max-output-chunk-tokens`
  - `--max-chunks-per-url`, `--max-chunks-per-category`
- Parser output dataset in JSONL (`data/rag_dataset.jsonl`) with metadata used by retrieval and evaluation.

## Embeddings and FAISS Management

- Embedding generation via sentence-transformer models.
- FAISS index build from JSONL embedding input.
- FAISS persistence supports configurable location:
  - `--faiss-path`
  - `--index`
- FAISS cleanup commands:
  - remove one index
  - drop full persist directory
- Legacy index migration logic for older index layouts.
- Important consistency rule: query embedding model and indexed document embedding model must match in vector dimension.

## Retrieval Features

- Retriever modes:
  - semantic
  - bm25
  - hybrid
- Hybrid retrieval supports weighted fusion and rank aggregation.
- Tunable retrieval controls:
  - `--k-values`
  - `--alpha`
  - `--hybrid-candidate-multiplier`
  - `--hybrid-rrf-k`
- Evaluation-time evidence filtering with `--require-evidence` (when used).

## Reranking Features

- Cross-encoder reranking is optional and enabled via `--rerank`.
- Custom reranker model selection via `--reranker-model`:
  - HF model ids
  - local model directories (e.g. `models/reranker-failure-driven`)
- Candidate pool size control via `--rerank-candidates`.
- Blending controls for rerank prior contributions:
  - `--rerank-alpha`
  - `--rerank-semantic-weight`
  - `--rerank-bm25-weight`
- Calibration controls:
  - `--ce-calibration`
  - `--ce-temperature`

## Failure-Driven Reranker Training

- Dataset export from retrieval failures and near-misses.
- Hard-negative sampling from non-ground-truth top-ranked chunks.
- Per-negative weighting to emphasize informative mistakes.
- In-loop training support from `reranker_pipeline --train-reranker`.
- Training controls:
  - `--train-reranker-model`
  - `--train-reranker-out-dir`
  - `--train-reranker-epochs`
  - `--train-reranker-batch-size`
  - `--train-reranker-warmup-steps`
  - `--train-reranker-val-ratio`

## Evaluation Dataset and Metrics

- Evaluation dataset build from base eval file + RAG chunks.
- Matching / evidence-resolution settings:
  - `--fuzzy-ratio`
  - `--lexical-min-hits`
  - `--max-chunk-ids`
  - `--max-gt-url-share`
  - `--target-multi-gt-share`
  - `--keep-max-ids-for-multi`
- Retrieval metrics support includes:
  - hit rate at K
  - recall/precision at K
  - MRR
  - nDCG at K
- JSON report export via `--out-json`.

## RAG Generation Features

- End-to-end question answering on retrieved context chunks.
- Pluggable LLM provider configuration (`llm.config.json`).
- Supports reranking before final context selection.
- Runtime controls include provider/model/timeouts and related generation knobs.

## Caching Features

- Retrieval cache available for repeated query/top-k combinations:
  - `--retrieval-cache-enabled`
  - `--retrieval-cache-capacity`
  - `--retrieval-cache-ttl-seconds`
- LLM cache available for repeated generation/query expansion requests:
  - `--llm-cache-enabled`
  - `--llm-cache-capacity`
  - `--llm-cache-ttl-seconds`
- Cache implementation uses LRU + TTL semantics.

## Logging and Observability

- Configurable runtime logging level.
- Optional log file output and structured JSON logging.
- Progress logging for evaluation loops and major pipeline stages.
- Audit command (`dataset_audit`) generates quality diagnostics over RAG/eval datasets.

## Configuration Assets

- `sources.config.json`: source registry and ingestion configuration.
- `llm.config.json`: default provider/model settings for LLM calls.
- Script-level env overrides used by stable pipeline:
  - `EMBEDDING_MODEL`
  - `FAISS_PATH`
  - `FAISS_INDEX_NAME`
  - `RERANKER_MODEL`

## Artifacts and Output Layout

- Core datasets:
  - `data/rag_dataset.jsonl`
  - `data/evaluation_with_evidence.jsonl`
- Retrieval reports:
  - `experiments/results/retrieval_report_best.json`
- Reranker artifacts:
  - training dataset exports in `artifacts/datasets/`
  - trained models in `models/` or `artifacts/models/` (depending on command flags)
