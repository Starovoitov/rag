## Project Features Catalog

This document lists core project features by category with short practical descriptions.

## Unified CLI and Pipeline Entry Points

- `main.py` is the single CLI entry point for parser, retrieval, evaluation, RAG generation, FAISS maintenance, and reranker workflows.
- `scripts/run_best_eval.sh` is the stable end-to-end reproducible benchmark pipeline (cleanup -> parser -> FAISS -> eval dataset -> evaluation runner -> dataset audit).
- `reranker_pipeline` runs tuned retrieval evaluation, exports failure-driven training data, and can train a reranker in one pass.
- `build_faiss` provides a dedicated CLI for FAISS index build (including optional `--prepare-input` from `rag_dataset`).

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

- Dataset evaluation runs in two stages:
  - Ground-truth evidence construction (`build_evaluation_dataset`) from base eval blocks plus `rag_dataset` chunks.
  - Retrieval benchmark scoring (`evaluation_runner`) that compares ranked retrieved ids against expected evidence ids.
- Ground-truth resolution criteria (in order):
  - exact question match against QA map
  - fuzzy question match (`--fuzzy-ratio`)
  - optional semantic fallback (`--semantic-min-score`)
  - lexical fallback with keyword/phrase overlap (`--lexical-min-hits`, `--max-chunk-ids`)
- Ground-truth balancing criteria:
  - per-source evidence concentration cap (`--max-gt-url-share`)
  - multi-ground-truth share cap (`--target-multi-gt-share`, `--keep-max-ids-for-multi`)
- Filtering criterion at scoring time:
  - optional `--require-evidence` to score only samples with non-empty `expected_evidence.chunk_ids`
- Retrieval quality formulas (project metrics implementation):
## recall@k = |top_k(retrieved_doc_ids) intersection relevant_doc_ids| / |relevant_doc_ids|
## precision@k = relevant_hits_in_top_k / |top_k(retrieved_doc_ids)|
## hit_rate@k = 1.0 if any relevant doc appears in top_k, else 0.0
## reciprocal_rank = 1 / rank(first_relevant_doc), else 0.0 if no relevant doc retrieved
## mrr = mean(reciprocal_rank over all queries)
## dcg@k = sum over rank i in top_k of ((2^rel_i - 1) / log2(i + 1))
## ndcg@k = dcg@k / idcg@k
- Aggregation rule:
  - each metric is computed per query and then averaged across the evaluated sample set.
- JSON report export via `--out-json` includes metrics, sample counts, filtering stats, and per-query runs.

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
