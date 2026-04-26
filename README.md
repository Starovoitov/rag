# RAG Agent

RAG-focused project for:
- parsing and preparing a JSONL dataset,
- building local FAISS embeddings index,
- running retrieval demos,
- running retrieval evaluation,
- running end-to-end RAG generation.

## Unified entrypoint

Use the root CLI:

```bash
python main.py --help
```

Recommended daily command (full tuned pipeline + reranker training):

```bash
python main.py reranker_pipeline --train-reranker
```

Available commands:
- `build_parser` - build `data/rag_dataset.jsonl`
- `demo_retrieval` - run BM25 + semantic + hybrid retrieval demo
- `evaluation_runner` - run retrieval benchmark on evaluation dataset
- `reranker_pipeline` - run tuned retrieval eval + export failure-driven reranker dataset (+ optional train)
- `run_rag` - run full RAG query against selected LLM provider
- `cleanup_faiss` - remove FAISS index (optionally remove full FAISS directory)

Caching support:
- `caching/lru_ttl_cache.py` provides in-memory LRU + TTL cache primitives
- retrieval cache available in `evaluation_runner` / `reranker_pipeline`
- LLM cache available in `evaluation_runner`, `reranker_pipeline`, and `run_rag`

Reranking support:
- `reranking/cross_encoder.py` provides `CrossEncoderReranker`
- available in `demo_retrieval`, `evaluation_runner`, and `run_rag` via `--rerank`

## Install

```bash
poetry install
```

## Run tests

Run all unit tests (unittest discovery):

```bash
python -m unittest discover -s tests -v
```

Run a single test module:

```bash
python -m unittest tests.test_utils_common -v
```

## Architecture

Sources (URLs / GitHub / docs / community pages)
-> Scraper (`requests` + `trafilatura`)
-> Clean HTML to text
-> Normalize
-> Chunking (token-based, default 300-800 with 15% overlap)
-> Metadata enrichment
-> JSONL output
-> Optional embeddings + FAISS

## Path conventions

- `data/` - source inputs and core datasets (e.g. `data/rag_dataset.jsonl`, `data/evaluation_with_evidence.jsonl`, `data/faiss`)
- `artifacts/` - generated assets used by pipelines (e.g. `artifacts/datasets/reranker_train.jsonl`, `artifacts/models/reranker-failure-driven`)
- `experiments/` - run outputs for analysis (e.g. `experiments/results/retrieval_report_best.json`, `experiments/logs/*.jsonl`)
- `caching/` - reusable cache implementations (currently in-memory LRU + TTL)

## Root configs

- `sources.config.json` - parser source registry (maps to `SourceSpec` entries)
- `llm.config.json` - named LLM provider defaults (maps to `LLMConfig` base fields)
- Config type suggestion: keep both as JSON objects (typed + validated at runtime) for easy CLI overrides and versioning.

## Common workflows

### 1) Build parser dataset

```bash
python main.py build_parser --output data/rag_dataset.jsonl
```

Use custom source config:

```bash
python main.py build_parser --sources-config sources.config.json
```

Optional chunk config:

```bash
python main.py build_parser --min-tokens 300 --max-tokens 800 --overlap-ratio 0.15
```

Balanced dataset config (recommended to reduce source/topic skew):

```bash
python main.py build_parser \
  --output data/rag_dataset.jsonl \
  --min-tokens 150 \
  --max-tokens 300 \
  --overlap-ratio 0.25 \
  --min-output-chunk-tokens 120 \
  --max-output-chunk-tokens 650 \
  --max-chunks-per-url 12 \
  --max-chunks-per-category 45
```

### 2) Build embeddings + FAISS

```bash
python -c "from embeddings.embedder import prepare_embedding_input, build_faiss_index; prepare_embedding_input('data/rag_dataset.jsonl', 'data/embeddings_input.jsonl'); build_faiss_index(input_jsonl='data/embeddings_input.jsonl', persist_directory='data/faiss', index_name='.')"
```

### 3) Retrieval demo

```bash
python main.py demo_retrieval --query "what is rag" --top-k 5
```

With cross-encoder reranking:

```bash
python main.py demo_retrieval --query "what is rag" --top-k 5 --rerank --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 --rerank-candidates 20
```

### 4) Retrieval evaluation

Generate evaluation dataset with evidence links:

```bash
python main.py build_evaluation_dataset --rag data/rag_dataset.jsonl --eval evaluation/evaluation.json --out data/evaluation_with_evidence.jsonl --fuzzy-ratio 0.78 --lexical-min-hits 1 --max-chunk-ids 3
```

Then run retrieval benchmark:

```bash
python main.py evaluation_runner --dataset data/evaluation_with_evidence.jsonl --retriever hybrid --k-values 1,3,5 --out-json data/retrieval_report.json
```

With retrieval + query-expansion LLM caching:

```bash
python main.py evaluation_runner --dataset data/evaluation_with_evidence.jsonl --retriever hybrid --k-values 1,3,5 --retrieval-cache-enabled --llm-cache-enabled --out-json data/retrieval_report.json
```

Tune cache capacity/TTL:

```bash
python main.py evaluation_runner \
  --dataset data/evaluation_with_evidence.jsonl \
  --retriever hybrid \
  --k-values 1,3,5 \
  --retrieval-cache-enabled \
  --retrieval-cache-capacity 15000 \
  --retrieval-cache-ttl-seconds 600 \
  --llm-cache-enabled \
  --llm-cache-capacity 1024 \
  --llm-cache-ttl-seconds 900 \
  --out-json data/retrieval_report.json
```

Evaluate only questions that have non-empty `chunk_ids`:

```bash
python main.py evaluation_runner --dataset data/evaluation_with_evidence.jsonl --retriever hybrid --k-values 1,3,5 --require-evidence --out-json data/retrieval_report.json
```

With cross-encoder reranking:

```bash
python main.py evaluation_runner --dataset data/evaluation_with_evidence.jsonl --retriever hybrid --k-values 1,3,5 --rerank --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 --rerank-candidates 20 --out-json data/retrieval_report.json
```

### 5) Run end-to-end RAG

```bash
python main.py run_rag --question "What is RAG?" --provider openai
```

Use custom LLM provider config:

```bash
python main.py run_rag --question "What is RAG?" --provider openai --llm-config-path llm.config.json
```

With cross-encoder reranking:

```bash
python main.py run_rag --question "What is RAG?" --provider openai --rerank --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 --rerank-candidates 20
```

With LLM response caching (LRU + TTL):

```bash
python main.py run_rag --question "What is RAG?" --provider openai --llm-cache-enabled --llm-cache-capacity 512 --llm-cache-ttl-seconds 300
```

With advanced runtime logging:

```bash
python main.py run_rag \
  --question "What is RAG?" \
  --provider openai \
  --log-level INFO \
  --log-path experiments/logs/run_rag.log \
  --log-json
```

### 6) One-shot reranker pipeline

Run tuned retrieval evaluation and export hard-negative reranker training data in one command:

```bash
python main.py reranker_pipeline
```

Run full pipeline with in-loop reranker training:

```bash
python main.py reranker_pipeline --train-reranker
```

With custom LLM config for multi-query expansion:

```bash
python main.py reranker_pipeline --multi-query-llm-expansion --llm-config-path llm.config.json
```

Run with retrieval + LLM caches enabled:

```bash
python main.py reranker_pipeline --retrieval-cache-enabled --llm-cache-enabled
```

Default outputs:
- report: `experiments/results/retrieval_report_best.json`
- pairwise training data (`reranker_pairwise_v1`): `artifacts/datasets/reranker_train.jsonl`
- trained model (if `--train-reranker`): `artifacts/models/reranker-failure-driven`

Reranker dataset schema (`reranker_context_v1`):

```json
{
  "schema_version": "reranker_context_v1",
  "query": "How does RAG work in simple terms?",
  "positives": ["cce893f9c187c4e4"],
  "negatives": ["79eae0092b291ea5", "ff67889b4dab0ad5"],
  "weights": {
    "79eae0092b291ea5": 1.5,
    "ff67889b4dab0ad5": 1.11
  },
  "failure_bucket": "true_recall_failure",
  "source_miss_type": "both_hit"
}
```

Field notes:
- `positives`: ground-truth chunk ids for the query.
- `negatives`: hard negatives from retrieved top-k (non-GT only).
- `weights`: per-negative training weight (bucket-weighted and rank-aware).
- `failure_bucket` / `source_miss_type`: failure diagnostics retained for analysis and reweighting.

### 7) Cleanup FAISS

Delete one FAISS index:

```bash
python main.py cleanup_faiss --faiss-path data/faiss --index .
```

Delete full FAISS directory:

```bash
python main.py cleanup_faiss --faiss-path data/faiss --drop-persist-directory
```

## Notes

- If a source fails to parse, a `source_error` record is still written.
- For semantic retrieval and RAG, `intfloat/e5-base-v2` is used by default.
- If network is restricted, model loading may require local cache/offline mode.
- Cache behavior:
  - retrieval cache keys include query text and `top_k`,
  - LLM cache keys include model/provider, generation params, and prompts,
  - expired entries are cleaned automatically by TTL cleanup.
- Logging behavior:
  - stage-level `INFO` logs are emitted for major pipeline steps,
  - potential issues are emitted as `WARNING`,
  - failures are emitted as `ERROR` with exception traces,
  - logging controls: `--log-level`, `--log-path`, `--log-json`.
