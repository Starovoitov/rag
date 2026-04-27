# RAG Agent

## Installation and Tests

### Requirements

- Python `3.12+` (recommended: `3.12.x`)
- Poetry (`poetry` command available in shell)

### Install dependencies

```bash
poetry install
```

If needed, select Python 3.12 explicitly for this project:

```bash
poetry env use python3.12
poetry install
```

### Run tests

Run all tests:

```bash
poetry run python -m unittest discover -s tests -v
```

Run one test module:

```bash
poetry run python -m unittest tests.test_utils_common -v
```

Main entry point for a reproducible retrieval-evaluation workflow with optional failure-driven reranker training.

## What `scripts/run_best_eval.sh` is

`scripts/run_best_eval.sh` is the repository's "best/stable" end-to-end evaluation script.  
It rebuilds the retrieval artifacts from scratch, runs the benchmark with tuned hybrid+rereank settings, and produces audit outputs.

Pipeline stages:
1. Clean FAISS storage (`cleanup_faiss`).
2. Build parser dataset (`build_parser`).
3. Build FAISS index from selected embedding model.
4. Build evaluation dataset with evidence links.
5. Run retrieval benchmark (`evaluation_runner`) with tuned settings.
6. Run dataset audit (`dataset_audit`).

Primary outputs:
- `data/rag_dataset.jsonl`
- `data/evaluation_with_evidence.jsonl`
- `experiments/results/retrieval_report_best.json`
- `data/dataset_audit_report.json`

## Main commands

Show all CLI commands:

```bash
python main.py --help
```

Build parser dataset:

```bash
python main.py build_parser --output data/rag_dataset.jsonl --embedding-model intfloat/e5-base-v2
```

Build FAISS index:

```bash
python -c "from embeddings.embedder import prepare_embedding_input, build_faiss_index; prepare_embedding_input('data/rag_dataset.jsonl', 'data/embeddings_input.jsonl'); build_faiss_index(input_jsonl='data/embeddings_input.jsonl', persist_directory='data/faiss', index_name='.', model_name='intfloat/e5-base-v2')"
```

Build evaluation dataset:

```bash
python main.py build_evaluation_dataset --rag data/rag_dataset.jsonl --eval evaluation/evaluation.json --out data/evaluation_with_evidence.jsonl
```

Run benchmark:

```bash
python main.py evaluation_runner --dataset data/evaluation_with_evidence.jsonl --retriever hybrid --k-values 1,3,5,10,20,30 --rerank --reranker-model models/reranker-failure-driven --out-json experiments/results/retrieval_report_best.json
```

Run the stable all-in-one script:

```bash
bash scripts/run_best_eval.sh
```

## Training `reranker-failure-driven`

Use `reranker_pipeline` to generate failure-driven training pairs and optionally train a new cross-encoder reranker.

Minimal training command:

```bash
python main.py reranker_pipeline \
  --train-reranker \
  --train-reranker-model cross-encoder/ms-marco-MiniLM-L-12-v2 \
  --train-reranker-out-dir models/reranker-failure-driven
```

What this training does:
- runs retrieval evaluation in a tuned configuration;
- collects failure-focused hard negatives from misses/near-misses;
- exports reranker training dataset;
- fine-tunes a cross-encoder to better separate relevant and non-relevant chunks for this corpus.

Important requirement:
- `--embedding-model` used at evaluation time must match the model used when building FAISS embeddings, otherwise semantic cosine scoring fails due to vector dimension mismatch.

## Legacy README moved

Previous root README content was moved to:
- `docs/project-handbook-legacy.md`


## Notes

- Default semantic model: `intfloat/e5-base-v2`
- Retrieval and LLM caches are supported by `evaluation_runner`, `reranker_pipeline`, and `run_rag`
- Logging controls: `--log-level`, `--log-path`, `--log-json`


## Latest benchmark report

```text
Retrieval benchmark report
- dataset: data/evaluation_with_evidence.jsonl
- retriever: hybrid
- samples: 81
- hit_rate@1: 0.4815
- hit_rate@10: 0.9383
- hit_rate@20: 1.0000
- hit_rate@3: 0.7654
- hit_rate@30: 1.0000
- hit_rate@5: 0.8395
- mrr: 0.6410
- ndcg@1: 0.4815
- ndcg@10: 0.5323
- ndcg@20: 0.5989
- ndcg@3: 0.3899
- ndcg@30: 0.6140
- ndcg@5: 0.4413
- precision@1: 0.4815
- precision@10: 0.2123
- precision@20: 0.1352
- precision@3: 0.3621
- precision@30: 0.0955
- precision@5: 0.2963
- recall@1: 0.1502
- recall@10: 0.6492
- recall@20: 0.8282
- recall@3: 0.3344
- recall@30: 0.8765
- recall@5: 0.4547
Saved JSON report to experiments/results/retrieval_report_best.json
2026-04-27 14:19:35 | INFO | rag.evaluation_runner | saved json report: experiments/results/retrieval_report_best.json
2026-04-27 14:19:35 | INFO | rag.evaluation_runner | evaluation runner completed successfully
{
  "inputs": {
    "rag_path": "data/rag_dataset.jsonl",
    "eval_path": "data/evaluation_with_evidence.jsonl"
  },
  "rag": {
    "raw_chunks": 224,
    "token_count_min": 105,
    "token_count_max": 231,
    "token_count_avg": 143.3125,
    "overlap_tokens_unique": [
      0,
      25
    ],
    "duplicate_chunk_text_entries": 0,
    "top_categories": [
      [
        "advanced_rag_ideas",
        44
      ],
      [
        "production_ops_safety",
        32
      ],
      [
        "retrieval_indexing_core",
        30
      ],
      [
        "practical_repositories",
        25
      ],
      [
        "practical_tutorials",
        20
      ],
      [
        "embedding_theory",
        18
      ],
      [
        "evaluation_metrics",
        18
      ],
      [
        "basic_theory",
        15
      ],
      [
        "embeddings_vector_db_core",
        10
      ],
      [
        "data_quality_chunking",
        10
      ]
    ],
    "top_urls": [
      [
        "https://huggingface.co/learn/cookbook/advanced_rag",
        10
      ],
      [
        "https://huggingface.co/learn/cookbook/rag_zephyr_langchain",
        10
      ],
      [
        "https://langchain-tutorials.github.io/langchain-rag-tutorial-2026/",
        10
      ],
      [
        "https://github.com/run-llama/llama_index",
        10
      ],
      [
        "https://github.com/huggingface/transformers",
        10
      ],
      [
        "https://www.sbert.net/",
        10
      ],
      [
        "https://python.langchain.com/docs/tutorials/rag/",
        10
      ],
      [
        "https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide",
        10
      ],
      [
        "https://www.evidentlyai.com/llm-guide/rag-evaluation",
        10
      ],
      [
        "https://huggingface.co/learn/cookbook/en/rag_evaluation",
        10
      ]
    ]
  },
  "evaluation": {
    "rows_total": 81,
    "evidence_count_distribution": {
      "3": 58,
      "4": 23
    },
    "queries_with_multi_gt": 81,
    "queries_with_single_gt": 0,
    "queries_with_no_gt": 0,
    "resolution_method_distribution": {
      "semantic": 77,
      "fuzzy": 3,
      "exact": 1
    },
    "unknown_chunk_refs": 0,
    "top_gt_chunks": [
      [
        "e81b1c1522fd42fc",
        26
      ],
      [
        "97ab424a10b4e447",
        17
      ],
      [
        "118660371fc53380",
        12
      ],
      [
        "556b958dcf032309",
        10
      ],
      [
        "3471094e31bec51f",
        9
      ],
      [
        "091f7e04586ecc88",
        8
      ],
      [
        "52415bc308031836",
        8
      ],
      [
        "32becd10812c4a6d",
        8
      ],
      [
        "3a79f41179f17992",
        8
      ],
      [
        "ecd91504f76e2141",
        7
      ]
    ],
    "top_gt_urls": [
      [
        "https://www.evidentlyai.com/llm-guide/rag-evaluation",
        52
      ],
      [
        "https://airbyte.com/agentic-data/graph-rag-vs-vector-rag",
        39
      ],
      [
        "https://huggingface.co/learn/cookbook/advanced_rag",
        28
      ],
      [
        "https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide",
        27
      ],
      [
        "https://www.promptfoo.dev/docs/guides/evaluate-rag/",
        17
      ],
      [
        "seed://multi-hop-retrieval",
        15
      ],
      [
        "https://docs.llamaindex.ai/en/stable/module_guides/evaluating/",
        15
      ],
      [
        "https://huggingface.co/learn/cookbook/rag_zephyr_langchain",
        13
      ],
      [
        "https://python.langchain.com/docs/tutorials/rag/",
        9
      ],
      [
        "https://www.ibm.com/think/topics/retrieval-augmented-generation",
        9
      ]
    ],
    "top1_gt_chunk_share": 0.09774436090225563,
    "top10_gt_chunk_share": 0.424812030075188,
    "top1_gt_url_share": 0.19548872180451127
  },
  "quality_score": 0.7106111142517949
}
```
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
