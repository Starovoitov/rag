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

Reranking support:
- `reranking/cross_encoder.py` provides `CrossEncoderReranker`
- available in `demo_retrieval`, `evaluation_runner`, and `run_rag` via `--rerank`

## Install

```bash
poetry install
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

## Common workflows

### 1) Build parser dataset

```bash
python main.py build_parser --output data/rag_dataset.jsonl
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
python -c "from embeddings.embedder import prepare_embedding_input, build_faiss_index; prepare_embedding_input('data/rag_dataset.jsonl', 'data/embeddings_input.jsonl'); build_faiss_index(input_jsonl='data/embeddings_input.jsonl', persist_directory='data/faiss', index_name='rag_chunks')"
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
python evaluation/dataset.py --rag data/rag_dataset.jsonl --eval data/evaluation.txt --out data/evaluation_with_evidence.jsonl --fuzzy-ratio 0.78 --lexical-min-hits 1 --max-chunk-ids 3
```

Then run retrieval benchmark:

```bash
python main.py evaluation_runner --dataset data/evaluation_with_evidence.jsonl --retriever hybrid --k-values 1,3,5 --out-json data/retrieval_report.json
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

With cross-encoder reranking:

```bash
python main.py run_rag --question "What is RAG?" --provider openai --rerank --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 --rerank-candidates 20
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

Default outputs:
- report: `data/retrieval_report_best.json`
- pairwise training data (`reranker_pairwise_v1`): `data/reranker_train.jsonl`
- trained model (if `--train-reranker`): `models/reranker-failure-driven`

### 7) Cleanup FAISS

Delete one FAISS index:

```bash
python main.py cleanup_faiss --faiss-path data/faiss --index rag_chunks
```

Delete full FAISS directory:

```bash
python main.py cleanup_faiss --faiss-path data/faiss --drop-persist-directory
```

## Notes

- If a source fails to parse, a `source_error` record is still written.
- For semantic retrieval and RAG, `intfloat/e5-small-v2` is used by default.
- If network is restricted, model loading may require local cache/offline mode.
