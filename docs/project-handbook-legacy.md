# Legacy Project Handbook

This document contains the previous root `README.md` content, moved to `docs/` to keep historical setup and workflow notes in one place.

## RAG Agent

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

## Install

```bash
poetry install
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Common workflows

### Build parser dataset

```bash
python main.py build_parser --output data/rag_dataset.jsonl
```

### Build embeddings + FAISS

```bash
python -c "from embeddings.embedder import prepare_embedding_input, build_faiss_index; prepare_embedding_input('data/rag_dataset.jsonl', 'data/embeddings_input.jsonl'); build_faiss_index(input_jsonl='data/embeddings_input.jsonl', persist_directory='data/faiss', index_name='.')"
```

### Retrieval evaluation

```bash
python main.py evaluation_runner --dataset data/evaluation_with_evidence.jsonl --retriever hybrid --k-values 1,3,5 --out-json data/retrieval_report.json
```

### Reranker pipeline

```bash
python main.py reranker_pipeline
python main.py reranker_pipeline --train-reranker
```

### Run RAG

```bash
python main.py run_rag --question "What is RAG?" --provider openai
```

### Cleanup FAISS

```bash
python main.py cleanup_faiss --faiss-path data/faiss --index .
python main.py cleanup_faiss --faiss-path data/faiss --drop-persist-directory
```

## Notes

- Default semantic model: `intfloat/e5-base-v2`
- Retrieval and LLM caches are supported by `evaluation_runner`, `reranker_pipeline`, and `run_rag`
- Logging controls: `--log-level`, `--log-path`, `--log-json`
