# RAG Parser Pipeline

Simple pipeline for collecting RAG-focused data from URLs/GitHub/docs/community pages and exporting a single JSONL dataset.

## Architecture

URLs / GitHub / Docs  
-> Scraper (`requests` + `trafilatura`)  
-> Clean HTML to text  
-> Normalize (remove common noise)  
-> Chunking (token-based, default 300-800 with 15% overlap)  
-> Metadata enrichment  
-> JSONL output  
-> Optional embeddings input export

## Data types per source

1. **RAW CHUNKS**  
   - 300-800 tokens  
   - overlap 10-20% (default 15%)
2. **Q/A PAIRS**  
   - simple explain-like answers generated from chunk summary + priority topics
3. **EDGE CASES**  
   - bad chunking example  
   - hallucination example  
   - wrong retrieval example

## Install

```bash
poetry install
```

## Run parser

```bash
poetry run rag-parser --output parser/output/rag_dataset.jsonl
```

Optional chunk config:

```bash
poetry run rag-parser --min-tokens 300 --max-tokens 800 --overlap-ratio 0.15
```

## Optional embeddings input

```python
from parser.embeddings import prepare_embedding_input
prepare_embedding_input(
    "parser/output/rag_dataset.jsonl",
    "parser/output/embeddings_input.jsonl",
)
```

This prepares records for later embedding generation and vector DB ingestion.

If you want to run this helper from Poetry:

```bash
poetry run python -c "from parser.embeddings import prepare_embedding_input; prepare_embedding_input()"
```

## Notes

- If a source fails to parse, a `source_error` record is still written to the dataset.
- Current tokenizer is lightweight regex-based (no external tokenizer required).
- You can replace QA/edge-case generation with LLM-based enrichment later.

