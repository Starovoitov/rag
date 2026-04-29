# RAG FD

Minimal RAG toolkit for dataset prep, FAISS indexing, retrieval evaluation, and API-based runs.

## Quick Start

### Docker (backend + frontend)

```bash
docker compose up --build
```

Services:
- API: `http://localhost:8000`
- Frontend: `http://localhost:5173`

Optional custom ports:

```bash
EXTERNAL_PORT=4173 BACKEND_EXTERNAL_PORT=9000 docker compose up --build
```

GPU variant (NVIDIA runtime installed):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

### Local setup

Requirements:
- Python `3.12+`
- Poetry

```bash
make install
```

## Main CLI

List commands:

```bash
python main.py --help
```

Common pipeline:

```bash
python main.py reranker_pipeline --train-reranker
```

Core commands:
- `build_parser`
- `build_faiss`
- `build_evaluation_dataset`
- `evaluation_runner`
- `reranker_pipeline`
- `run_rag`
- `cleanup_faiss`

## API Server

```bash
poetry run uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

Docs:
- Swagger: `http://localhost:8000/docs`
- OpenAPI: `http://localhost:8000/openapi.json`

### Frontend (bare metal)

```bash
cd FE
npm install
npm run dev
```

Frontend default URL: `http://127.0.0.1:5173`  
Set backend URL in UI to: `http://127.0.0.1:8000`

## Latest Benchmark Report

Example of run with failure-driven reranker and as usual:
https://colab.research.google.com/drive/1Ovoo1aGeX_kdpxP1d814QwqqF1F3kXtd?usp=sharing

Runs: `hybrid` retriever, comparison of reranker variants  
Dataset: `data/evaluation_with_evidence.jsonl`  
Samples: `81`

| Metric | Failure-driven reranker | Reranker (not failure-driven) |
|---|---:|---:|
| hit_rate@1 | 0.4691 | 0.4444 |
| hit_rate@3 | 0.7284 | 0.6790 |
| hit_rate@5 | 0.8519 | 0.7901 |
| hit_rate@10 | 0.9506 | 0.8889 |
| hit_rate@20 | 1.0000 | 0.9506 |
| hit_rate@30 | 1.0000 | 0.9877 |
| mrr | 0.6346 | 0.5957 |
| ndcg@1 | 0.4691 | 0.4444 |
| ndcg@3 | 0.3754 | 0.3608 |
| ndcg@5 | 0.4353 | 0.4076 |
| ndcg@10 | 0.5227 | 0.4869 |
| ndcg@20 | 0.5819 | 0.5401 |
| ndcg@30 | 0.6044 | 0.5658 |
| precision@1 | 0.4691 | 0.4444 |
| precision@3 | 0.3457 | 0.3374 |
| precision@5 | 0.2938 | 0.2716 |
| precision@10 | 0.2099 | 0.1938 |
| precision@20 | 0.1309 | 0.1204 |
| precision@30 | 0.0951 | 0.0893 |
| recall@1 | 0.1451 | 0.1389 |
| recall@3 | 0.3189 | 0.3138 |
| recall@5 | 0.4516 | 0.4208 |
| recall@10 | 0.6409 | 0.5905 |
| recall@20 | 0.8004 | 0.7356 |
| recall@30 | 0.8724 | 0.8169 |

### Failure Buckets

| Bucket | Count |
|---|---:|
| near_miss | 0 |
| fragmentation | 0 |
| ranking_cutoff_failure | 1 |
| true_recall_failure | 0 |

### Failure Source Miss

| Source | Count |
|---|---:|
| embedding_miss | 0 |
| bm25_miss | 0 |
| both_miss | 0 |
| both_hit | 1 |

### Failure Bucket x Source Miss

| Bucket | embedding_miss | bm25_miss | both_miss | both_hit |
|---|---:|---:|---:|---:|
| near_miss | 0 | 0 | 0 | 0 |
| fragmentation | 0 | 0 | 0 | 0 |
| ranking_cutoff_failure | 0 | 0 | 0 | 1 |
| true_recall_failure | 0 | 0 | 0 | 0 |

Failed queries for manual inspection: `1`.

Detailed evaluation and comparison data is documented in `docs/architecture.md` and related docs.

## Quality Checks

```bash
make lint
make fix
poetry run python -m unittest discover -s tests -v
```

## More Docs

- Development notes: `docs/development.md`
- Architecture: `docs/architecture.md`
- Feature catalog: `docs/features-catalog.md`
- Handbook: `docs/handbook.md`
