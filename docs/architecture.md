# RAG Agent Architecture

## Purpose

This project is an end-to-end Retrieval-Augmented Generation (RAG) system for:
- building a chunked knowledge corpus,
- indexing it with dense embeddings + FAISS,
- retrieving with BM25 / semantic / hybrid fusion,
- improving ranking with cross-encoder reranking,
- evaluating retrieval quality and failure modes,
- exporting failure-driven reranker training data,
- retraining reranker models,
- generating grounded answers with source citations.

---

## System Diagram

```mermaid
flowchart TD
    A[Sources / Ingestion Inputs] --> B[Parser + Chunking]
    B --> C[data/rag_dataset.jsonl]

    C --> D[BM25 Index]
    C --> E[Embedding Input + SentenceTransformer]
    E --> F[FAISS IndexFlatIP + store.json]

    D --> G[Retriever Layer]
    F --> G

    G --> H[Optional Multi-Query Expansion]
    H --> I[Optional RRF Across Query Variants]
    I --> J[Candidate Pool]

    J --> K[Optional Soft Recall Rescue BM25 Tail]
    K --> L[Optional MMR Before Rerank]
    L --> M[Optional Cross-Encoder Reranker]
    M --> N[Final Ranked Chunks]

    N --> O[Evaluation + Diagnostics]
    N --> P[RAG Prompt Builder + LLM]

    O --> Q[Failure Records + Buckets]
    Q --> R[Export reranker_context_v1 JSONL]
    R --> S[Train Cross-Encoder]
    S --> M
```

---

## Core Modules

- `parser/`  
  Cleans text and chunks documents with configurable token bounds and overlap.

- `embeddings/embedder.py` and `embeddings/faiss_store.py`  
  Builds dense vectors with `SentenceTransformer`, stores vectors in FAISS (`IndexFlatIP`) and metadata in sidecar JSON.

- `retrieval/bm25.py`  
  In-memory lexical BM25 ranking over raw chunks.

- `retrieval/semantic.py`  
  Cosine-based dense retrieval over precomputed embeddings.

- `retrieval/hybrid.py`  
  Hybrid rank fusion using weighted reciprocal-rank fusion (RRF), plus optional per-source diversity limits.

- `reranking/cross_encoder.py`  
  Cross-encoder reranking with score calibration (`minmax`, `softmax`, `zscore`) and fusion with base retrieval priors.

- `evaluation/metrics.py` and `evaluation/dataset.py`  
  Metrics (`hit_rate@k`, `precision@k`, `recall@k`, `ndcg@k`, `mrr`) and evidence-linked evaluation dataset generation.

- `generation/run_rag.py`, `generation/prompt.py`, `generation/llm.py`  
  Builds grounded prompts from top chunks and queries OpenAI-compatible providers (`openai`, `gigachat`, `ollama`, `qwen`).

- `caching/lru_ttl_cache.py`  
  Generic in-memory LRU + TTL cache with periodic cleanup, access counters, and hit/miss telemetry.

- `main.py`  
  Unified orchestrator for parsing, retrieval demo, evaluation, reranker pipeline, and RAG execution.

---

## Retrieval and Ranking Formulas

## 1) BM25

For term $t$ and document $d$:

$$
\text{IDF}(t) = \log\left(1 + \frac{N - df_t + 0.5}{df_t + 0.5}\right)

\text{BM25}(q,d) = \sum_{t \in q} \text{IDF}(t)\cdot
\frac{tf_{t,d}(k_1+1)}{tf_{t,d}+k_1\left(1-b+b\frac{|d|}{avgdl}\right)}
$$

Defaults: $k_1=1.5,\; b=0.75$.

## 2) Semantic similarity (cosine)

$$
\cos(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q}\cdot\mathbf{d}}{\|\mathbf{q}\|\|\mathbf{d}\|}
$$

Embeddings are normalized; FAISS uses inner product, so IP ~= cosine.

## 3) Hybrid RRF fusion

Per branch rank contribution:

$$
\text{RRF}(r)=\frac{1}{k_{rrf}+r}
$$

Combined hybrid score:

$$
S_{\text{hybrid}}(d)=\alpha\cdot\text{RRF}_{\text{semantic}}(d) + (1-\alpha)\cdot\text{RRF}_{\text{bm25}}(d)
$$

## 4) Multi-query RRF

If query variants produce ranks $r_i(d)$:

$$
S(d)=\sum_i \frac{1}{k_{rrf}+r_i(d)}
$$

## 5) MMR (diversification before rerank)

$$
\underset{d \in C \setminus S}{\arg\max}
\left[
\lambda\cdot\text{Rel}(q,d) - (1-\lambda)\cdot\max_{s \in S}\text{Sim}(d,s)
\right]
$$

## 6) Cross-encoder fusion

After CE calibration and base-score normalization:

$$
S_{\text{final}}(d)=\alpha_{ce}\cdot CE_{\text{norm}}(d)+(1-\alpha_{ce})\cdot Base_{\text{norm}}(d)
$$

Calibration options:
- `minmax`
- `softmax` with temperature $T$: $\text{softmax}(z/T)$
- `zscore` + sigmoid: $\sigma((z-\mu)/(\sigma T))$

---

## Evaluation Metrics

For a query with `TopK` and `Relevant`:

- Recall@k: $$\frac{|TopK \cap Relevant|}{|Relevant|}$$
- Precision@k: $$\frac{|TopK \cap Relevant|}{k}$$
- HitRate@k: $$\mathbb{1}[TopK \cap Relevant \neq \emptyset]$$
- Reciprocal Rank: $$RR = \frac{1}{\text{rank of first relevant}}$$
- MRR: average of RR across queries.
- DCG@k: $$\sum_{i=1}^{k}\frac{2^{rel_i}-1}{\log_2(i+1)}$$
- NDCG@k: $$\frac{DCG@k}{IDCG@k}$$
---

## Advanced Features

- Multi-query expansion:
  - rule-based layers (paraphrase + decomposition + entity/concept),
  - optional LLM structured expansions,
  - optional batched pre-generation for all queries in one upfront request.

- Failure analysis:
  - buckets: `near_miss`, `fragmentation`, `ranking_cutoff_failure`, `true_recall_failure`,
  - source miss attribution: `embedding_miss`, `bm25_miss`, `both_miss`, `both_hit`,
  - manual inspection sample export in report JSON.

- Failure-driven reranker training:
  - exports `reranker_context_v1` with positives, negatives, and per-negative weights,
  - rank-aware weighting and bucket-aware weighting for hard negatives,
  - in-loop training option from `main.py reranker_pipeline`.

- Caching subsystem:
  - retrieval-side cache for repeated query/top-k lookups in evaluation pipelines,
  - LLM response cache for query expansion and `run_rag`,
  - LRU eviction + TTL expiry + periodic cleanup to balance memory usage and freshness.

---

## Operational Entrypoints

Use:

```bash
python main.py --help
```

Primary commands:
- `build_parser`
- `demo_retrieval`
- `evaluation_runner`
- `reranker_pipeline`
- `run_rag`
- `cleanup_faiss`

Recommended full loop:

```bash
python main.py reranker_pipeline --train-reranker
```

