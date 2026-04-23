#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EMBEDDING_MODEL="intfloat/e5-base-v2"

# Current stable config:
# - balanced parser dataset build
# - evaluation dataset:
#   - fuzzy-ratio=0.86, lexical-min-hits=1, max-chunk-ids=2
#   - no semantic fallback
#   - max-gt-url-share=0.25, target-multi-gt-share=0.35
# - retriever=hybrid, k-values=1,3,5,10,20, alpha=0.3
# - rerank=true, reranker=BAAI/bge-reranker-large
# - rerank-candidates=100, rerank-alpha=0.2
# - hybrid-candidate-multiplier=100, hybrid-rrf-k=10
# - require-evidence=true, two-stage-rerank=true, multi-query=true

python main.py cleanup_faiss --faiss-path data/faiss --drop-persist-directory

python main.py build_parser \
  --output data/rag_dataset.jsonl \
  --min-tokens 90 \
  --max-tokens 160 \
  --overlap-ratio 0.12 \
  --min-output-chunk-tokens 120 \
  --max-output-chunk-tokens 650 \
  --max-chunks-per-url 12 \
  --max-chunks-per-category 45 \
  --embedding-model "$EMBEDDING_MODEL" \

python -c "from embeddings.embedder import prepare_embedding_input, build_faiss_index; prepare_embedding_input('data/rag_dataset.jsonl', 'data/embeddings_input.jsonl'); build_faiss_index(input_jsonl='data/embeddings_input.jsonl', persist_directory='data/faiss', index_name='rag_chunks', model_name='$EMBEDDING_MODEL')"

python evaluation/dataset.py \
  --rag data/rag_dataset.jsonl \
  --eval data/evaluation.txt \
  --out data/evaluation_with_evidence.jsonl \
  --fuzzy-ratio 0.86 \
  --lexical-min-hits 1 \
  --max-chunk-ids 2 \
  --max-gt-url-share 0.27 \
  --target-multi-gt-share 0.35 \
  --keep-max-ids-for-multi 1 

python main.py evaluation_runner \
  --dataset data/evaluation_with_evidence.jsonl \
  --retriever hybrid \
  --k-values 1,3,5,10,20 \
  --alpha 0.65 \
  --rerank \
  --embedding-model "$EMBEDDING_MODEL" \
  --rerank-candidates 40 \
  --rerank-alpha 0.45 \
  --hybrid-candidate-multiplier 80 \
  --hybrid-rrf-k 80 \
  --multi-query \
  --multi-query-variants 3 \
  --multi-query-rrf-k 60 \
  --stratified-rerank-pool \
  --hard-negative-semantic-floor 0.12 \
  --rerank-semantic-weight 0.45 \
  --rerank-bm25-weight 0.35 \
  --rerank-rank-weight 0.20 \
  --soft-recall-rescue \
  --soft-recall-rescue-tail-k 20 \
  --soft-recall-rescue-bm25-depth 200 \
  --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --require-evidence \
  --two-stage-rerank \
  --out-json data/retrieval_report_best.json

python scripts/dataset_audit.py \
  --rag data/rag_dataset.jsonl \
  --eval data/evaluation_with_evidence.jsonl \
  --out data/dataset_audit_report.json


#BAAI/bge-reranker-large \
#  --require-evidence 
# hybrid \
# cross-encoder/ms-marco-MiniLM-L-6-v2
#   --multi-query \
#  --multi-query-variants 3 \
#  --multi-query-rrf-k 50 \

#  --hybrid-max-per-group 3 \
#  --two-stage-rerank \
#  --prefilter-candidates 45 \




 # --no-semantic-fallback