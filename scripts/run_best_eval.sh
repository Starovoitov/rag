set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EMBEDDING_MODEL="${EMBEDDING_MODEL:-intfloat/e5-base-v2}"
if [[ -z "${EMBEDDING_MODEL// }" ]]; then
  EMBEDDING_MODEL="intfloat/e5-base-v2"
fi
RERANKER_MODEL="${RERANKER_MODEL:-models/reranker-failure-driven}"
if [[ ! -d "$RERANKER_MODEL" ]]; then
  RERANKER_MODEL="BAAI/bge-reranker-v2-m3"
fi

# BEST / STABLE CONFIG (source of truth)
# Keep these values aligned with the commands below.
# Parser:
# - min-tokens=100, max-tokens=140, overlap-ratio=0.12
# - min-output-chunk-tokens=120, max-output-chunk-tokens=650
# - max-chunks-per-url=10, max-chunks-per-category=45
# Evaluation dataset:
# - fuzzy-ratio=0.86, lexical-min-hits=1, max-chunk-ids=2
# - max-gt-url-share=0.27, target-multi-gt-share=0.35
# Retrieval / rerank:
# - retriever=hybrid, k-values=1,3,5,10,20, alpha=0.65
# - rerank=true, rerank-candidates=80, rerank-alpha=0.45
# - ce-calibration=zscore, ce-temperature=1.0
# - hybrid-candidate-multiplier=80, hybrid-rrf-k=80
# - multi-query=true, variants=3, multi-query-rrf-k=60
# - stratified-rerank-pool=true, hard-negative-semantic-floor=0.12
# - rerank prior weights: semantic=0.55, bm25=0.45
# - soft-recall-rescue=true, tail-k=30, bm25-depth=200
# - mmr-before-rerank=true, mmr-lambda=0.82, mmr-k=30
# - require-evidence=true, two-stage-rerank=true

python main.py cleanup_faiss --faiss-path data/faiss --drop-persist-directory

python main.py build_parser \
  --output data/rag_dataset.jsonl \
  --min-tokens 100 \
  --max-tokens 140 \
  --overlap-ratio 0.18 \
  --min-output-chunk-tokens 60 \
  --max-output-chunk-tokens 750 \
  --max-chunks-per-url 10 \
  --max-chunks-per-category 45 \
  --embedding-model "$EMBEDDING_MODEL" \

python -c "from embeddings.embedder import prepare_embedding_input, build_faiss_index; prepare_embedding_input('data/rag_dataset.jsonl', 'data/embeddings_input.jsonl'); build_faiss_index(input_jsonl='data/embeddings_input.jsonl', persist_directory='data/faiss', index_name='rag_chunks', model_name='$EMBEDDING_MODEL')"

python evaluation/dataset.py \
  --rag data/rag_dataset.jsonl \
  --eval data/evaluation.txt \
  --out data/evaluation_with_evidence.jsonl \
  --fuzzy-ratio 0.80 \
  --lexical-min-hits 1 \
  --max-chunk-ids 3 \
  --max-gt-url-share 0.27 \
  --target-multi-gt-share 0.30 \
  --keep-max-ids-for-multi 2

python main.py evaluation_runner \
  --dataset data/evaluation_with_evidence.jsonl \
  --retriever hybrid \
  --k-values 1,3,5,10,20 \
  --alpha 0.65 \
  --rerank \
  --rag-dataset data/rag_dataset.jsonl \
  --faiss-path data/faiss \
  --index rag_chunks \
  --embedding-model "$EMBEDDING_MODEL" \
  --rerank-candidates 150 \
  --rerank-alpha 0.35 \
  --ce-calibration zscore \
  --ce-temperature 1.0 \
  --hybrid-candidate-multiplier 100 \
  --hybrid-rrf-k 80 \
  --multi-query \
  --multi-query-variants 3 \
  --multi-query-rrf-k 60 \
  --stratified-rerank-pool \
  --hard-negative-semantic-floor 0.12 \
  --rerank-semantic-weight 0.75 \
  --rerank-bm25-weight 0.25 \
  --soft-recall-rescue \
  --soft-recall-rescue-tail-k 30 \
  --soft-recall-rescue-bm25-depth 300 \
  --reranker-model "$RERANKER_MODEL" \
  --mmr-before-rerank \
  --mmr-lambda 0.82 \
  --mmr-k 35 \
  --require-evidence \
  --two-stage-rerank \
  --out-json data/retrieval_report_best.json

python scripts/dataset_audit.py \
  --rag data/rag_dataset.jsonl \
  --eval data/evaluation_with_evidence.jsonl \
  --out data/dataset_audit_report.json


#  --multi-query-llm-timeout-seconds 20 \
#  --multi-query-llm-retries 0 \
#  --multi-query-llm-debug \


 # --no-semantic-fallback

   #--mmr-before-rerank \
  #--mmr-lambda 0.82 \
  #--mmr-k 35 \