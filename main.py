#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from reranking.failure_driven import (
    build_reranker_training_contexts_from_failures,
    build_stratified_rerank_pool,
    classify_failure,
    inject_bm25_tail_candidates,
    mmr_select_candidates,
    prefilter_rerank_candidates,
    rrf_fuse_doc_ids,
    source_miss_type,
    train_reranker_from_contexts_jsonl,
)
from utils.cli_config import (
    apply_config_defaults,
    load_cli_defaults,
    validate_required_command_params,
)
from utils.common import min_max_normalize
from utils.logger import configure_runtime_logger
from utils.query_manipulation import (
    build_query_variants_with_debug,
    llm_structured_query_expansion_batch,
)

if TYPE_CHECKING:
    pass


DEFAULT_EMBEDDING_MODEL = "intfloat/e5-base-v2"
DEFAULT_LLM_CONFIG_PATH = "llm.config.json"
DEFAULT_CLI_PARAMS_CONFIG = "cli.defaults.json"
REQUIRED_COMMAND_PARAMS: dict[str, tuple[str, ...]] = {
    "run_rag": ("question",),
    "run_experiments": ("question",),
}


def cmd_build_parser(args: argparse.Namespace) -> None:
    # Command-scoped import avoids loading parser stack for unrelated commands.
    from parser.pipeline import run_pipeline

    stats = run_pipeline(
        output_path=args.output,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        overlap_ratio=args.overlap_ratio,
        min_output_chunk_tokens=args.min_output_chunk_tokens,
        max_output_chunk_tokens=args.max_output_chunk_tokens,
        max_chunks_per_url=args.max_chunks_per_url,
        max_chunks_per_category=args.max_chunks_per_category,
        sources_config=args.sources_config,
        chunker_mode=args.chunker_mode,
        near_duplicate_jaccard=args.near_duplicate_jaccard,
        log_level=args.log_level,
        log_path=args.log_path,
        log_json=args.log_json,
    )
    stats["embedding_model"] = args.embedding_model
    print(json.dumps(stats, indent=2))


def cmd_demo_retrieval(args: argparse.Namespace) -> None:
    # Command-scoped import keeps startup lightweight.
    from demo_retrieval import run_demo

    run_demo(
        query=args.query,
        top_k=args.top_k,
        model_name=args.model,
        dataset_path=args.dataset,
        faiss_path=args.faiss_path,
        index_name=args.index,
        rerank=args.rerank,
        reranker_model=args.reranker_model,
        rerank_candidates=args.rerank_candidates,
    )


def cmd_evaluation_runner(args: argparse.Namespace) -> None:
    # Command-scoped imports keep module import side effects localized.
    from evaluation.dataset import load_eval_samples
    from evaluation.metrics import RetrievalResult, evaluate_retrieval
    from evaluation.runner import QueryRun, build_retriever, parse_k_values
    from ingestion.loaders import load_bm25_documents_from_dataset
    from retrieval.semantic import search_semantic
    from utils.embedding_format import format_query_for_embedding

    logger = configure_runtime_logger(
        "rag.evaluation_runner",
        level=args.log_level,
        log_path=args.log_path,
        json_logs=args.log_json,
    )
    logger.info("starting evaluation runner")
    samples = load_eval_samples(Path(args.dataset))
    if not samples:
        logger.error("no evaluation samples found: %s", args.dataset)
        raise ValueError(f"No samples found in dataset: {args.dataset}")
    total_samples_before_filter = len(samples)
    logger.info("loaded evaluation samples: count=%s", total_samples_before_filter)
    if args.require_evidence:
        samples = [sample for sample in samples if sample.relevant_docs]
    filtered_out_samples = total_samples_before_filter - len(samples)
    if filtered_out_samples > 0:
        logger.warning("filtered out samples without evidence: count=%s", filtered_out_samples)
    if not samples:
        logger.error("no samples left after filtering")
        raise ValueError(
            "No samples left after filtering. "
            "Try running without --require-evidence or regenerate dataset with more evidence links."
        )

    k_values = parse_k_values(args.k_values)
    max_k = max(k_values)
    logger.info(
        "building retriever: mode=%s max_k=%s retrieval_cache_enabled=%s",
        args.retriever,
        max_k,
        args.retrieval_cache_enabled,
    )
    retriever = build_retriever(
        args.retriever,
        rag_dataset_path=args.rag_dataset,
        faiss_path=args.faiss_path,
        index_name=args.index,
        embedding_model=args.embedding_model,
        alpha=args.alpha,
        hybrid_candidate_multiplier=args.hybrid_candidate_multiplier,
        hybrid_max_per_group=args.hybrid_max_per_group,
        hybrid_rrf_k=args.hybrid_rrf_k,
        cache_enabled=args.retrieval_cache_enabled,
        cache_capacity=args.retrieval_cache_capacity,
        cache_ttl_seconds=args.retrieval_cache_ttl_seconds,
    )
    semantic_embedding_map: dict[str, list[float]] = {}
    if args.retriever == "hybrid":
        semantic_obj = getattr(retriever, "semantic", None)
        documents = getattr(semantic_obj, "documents", None)
        if documents:
            semantic_embedding_map = {doc.doc_id: doc.embedding for doc in documents}
    doc_text_map = {
        item["id"]: item["text"] for item in load_bm25_documents_from_dataset(args.rag_dataset)
    }
    logger.info("loaded bm25 text map: count=%s", len(doc_text_map))
    reranker = None
    rerank_candidate_cls = None
    if args.rerank:
        # Heavy reranker deps are loaded only when reranking is enabled.
        from reranking.cross_encoder import CrossEncoderReranker, RerankCandidate

        logger.info("initializing reranker: model=%s", args.reranker_model)
        reranker = CrossEncoderReranker(model_name=args.reranker_model)
        rerank_candidate_cls = RerankCandidate

    query_runs: list[QueryRun] = []
    metric_inputs: list[RetrievalResult] = []
    llm_expansion_stats = {
        "requested_queries": 0,
        "generated_queries": 0,
        "fallback_queries": 0,
    }
    llm_expansion_examples: list[dict[str, object]] = []
    failure_records: list[dict[str, object]] = []
    miss_type_counts: Counter[str] = Counter()
    failure_bucket_source_counts: dict[str, Counter[str]] = {}
    llm_expansion_cache: dict[str, tuple[list[str], list[str], list[str]]] = {}
    if args.multi_query and args.multi_query_llm_expansion:
        llm_expansion_cache = llm_structured_query_expansion_batch(
            [sample.query for sample in samples],
            provider=args.multi_query_llm_provider,
            model=args.multi_query_llm_model,
            api_base=args.multi_query_llm_api_base,
            api_key=args.multi_query_llm_api_key,
            timeout_seconds=args.multi_query_llm_timeout_seconds,
            retries=args.multi_query_llm_retries,
            llm_config_path=args.llm_config_path,
            cache_enabled=args.llm_cache_enabled,
            cache_capacity=args.llm_cache_capacity,
            cache_ttl_seconds=args.llm_cache_ttl_seconds,
        )

    total_samples = len(samples)
    try:
        from tqdm import tqdm  # type: ignore

        sample_iter = tqdm(samples, total=total_samples, desc="evaluation_runner", unit="sample")
        logger.info("tqdm progress bar enabled for evaluation loop")
    except ImportError:
        sample_iter = samples
        logger.warning("tqdm is not available; falling back to plain loop progress logging")

    progress_log_step = max(1, total_samples // 20)  # ~5% increments
    for sample_idx, sample in enumerate(sample_iter, start=1):
        if sample_idx == 1 or sample_idx % progress_log_step == 0 or sample_idx == total_samples:
            logger.info("evaluation progress: %s/%s", sample_idx, total_samples)
        retrieve_k = max(max_k, args.rerank_candidates) if args.rerank else max_k
        semantic_branch_doc_ids: list[str] | None = None
        bm25_branch_doc_ids: list[str] | None = None
        semantic_score_map: dict[str, float] = {}
        bm25_score_map: dict[str, float] = {}
        query_embedding_for_mmr: list[float] | None = None
        if args.retriever == "hybrid":
            source_probe_k = max(retrieve_k, args.soft_recall_rescue_bm25_depth)
            semantic_obj = getattr(retriever, "semantic", None)
            bm25_obj = getattr(retriever, "bm25", None)
            if (
                semantic_obj is not None
                and hasattr(semantic_obj, "search")
                and hasattr(semantic_obj, "embedder")
            ):
                query_embedding = semantic_obj.embedder.encode(
                    [format_query_for_embedding(sample.query, semantic_obj.embedding_model)],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )[0].tolist()
                query_embedding_for_mmr = query_embedding
                semantic_hits = search_semantic(
                    query_embedding, semantic_obj.documents, top_k=source_probe_k
                )
                semantic_branch_doc_ids = [item.doc_id for item in semantic_hits]
                semantic_score_map = {item.doc_id: float(item.score) for item in semantic_hits}
            if bm25_obj is not None and hasattr(bm25_obj, "index"):
                bm25_hits = bm25_obj.index.search(sample.query, top_k=source_probe_k)
                bm25_branch_doc_ids = [item.doc_id for item in bm25_hits]
                bm25_score_map = {item.doc_id: float(item.score) for item in bm25_hits}
        if args.multi_query:
            query_variants, llm_debug = build_query_variants_with_debug(
                query=sample.query,
                max_variants=args.multi_query_variants,
                use_llm_structured_expansion=args.multi_query_llm_expansion,
                llm_provider=args.multi_query_llm_provider,
                llm_model=args.multi_query_llm_model,
                llm_api_base=args.multi_query_llm_api_base,
                llm_api_key=args.multi_query_llm_api_key,
                llm_timeout_seconds=args.multi_query_llm_timeout_seconds,
                llm_retries=args.multi_query_llm_retries,
                llm_config_path=args.llm_config_path,
                llm_cache_enabled=args.llm_cache_enabled,
                llm_cache_capacity=args.llm_cache_capacity,
                llm_cache_ttl_seconds=args.llm_cache_ttl_seconds,
                llm_precomputed=llm_expansion_cache.get(sample.query),
            )
            if args.multi_query_llm_debug and llm_debug.get("llm_requested"):
                llm_expansion_stats["requested_queries"] += 1
                if int(llm_debug.get("llm_generated_count", 0)) > 0:
                    llm_expansion_stats["generated_queries"] += 1
                    if len(llm_expansion_examples) < 5:
                        llm_expansion_examples.append(
                            {
                                "query": sample.query,
                                "generated_preview": llm_debug.get("llm_generated_preview", []),
                            }
                        )
                if bool(llm_debug.get("fallback_used", False)):
                    llm_expansion_stats["fallback_queries"] += 1
            per_query_results = [
                retriever.search(query, top_k=retrieve_k) for query in query_variants if query
            ]
            retrieved = rrf_fuse_doc_ids(
                per_query_results, top_k=retrieve_k, rrf_k=args.multi_query_rrf_k
            )
        else:
            retrieved = retriever.search(sample.query, top_k=retrieve_k)
        if reranker is not None:
            if args.stratified_rerank_pool and args.retriever == "hybrid":
                retrieved = build_stratified_rerank_pool(
                    hybrid_doc_ids=retrieved,
                    semantic_doc_ids=semantic_branch_doc_ids,
                    bm25_doc_ids=bm25_branch_doc_ids,
                    limit=retrieve_k,
                )
            if args.soft_recall_rescue and args.retriever == "hybrid":
                retrieved = inject_bm25_tail_candidates(
                    query=sample.query,
                    merged_doc_ids=retrieved,
                    retriever=retriever,
                    bm25_search_depth=args.soft_recall_rescue_bm25_depth,
                    rescue_tail_k=args.soft_recall_rescue_tail_k,
                )
            if (
                args.mmr_before_rerank
                and query_embedding_for_mmr is not None
                and semantic_embedding_map
                and args.retriever == "hybrid"
            ):
                mmr_k = args.mmr_k if args.mmr_k > 0 else retrieve_k
                mmr_k = min(mmr_k, retrieve_k)
                diversity_threshold = (
                    args.mmr_diversity_threshold if args.mmr_diversity_threshold > 0 else None
                )
                retrieved = mmr_select_candidates(
                    candidate_doc_ids=retrieved,
                    query_embedding=query_embedding_for_mmr,
                    doc_embeddings=semantic_embedding_map,
                    lambda_=args.mmr_lambda,
                    max_k=mmr_k,
                    diversity_threshold=diversity_threshold,
                )
            sem_norm = min_max_normalize(semantic_score_map)
            bm25_norm = min_max_normalize(bm25_score_map)
            rerank_input = [
                rerank_candidate_cls(
                    doc_id=doc_id,
                    text=doc_text_map.get(doc_id, ""),
                    score=(
                        (args.rerank_semantic_weight * sem_norm.get(doc_id, 0.0))
                        + (args.rerank_bm25_weight * bm25_norm.get(doc_id, 0.0))
                    ),
                    metadata={
                        "semantic_norm": sem_norm.get(doc_id, 0.0),
                        "bm25_norm": bm25_norm.get(doc_id, 0.0),
                    },
                )
                for rank, doc_id in enumerate(retrieved, start=1)
                if doc_text_map.get(doc_id, "")
            ]
            if args.hard_negative_semantic_floor > 0.0:
                rerank_input = [
                    candidate
                    for candidate in rerank_input
                    if candidate.metadata.get("semantic_norm", 0.0)
                    >= args.hard_negative_semantic_floor
                ]
            if args.two_stage_rerank:
                rerank_input = prefilter_rerank_candidates(
                    sample.query,
                    rerank_input,
                    keep_top_n=args.prefilter_candidates,
                )
            reranked = reranker.rerank(
                sample.query,
                rerank_input,
                top_k=retrieve_k,
                alpha=args.rerank_alpha,
                ce_calibration=args.ce_calibration,
                ce_temperature=args.ce_temperature,
                top1_margin_lambda=args.rerank_top1_margin_lambda,
            )
            retrieved = [item.doc_id for item in reranked]
        else:
            retrieved = retrieved[:retrieve_k]
        retrieved_for_metrics = retrieved[:max_k]
        query_runs.append(
            QueryRun(
                query=sample.query,
                relevant_doc_ids=sample.relevant_docs,
                retrieved_doc_ids=retrieved_for_metrics,
            )
        )
        metric_inputs.append(
            RetrievalResult(
                query=sample.query,
                retrieved_doc_ids=retrieved_for_metrics,
                relevant_doc_ids=sample.relevant_docs,
            )
        )
        if sample.relevant_docs and not set(retrieved_for_metrics).intersection(
            sample.relevant_docs
        ):
            miss_type = source_miss_type(
                relevant_doc_ids=sample.relevant_docs,
                semantic_doc_ids=semantic_branch_doc_ids,
                bm25_doc_ids=bm25_branch_doc_ids,
            )
            miss_type_counts[miss_type] += 1
            failure = classify_failure(
                query=sample.query,
                gt_doc_ids=sample.relevant_docs,
                top_k_doc_ids=retrieved_for_metrics,
                all_ranked_doc_ids=retrieved,
                doc_text_map=doc_text_map,
                near_miss_threshold=args.failure_near_miss_threshold,
                top_k=max_k,
            )
            bucket = failure.bucket
            reasons = failure.reasons
            if bucket not in failure_bucket_source_counts:
                failure_bucket_source_counts[bucket] = Counter()
            failure_bucket_source_counts[bucket][miss_type] += 1
            failure_records.append(
                {
                    "query": sample.query,
                    "bucket": bucket,
                    "relevant_doc_ids": sample.relevant_docs,
                    "retrieved_top_k_doc_ids": retrieved_for_metrics,
                    "retrieved_full_doc_ids": retrieved,
                    "bm25_branch_doc_ids": (bm25_branch_doc_ids or [])[
                        : args.soft_recall_rescue_bm25_depth
                    ],
                    "source_miss_type": miss_type,
                    "reasons": reasons,
                }
            )

    metrics = evaluate_retrieval(metric_inputs, k_values)
    top1_doc_ids = [run.retrieved_doc_ids[0] for run in query_runs if run.retrieved_doc_ids]
    top1_counts = Counter(top1_doc_ids)
    top1_total = sum(top1_counts.values())
    top1_distribution = (
        {doc_id: count / top1_total for doc_id, count in top1_counts.most_common()}
        if top1_total
        else {}
    )
    max_top1_doc, max_top1_count = top1_counts.most_common(1)[0] if top1_counts else (None, 0)
    failure_bucket_counts = Counter(item["bucket"] for item in failure_records)
    failure_bucket_source_counts_json = {
        bucket: dict(counter) for bucket, counter in failure_bucket_source_counts.items()
    }
    manual_inspection_failed_queries = failure_records[: args.failure_sample_size]
    report = {
        "dataset": args.dataset,
        "retriever": args.retriever,
        "rerank_enabled": args.rerank,
        "rerank_alpha": args.rerank_alpha if args.rerank else None,
        "rerank_top1_margin_lambda": args.rerank_top1_margin_lambda if args.rerank else None,
        "two_stage_rerank_enabled": args.two_stage_rerank,
        "reranker_model": args.reranker_model if args.rerank else None,
        "stratified_rerank_pool_enabled": args.stratified_rerank_pool,
        "hard_negative_semantic_floor": args.hard_negative_semantic_floor,
        "rerank_prior_weights": {
            "semantic": args.rerank_semantic_weight,
            "bm25": args.rerank_bm25_weight,
        },
        "ce_calibration": args.ce_calibration if args.rerank else None,
        "ce_temperature": args.ce_temperature if args.rerank else None,
        "soft_recall_rescue_enabled": args.soft_recall_rescue,
        "soft_recall_rescue_tail_k": (
            args.soft_recall_rescue_tail_k if args.soft_recall_rescue else 0
        ),
        "soft_recall_rescue_bm25_depth": (
            args.soft_recall_rescue_bm25_depth if args.soft_recall_rescue else 0
        ),
        "multi_query_enabled": args.multi_query,
        "cache": {
            "retrieval_enabled": args.retrieval_cache_enabled,
            "retrieval_capacity": args.retrieval_cache_capacity,
            "retrieval_ttl_seconds": args.retrieval_cache_ttl_seconds,
            "llm_enabled": args.llm_cache_enabled,
            "llm_capacity": args.llm_cache_capacity,
            "llm_ttl_seconds": args.llm_cache_ttl_seconds,
        },
        "mmr_before_rerank": args.mmr_before_rerank,
        "mmr_lambda": args.mmr_lambda if args.mmr_before_rerank else None,
        "mmr_k": args.mmr_k if args.mmr_before_rerank else None,
        "mmr_diversity_threshold": (
            args.mmr_diversity_threshold
            if (args.mmr_before_rerank and args.mmr_diversity_threshold > 0)
            else None
        ),
        "require_evidence": args.require_evidence,
        "k_values": k_values,
        "samples_total": len(samples),
        "samples_total_before_filter": total_samples_before_filter,
        "samples_filtered_out": filtered_out_samples,
        "samples_with_ground_truth": sum(1 for s in samples if s.relevant_docs),
        "metrics": metrics,
        "evaluation": {
            "failed_queries_for_manual_inspection": manual_inspection_failed_queries,
            "failed_queries_for_manual_inspection_count": len(manual_inspection_failed_queries),
        },
        "diagnostics": {
            "top1_unique_docs": len(top1_counts),
            "top1_total_queries": top1_total,
            "top1_most_common_doc_id": max_top1_doc,
            "top1_most_common_doc_fraction": (max_top1_count / top1_total) if top1_total else 0.0,
            "top1_distribution": top1_distribution,
            "failure_analysis": {
                "enabled": True,
                "failure_count": len(failure_records),
                "failure_bucket_counts": dict(failure_bucket_counts),
                "failure_source_miss_counts": dict(miss_type_counts),
                "failure_bucket_source_miss_counts": failure_bucket_source_counts_json,
                "near_miss_threshold": args.failure_near_miss_threshold,
                "manual_inspection_samples": manual_inspection_failed_queries,
            },
        },
        "runs": [run.__dict__ for run in query_runs],
    }

    reranker_dataset_export: dict[str, object] | None = None
    reranker_training_result: dict[str, object] | None = None
    if args.export_reranker_train_jsonl or args.train_reranker:
        out_jsonl = Path(
            args.export_reranker_train_jsonl or "artifacts/datasets/reranker_train.jsonl"
        )
        context_build_result = build_reranker_training_contexts_from_failures(
            failure_records=failure_records,
            doc_text_map=doc_text_map,
            max_negative_rank=max(1, args.reranker_train_max_negative_rank),
            max_negatives=max(1, args.reranker_train_max_negatives),
            ranking_cutoff_weight=max(0.1, args.reranker_train_weight_ranking_cutoff),
            true_recall_weight=max(0.1, args.reranker_train_weight_true_recall),
            default_weight=max(0.1, args.reranker_train_weight_default),
        )
        contexts = context_build_result.contexts
        pair_stats = context_build_result.stats
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as fp:
            for row in contexts:
                fp.write(json.dumps(row.model_dump(), ensure_ascii=False) + "\n")
        reranker_dataset_export = {
            "path": str(out_jsonl),
            "contexts": len(contexts),
            "stats": pair_stats.model_dump(),
            "schema_version": "reranker_context_v1",
        }
        report["reranker_dataset_export"] = reranker_dataset_export
        print(f"- reranker_dataset_export: {out_jsonl} ({len(contexts)} contexts)")
        if args.train_reranker:
            reranker_training_result = train_reranker_from_contexts_jsonl(
                train_jsonl=out_jsonl,
                doc_text_map=doc_text_map,
                model_name=args.train_reranker_model,
                out_dir=Path(args.train_reranker_out_dir),
                epochs=max(1, args.train_reranker_epochs),
                batch_size=max(1, args.train_reranker_batch_size),
                warmup_steps=max(0, args.train_reranker_warmup_steps),
                val_ratio=args.train_reranker_val_ratio,
                seed=args.train_reranker_seed,
            )
            report["reranker_training"] = reranker_training_result.model_dump()
            print(f"- reranker_training_out: {reranker_training_result.out_dir}")

    print("Retrieval benchmark report")
    print(f"- dataset: {args.dataset}")
    print(f"- retriever: {args.retriever}")
    print(f"- samples: {len(samples)}")
    if args.require_evidence:
        print(f"- require_evidence: true (filtered_out={filtered_out_samples})")
    for key in sorted(metrics):
        print(f"- {key}: {metrics[key]:.4f}")
    if failure_records:
        print("- failure_buckets:")
        for bucket in (
            "near_miss",
            "fragmentation",
            "ranking_cutoff_failure",
            "true_recall_failure",
        ):
            print(f"  - {bucket}: {failure_bucket_counts.get(bucket, 0)}")
        print("- failure_source_miss:")
        for key in ("embedding_miss", "bm25_miss", "both_miss", "both_hit"):
            print(f"  - {key}: {miss_type_counts.get(key, 0)}")
        print("- failure_bucket_source_miss:")
        for bucket in (
            "near_miss",
            "fragmentation",
            "ranking_cutoff_failure",
            "true_recall_failure",
        ):
            per_bucket = failure_bucket_source_counts.get(bucket, Counter())
            print(
                "  - "
                f"{bucket}: embedding_miss={per_bucket.get('embedding_miss', 0)}, "
                f"bm25_miss={per_bucket.get('bm25_miss', 0)}, "
                f"both_miss={per_bucket.get('both_miss', 0)}, "
                f"both_hit={per_bucket.get('both_hit', 0)}"
            )
        print(
            f"- failed_queries_for_manual_inspection: {min(len(failure_records), args.failure_sample_size)}"
        )
    if args.multi_query_llm_debug and args.multi_query_llm_expansion:
        print(
            "- multi_query_llm_debug: "
            f"requested={llm_expansion_stats['requested_queries']}, "
            f"generated={llm_expansion_stats['generated_queries']}, "
            f"fallback={llm_expansion_stats['fallback_queries']}"
        )
        for item in llm_expansion_examples:
            print(f"  - llm_example_query: {item['query']}")
            print(f"    generated_preview: {item['generated_preview']}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved JSON report to {out_path}")
        logger.info("saved json report: %s", out_path)
    logger.info("evaluation runner completed successfully")


def cmd_run_rag(args: argparse.Namespace) -> None:
    # Command-scoped import keeps retrieval-only commands lean.
    from generation.run_rag import run_rag

    run_rag(
        question=args.question,
        provider=args.provider,
        model=args.model,
        top_k=args.top_k,
        max_context_tokens=args.max_context_tokens,
        faiss_path=args.faiss_path,
        index_name=args.index,
        embedding_model=args.embedding_model,
        stream=args.stream,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        rerank=args.rerank,
        reranker_model=args.reranker_model,
        rerank_candidates=args.rerank_candidates,
        llm_cache_enabled=args.llm_cache_enabled,
        llm_cache_capacity=args.llm_cache_capacity,
        llm_cache_ttl_seconds=args.llm_cache_ttl_seconds,
        log_level=args.log_level,
        log_path=args.log_path,
        log_json=args.log_json,
    )


def cmd_cleanup_faiss(args: argparse.Namespace) -> None:
    # Command-scoped import keeps cleanup deps isolated.
    from ingestion.cleaner import cleanup_faiss_db

    result = cleanup_faiss_db(
        persist_directory=args.faiss_path,
        index_name=args.index,
        drop_persist_directory=args.drop_persist_directory,
    )
    print(json.dumps(result, indent=2))


def cmd_build_faiss(args: argparse.Namespace) -> None:
    # Command-scoped import keeps embedding deps isolated.
    from embeddings.embedder import build_faiss_index, prepare_embedding_input

    input_jsonl = args.input_jsonl
    prepared_rows = 0
    if args.prepare_input:
        prepared_rows = prepare_embedding_input(
            input_jsonl=args.rag_dataset,
            output_jsonl=args.input_jsonl,
        )
        input_jsonl = args.input_jsonl

    indexed_rows = build_faiss_index(
        input_jsonl=input_jsonl,
        persist_directory=args.faiss_path,
        index_name=args.index,
        model_name=args.embedding_model,
    )
    print(
        json.dumps(
            {
                "prepare_input": args.prepare_input,
                "rag_dataset": args.rag_dataset if args.prepare_input else None,
                "input_jsonl": input_jsonl,
                "prepared_rows": prepared_rows,
                "indexed_rows": indexed_rows,
                "faiss_path": args.faiss_path,
                "index_name": args.index,
                "embedding_model": args.embedding_model,
            },
            indent=2,
        )
    )


def cmd_reranker_pipeline(args: argparse.Namespace) -> None:
    eval_args = argparse.Namespace(
        # Core IO
        dataset=args.dataset,
        rag_dataset=args.rag_dataset,
        faiss_path=args.faiss_path,
        index=args.index,
        out_json=args.out_json,
        # Retrieval stack defaults (current tuned setup)
        retriever="hybrid",
        k_values=args.k_values,
        embedding_model=args.embedding_model,
        alpha=0.65,
        hybrid_candidate_multiplier=80,
        hybrid_max_per_group=1,
        hybrid_rrf_k=80.0,
        # Reranker stack defaults
        rerank=True,
        reranker_model=args.reranker_model,
        rerank_candidates=40,
        rerank_alpha=0.45,
        rerank_top1_margin_lambda=args.rerank_top1_margin_lambda,
        ce_calibration="zscore",
        ce_temperature=1.0,
        stratified_rerank_pool=True,
        hard_negative_semantic_floor=0.12,
        rerank_semantic_weight=0.55,
        rerank_bm25_weight=0.45,
        two_stage_rerank=True,
        prefilter_candidates=40,
        # Multi-query + rescue + MMR defaults
        multi_query=True,
        multi_query_variants=3,
        multi_query_rrf_k=60,
        multi_query_llm_expansion=args.multi_query_llm_expansion,
        multi_query_llm_provider=args.multi_query_llm_provider,
        multi_query_llm_model=args.multi_query_llm_model,
        multi_query_llm_api_base=args.multi_query_llm_api_base,
        multi_query_llm_api_key=args.multi_query_llm_api_key,
        multi_query_llm_timeout_seconds=args.multi_query_llm_timeout_seconds,
        multi_query_llm_retries=args.multi_query_llm_retries,
        multi_query_llm_debug=args.multi_query_llm_debug,
        llm_config_path=args.llm_config_path,
        retrieval_cache_enabled=args.retrieval_cache_enabled,
        retrieval_cache_capacity=args.retrieval_cache_capacity,
        retrieval_cache_ttl_seconds=args.retrieval_cache_ttl_seconds,
        llm_cache_enabled=args.llm_cache_enabled,
        llm_cache_capacity=args.llm_cache_capacity,
        llm_cache_ttl_seconds=args.llm_cache_ttl_seconds,
        log_level=args.log_level,
        log_path=args.log_path,
        log_json=args.log_json,
        soft_recall_rescue=True,
        soft_recall_rescue_tail_k=20,
        soft_recall_rescue_bm25_depth=200,
        mmr_before_rerank=True,
        mmr_lambda=0.82,
        mmr_k=30,
        mmr_diversity_threshold=0.0,
        # Failure analysis defaults
        require_evidence=True,
        failure_near_miss_threshold=0.80,
        failure_sample_size=20,
        # Integrated reranker dataset + train toggles
        export_reranker_train_jsonl=args.export_reranker_train_jsonl,
        reranker_train_max_negative_rank=20,
        reranker_train_max_negatives=16,
        reranker_train_weight_ranking_cutoff=2.0,
        reranker_train_weight_true_recall=0.3,
        reranker_train_weight_default=1.0,
        train_reranker=args.train_reranker,
        train_reranker_model=args.train_reranker_model,
        train_reranker_out_dir=args.train_reranker_out_dir,
        train_reranker_epochs=args.train_reranker_epochs,
        train_reranker_batch_size=args.train_reranker_batch_size,
        train_reranker_warmup_steps=args.train_reranker_warmup_steps,
        train_reranker_val_ratio=args.train_reranker_val_ratio,
        train_reranker_seed=args.train_reranker_seed,
    )
    cmd_evaluation_runner(eval_args)


def cmd_build_evaluation_dataset(args: argparse.Namespace) -> None:
    # Command-scoped import keeps evaluation dataset builder optional.
    from evaluation.dataset import build_evaluation_dataset

    count, stats = build_evaluation_dataset(
        rag_path=Path(args.rag),
        eval_json_path=Path(args.eval),
        out_path=Path(args.out),
        fuzzy_ratio=args.fuzzy_ratio,
        lexical_min_hits=args.lexical_min_hits,
        max_chunk_ids=args.max_chunk_ids,
        semantic_fallback=not args.no_semantic_fallback,
        semantic_model=args.semantic_model,
        semantic_min_score=args.semantic_min_score,
        max_gt_url_share=args.max_gt_url_share,
        target_multi_gt_share=args.target_multi_gt_share,
        keep_max_ids_for_multi=args.keep_max_ids_for_multi,
        excerpt_max=args.excerpt_max,
    )
    print(f"Wrote {args.out} ({count} records).")
    print("Stats:", json.dumps(stats, indent=2))


def cmd_dataset_audit(args: argparse.Namespace) -> None:
    # Command-scoped import keeps audit utilities optional.
    from commands.dataset_audit import audit

    report = audit(Path(args.rag), Path(args.eval))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_build_reranker_dataset(args: argparse.Namespace) -> None:
    # Command-scoped import keeps reranker dataset utilities optional.
    from commands.build_reranker_dataset import build_contexts, load_chunk_texts

    report = json.loads(Path(args.eval_report).read_text(encoding="utf-8"))
    chunk_texts = load_chunk_texts(Path(args.rag_dataset))
    contexts, stats = build_contexts(
        report=report,
        chunk_texts=chunk_texts,
        max_negative_rank=max(1, args.max_negative_rank),
        max_negatives=max(1, args.max_negatives),
        ranking_cutoff_weight=max(0.1, args.ranking_cutoff_weight),
        true_recall_weight=max(0.1, args.true_recall_weight),
        default_weight=max(0.1, args.default_weight),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for row in contexts:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path} ({len(contexts)} contexts).")
    print("Stats:", json.dumps(stats, indent=2))


def cmd_train_reranker(args: argparse.Namespace) -> None:
    # Heavy training deps are imported lazily for non-training commands.
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    from torch.utils.data import DataLoader

    from commands.train_reranker import load_chunk_texts, load_pairwise_samples

    chunk_texts = load_chunk_texts(Path(args.rag_dataset))
    train_examples, val_examples = load_pairwise_samples(
        Path(args.train_jsonl),
        seed=args.seed,
        val_ratio=args.val_ratio,
        chunk_texts=chunk_texts,
    )
    if not train_examples:
        raise ValueError("No training examples loaded. Check --train-jsonl contents.")

    model = CrossEncoder(args.model, num_labels=1, max_length=512)
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    evaluator = None
    if val_examples:
        evaluator = CEBinaryClassificationEvaluator.from_input_examples(
            val_examples,
            name="failure-driven-val",
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_dataloader=train_loader,
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        output_path=str(out_dir),
        show_progress_bar=True,
    )
    model.save(str(out_dir))
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "train_examples": len(train_examples),
                "val_examples": len(val_examples),
                "model": args.model,
            },
            indent=2,
        )
    )


def cmd_run_experiments(args: argparse.Namespace) -> None:
    # Command-scoped import keeps experiment stack optional.
    from experiments.run_experiments import run_experiments

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    run_experiments(
        question=args.question,
        models=models,
        top_k=args.top_k,
        max_context_tokens=args.max_context_tokens,
        faiss_path=args.faiss_path,
        index_name=args.index,
        embedding_model=args.embedding_model,
        log_path=args.log_path,
        llm_config_path=args.llm_config_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single entrypoint for project workflows.")
    parser.add_argument(
        "--config",
        help="Path to CLI defaults JSON (default: project root cli.defaults.json).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser_cmd = subparsers.add_parser(
        "build_parser",
        aliases=["build-parser"],
        help="Run parser pipeline and build rag_dataset.jsonl",
    )
    build_parser_cmd.add_argument(
        "--output",
    )
    build_parser_cmd.add_argument(
        "--min-tokens",
        type=int,
    )
    build_parser_cmd.add_argument(
        "--max-tokens",
        type=int,
    )
    build_parser_cmd.add_argument(
        "--overlap-ratio",
        type=float,
    )
    build_parser_cmd.add_argument(
        "--min-output-chunk-tokens",
        type=int,
    )
    build_parser_cmd.add_argument(
        "--max-output-chunk-tokens",
        type=int,
    )
    build_parser_cmd.add_argument(
        "--max-chunks-per-url",
        type=int,
    )
    build_parser_cmd.add_argument(
        "--max-chunks-per-category",
        type=int,
    )
    build_parser_cmd.add_argument(
        "--sources-config",
    )
    build_parser_cmd.add_argument(
        "--chunker-mode",
        choices=("token", "semantic_dynamic"),
    )
    build_parser_cmd.add_argument(
        "--near-duplicate-jaccard",
        type=float,
        help="Skip near-duplicate chunks from the same URL when similarity >= threshold (0 disables).",
    )
    build_parser_cmd.add_argument(
        "--embedding-model",
    )
    build_parser_cmd.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    build_parser_cmd.add_argument(
        "--log-path",
    )
    build_parser_cmd.add_argument("--log-json", action="store_true")
    build_parser_cmd.set_defaults(handler=cmd_build_parser)

    demo_cmd = subparsers.add_parser(
        "demo_retrieval", help="Run BM25/semantic/hybrid retrieval demo."
    )
    demo_cmd.add_argument(
        "--query",
        "-q",
    )
    demo_cmd.add_argument(
        "--top-k",
        "-k",
        type=int,
    )
    demo_cmd.add_argument(
        "--model",
        "-m",
    )
    demo_cmd.add_argument(
        "--dataset",
    )
    demo_cmd.add_argument(
        "--faiss-path",
    )
    demo_cmd.add_argument(
        "--index",
    )
    demo_cmd.add_argument("--rerank", action="store_true")
    demo_cmd.add_argument(
        "--reranker-model",
    )
    demo_cmd.add_argument(
        "--rerank-candidates",
        type=int,
    )
    demo_cmd.set_defaults(handler=cmd_demo_retrieval)

    eval_cmd = subparsers.add_parser(
        "evaluation_runner",
        aliases=["evaluate"],
        help="Run retrieval benchmark over eval dataset.",
    )
    eval_cmd.add_argument(
        "--dataset",
    )
    eval_cmd.add_argument(
        "--retriever",
        choices=("semantic", "bm25", "hybrid"),
    )
    eval_cmd.add_argument(
        "--k-values",
    )
    eval_cmd.add_argument(
        "--rag-dataset",
    )
    eval_cmd.add_argument(
        "--faiss-path",
    )
    eval_cmd.add_argument(
        "--index",
    )
    eval_cmd.add_argument(
        "--embedding-model",
    )
    eval_cmd.add_argument(
        "--alpha",
        type=float,
    )
    eval_cmd.add_argument(
        "--hybrid-candidate-multiplier",
        type=int,
        help="Hybrid per-branch candidate pool multiplier before merge/rerank.",
    )
    eval_cmd.add_argument(
        "--hybrid-max-per-group",
        type=int,
        help="Max documents per source/section group in hybrid top-k (<=0 disables).",
    )
    eval_cmd.add_argument(
        "--hybrid-rrf-k",
        type=float,
        help="RRF k parameter used to fuse semantic and BM25 ranks.",
    )
    eval_cmd.add_argument("--rerank", action="store_true")
    eval_cmd.add_argument(
        "--reranker-model",
    )
    eval_cmd.add_argument(
        "--rerank-candidates",
        type=int,
    )
    eval_cmd.add_argument(
        "--rerank-alpha",
        type=float,
    )
    eval_cmd.add_argument(
        "--rerank-top1-margin-lambda",
        type=float,
        help="Post-process top-1 score with lambda * (top1 - top2).",
    )
    eval_cmd.add_argument(
        "--ce-calibration",
        choices=("minmax", "softmax", "zscore"),
        help="How to normalize CE scores per query before fusion with prior scores.",
    )
    eval_cmd.add_argument(
        "--ce-temperature",
        type=float,
        help="Temperature for CE calibration modes (softmax/zscore). >1 flattens scores.",
    )
    eval_cmd.add_argument(
        "--stratified-rerank-pool",
        action="store_true",
        help="Interleave semantic/BM25 candidates before reranking.",
    )
    eval_cmd.add_argument(
        "--hard-negative-semantic-floor",
        type=float,
        help="Drop rerank candidates with normalized semantic score below threshold.",
    )
    eval_cmd.add_argument(
        "--rerank-semantic-weight",
        type=float,
    )
    eval_cmd.add_argument(
        "--rerank-bm25-weight",
        type=float,
    )
    eval_cmd.add_argument(
        "--two-stage-rerank",
        action="store_true",
        help="Apply lexical prefilter before cross-encoder reranking.",
    )
    eval_cmd.add_argument(
        "--prefilter-candidates",
        type=int,
        help="How many candidates to keep for stage-2 cross-encoder reranking.",
    )
    eval_cmd.add_argument("--multi-query", action="store_true")
    eval_cmd.add_argument(
        "--multi-query-variants",
        type=int,
    )
    eval_cmd.add_argument(
        "--multi-query-rrf-k",
        type=int,
    )
    eval_cmd.add_argument(
        "--multi-query-llm-expansion",
        action="store_true",
        help="Use LLM-based structured expansion (paraphrase + decomposition + concept queries).",
    )
    eval_cmd.add_argument(
        "--multi-query-llm-provider",
    )
    eval_cmd.add_argument(
        "--multi-query-llm-model",
    )
    eval_cmd.add_argument(
        "--multi-query-llm-api-base",
    )
    eval_cmd.add_argument(
        "--multi-query-llm-api-key",
    )
    eval_cmd.add_argument(
        "--multi-query-llm-timeout-seconds",
        type=int,
    )
    eval_cmd.add_argument(
        "--multi-query-llm-retries",
        type=int,
    )
    eval_cmd.add_argument(
        "--llm-config-path",
    )
    eval_cmd.add_argument("--multi-query-llm-debug", action="store_true")
    eval_cmd.add_argument(
        "--retrieval-cache-enabled",
        action="store_true",
        help="Enable in-memory cache for retrieval query results.",
    )
    eval_cmd.add_argument(
        "--retrieval-cache-capacity",
        type=int,
    )
    eval_cmd.add_argument(
        "--retrieval-cache-ttl-seconds",
        type=float,
    )
    eval_cmd.add_argument(
        "--llm-cache-enabled",
        action="store_true",
        help="Enable in-memory cache for LLM calls (query expansion / generation).",
    )
    eval_cmd.add_argument(
        "--llm-cache-capacity",
        type=int,
    )
    eval_cmd.add_argument(
        "--llm-cache-ttl-seconds",
        type=float,
    )
    eval_cmd.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    eval_cmd.add_argument("--log-path", help="Optional runtime log file path.")
    eval_cmd.add_argument(
        "--log-json", action="store_true", help="Emit runtime logs in JSON format."
    )
    eval_cmd.add_argument(
        "--soft-recall-rescue",
        action="store_true",
        help="Inject BM25-only tail candidates into reranker pool after hybrid retrieval.",
    )
    eval_cmd.add_argument(
        "--soft-recall-rescue-tail-k",
        type=int,
        help="How many BM25-only candidates to inject into reranker pool.",
    )
    eval_cmd.add_argument(
        "--soft-recall-rescue-bm25-depth",
        type=int,
        help="How deep to search BM25 before extracting BM25-only tail candidates.",
    )
    eval_cmd.add_argument(
        "--mmr-before-rerank",
        action="store_true",
        help="Apply MMR diversity selection after fusion/rescue and before CE reranking.",
    )
    eval_cmd.add_argument(
        "--mmr-lambda",
        type=float,
    )
    eval_cmd.add_argument(
        "--mmr-k",
        type=int,
        help="Candidate pool size kept after MMR (<=0 uses rerank candidate size).",
    )
    eval_cmd.add_argument(
        "--mmr-diversity-threshold",
        type=float,
        help="Optional hard max cosine similarity to selected docs (<=0 disables).",
    )
    eval_cmd.add_argument(
        "--require-evidence",
        action="store_true",
        help="Evaluate only samples with non-empty expected_evidence.chunk_ids.",
    )
    eval_cmd.add_argument(
        "--failure-near-miss-threshold",
        type=float,
        help="Similarity threshold used to classify failures as near_miss.",
    )
    eval_cmd.add_argument(
        "--failure-sample-size",
        type=int,
        help="Number of failed queries to include for manual inspection.",
    )
    eval_cmd.add_argument(
        "--export-reranker-train-jsonl",
        help="Optional path to export pairwise hard-negative reranker training dataset.",
    )
    eval_cmd.add_argument(
        "--reranker-train-max-negative-rank",
        type=int,
    )
    eval_cmd.add_argument(
        "--reranker-train-max-negatives",
        type=int,
    )
    eval_cmd.add_argument(
        "--reranker-train-weight-ranking-cutoff",
        type=float,
    )
    eval_cmd.add_argument(
        "--reranker-train-weight-true-recall",
        type=float,
    )
    eval_cmd.add_argument(
        "--reranker-train-weight-default",
        type=float,
    )
    eval_cmd.add_argument(
        "--train-reranker",
        action="store_true",
        help="Train reranker in the same run after exporting hard-negative dataset.",
    )
    eval_cmd.add_argument(
        "--train-reranker-model",
    )
    eval_cmd.add_argument(
        "--train-reranker-out-dir",
    )
    eval_cmd.add_argument(
        "--train-reranker-epochs",
        type=int,
    )
    eval_cmd.add_argument(
        "--train-reranker-batch-size",
        type=int,
    )
    eval_cmd.add_argument(
        "--train-reranker-warmup-steps",
        type=int,
    )
    eval_cmd.add_argument(
        "--train-reranker-val-ratio",
        type=float,
    )
    eval_cmd.add_argument(
        "--train-reranker-seed",
        type=int,
    )
    eval_cmd.add_argument(
        "--out-json",
    )
    eval_cmd.set_defaults(handler=cmd_evaluation_runner)

    rerank_pipeline_cmd = subparsers.add_parser(
        "reranker_pipeline",
        help="One-shot eval + failure dataset export + optional reranker training.",
    )
    rerank_pipeline_cmd.add_argument(
        "--dataset",
    )
    rerank_pipeline_cmd.add_argument(
        "--rag-dataset",
    )
    rerank_pipeline_cmd.add_argument(
        "--faiss-path",
    )
    rerank_pipeline_cmd.add_argument(
        "--index",
    )
    rerank_pipeline_cmd.add_argument(
        "--embedding-model",
    )
    rerank_pipeline_cmd.add_argument(
        "--reranker-model",
    )
    rerank_pipeline_cmd.add_argument(
        "--rerank-top1-margin-lambda",
        type=float,
        help="Post-process top-1 score with lambda * (top1 - top2).",
    )
    rerank_pipeline_cmd.add_argument(
        "--k-values",
    )
    rerank_pipeline_cmd.add_argument("--multi-query-llm-expansion", action="store_true")
    rerank_pipeline_cmd.add_argument(
        "--multi-query-llm-provider",
    )
    rerank_pipeline_cmd.add_argument(
        "--multi-query-llm-model",
    )
    rerank_pipeline_cmd.add_argument(
        "--multi-query-llm-api-base",
    )
    rerank_pipeline_cmd.add_argument(
        "--multi-query-llm-api-key",
    )
    rerank_pipeline_cmd.add_argument(
        "--multi-query-llm-timeout-seconds",
        type=int,
    )
    rerank_pipeline_cmd.add_argument(
        "--multi-query-llm-retries",
        type=int,
    )
    rerank_pipeline_cmd.add_argument(
        "--llm-config-path",
    )
    rerank_pipeline_cmd.add_argument("--multi-query-llm-debug", action="store_true")
    rerank_pipeline_cmd.add_argument("--retrieval-cache-enabled", action="store_true")
    rerank_pipeline_cmd.add_argument(
        "--retrieval-cache-capacity",
        type=int,
    )
    rerank_pipeline_cmd.add_argument(
        "--retrieval-cache-ttl-seconds",
        type=float,
    )
    rerank_pipeline_cmd.add_argument("--llm-cache-enabled", action="store_true")
    rerank_pipeline_cmd.add_argument(
        "--llm-cache-capacity",
        type=int,
    )
    rerank_pipeline_cmd.add_argument(
        "--llm-cache-ttl-seconds",
        type=float,
    )
    rerank_pipeline_cmd.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    rerank_pipeline_cmd.add_argument(
        "--log-path",
    )
    rerank_pipeline_cmd.add_argument("--log-json", action="store_true")
    rerank_pipeline_cmd.add_argument(
        "--out-json",
    )
    rerank_pipeline_cmd.add_argument(
        "--export-reranker-train-jsonl",
        help="Path for exported reranker pairwise training dataset.",
    )
    rerank_pipeline_cmd.add_argument("--train-reranker", action="store_true")
    rerank_pipeline_cmd.add_argument(
        "--train-reranker-model",
    )
    rerank_pipeline_cmd.add_argument(
        "--train-reranker-out-dir",
    )
    rerank_pipeline_cmd.add_argument(
        "--train-reranker-epochs",
        type=int,
    )
    rerank_pipeline_cmd.add_argument(
        "--train-reranker-batch-size",
        type=int,
    )
    rerank_pipeline_cmd.add_argument(
        "--train-reranker-warmup-steps",
        type=int,
    )
    rerank_pipeline_cmd.add_argument(
        "--train-reranker-val-ratio",
        type=float,
    )
    rerank_pipeline_cmd.add_argument(
        "--train-reranker-seed",
        type=int,
    )
    rerank_pipeline_cmd.set_defaults(handler=cmd_reranker_pipeline)

    rag_cmd = subparsers.add_parser(
        "run_rag",
        aliases=["run-rag"],
        help="Run full RAG query against selected LLM provider.",
    )
    rag_cmd.add_argument(
        "--question",
        "-q",
    )
    rag_cmd.add_argument("--provider", choices=("openai", "gigachat", "ollama", "qwen"))
    rag_cmd.add_argument(
        "--model",
    )
    rag_cmd.add_argument(
        "--top-k",
        type=int,
    )
    rag_cmd.add_argument(
        "--max-context-tokens",
        type=int,
    )
    rag_cmd.add_argument(
        "--faiss-path",
    )
    rag_cmd.add_argument(
        "--index",
    )
    rag_cmd.add_argument(
        "--embedding-model",
    )
    rag_cmd.add_argument("--stream", action="store_true")
    rag_cmd.add_argument(
        "--max-tokens",
        type=int,
    )
    rag_cmd.add_argument(
        "--temperature",
        type=float,
    )
    rag_cmd.add_argument(
        "--top-p",
        type=float,
    )
    rag_cmd.add_argument("--rerank", action="store_true")
    rag_cmd.add_argument(
        "--reranker-model",
    )
    rag_cmd.add_argument(
        "--rerank-candidates",
        type=int,
    )
    rag_cmd.add_argument("--llm-cache-enabled", action="store_true")
    rag_cmd.add_argument(
        "--llm-cache-capacity",
        type=int,
    )
    rag_cmd.add_argument(
        "--llm-cache-ttl-seconds",
        type=float,
    )
    rag_cmd.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    rag_cmd.add_argument("--log-path", help="Optional runtime log file path.")
    rag_cmd.add_argument(
        "--log-json", action="store_true", help="Emit runtime logs in JSON format."
    )
    rag_cmd.add_argument(
        "--llm-config-path",
    )
    rag_cmd.set_defaults(handler=cmd_run_rag)

    clean_cmd = subparsers.add_parser(
        "cleanup_faiss", help="Delete FAISS index and optionally full directory."
    )
    clean_cmd.add_argument(
        "--faiss-path",
    )
    clean_cmd.add_argument(
        "--index",
    )
    clean_cmd.add_argument("--drop-persist-directory", action="store_true")
    clean_cmd.set_defaults(handler=cmd_cleanup_faiss)

    build_faiss_cmd = subparsers.add_parser(
        "build_faiss",
        aliases=["build-faiss"],
        help="Build FAISS index from embeddings input or directly from rag dataset.",
    )
    build_faiss_cmd.add_argument(
        "--input-jsonl",
        help="Embeddings input JSONL with id/text records.",
    )
    build_faiss_cmd.add_argument(
        "--prepare-input",
        action="store_true",
        help="Generate --input-jsonl from --rag-dataset before indexing.",
    )
    build_faiss_cmd.add_argument(
        "--rag-dataset",
        help="RAG dataset used when --prepare-input is enabled.",
    )
    build_faiss_cmd.add_argument(
        "--faiss-path",
    )
    build_faiss_cmd.add_argument(
        "--index",
    )
    build_faiss_cmd.add_argument(
        "--embedding-model",
    )
    build_faiss_cmd.set_defaults(handler=cmd_build_faiss)

    build_eval_cmd = subparsers.add_parser(
        "build_evaluation_dataset",
        aliases=["build-eval-dataset"],
        help="Build structured evaluation JSONL from eval JSON and rag dataset.",
    )
    build_eval_cmd.add_argument(
        "--rag",
    )
    build_eval_cmd.add_argument(
        "--eval",
    )
    build_eval_cmd.add_argument(
        "--out",
    )
    build_eval_cmd.add_argument(
        "--fuzzy-ratio",
        type=float,
    )
    build_eval_cmd.add_argument(
        "--lexical-min-hits",
        type=int,
    )
    build_eval_cmd.add_argument(
        "--max-chunk-ids",
        type=int,
    )
    build_eval_cmd.add_argument("--no-semantic-fallback", action="store_true")
    build_eval_cmd.add_argument(
        "--semantic-model",
    )
    build_eval_cmd.add_argument(
        "--semantic-min-score",
        type=float,
    )
    build_eval_cmd.add_argument(
        "--max-gt-url-share",
        type=float,
    )
    build_eval_cmd.add_argument(
        "--target-multi-gt-share",
        type=float,
    )
    build_eval_cmd.add_argument(
        "--keep-max-ids-for-multi",
        type=int,
    )
    build_eval_cmd.add_argument(
        "--excerpt-max",
        type=int,
    )
    build_eval_cmd.set_defaults(handler=cmd_build_evaluation_dataset)

    audit_cmd = subparsers.add_parser(
        "dataset_audit",
        aliases=["audit-dataset"],
        help="Audit rag/evaluation datasets and print quality report.",
    )
    audit_cmd.add_argument(
        "--rag",
    )
    audit_cmd.add_argument(
        "--eval",
    )
    audit_cmd.add_argument(
        "--out",
    )
    audit_cmd.set_defaults(handler=cmd_dataset_audit)

    build_reranker_ds_cmd = subparsers.add_parser(
        "build_reranker_dataset",
        aliases=["build-reranker-dataset"],
        help="Build context-aware reranker training JSONL from retrieval report.",
    )
    build_reranker_ds_cmd.add_argument(
        "--eval-report",
    )
    build_reranker_ds_cmd.add_argument(
        "--rag-dataset",
    )
    build_reranker_ds_cmd.add_argument(
        "--out",
    )
    build_reranker_ds_cmd.add_argument(
        "--max-negative-rank",
        type=int,
    )
    build_reranker_ds_cmd.add_argument(
        "--max-negatives",
        type=int,
    )
    build_reranker_ds_cmd.add_argument(
        "--ranking-cutoff-weight",
        type=float,
    )
    build_reranker_ds_cmd.add_argument(
        "--true-recall-weight",
        type=float,
    )
    build_reranker_ds_cmd.add_argument(
        "--default-weight",
        type=float,
    )
    build_reranker_ds_cmd.set_defaults(handler=cmd_build_reranker_dataset)

    train_reranker_cmd = subparsers.add_parser(
        "train_reranker",
        aliases=["train-reranker"],
        help="Fine-tune cross-encoder reranker on context dataset.",
    )
    train_reranker_cmd.add_argument(
        "--train-jsonl",
    )
    train_reranker_cmd.add_argument(
        "--rag-dataset",
    )
    train_reranker_cmd.add_argument(
        "--model",
    )
    train_reranker_cmd.add_argument(
        "--out-dir",
    )
    train_reranker_cmd.add_argument(
        "--epochs",
        type=int,
    )
    train_reranker_cmd.add_argument(
        "--batch-size",
        type=int,
    )
    train_reranker_cmd.add_argument(
        "--warmup-steps",
        type=int,
    )
    train_reranker_cmd.add_argument(
        "--val-ratio",
        type=float,
    )
    train_reranker_cmd.add_argument(
        "--seed",
        type=int,
    )
    train_reranker_cmd.set_defaults(handler=cmd_train_reranker)

    experiments_cmd = subparsers.add_parser(
        "run_experiments",
        aliases=["run-experiments"],
        help="Run RAG answer comparison across configured LLM providers.",
    )
    experiments_cmd.add_argument(
        "--question",
        "-q",
    )
    experiments_cmd.add_argument(
        "--models",
    )
    experiments_cmd.add_argument(
        "--top-k",
        type=int,
    )
    experiments_cmd.add_argument(
        "--max-context-tokens",
        type=int,
    )
    experiments_cmd.add_argument(
        "--faiss-path",
    )
    experiments_cmd.add_argument(
        "--index",
    )
    experiments_cmd.add_argument(
        "--embedding-model",
    )
    experiments_cmd.add_argument(
        "--log-path",
    )
    experiments_cmd.add_argument(
        "--llm-config-path",
    )
    experiments_cmd.set_defaults(handler=cmd_run_experiments)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_arg = getattr(args, "config", None)
    if config_arg:
        config_path = Path(config_arg).expanduser()
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
    else:
        config_path = Path.cwd() / DEFAULT_CLI_PARAMS_CONFIG
    config_defaults = load_cli_defaults(config_path)
    apply_config_defaults(parser, args, sys.argv[1:], config_defaults)
    validate_required_command_params(parser, args, REQUIRED_COMMAND_PARAMS)
    args.handler(args)


if __name__ == "__main__":
    main()
