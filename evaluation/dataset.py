from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Any

COMMON_RAG_LEXEMES = frozenset(
    {
        "rag",
        "retrieval",
        "generation",
        "retrieval-augmented",
        "llm",
        "llms",
        "model",
        "models",
        "embedding",
        "embeddings",
        "vector",
        "vectors",
        "chunk",
        "chunks",
        "chunking",
        "document",
        "documents",
        "data",
        "search",
        "query",
        "queries",
        "index",
        "indexing",
        "database",
        "databases",
        "knowledge",
        "text",
        "language",
    }
)

SUPPORT_CONCEPT_TERMS = (
    "grounding",
    "groundedness",
    "grounded",
    "citation",
    "citations",
    "hallucination",
    "hallucinations",
    "fresh",
    "up-to-date",
    "auditable",
    "sources",
    "source",
    "refusal",
    "refuse",
    "noise",
    "overlap",
    "duplicate",
    "supported",
    "claim",
    "claims",
)

DEFAULT_STOPWORDS = frozenset(
    """
    a an the is are was were be been being to of and or for in on at by with from as it its
    this that these those what how when where why which who can does do did will would should
    could may might must shall not no nor if then than so such into about over under between
    both each few more most other some such only own same so than too very just also
    """.split()
)


@dataclass(frozen=True)
class EvalBlock:
    section: str
    question: str
    answer: str
    distractor: str
    noise: str


@dataclass(frozen=True)
class EvalSample:
    query: str
    relevant_docs: list[str]
    reference_answer: str | None = None
    metadata: dict[str, object] | None = None


def load_dataset(rag_jsonl: Path) -> tuple[dict[str, list[str]], dict[str, str], dict[str, str], list[str]]:
    qa_map: dict[str, list[str]] = defaultdict(list)
    chunk_text: dict[str, str] = {}
    chunk_url: dict[str, str] = {}

    with rag_jsonl.open("r", encoding="utf-8") as dataset:
        for line in dataset:
            row = json.loads(line)
            rtype = row.get("record_type")
            if rtype == "qa_pair":
                question = str(row.get("question", "")).strip()
                meta = row.get("metadata") or {}
                chunk_id = meta.get("chunk_id")
                if not question or not chunk_id:
                    continue
                chunk_id = str(chunk_id)
                if chunk_id not in qa_map[question]:
                    qa_map[question].append(chunk_id)
            elif rtype == "raw_chunk":
                chunk_id = row.get("chunk_id")
                text = row.get("text")
                if chunk_id and text is not None:
                    chunk_id_s = str(chunk_id)
                    chunk_text[chunk_id_s] = str(text)
                    meta = row.get("metadata") or {}
                    chunk_url[chunk_id_s] = str(meta.get("url", ""))

    return dict(qa_map), chunk_text, chunk_url, sorted(qa_map.keys())


def keywords(text: str) -> list[str]:
    text = re.sub(r"[^\w\s-]", " ", text.lower())
    out: list[str] = []
    for raw in text.split():
        w = raw.strip("-")
        if not w or w in DEFAULT_STOPWORDS:
            continue
        if len(w) >= 3 or w == "rag":
            out.append(w)
    return out


def word_in_text(word: str, text: str) -> bool:
    return (
        re.search(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", text, re.IGNORECASE)
        is not None
    )


def _lexeme_weight(word: str) -> float:
    return 0.35 if word in COMMON_RAG_LEXEMES else 1.0


def _ngrams(words: list[str], n: int) -> set[str]:
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _phrase_hits(phrases: set[str], text: str) -> int:
    if not phrases:
        return 0
    return sum(1 for phrase in phrases if phrase in text)


def _sentence_split(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    parts = re.split(r"(?<=[.!?])\s+| (?=[-•])", compact)
    out = [p.strip() for p in parts if p and p.strip()]
    if out:
        return out
    return [compact]


def build_keyword_df(chunk_text: dict[str, str]) -> tuple[dict[str, int], int]:
    df: dict[str, int] = defaultdict(int)
    total = len(chunk_text)
    for text in chunk_text.values():
        seen = set(keywords(text))
        for w in seen:
            df[w] += 1
    return dict(df), total


def _idf(word: str, keyword_df: dict[str, int], total_chunks: int) -> float:
    if total_chunks <= 0:
        return 1.0
    doc_freq = keyword_df.get(word, 0)
    return math.log1p(total_chunks / (1 + doc_freq))


def _metric_terms(answer: str) -> set[str]:
    src = answer.lower().replace(" @ ", "@").replace(" / ", "/")
    terms: set[str] = set()
    for term in ("recall@k", "precision@k", "mrr", "ndcg", "em/f1", "faithfulness", "groundedness"):
        if term in src:
            terms.add(term)
    return terms


def _expanded_keywords(words: list[str]) -> list[str]:
    expansions = {
        "generator": ("generation", "generate"),
        "retriever": ("retrieval", "retrieve"),
        "citation": ("citations", "cite"),
        "grounding": ("grounded",),
    }
    out: list[str] = list(words)
    for w in words:
        for alt in expansions.get(w, ()):
            out.append(alt)
    return list(dict.fromkeys(out))


def lexical_chunk_ids(
    question: str,
    answer: str | None,
    chunk_text: dict[str, str],
    keyword_df: dict[str, int],
    total_chunks: int,
    min_hits: int,
    max_chunk_ids: int,
) -> list[str]:
    kws_q = keywords(question)
    kws_a = keywords(answer or "")
    metric_terms = _metric_terms(answer or "")
    answer_text_low = (answer or "").lower()
    concept_terms = {t for t in SUPPORT_CONCEPT_TERMS if t in answer_text_low}
    answer_bigrams = _ngrams(kws_a, 2)
    if max_chunk_ids <= 0:
        return []
    if not kws_q and not kws_a:
        return []

    rows: list[tuple[float, int, float, int, int, str]] = []
    uniq_a = set(kws_a)
    min_answer_hits = max(2, math.ceil(len(uniq_a) * 0.22)) if uniq_a else 0
    if uniq_a and (metric_terms or concept_terms):
        min_answer_hits = max(1, math.ceil(len(uniq_a) * 0.16))

    for cid, text in chunk_text.items():
        text_low = text.lower()
        text_norm = text_low.replace(" @ ", "@").replace(" / ", "/")
        matched_q = {w for w in kws_q if word_in_text(w, text)}
        matched_a = {w for w in kws_a if word_in_text(w, text)}
        distinct = len(matched_q) + (2 * len(matched_a))
        if distinct < min_hits:
            continue
        if uniq_a and len(matched_a) < min_answer_hits:
            continue
        if metric_terms and sum(1 for t in metric_terms if t in text_norm) < 1:
            continue
        if len(concept_terms) >= 2 and sum(1 for t in concept_terms if t in text_norm) < 1:
            continue

        phrase_bonus = _phrase_hits(answer_bigrams, text_low)
        q_weight = sum(_idf(w, keyword_df, total_chunks) * _lexeme_weight(w) for w in matched_q)
        a_weight = sum(_idf(w, keyword_df, total_chunks) * _lexeme_weight(w) for w in matched_a)
        score = q_weight + (3.5 * a_weight) + (2.5 * phrase_bonus)
        weighted = sum(text_low.count(w) for w in matched_q | matched_a if len(w) >= 3)

        sentence_best = 0
        for sentence in _sentence_split(text):
            sentence_low = sentence.lower()
            sent_a = {w for w in kws_a if word_in_text(w, sentence)}
            sent_q = {w for w in kws_q if word_in_text(w, sentence)}
            if kws_a and not sent_a:
                continue
            sent_score = (4 * len(sent_a)) + len(sent_q) + (3 * _phrase_hits(answer_bigrams, sentence_low))
            sentence_best = max(sentence_best, sent_score)

        min_sentence_score = 3 if metric_terms or concept_terms else 5
        if uniq_a and sentence_best < min_sentence_score:
            continue
        rare_answer_terms = {w for w in uniq_a if _idf(w, keyword_df, total_chunks) >= 1.8}
        if rare_answer_terms and not (rare_answer_terms & matched_a) and not (metric_terms or concept_terms):
            continue
        answer_ratio = (len(matched_a) / len(uniq_a)) if uniq_a else 0.0
        rows.append((score, sentence_best, answer_ratio, weighted, -len(text), cid))

    if not rows:
        return []

    rows.sort(reverse=True)
    top = rows[0]
    top_score, top_sentence, top_answer_ratio = top[0], top[1], top[2]
    min_ratio = 0.12 if metric_terms or concept_terms else 0.24
    min_top_sentence = 4 if metric_terms or concept_terms else 6
    if uniq_a and (top_answer_ratio < min_ratio or top_sentence < min_top_sentence or top_score < 8):
        return []
    selected: list[str] = []
    min_score_ratio = 0.78
    min_sentence_ratio = 0.72
    min_answer_ratio_drop = 0.18
    top_ids = {top[5]}
    for row in rows:
        cid = row[5]
        if cid in top_ids:
            selected.append(cid)
            continue
        if len(selected) >= max_chunk_ids:
            break
        row_score, row_sentence, row_answer_ratio = row[0], row[1], row[2]
        if row_score < (top_score * min_score_ratio):
            continue
        if top_sentence > 0 and row_sentence < (top_sentence * min_sentence_ratio):
            continue
        if uniq_a and row_answer_ratio + min_answer_ratio_drop < top_answer_ratio:
            continue
        selected.append(cid)
    return selected[:max_chunk_ids]


def fuzzy_match_question(question: str, qa_questions: list[str], min_ratio: float) -> str | None:
    if not qa_questions:
        return None
    q = question.strip()
    candidates = get_close_matches(q, qa_questions, n=1, cutoff=max(0.0, min_ratio - 0.01))
    if not candidates:
        return None
    candidate = candidates[0]
    ratio = SequenceMatcher(None, q.lower(), candidate.lower()).ratio()
    if ratio < min_ratio:
        return None
    return candidate


def _dot(vec_a: list[float], vec_b: list[float]) -> float:
    return sum(a * b for a, b in zip(vec_a, vec_b))


def build_semantic_index(
    chunk_text: dict[str, str],
    *,
    model_name: str,
) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    chunk_ids = list(chunk_text.keys())
    chunk_inputs = [f"passage: {chunk_text[cid]}" for cid in chunk_ids]
    embedder = SentenceTransformer(model_name)
    chunk_embeddings = embedder.encode(
        chunk_inputs,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return {
        "embedder": embedder,
        "chunk_ids": chunk_ids,
        "chunk_embeddings": [list(vec) for vec in chunk_embeddings],
    }


def semantic_chunk_ids(
    question: str,
    answer: str | None,
    semantic_index: dict[str, Any],
    *,
    max_chunk_ids: int,
    min_score: float,
) -> list[str]:
    if max_chunk_ids <= 0:
        return []
    query_text = f"query: {question.strip()} {answer or ''}".strip()
    query_embedding = semantic_index["embedder"].encode(
        [query_text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    query_vec = list(query_embedding)
    scored: list[tuple[float, str]] = []
    for cid, chunk_vec in zip(semantic_index["chunk_ids"], semantic_index["chunk_embeddings"]):
        score = _dot(query_vec, chunk_vec)
        if score >= min_score:
            scored.append((score, cid))
    scored.sort(reverse=True)
    return [cid for _, cid in scored[:max_chunk_ids]]


def _clip_ids(ids: list[str], max_chunk_ids: int) -> list[str]:
    if max_chunk_ids <= 0:
        return []
    return ids[:max_chunk_ids]


def resolve_chunk_ids(
    question: str,
    answer: str | None,
    qa_map: dict[str, list[str]],
    chunk_text: dict[str, str],
    keyword_df: dict[str, int],
    total_chunks: int,
    qa_questions: list[str],
    *,
    fuzzy_ratio: float,
    lexical_min_hits: int,
    max_chunk_ids: int,
    semantic_index: dict[str, Any] | None,
    semantic_min_score: float,
) -> tuple[list[str], str]:
    q = question.strip()
    if q in qa_map:
        return _clip_ids(qa_map[q], max_chunk_ids), "exact"

    fuzzy_q = fuzzy_match_question(q, qa_questions, fuzzy_ratio)
    if fuzzy_q and fuzzy_q in qa_map:
        return _clip_ids(qa_map[fuzzy_q], max_chunk_ids), "fuzzy"

    if semantic_index is not None:
        semantic_ids = semantic_chunk_ids(
            q,
            answer,
            semantic_index,
            max_chunk_ids=max_chunk_ids,
            min_score=semantic_min_score,
        )
        if semantic_ids:
            return semantic_ids, "semantic"

    lexical = lexical_chunk_ids(q, answer, chunk_text, keyword_df, total_chunks, lexical_min_hits, max_chunk_ids)
    if lexical:
        return lexical, "lexical"
    return [], "none"


def excerpt_for_chunk(
    chunk_text: dict[str, str],
    chunk_id: str,
    query_hint: str,
    answer_hint: str,
    *,
    max_len: int,
) -> str:
    text = chunk_text.get(chunk_id, "")
    one_line = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    if not one_line:
        return ""
    if len(one_line) <= max_len:
        return one_line

    answer_words = _expanded_keywords(keywords(answer_hint))
    query_words = keywords(query_hint)
    answer_bigrams = _ngrams(answer_words, 2)
    need_pair = "retriever" in answer_hint.lower() and "generator" in answer_hint.lower()
    sentences = _sentence_split(one_line)
    best_sentence = ""
    best_score = -1
    for idx in range(len(sentences)):
        window = " ".join(sentences[idx : idx + 3]).strip()
        window_low = window.lower()
        a_hits = {w for w in answer_words if word_in_text(w, window)}
        q_hits = {w for w in query_words if word_in_text(w, window)}
        score = (4 * len(a_hits)) + len(q_hits) + (3 * _phrase_hits(answer_bigrams, window_low))
        if need_pair:
            has_retrieval = any(word_in_text(w, window) for w in ("retriever", "retrieval", "retrieve"))
            has_generation = any(word_in_text(w, window) for w in ("generator", "generation", "generate"))
            if has_retrieval and has_generation:
                score += 10
        if score > best_score:
            best_score = score
            best_sentence = window

    source = best_sentence if best_sentence else one_line
    if len(source) <= max_len:
        return source

    kws = (answer_words + query_words)[:24]
    start = 0
    for w in kws:
        match = re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", source, re.IGNORECASE)
        if match:
            start = max(0, match.start() - max_len // 5)
            break

    snippet = source[start : start + max_len].strip()
    if start > 0:
        snippet = f"...{snippet}"
    if start + max_len < len(one_line):
        snippet = f"{snippet.rstrip()}..."
    return snippet


def parse_evaluation_blocks(lines: list[str]) -> list[EvalBlock]:
    blocks: list[EvalBlock] = []
    section = ""
    idx = 0
    total = len(lines)

    while idx < total:
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        if line.startswith(("Expected Evidence:", "Excerpt:")):
            idx += 1
            continue

        if line.endswith("?") and not line.startswith(("Distractor:", "Noise:")):
            question = line
            idx += 1
            while idx < total and not lines[idx].strip():
                idx += 1
            if idx >= total:
                break
            answer = lines[idx].strip()
            idx += 1
            while idx < total and not lines[idx].strip():
                idx += 1
            if idx >= total:
                break
            distractor_line = lines[idx].strip()
            idx += 1
            if not distractor_line.startswith("Distractor:"):
                raise ValueError(f"Expected Distractor after answer, got: {distractor_line[:80]!r}")
            distractor = distractor_line.split(":", 1)[1].strip()
            while idx < total and not lines[idx].strip():
                idx += 1
            if idx >= total:
                break
            noise_line = lines[idx].strip()
            idx += 1
            if not noise_line.startswith("Noise:"):
                raise ValueError(f"Expected Noise after distractor, got: {noise_line[:80]!r}")
            noise = noise_line.split(":", 1)[1].strip()
            blocks.append(EvalBlock(section=section, question=question, answer=answer, distractor=distractor, noise=noise))
            continue

        section = line
        idx += 1

    return blocks


def build_jsonl_records(
    blocks: list[EvalBlock],
    qa_map: dict[str, list[str]],
    chunk_text: dict[str, str],
    keyword_df: dict[str, int],
    total_chunks: int,
    qa_questions: list[str],
    *,
    fuzzy_ratio: float,
    lexical_min_hits: int,
    max_chunk_ids: int,
    excerpt_max: int,
    semantic_index: dict[str, Any] | None,
    semantic_min_score: float,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    stats: dict[str, int] = defaultdict(int)
    records: list[dict[str, object]] = []

    for index, block in enumerate(blocks):
        ids, method = resolve_chunk_ids(
            block.question,
            block.answer,
            qa_map,
            chunk_text,
            keyword_df,
            total_chunks,
            qa_questions,
            fuzzy_ratio=fuzzy_ratio,
            lexical_min_hits=lexical_min_hits,
            max_chunk_ids=max_chunk_ids,
            semantic_index=semantic_index,
            semantic_min_score=semantic_min_score,
        )
        stats[f"blocks_{method}"] += 1
        stats["blocks_total"] += 1

        hint = f"{block.question} {block.answer}"
        excerpt = ""
        if ids:
            stats["with_evidence"] += 1
            excerpt = excerpt_for_chunk(
                chunk_text,
                ids[0],
                hint,
                block.answer,
                max_len=excerpt_max,
            )
        else:
            stats["no_evidence"] += 1

        section_label = block.section or "Fundamentals of RAG"
        records.append(
            {
                "index": index,
                "section": section_label,
                "question": {"stem": block.question, "kind": "open_ended"},
                "reference_answer": block.answer,
                "distractor": block.distractor,
                "noise": block.noise,
                "expected_evidence": {
                    "chunk_ids": ids,
                    "excerpt": excerpt,
                    "resolution_method": method,
                },
            }
        )

    return records, dict(stats)


def rebalance_url_share(
    records: list[dict[str, object]],
    chunk_url: dict[str, str],
    *,
    max_url_share: float,
) -> list[dict[str, object]]:
    if max_url_share <= 0.0 or max_url_share >= 1.0:
        return records
    total_refs = sum(
        len((rec.get("expected_evidence") or {}).get("chunk_ids", []))  # type: ignore[union-attr]
        for rec in records
    )
    if total_refs <= 0:
        return records
    cap = max(1, int(total_refs * max_url_share))
    per_url_count: dict[str, int] = defaultdict(int)
    out: list[dict[str, object]] = []
    for rec in records:
        expected = dict(rec.get("expected_evidence") or {})
        ids = [str(cid) for cid in expected.get("chunk_ids", [])]
        if not ids:
            out.append(rec)
            continue
        kept: list[str] = []
        for cid in ids:
            url = chunk_url.get(cid, "")
            if not url or per_url_count[url] < cap:
                kept.append(cid)
        if not kept:
            fallback = min(ids, key=lambda cid: per_url_count.get(chunk_url.get(cid, ""), 0))
            kept = [fallback]
        for cid in kept:
            url = chunk_url.get(cid, "")
            if url:
                per_url_count[url] = per_url_count.get(url, 0) + 1
        expected["chunk_ids"] = kept
        rec = dict(rec)
        rec["expected_evidence"] = expected
        out.append(rec)
    return out


def rebalance_multi_gt_share(
    records: list[dict[str, object]],
    *,
    target_multi_gt_share: float,
    keep_max_ids_for_multi: int,
) -> list[dict[str, object]]:
    if target_multi_gt_share >= 1.0:
        return records
    if target_multi_gt_share < 0.0:
        target_multi_gt_share = 0.0
    keep_max_ids_for_multi = max(1, keep_max_ids_for_multi)
    total = len(records)
    if total <= 0:
        return records
    current_multi = sum(
        1 for rec in records if len((rec.get("expected_evidence") or {}).get("chunk_ids", [])) > 1  # type: ignore[union-attr]
    )
    allowed_multi = int(total * target_multi_gt_share)
    if current_multi <= allowed_multi:
        return records

    # Reduce multi-GT first on less reliable methods.
    method_rank = {"lexical": 0, "semantic": 1, "fuzzy": 2, "exact": 3, "manual_fill": 4}
    reducible: list[tuple[int, int, int]] = []
    for idx, rec in enumerate(records):
        expected = rec.get("expected_evidence") or {}
        ids = expected.get("chunk_ids", [])
        if len(ids) <= 1:
            continue
        method = str(expected.get("resolution_method", "lexical"))
        reducible.append((method_rank.get(method, 0), -len(ids), idx))
    reducible.sort()
    need_reduce = current_multi - allowed_multi

    for _, _, idx in reducible[:need_reduce]:
        rec = dict(records[idx])
        expected = dict(rec.get("expected_evidence") or {})
        ids = [str(cid) for cid in expected.get("chunk_ids", [])]
        expected["chunk_ids"] = ids[:keep_max_ids_for_multi]
        rec["expected_evidence"] = expected
        records[idx] = rec
    return records


def load_eval_samples(path: Path) -> list[EvalSample]:
    samples: list[EvalSample] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            row = json.loads(line)
            question = row.get("question", {})
            expected = row.get("expected_evidence", {})
            query = str(question.get("stem", "")).strip()
            relevant_docs = [str(doc_id) for doc_id in expected.get("chunk_ids", [])]
            if not query:
                continue
            samples.append(
                EvalSample(
                    query=query,
                    relevant_docs=relevant_docs,
                    reference_answer=row.get("reference_answer"),
                    metadata={
                        "index": row.get("index"),
                        "section": row.get("section"),
                        "resolution_method": expected.get("resolution_method"),
                    },
                )
            )
    return samples


def split_eval_samples(
    samples: list[EvalSample],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[EvalSample], list[EvalSample]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in range [0.0, 1.0).")
    data = list(samples)
    rng = random.Random(seed)
    rng.shuffle(data)
    val_size = int(len(data) * val_ratio)
    return data[val_size:], data[:val_size]


def build_evaluation_dataset(
    rag_path: Path,
    eval_txt_path: Path,
    out_path: Path,
    *,
    fuzzy_ratio: float = 0.86,
    lexical_min_hits: int = 2,
    max_chunk_ids: int = 2,
    semantic_fallback: bool = True,
    semantic_model: str = "intfloat/e5-small-v2",
    semantic_min_score: float = 0.56,
    max_gt_url_share: float = 0.25,
    target_multi_gt_share: float = 0.4,
    keep_max_ids_for_multi: int = 1,
    excerpt_max: int = 320,
) -> tuple[int, dict[str, int]]:
    qa_map, chunk_text, chunk_url, qa_questions = load_dataset(rag_path)
    keyword_df, total_chunks = build_keyword_df(chunk_text)
    semantic_index = None
    if semantic_fallback:
        semantic_index = build_semantic_index(chunk_text, model_name=semantic_model)
    blocks = parse_evaluation_blocks(eval_txt_path.read_text(encoding="utf-8").splitlines())
    records, stats = build_jsonl_records(
        blocks,
        qa_map,
        chunk_text,
        keyword_df,
        total_chunks,
        qa_questions,
        fuzzy_ratio=fuzzy_ratio,
        lexical_min_hits=lexical_min_hits,
        max_chunk_ids=max_chunk_ids,
        semantic_index=semantic_index,
        semantic_min_score=semantic_min_score,
        excerpt_max=excerpt_max,
    )
    records = rebalance_url_share(records, chunk_url, max_url_share=max_gt_url_share)
    records = rebalance_multi_gt_share(
        records,
        target_multi_gt_share=target_multi_gt_share,
        keep_max_ids_for_multi=keep_max_ids_for_multi,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as output:
        for rec in records:
            output.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records), stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build structured evaluation JSONL from evaluation.txt + rag_dataset.jsonl.",
    )
    parser.add_argument("--rag", type=Path, default=Path("data/rag_dataset.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("data/evaluation.txt"))
    parser.add_argument("--out", type=Path, default=Path("data/evaluation_with_evidence.jsonl"))
    parser.add_argument("--fuzzy-ratio", type=float, default=0.86)
    parser.add_argument("--lexical-min-hits", type=int, default=2)
    parser.add_argument("--max-chunk-ids", type=int, default=2)
    parser.add_argument("--no-semantic-fallback", action="store_true")
    parser.add_argument("--semantic-model", default="intfloat/e5-small-v2")
    parser.add_argument("--semantic-min-score", type=float, default=0.56)
    parser.add_argument("--max-gt-url-share", type=float, default=0.25)
    parser.add_argument("--target-multi-gt-share", type=float, default=0.4)
    parser.add_argument("--keep-max-ids-for-multi", type=int, default=1)
    parser.add_argument("--excerpt-max", type=int, default=320)
    args = parser.parse_args()

    count, stats = build_evaluation_dataset(
        rag_path=args.rag,
        eval_txt_path=args.eval,
        out_path=args.out,
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


if __name__ == "__main__":
    main()
