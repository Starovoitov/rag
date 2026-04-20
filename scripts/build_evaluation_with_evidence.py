#!/usr/bin/env python3
"""Build a structured evaluation dataset (JSONL) from data/evaluation.txt + rag_dataset.jsonl.

Each line is one JSON object with section, structured question, gold answer, distractor, noise,
and expected_evidence (chunk_ids, excerpt, resolution method). Items before the first section
heading use section \"Fundamentals of RAG\".

Resolution order for expected chunk_ids:
1. Exact match on qa_pair.question → metadata.chunk_id(s)
2. difflib fuzzy match when ratio >= --fuzzy-ratio
3. Lexical overlap of keywords from question + answer against raw_chunk (+ metadata)
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path


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


def load_dataset(
    rag_jsonl: Path,
) -> tuple[dict[str, list[str]], dict[str, str], list[str]]:
    qa_map: dict[str, list[str]] = defaultdict(list)
    chunk_text: dict[str, str] = {}

    with rag_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rtype = row.get("record_type")
            if rtype == "qa_pair":
                q = str(row.get("question", "")).strip()
                meta = row.get("metadata") or {}
                cid = meta.get("chunk_id")
                if not q or not cid:
                    continue
                cid = str(cid)
                if cid not in qa_map[q]:
                    qa_map[q].append(cid)
            elif rtype == "raw_chunk":
                cid = row.get("chunk_id")
                text = row.get("text")
                if cid and text is not None:
                    cid = str(cid)
                    body = str(text)
                    chunk_text[cid] = body

    return dict(qa_map), chunk_text, sorted(qa_map.keys())


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
    df = keyword_df.get(word, 0)
    return math.log1p(total_chunks / (1 + df))


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
    # Preserve order while deduplicating
    return list(dict.fromkeys(out))


def lexical_chunk_ids(
    question: str,
    answer: str | None,
    chunk_text: dict[str, str],
    keyword_df: dict[str, int],
    total_chunks: int,
    min_hits: int,
) -> list[str]:
    kws_q = keywords(question)
    kws_a = keywords(answer or "")
    metric_terms = _metric_terms(answer or "")
    answer_text_low = (answer or "").lower()
    concept_terms = {t for t in SUPPORT_CONCEPT_TERMS if t in answer_text_low}
    answer_bigrams = _ngrams(kws_a, 2)
    if not kws_q and not kws_a:
        return []

    rows: list[tuple[float, int, float, int, int, str]] = []
    uniq_a = set(kws_a)
    min_answer_hits = max(2, math.ceil(len(uniq_a) * 0.22)) if uniq_a else 0
    if uniq_a and (metric_terms or concept_terms):
        min_answer_hits = max(1, math.ceil(len(uniq_a) * 0.16))
    for cid, text in chunk_text.items():
        tlow = text.lower()
        tnorm = tlow.replace(" @ ", "@").replace(" / ", "/")
        matched_q = {w for w in kws_q if word_in_text(w, text)}
        matched_a = {w for w in kws_a if word_in_text(w, text)}
        distinct = len(matched_q) + (2 * len(matched_a))
        if distinct < min_hits:
            continue
        if uniq_a and len(matched_a) < min_answer_hits:
            continue
        if metric_terms:
            metric_hits = sum(1 for t in metric_terms if t in tnorm)
            min_metric_hits = 1
            if metric_hits < min_metric_hits:
                continue
        if len(concept_terms) >= 2:
            concept_hits = sum(1 for t in concept_terms if t in tnorm)
            min_concept_hits = 1
            if concept_hits < min_concept_hits:
                continue

        phrase_bonus = _phrase_hits(answer_bigrams, tlow)
        q_weight = sum(_idf(w, keyword_df, total_chunks) * _lexeme_weight(w) for w in matched_q)
        a_weight = sum(_idf(w, keyword_df, total_chunks) * _lexeme_weight(w) for w in matched_a)
        score = (
            q_weight
            + (3.5 * a_weight)
            + (2.5 * phrase_bonus)
        )
        weighted = sum(tlow.count(w) for w in matched_q | matched_a if len(w) >= 3)
        sentence_best = 0
        for sent in _sentence_split(text):
            slow = sent.lower()
            sent_a = {w for w in kws_a if word_in_text(w, sent)}
            sent_q = {w for w in kws_q if word_in_text(w, sent)}
            if kws_a and not sent_a:
                continue
            sent_score = (4 * len(sent_a)) + len(sent_q) + (3 * _phrase_hits(answer_bigrams, slow))
            sentence_best = max(sentence_best, sent_score)
        min_sentence_score = 5
        if metric_terms or concept_terms:
            min_sentence_score = 3
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
    min_ratio = 0.24
    if metric_terms or concept_terms:
        min_ratio = 0.12
    min_top_sentence = 6
    if metric_terms or concept_terms:
        min_top_sentence = 4
    if uniq_a and (top_answer_ratio < min_ratio or top_sentence < min_top_sentence or top_score < 8):
        return []
    return [top[5]]


def fuzzy_match_question(
    question: str,
    qa_questions: list[str],
    min_ratio: float,
) -> str | None:
    if not qa_questions:
        return None
    q = question.strip()
    candidates = get_close_matches(q, qa_questions, n=1, cutoff=max(0.0, min_ratio - 0.01))
    if not candidates:
        return None
    cand = candidates[0]
    ratio = SequenceMatcher(None, q.lower(), cand.lower()).ratio()
    if ratio < min_ratio:
        return None
    return cand


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
) -> tuple[list[str], str]:
    q = question.strip()
    if q in qa_map:
        return qa_map[q], "exact"

    fuzzy_q = fuzzy_match_question(q, qa_questions, fuzzy_ratio)
    if fuzzy_q and fuzzy_q in qa_map:
        return qa_map[fuzzy_q], "fuzzy"

    lex = lexical_chunk_ids(q, answer, chunk_text, keyword_df, total_chunks, lexical_min_hits)
    if lex:
        return lex, "lexical"

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
    need_retrieval_generator_pair = "retriever" in answer_hint.lower() and "generator" in answer_hint.lower()
    sentences = _sentence_split(one_line)
    best_sentence = ""
    best_score = -1
    for i in range(len(sentences)):
        window = " ".join(sentences[i : i + 3]).strip()
        sent_low = window.lower()
        a_hits = {w for w in answer_words if word_in_text(w, window)}
        q_hits = {w for w in query_words if word_in_text(w, window)}
        score = (4 * len(a_hits)) + len(q_hits) + (3 * _phrase_hits(answer_bigrams, sent_low))
        if need_retrieval_generator_pair:
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
        m = re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", source, re.IGNORECASE)
        if m:
            start = max(0, m.start() - max_len // 5)
            break

    snippet = source[start : start + max_len].strip()
    if start > 0:
        snippet = f"...{snippet}"
    if start + max_len < len(one_line):
        snippet = f"{snippet.rstrip()}..."
    return snippet


def parse_evaluation_blocks(lines: list[str]) -> list[EvalBlock]:
    """Parse plain evaluation.txt (or legacy .txt with evidence lines — those are skipped)."""
    blocks: list[EvalBlock] = []
    section = ""
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith(("Expected Evidence:", "Excerpt:")):
            i += 1
            continue

        if line.endswith("?") and not line.startswith(("Distractor:", "Noise:")):
            question = line
            i += 1
            while i < n and not lines[i].strip():
                i += 1
            if i >= n:
                break
            answer = lines[i].strip()
            i += 1
            while i < n and not lines[i].strip():
                i += 1
            if i >= n:
                break
            d_line = lines[i].strip()
            i += 1
            if not d_line.startswith("Distractor:"):
                raise ValueError(f"Expected Distractor after answer, got: {d_line[:80]!r}")
            distractor = d_line.split(":", 1)[1].strip()
            while i < n and not lines[i].strip():
                i += 1
            if i >= n:
                break
            n_line = lines[i].strip()
            i += 1
            if not n_line.startswith("Noise:"):
                raise ValueError(f"Expected Noise after distractor, got: {n_line[:80]!r}")
            noise = n_line.split(":", 1)[1].strip()
            blocks.append(EvalBlock(section, question, answer, distractor, noise))
            continue

        section = line
        i += 1

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
    excerpt_max: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    stats: dict[str, int] = defaultdict(int)
    records: list[dict[str, object]] = []

    for idx, block in enumerate(blocks):
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
        record: dict[str, object] = {
            "index": idx,
            "section": section_label,
            "question": {
                "stem": block.question,
                "kind": "open_ended",
            },
            "reference_answer": block.answer,
            "distractor": block.distractor,
            "noise": block.noise,
            "expected_evidence": {
                "chunk_ids": ids,
                "excerpt": excerpt,
                "resolution_method": method,
            },
        }
        records.append(record)

    return records, dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build structured evaluation JSONL from evaluation.txt + rag_dataset.jsonl.",
    )
    parser.add_argument(
        "--rag",
        type=Path,
        default=Path("data/rag_dataset.jsonl"),
        help="Path to rag_dataset.jsonl",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=Path("data/evaluation.txt"),
        help="Path to evaluation.txt (plain blocks)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/evaluation_with_evidence.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--fuzzy-ratio", type=float, default=0.82)
    parser.add_argument("--lexical-min-hits", type=int, default=2)
    parser.add_argument("--excerpt-max", type=int, default=320)
    args = parser.parse_args()

    qa_map, chunk_text, qa_questions = load_dataset(args.rag)
    keyword_df, total_chunks = build_keyword_df(chunk_text)
    raw_lines = args.eval.read_text(encoding="utf-8").splitlines()
    blocks = parse_evaluation_blocks(raw_lines)
    records, stats = build_jsonl_records(
        blocks,
        qa_map,
        chunk_text,
        keyword_df,
        total_chunks,
        qa_questions,
        fuzzy_ratio=args.fuzzy_ratio,
        lexical_min_hits=args.lexical_min_hits,
        excerpt_max=args.excerpt_max,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f_out:
        for rec in records:
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {args.out} ({len(records)} records).")
    print("Stats:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
