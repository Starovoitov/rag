from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SYSTEM_PROMPT = """
You are a retrieval-augmented assistant.

Critical rules:
1. Only use provided context.
2. If answer is not in context, say: "Not found in provided sources."
3. Do not use prior knowledge.
4. Be concise and factual.
5. Always cite sources like [1], [2].
""".strip()

USER_PROMPT = """
You will be given numbered sources.

Sources:
{context}

Question:
{question}

Answer using only the sources above.
If multiple sources are relevant, combine them.
Cite sources after each sentence.
""".strip()


@dataclass
class SourceChunk:
    """RAG source chunk with optional retrieval score and metadata."""

    doc_id: str
    text: str
    score: float | None = None
    metadata: dict[str, Any] | None = None


def estimate_tokens(text: str) -> int:
    """
    Lightweight token estimate for context truncation.

    Rough rule of thumb: ~4 characters per token for English text.
    """
    return max(1, len(text) // 4)


def merge_top_k_documents(chunks: list[SourceChunk], top_k: int) -> list[SourceChunk]:
    """Keep top-k chunks by score when available, preserving stable order on ties."""
    if top_k <= 0:
        return []
    ranked = sorted(
        chunks,
        key=lambda c: c.score if c.score is not None else float("-inf"),
        reverse=True,
    )
    return ranked[:top_k]


def _source_header(idx: int, chunk: SourceChunk) -> str:
    meta = chunk.metadata or {}
    title = str(meta.get("title", "")).strip()
    url = str(meta.get("url", "")).strip()
    parts = [f"[{idx}]"]
    if title:
        parts.append(f"title={title}")
    if url:
        parts.append(f"url={url}")
    return " ".join(parts)


def format_context_with_citations(
    chunks: list[SourceChunk],
    max_context_tokens: int = 2500,
) -> tuple[str, list[SourceChunk]]:
    """
    Build numbered context block and truncate to token budget.

    Returns:
    - context text for the prompt
    - the subset of chunks actually included (for traceability)
    """
    if max_context_tokens <= 0 or not chunks:
        return "", []

    selected: list[SourceChunk] = []
    lines: list[str] = []
    used_tokens = 0

    for idx, chunk in enumerate(chunks, start=1):
        header = _source_header(idx, chunk)
        body = chunk.text.strip()
        if not body:
            continue

        block = f"{header}\n{body}"
        block_tokens = estimate_tokens(block)
        if used_tokens + block_tokens > max_context_tokens:
            remaining = max_context_tokens - used_tokens
            if remaining <= 20:
                break
            max_chars = remaining * 4
            clipped = body[:max_chars].rstrip()
            if clipped:
                block = f"{header}\n{clipped}"
                block_tokens = estimate_tokens(block)
            else:
                break

        lines.append(block)
        selected.append(chunk)
        used_tokens += block_tokens

        if used_tokens >= max_context_tokens:
            break

    return "\n\n".join(lines), selected


def build_rag_messages(
    question: str,
    chunks: list[SourceChunk],
    top_k: int = 5,
    max_context_tokens: int = 2500,
) -> dict[str, Any]:
    """
    Build system/user prompts with anti-hallucination guardrails and citations.
    """
    candidates = merge_top_k_documents(chunks, top_k=top_k)
    context, used_chunks = format_context_with_citations(
        candidates,
        max_context_tokens=max_context_tokens,
    )

    anti_hallucination_guard = (
        "If the sources are insufficient or contradictory, answer exactly: "
        "\"Not found in provided sources.\""
    )
    system = f"{SYSTEM_PROMPT}\n\nAdditional guardrail:\n{anti_hallucination_guard}"
    user = USER_PROMPT.format(context=context or "[no sources provided]", question=question)

    return {
        "system_prompt": system,
        "user_prompt": user,
        "used_chunks": used_chunks,
        "context_tokens_estimate": estimate_tokens(context) if context else 0,
    }
