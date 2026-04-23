from __future__ import annotations


def _is_e5_model(model_name: str) -> bool:
    name = (model_name or "").lower()
    return "e5" in name


def format_query_for_embedding(query: str, model_name: str) -> str:
    if _is_e5_model(model_name):
        return f"query: {query}"
    return query


def format_passage_for_embedding(passage: str, model_name: str) -> str:
    if _is_e5_model(model_name):
        return f"passage: {passage}"
    return passage
