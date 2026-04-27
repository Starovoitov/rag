from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from parser.models import SourceSpec

DEFAULT_SOURCES_CONFIG_PATH = "sources.config.json"


class SourceSpecPayload(TypedDict):
    category: str
    subtopic: str
    url: str
    source_type: str
    priority_topics: list[str]


class SourcesConfigPayload(TypedDict):
    sources: list[SourceSpecPayload]


class AliasGroupPayload(TypedDict):
    primary: str
    aliases: list[str]


class SeedChunkPayload(TypedDict):
    title: str
    content: str


def _load_sources_config(config_path: str = DEFAULT_SOURCES_CONFIG_PATH) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Sources config not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid sources config '{path}': expected top-level JSON object")
    return raw


def _parse_source_spec(payload: SourceSpecPayload, idx: int) -> SourceSpec:
    required_keys = ("category", "subtopic", "url", "source_type", "priority_topics")
    for key in required_keys:
        if key not in payload:
            raise ValueError(f"sources[{idx}] missing required key '{key}'")
    priority_topics = payload["priority_topics"]
    if not isinstance(priority_topics, list) or not all(isinstance(item, str) for item in priority_topics):
        raise ValueError(f"sources[{idx}].priority_topics must be a list[str]")
    return SourceSpec(
        category=str(payload["category"]).strip(),
        subtopic=str(payload["subtopic"]).strip(),
        url=str(payload["url"]).strip(),
        source_type=str(payload["source_type"]).strip(),
        priority_topics=[item.strip() for item in priority_topics if str(item).strip()],
    )


def build_sources(config_path: str = DEFAULT_SOURCES_CONFIG_PATH) -> list[SourceSpec]:
    """Load source specs from a JSON config file."""
    raw = _load_sources_config(config_path=config_path)
    sources = raw.get("sources")
    if not isinstance(sources, list):
        raise ValueError(f"Invalid sources config '{config_path}': top-level 'sources' must be a list")
    return [_parse_source_spec(payload, idx) for idx, payload in enumerate(sources)]


def build_alias_groups(config_path: str = DEFAULT_SOURCES_CONFIG_PATH) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Load alias groups used for chunk enrichment."""
    raw = _load_sources_config(config_path=config_path)
    payload = raw.get("alias_groups", [])
    if not isinstance(payload, list):
        raise ValueError(f"Invalid sources config '{config_path}': 'alias_groups' must be a list")

    result: list[tuple[str, tuple[str, ...]]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"alias_groups[{idx}] must be an object")
        primary = str(item.get("primary", "")).strip()
        aliases = item.get("aliases", [])
        if not primary:
            raise ValueError(f"alias_groups[{idx}].primary must be a non-empty string")
        if not isinstance(aliases, list) or not all(isinstance(alias, str) for alias in aliases):
            raise ValueError(f"alias_groups[{idx}].aliases must be a list[str]")
        cleaned_aliases = tuple(alias.strip() for alias in aliases if alias.strip())
        result.append((primary, cleaned_aliases))
    return tuple(result)


def build_seed_chunks(config_path: str = DEFAULT_SOURCES_CONFIG_PATH) -> tuple[dict[str, str], ...]:
    """Load synthetic seed chunks that are always appended to dataset output."""
    raw = _load_sources_config(config_path=config_path)
    payload = raw.get("multi_hop_seed_chunks", [])
    if not isinstance(payload, list):
        raise ValueError(f"Invalid sources config '{config_path}': 'multi_hop_seed_chunks' must be a list")

    result: list[dict[str, str]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"multi_hop_seed_chunks[{idx}] must be an object")
        title = str(item.get("title", "")).strip()
        content = str(item.get("content", "")).strip()
        if not title:
            raise ValueError(f"multi_hop_seed_chunks[{idx}].title must be a non-empty string")
        if not content:
            raise ValueError(f"multi_hop_seed_chunks[{idx}].content must be a non-empty string")
        result.append({"title": title, "content": content})
    return tuple(result)

