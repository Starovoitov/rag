from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

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
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Sources config not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    sources = raw.get("sources")
    if not isinstance(sources, list):
        raise ValueError(f"Invalid sources config '{path}': top-level 'sources' must be a list")
    return [_parse_source_spec(payload, idx) for idx, payload in enumerate(sources)]

