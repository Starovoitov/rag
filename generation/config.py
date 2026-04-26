from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypedDict

from generation.llm import LLMConfig

DEFAULT_LLM_CONFIG_PATH = "llm.config.json"


class LLMProviderPayload(TypedDict):
    provider: str
    model: str
    api_base: str
    api_key_env: str


class LLMProvidersConfigPayload(TypedDict):
    providers: dict[str, LLMProviderPayload]


def _validate_provider_payload(name: str, payload: LLMProviderPayload) -> LLMProviderPayload:
    required = ("provider", "model", "api_base", "api_key_env")
    for key in required:
        if key not in payload:
            raise ValueError(f"Provider '{name}' missing required key '{key}'")
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"Provider '{name}' key '{key}' must be a non-empty string")
    return payload


def load_llm_provider_configs(config_path: str = DEFAULT_LLM_CONFIG_PATH) -> dict[str, LLMConfig]:
    """Load named LLM provider defaults from JSON config."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"LLM config not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    providers = raw.get("providers")
    if not isinstance(providers, dict) or not providers:
        raise ValueError(f"Invalid llm config '{path}': top-level 'providers' must be a non-empty object")

    result: dict[str, LLMConfig] = {}
    for name, payload in providers.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Provider names must be non-empty strings")
        if not isinstance(payload, dict):
            raise ValueError(f"Provider '{name}' must be an object")
        validated = _validate_provider_payload(name, payload)
        env_name = validated["api_key_env"]
        result[name] = LLMConfig(
            provider=validated["provider"],
            model=validated["model"],
            api_base=validated["api_base"],
            api_key=os.getenv(env_name),
        )
    return result
