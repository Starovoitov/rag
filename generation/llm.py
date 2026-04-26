from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass
from typing import Any, Iterator

import requests
from caching import LRUTTLCache
from retry import retry
from utils.logger import get_json_logger, log_event


@dataclass
class LLMConfig:
    """Generic runtime settings for LLM calls across providers."""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_base: str = "https://api.openai.com/v1/chat/completions"
    api_key: str | None = None
    timeout_seconds: int = 60
    retries: int = 2
    retry_backoff_seconds: float = 1.5
    max_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95
    enable_streaming: bool = False
    log_path: str = "experiments/logs/llm_api_calls.jsonl"
    cache_enabled: bool = False
    cache_capacity: int = 512
    cache_ttl_seconds: float = 300.0


_LLM_RESPONSE_CACHE: LRUTTLCache[str, str] | None = None


def _llm_cache_key(system_prompt: str, user_prompt: str, config: LLMConfig) -> str:
    payload = {
        "provider": config.provider,
        "model": config.model,
        "api_base": config.api_base,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _get_llm_cache(config: LLMConfig) -> LRUTTLCache[str, str]:
    global _LLM_RESPONSE_CACHE
    if _LLM_RESPONSE_CACHE is None:
        _LLM_RESPONSE_CACHE = LRUTTLCache(
            capacity=max(1, config.cache_capacity),
            ttl_seconds=max(0.1, config.cache_ttl_seconds),
            cleanup_interval_seconds=30.0,
        )
        return _LLM_RESPONSE_CACHE

    if (
        _LLM_RESPONSE_CACHE.capacity != max(1, config.cache_capacity)
        or _LLM_RESPONSE_CACHE.default_ttl_seconds != max(0.1, config.cache_ttl_seconds)
    ):
        _LLM_RESPONSE_CACHE = LRUTTLCache(
            capacity=max(1, config.cache_capacity),
            ttl_seconds=max(0.1, config.cache_ttl_seconds),
            cleanup_interval_seconds=30.0,
        )
    return _LLM_RESPONSE_CACHE


def _headers(config: LLMConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        # OpenAI-compatible and many hosted endpoints accept Bearer auth.
        headers["Authorization"] = f"Bearer {config.api_key}"
    return headers


def _payload(system_prompt: str, user_prompt: str, config: LLMConfig, stream: bool) -> dict[str, Any]:
    return {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "stream": stream,
    }


def _extract_text_from_json(data: dict[str, Any]) -> str:
    choices = data.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    if isinstance(message, dict):
        return str(message.get("content", "")).strip()
    return ""


def _stream_openai_compatible(response: requests.Response) -> Iterator[str]:
    """
    Parse SSE chunks for OpenAI-compatible chat completion streaming.
    """
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = event.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        token = delta.get("content")
        if token:
            yield str(token)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    config: LLMConfig | None = None,
) -> str:
    """
    Call an LLM endpoint with retry, token settings, and logging.

    This method expects an OpenAI-compatible chat-completions endpoint.
    """
    conf = config or LLMConfig()
    start = time.perf_counter()
    logger = get_json_logger("generation.llm", conf.log_path)
    cache_key = ""
    cache: LRUTTLCache[str, str] | None = None
    if conf.cache_enabled:
        cache = _get_llm_cache(conf)
        cache_key = _llm_cache_key(system_prompt, user_prompt, conf)
        cached_answer = cache.get(cache_key)
        if cached_answer is not None:
            log_event(
                logger,
                {
                    "provider": conf.provider,
                    "model": conf.model,
                    "stream": False,
                    "elapsed_ms": int((time.perf_counter() - start) * 1000),
                    "ok": True,
                    "cache_hit": True,
                    "answer_preview": cached_answer[:200],
                    "max_tokens": conf.max_tokens,
                    "temperature": conf.temperature,
                    "top_p": conf.top_p,
                    "retries_configured": conf.retries,
                },
            )
            return cached_answer

    @retry(
        exceptions=Exception,
        tries=conf.retries + 1,
        delay=conf.retry_backoff_seconds,
        backoff=1,
    )
    def _request_once_impl() -> str:
        response = requests.post(
            conf.api_base,
            headers=_headers(conf),
            json=_payload(system_prompt, user_prompt, conf, stream=False),
            timeout=conf.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return _extract_text_from_json(data)

    try:
        answer = _request_once_impl()
        if cache is not None and cache_key:
            cache.set(cache_key, answer)
        log_event(
            logger,
            {
                "provider": conf.provider,
                "model": conf.model,
                "stream": False,
                "elapsed_ms": int((time.perf_counter() - start) * 1000),
                "ok": True,
                "cache_hit": False,
                "answer_preview": answer[:200],
                "max_tokens": conf.max_tokens,
                "temperature": conf.temperature,
                "top_p": conf.top_p,
                "retries_configured": conf.retries,
            },
        )
        return answer
    except Exception as exc:  # noqa: BLE001
        log_event(
            logger,
            {
                "provider": conf.provider,
                "model": conf.model,
                "stream": False,
                "ok": False,
                "elapsed_ms": int((time.perf_counter() - start) * 1000),
                "error": str(exc),
                "retries_configured": conf.retries,
            },
        )
        raise RuntimeError(f"LLM request failed after retries: {exc}") from exc


def stream_llm(
    system_prompt: str,
    user_prompt: str,
    config: LLMConfig | None = None,
) -> Iterator[str]:
    """
    Stream answer tokens for OpenAI-compatible endpoints.

    Caller can print each yielded token for live output.
    """
    conf = config or LLMConfig(enable_streaming=True)
    start = time.perf_counter()
    logger = get_json_logger("generation.llm", conf.log_path)
    response = requests.post(
        conf.api_base,
        headers=_headers(conf),
        json=_payload(system_prompt, user_prompt, conf, stream=True),
        timeout=conf.timeout_seconds,
        stream=True,
    )
    response.raise_for_status()

    full_text_parts: list[str] = []
    for token in _stream_openai_compatible(response):
        full_text_parts.append(token)
        yield token

    log_event(
        logger,
        {
            "provider": conf.provider,
            "model": conf.model,
            "stream": True,
            "elapsed_ms": int((time.perf_counter() - start) * 1000),
            "ok": True,
            "answer_preview": "".join(full_text_parts)[:200],
            "max_tokens": conf.max_tokens,
            "temperature": conf.temperature,
            "top_p": conf.top_p,
        },
    )
