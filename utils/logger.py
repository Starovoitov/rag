from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """Structured JSON formatter for logs with optional payload."""

    def format(self, record: logging.LogRecord) -> str:
        base: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }
        payload = getattr(record, "payload", None)
        if isinstance(payload, dict):
            base.update(payload)
        return json.dumps(base, ensure_ascii=False)


def get_json_logger(name_prefix: str, log_path: str) -> logging.Logger:
    """Create/reuse a file logger writing one JSON event per line."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger_name = f"{name_prefix}.{hash(path.resolve())}"
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, payload: dict[str, Any]) -> None:
    """Emit structured event payload through standard logging."""
    logger.info("event", extra={"payload": payload})


def _parse_log_level(level: str) -> int:
    normalized = level.strip().upper()
    value = getattr(logging, normalized, None)
    if not isinstance(value, int):
        raise ValueError(f"Unsupported log level: {level}")
    return value


def configure_runtime_logger(
    name: str,
    *,
    level: str = "INFO",
    log_path: str | None = None,
    json_logs: bool = False,
) -> logging.Logger:
    """
    Create/reuse a runtime logger for CLI flows.

    - Always logs to stderr.
    - Optionally writes to file when `log_path` is provided.
    - Supports text or JSON formatting.
    """
    logger = logging.getLogger(name)
    logger.setLevel(_parse_log_level(level))
    logger.propagate = False

    if logger.handlers:
        return logger

    if json_logs:
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
