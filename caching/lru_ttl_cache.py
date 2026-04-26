from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import time
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    size: int
    capacity: int
    hit_rate: float
    cleanup_runs: int
    expired_removed: int
    evicted_lru: int


@dataclass
class CacheEntry(Generic[K, V]):
    key: K
    value: V
    created_at: float
    access_count: int
    ttl_seconds: float

    def is_expired(self, now: float) -> bool:
        return (now - self.created_at) >= self.ttl_seconds


class LRUTTLCache(Generic[K, V]):
    """
    In-memory LRU cache with per-entry TTL expiration.

    Core behavior:
    - LRU eviction when capacity is exceeded
    - TTL expiration on read and periodic cleanup
    - Opportunistic auto-cleanup every `cleanup_interval_seconds`
    - Hit/miss counters and eviction/cleanup telemetry
    """

    def __init__(
        self,
        capacity: int,
        ttl_seconds: float,
        cleanup_interval_seconds: float = 30.0,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        if cleanup_interval_seconds <= 0:
            raise ValueError("cleanup_interval_seconds must be > 0")

        self.capacity = capacity
        self.default_ttl_seconds = ttl_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds

        self._entries: OrderedDict[K, CacheEntry[K, V]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._cleanup_runs = 0
        self._expired_removed = 0
        self._evicted_lru = 0
        self._last_cleanup_at = time.monotonic()

    def _run_cleanup_if_due(self) -> None:
        now_mono = time.monotonic()
        if (now_mono - self._last_cleanup_at) < self.cleanup_interval_seconds:
            return
        self.cleanup_expired()
        self._last_cleanup_at = time.monotonic()

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
            self._evicted_lru += 1

    def set(self, key: K, value: V, ttl_seconds: float | None = None) -> None:
        self._run_cleanup_if_due()
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        if ttl <= 0:
            raise ValueError("ttl_seconds must be > 0")

        now = time.time()
        if key in self._entries:
            entry = self._entries[key]
            entry.value = value
            entry.created_at = now
            entry.ttl_seconds = ttl
            self._entries.move_to_end(key)
        else:
            self._entries[key] = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                access_count=0,
                ttl_seconds=ttl,
            )
        self._evict_if_needed()

    def get(self, key: K, default: V | None = None) -> V | None:
        self._run_cleanup_if_due()
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return default

        now = time.time()
        if entry.is_expired(now):
            del self._entries[key]
            self._expired_removed += 1
            self._misses += 1
            return default

        entry.access_count += 1
        self._entries.move_to_end(key)
        self._hits += 1
        return entry.value

    def contains(self, key: K) -> bool:
        self._run_cleanup_if_due()
        entry = self._entries.get(key)
        if entry is None:
            return False
        if entry.is_expired(time.time()):
            del self._entries[key]
            self._expired_removed += 1
            return False
        return True

    def delete(self, key: K) -> bool:
        self._run_cleanup_if_due()
        if key not in self._entries:
            return False
        del self._entries[key]
        return True

    def cleanup_expired(self) -> int:
        now = time.time()
        expired_keys = [key for key, entry in self._entries.items() if entry.is_expired(now)]
        for key in expired_keys:
            del self._entries[key]
        removed = len(expired_keys)
        self._cleanup_runs += 1
        self._expired_removed += removed
        return removed

    def clear(self) -> None:
        self._entries.clear()

    def get_entry_metadata(self, key: K) -> dict[str, float | int | str] | None:
        self._run_cleanup_if_due()
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.is_expired(time.time()):
            del self._entries[key]
            self._expired_removed += 1
            return None
        return {
            "key": str(entry.key),
            "created_at": entry.created_at,
            "access_count": entry.access_count,
            "ttl_seconds": entry.ttl_seconds,
        }

    def stats(self) -> CacheStats:
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total else 0.0
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            size=len(self._entries),
            capacity=self.capacity,
            hit_rate=hit_rate,
            cleanup_runs=self._cleanup_runs,
            expired_removed=self._expired_removed,
            evicted_lru=self._evicted_lru,
        )

    def __len__(self) -> int:
        return len(self._entries)

