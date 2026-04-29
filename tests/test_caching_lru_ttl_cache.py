from __future__ import annotations

import time
import unittest

from caching.lru_ttl_cache import LRUTTLCache


class TestLRUTTLCache(unittest.TestCase):
    def test_init_validation(self) -> None:
        with self.assertRaises(ValueError):
            LRUTTLCache[int, int](capacity=0, ttl_seconds=1.0)
        with self.assertRaises(ValueError):
            LRUTTLCache[int, int](capacity=1, ttl_seconds=0.0)

    def test_lru_eviction(self) -> None:
        cache = LRUTTLCache[str, int](capacity=2, ttl_seconds=10.0)
        cache.set("a", 1)
        cache.set("b", 2)
        self.assertEqual(cache.get("a"), 1)  # mark "a" as most recent
        cache.set("c", 3)  # evicts "b"
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), 1)
        self.assertEqual(cache.get("c"), 3)

    def test_ttl_expiration(self) -> None:
        cache = LRUTTLCache[str, str](capacity=2, ttl_seconds=0.05)
        cache.set("k", "v")
        time.sleep(0.06)
        self.assertIsNone(cache.get("k"))

    def test_stats_tracking(self) -> None:
        cache = LRUTTLCache[str, int](capacity=2, ttl_seconds=10.0)
        cache.set("x", 10)
        self.assertEqual(cache.get("x"), 10)  # hit
        self.assertIsNone(cache.get("missing"))  # miss
        stats = cache.stats()
        self.assertEqual(stats.hits, 1)
        self.assertEqual(stats.misses, 1)
        self.assertAlmostEqual(stats.hit_rate, 0.5)

    def test_delete_and_contains(self) -> None:
        cache = LRUTTLCache[str, int](capacity=2, ttl_seconds=10.0)
        cache.set("k", 1)
        self.assertTrue(cache.contains("k"))
        self.assertTrue(cache.delete("k"))
        self.assertFalse(cache.contains("k"))
        self.assertFalse(cache.delete("k"))

    def test_cleanup_expired_removes_entries(self) -> None:
        cache = LRUTTLCache[str, int](capacity=3, ttl_seconds=0.05)
        cache.set("a", 1)
        cache.set("b", 2)
        time.sleep(0.06)
        removed = cache.cleanup_expired()
        self.assertEqual(removed, 2)
        self.assertEqual(len(cache), 0)

    def test_set_custom_ttl_validation(self) -> None:
        cache = LRUTTLCache[str, int](capacity=2, ttl_seconds=10.0)
        with self.assertRaises(ValueError):
            cache.set("a", 1, ttl_seconds=0.0)

    def test_get_entry_metadata_and_clear(self) -> None:
        cache = LRUTTLCache[str, int](capacity=2, ttl_seconds=10.0)
        cache.set("k", 7)
        meta = cache.get_entry_metadata("k")
        self.assertIsNotNone(meta)
        assert meta is not None
        self.assertEqual(meta["key"], "k")
        cache.clear()
        self.assertEqual(len(cache), 0)

    def test_stats_tracks_evictions(self) -> None:
        cache = LRUTTLCache[str, int](capacity=1, ttl_seconds=10.0)
        cache.set("a", 1)
        cache.set("b", 2)
        stats = cache.stats()
        self.assertEqual(stats.evicted_lru, 1)
        self.assertEqual(stats.size, 1)


if __name__ == "__main__":
    unittest.main()

