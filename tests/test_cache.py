"""
Tests for core/cache.py -- stub module (caching disabled).

Full cache tests are backed up in tests/test_cache.py.bak.
"""

import unittest

from core.cache import log_cache_stats, clear_all_caches


class TestCacheStubs(unittest.TestCase):
    """Verify the no-op stubs don't raise."""

    def test_log_cache_stats_noop(self):
        log_cache_stats()

    def test_clear_all_caches_noop(self):
        clear_all_caches()


if __name__ == "__main__":
    unittest.main()
