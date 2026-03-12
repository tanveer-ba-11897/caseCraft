"""
Caching infrastructure for the CaseCraft pipeline -- DISABLED.

The in-memory LRU caches (condensation, retrieval, prompt) have been
removed to measure baseline latency.  The full implementation is backed
up in core/cache.py.bak.

To restore caching, replace this file with cache.py.bak and revert the
related edits in generator.py, prompts.py, retriever.py, config.py, and
casecraft.yaml.
"""

import logging

logger = logging.getLogger("casecraft.cache")


def log_cache_stats() -> None:
    """No-op -- caching is disabled."""
    pass


def clear_all_caches() -> None:
    """No-op -- caching is disabled."""
    pass
