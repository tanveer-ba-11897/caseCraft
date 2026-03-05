"""
Caching infrastructure for the CaseCraft pipeline.

Three specialised cache layers reduce redundant compute and LLM calls:

1. **CondensationCache** — Caches LLM condensation results by chunk content
   hash.  When the same chunk text (or overlapping text) appears again, the
   cached condensation is returned without an LLM roundtrip.

2. **RetrievalCache** — Caches RAG retrieval results by query + top_k hash.
   When per-chunk retrieval produces identical queries (e.g. similar chunks),
   the cached results are returned without re-running the hybrid pipeline.

3. **PromptCache** — Caches fully rendered Jinja2 prompt templates by content
   hash of their parameters.  Avoids redundant template rendering when the
   same prompt is built multiple times.

All caches are thread-safe (LRU with optional TTL) and designed for
single-run in-memory use.  Call ``clear_all_caches()`` between CLI
invocations or after knowledge base changes.
"""

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

logger = logging.getLogger("casecraft.cache")


# ── Utilities ────────────────────────────────────────────────────────


def content_hash(text: str) -> str:
    """Fast 16-char hex digest of text content (SHA-256 prefix)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── Generic LRU Cache ────────────────────────────────────────────────


class LRUCache:
    """Thread-safe LRU cache with optional TTL expiration.

    Parameters
    ----------
    max_size : int
        Maximum number of entries.  When full, the least-recently-used
        entry is evicted.
    ttl : float
        Time-to-live in seconds.  0 = no expiration.
    """

    def __init__(self, max_size: int = 128, ttl: float = 0):
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            value, timestamp = self._cache[key]
            if self._ttl > 0 and time.monotonic() - timestamp > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = (value, time.monotonic())

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "max_size": self._max_size,
            "hit_rate_pct": round(hit_rate, 1),
        }


# ── Specialised Caches ──────────────────────────────────────────────


class CondensationCache:
    """Cache LLM condensation results by chunk content hash.

    Avoids redundant LLM calls when:
    - Chunk overlap causes near-identical text to be condensed twice.
    - The same feature file is re-processed (e.g. during reviewer pass).
    - KB batches from the same source contain identical text.
    """

    def __init__(self, max_size: int = 256):
        self._cache = LRUCache(max_size=max_size)

    def get(self, chunk_text: str) -> Optional[str]:
        key = content_hash(chunk_text)
        return self._cache.get(key)

    def put(self, chunk_text: str, condensed: str) -> None:
        key = content_hash(chunk_text)
        self._cache.put(key, condensed)

    def clear(self) -> None:
        self._cache.clear()

    @property
    def stats(self) -> Dict[str, Any]:
        return self._cache.stats


class RetrievalCache:
    """Cache RAG retrieval results by query + top_k hash.

    During per-chunk retrieval, chunks that overlap substantially will
    produce identical or near-identical queries.  This cache returns
    previously retrieved results without re-running the full hybrid
    pipeline (dense + sparse + rerank).

    TTL prevents stale results if the knowledge base is modified
    mid-session (default: 5 minutes).
    """

    def __init__(self, max_size: int = 64, ttl: float = 300.0):
        self._cache = LRUCache(max_size=max_size, ttl=ttl)

    def get(self, query: str, top_k: int) -> Optional[List]:
        key = content_hash(f"{query}::top_k={top_k}")
        return self._cache.get(key)

    def put(self, query: str, top_k: int, results: List) -> None:
        key = content_hash(f"{query}::top_k={top_k}")
        self._cache.put(key, results)

    def clear(self) -> None:
        self._cache.clear()

    @property
    def stats(self) -> Dict[str, Any]:
        return self._cache.stats


class PromptCache:
    """Cache rendered prompt templates by parameter content hash.

    The generation template (~5 KB of static instructions) is re-rendered
    for every LLM call with different feature text and KB context.  When
    identical parameters produce the same rendered string, this cache
    returns it without re-executing Jinja2 rendering.

    High hit rate for condensation prompts (identical template, different
    chunk — but identical chunk text → identical prompt).  Low hit rate
    for generation prompts (unique data each time).
    """

    def __init__(self, max_size: int = 32):
        self._cache = LRUCache(max_size=max_size)

    def get(self, template_name: str, **kwargs: Any) -> Optional[str]:
        key = self._build_key(template_name, kwargs)
        return self._cache.get(key)

    def put(self, template_name: str, rendered: str, **kwargs: Any) -> None:
        key = self._build_key(template_name, kwargs)
        self._cache.put(key, rendered)

    def _build_key(self, template_name: str, kwargs: Dict[str, Any]) -> str:
        parts = [template_name]
        for k in sorted(kwargs.keys()):
            v = kwargs[k]
            if isinstance(v, str) and len(v) > 200:
                parts.append(f"{k}={content_hash(v)}")
            else:
                parts.append(f"{k}={v}")
        return content_hash("||".join(parts))

    def clear(self) -> None:
        self._cache.clear()

    @property
    def stats(self) -> Dict[str, Any]:
        return self._cache.stats


# ── Global Singleton Instances ───────────────────────────────────────

_condensation_cache: Optional[CondensationCache] = None
_retrieval_cache: Optional[RetrievalCache] = None
_prompt_cache: Optional[PromptCache] = None
_cache_lock = threading.Lock()


def get_condensation_cache() -> CondensationCache:
    """Return the global condensation cache (lazy-init from config)."""
    global _condensation_cache
    if _condensation_cache is not None:
        return _condensation_cache
    with _cache_lock:
        if _condensation_cache is None:
            from core.config import config
            _condensation_cache = CondensationCache(
                max_size=config.cache.condensation_cache_size,
            )
    return _condensation_cache


def get_retrieval_cache() -> RetrievalCache:
    """Return the global retrieval cache (lazy-init from config)."""
    global _retrieval_cache
    if _retrieval_cache is not None:
        return _retrieval_cache
    with _cache_lock:
        if _retrieval_cache is None:
            from core.config import config
            _retrieval_cache = RetrievalCache(
                max_size=config.cache.retrieval_cache_size,
                ttl=config.cache.retrieval_cache_ttl,
            )
    return _retrieval_cache


def get_prompt_cache() -> PromptCache:
    """Return the global prompt cache (lazy-init from config)."""
    global _prompt_cache
    if _prompt_cache is not None:
        return _prompt_cache
    with _cache_lock:
        if _prompt_cache is None:
            from core.config import config
            _prompt_cache = PromptCache(
                max_size=config.cache.prompt_cache_size,
            )
    return _prompt_cache


def log_cache_stats() -> None:
    """Log summary statistics for all active caches."""
    for name, cache_obj in [
        ("Condensation", _condensation_cache),
        ("Retrieval", _retrieval_cache),
        ("Prompt", _prompt_cache),
    ]:
        if cache_obj is None:
            continue
        stats = cache_obj.stats
        total = stats["hits"] + stats["misses"]
        if total > 0:
            logger.info(
                "%s cache: %d hits, %d misses (%.1f%% hit rate, %d/%d entries)",
                name,
                stats["hits"],
                stats["misses"],
                stats["hit_rate_pct"],
                stats["size"],
                stats["max_size"],
            )


def clear_all_caches() -> None:
    """Reset all caches (between runs or after KB changes)."""
    for cache_obj in (_condensation_cache, _retrieval_cache, _prompt_cache):
        if cache_obj is not None:
            cache_obj.clear()
    logger.info("All pipeline caches cleared")
