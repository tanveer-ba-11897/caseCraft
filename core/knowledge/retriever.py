"""
Advanced hybrid retriever backed by ChromaDB + BM25 + knowledge graph.

Dense search runs inside ChromaDB's native HNSW index.
Sparse (BM25) search is built lazily from the stored documents.
Re-ranking uses a CrossEncoder model.
Optional knowledge-graph expansion adds structurally related chunks
that embedding similarity might miss.

Init is near-instant (~0.05 s) because ChromaDB persists on disk and
ML models are loaded in a background thread.
"""

import logging
import os
import pickle
import re
import threading
import time
from typing import Any, List, Optional
from pathlib import Path

import numpy as np

from core.knowledge.models import KnowledgeChunk
from core.knowledge.vector_store import (
    VectorStore,
    VectorStoreError,
    migrate_json_to_chroma,
    DEFAULT_PERSIST_DIR,
    DEFAULT_COLLECTION,
)

logger = logging.getLogger("casecraft.retriever")

# Precompiled regex helpers
_BM25_TOKEN_PATTERN = re.compile(r'\w+')

_KEY_TERM_PATTERNS = [
    re.compile(r'"([^"]+)"'),
    re.compile(r"'([^']+)'"),
    re.compile(r'\b([A-Z]{2,})\b'),
]

_DOMAIN_EXPANSIONS = {
    "login": ["authentication", "sign in", "credentials"],
    "signup": ["registration", "create account", "sign up"],
    "password": ["credential", "passphrase", "secret"],
    "error": ["failure", "exception", "fault"],
    "button": ["control", "action", "click"],
    "form": ["input", "field", "submission"],
    "api": ["endpoint", "service", "request"],
    "notification": ["alert", "message", "push"],
    "payment": ["transaction", "billing", "checkout"],
    "search": ["query", "filter", "find"],
    "upload": ["import", "attachment", "file"],
    "download": ["export", "save", "retrieve"],
    "delete": ["remove", "cancel", "discard"],
    "update": ["edit", "modify", "change"],
    "permission": ["access", "role", "authorization"],
    "settings": ["configuration", "preferences", "options"],
    "dashboard": ["home", "overview", "summary"],
    "profile": ["account", "user info", "user details"],
    "navigation": ["menu", "sidebar", "breadcrumb"],
    "validation": ["verification", "check", "constraint"],
}


class RetrievalError(Exception):
    pass


class KnowledgeRetriever:
    """
    Hybrid retriever: ChromaDB HNSW (dense) + BM25 (sparse) + CrossEncoder re-ranking.

    All heavy work (ML model loading, BM25 index building) is deferred
    to background pre-loading.  Init itself only opens the persistent
    ChromaDB client (~0.05 s).
    """

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_reranker: bool = True,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        min_score_threshold: float = 0.1,
        debug: bool = False,
    ):
        self.use_reranker = use_reranker
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.min_score_threshold = min_score_threshold
        self.debug = debug

        # Open (or create) persistent vector store
        try:
            self.store = VectorStore(
                persist_dir=persist_dir,
                collection_name=collection_name,
            )
        except VectorStoreError as exc:
            raise RetrievalError(str(exc)) from exc

        # Auto-migrate legacy JSON index if ChromaDB is empty
        if self.store.count() == 0:
            legacy = Path("knowledge_base/index.json")
            if legacy.exists():
                logger.info("Empty ChromaDB detected - auto-migrating from %s", legacy)
                migrate_json_to_chroma(legacy, persist_dir, collection_name)
                # Refresh collection reference
                self.store = VectorStore(persist_dir, collection_name)

        if self.store.count() == 0:
            raise RetrievalError(
                "Knowledge base is empty. Run ingestion first "
                "(python cli/ingest.py docs <dir>)."
            )

        # Model config (lazy-loaded)
        self._model_name = model_name
        self._reranker_name = reranker_name
        self.embedder = None
        self.reranker = None
        self._models_lock = threading.Lock()
        self._models_loaded = False

        # BM25 sparse index (lazy-built)
        self.bm25 = None
        self._bm25_ids: List[str] = []
        self._bm25_texts: List[str] = []
        self._bm25_metadatas: list[Any] = []
        self._bm25_id_to_idx: dict = {}
        self._bm25_lock = threading.Lock()
        self._bm25_built = False

        # Secondary metadata indexes (built alongside BM25)
        self._source_index: dict[str, list[int]] = {}   # source_file → [idx]
        self._type_index: dict[str, list[int]] = {}     # chunk_type → [idx]

        # BM25 persistence path (next to the vector DB)
        self._bm25_cache_path = Path(persist_dir) / "bm25_cache.pkl"

        # Background pre-load
        self._preload_thread = threading.Thread(
            target=self._preload_all, daemon=True,
        )
        self._preload_thread.start()

        # Knowledge graph (lazy-loaded from config)
        self._knowledge_graph = None  # type: ignore[assignment]
        self._kg_loaded = False
        self._kg_lock = threading.Lock()

    # -- Lazy initialisers ---------------------------------------------

    def _preload_all(self) -> None:
        """Background thread: load ML models + build BM25 concurrently."""
        self._ensure_models_loaded()
        self._ensure_bm25_built()
        self._ensure_kg_loaded()

    def _ensure_models_loaded(self) -> None:
        if self._models_loaded:
            return
        with self._models_lock:
            if self._models_loaded:
                return
            # Timeout for model downloads: 5 minutes should be enough for
            # cached models; first download of a 400 MB+ model may still
            # take longer but will not block the main thread forever.
            _MODEL_LOAD_TIMEOUT = 300  # seconds
            try:
                from sentence_transformers import SentenceTransformer
                import concurrent.futures

                logger.info("Loading embedding model (lazy)...")
                t0 = time.time()
                with concurrent.futures.ThreadPoolExecutor(1) as _pool:
                    _fut = _pool.submit(SentenceTransformer, self._model_name)
                    self.embedder = _fut.result(timeout=_MODEL_LOAD_TIMEOUT)
                logger.info("Embedding model loaded in %.1fs", time.time() - t0)

                if self.use_reranker:
                    try:
                        from sentence_transformers import CrossEncoder
                        logger.info("Loading re-ranker model (lazy)...")
                        t0 = time.time()
                        with concurrent.futures.ThreadPoolExecutor(1) as _pool:
                            _fut = _pool.submit(CrossEncoder, self._reranker_name)
                            self.reranker = _fut.result(timeout=_MODEL_LOAD_TIMEOUT)
                        logger.info("Re-ranker loaded in %.1fs", time.time() - t0)
                    except ImportError:
                        logger.warning("CrossEncoder not available, re-ranking disabled.")
                        self.use_reranker = False
            except concurrent.futures.TimeoutError:
                raise RetrievalError(
                    f"Model loading timed out after {_MODEL_LOAD_TIMEOUT}s. "
                    "Check your network or use a cached model."
                )
            except Exception as exc:
                raise RetrievalError(
                    "Failed to load embedding/reranking models"
                ) from exc
            self._models_loaded = True

    def _ensure_bm25_built(self) -> None:
        if self._bm25_built:
            return
        with self._bm25_lock:
            if self._bm25_built:
                return

            current_count = self.store.count()

            # --- Try loading persisted BM25 from disk ---
            try:
                from core.config import config as _app_cfg
                persist_enabled = _app_cfg.cache.persist_bm25_index
            except Exception:
                persist_enabled = True

            _BM25_LOAD_TIMEOUT = 120  # seconds
            if persist_enabled and self._bm25_cache_path.exists():
                try:
                    import concurrent.futures
                    def _load_pickle(path):
                        with open(path, "rb") as f:
                            return pickle.load(f)
                    with concurrent.futures.ThreadPoolExecutor(1) as _pool:
                        cached = _pool.submit(_load_pickle, self._bm25_cache_path).result(timeout=_BM25_LOAD_TIMEOUT)
                    if cached.get("count") == current_count:
                        self.bm25 = cached["bm25"]
                        self._bm25_ids = cached["ids"]
                        self._bm25_texts = cached["texts"]
                        self._bm25_metadatas = cached["metadatas"]
                        self._bm25_id_to_idx = cached["id_to_idx"]
                        self._source_index = cached.get("source_index", {})
                        self._type_index = cached.get("type_index", {})
                        logger.info(
                            "BM25 index loaded from cache: %d docs (%.1f KB)",
                            current_count,
                            self._bm25_cache_path.stat().st_size / 1024,
                        )
                        self._bm25_built = True
                        return
                    else:
                        logger.info(
                            "BM25 cache stale (cached=%d, current=%d), rebuilding",
                            cached.get("count", 0), current_count,
                        )
                except Exception as e:
                    logger.debug("BM25 cache load failed: %s, rebuilding", e)

            # --- Build from scratch ---
            try:
                from rank_bm25 import BM25Okapi
                logger.info("Building BM25 index from vector store...")
                t0 = time.time()
                all_data = self.store.get_all(include=["documents", "metadatas"])
                self._bm25_ids = all_data["ids"]
                self._bm25_texts = list(all_data["documents"] or [])
                self._bm25_metadatas = list(all_data["metadatas"] or [])
                self._bm25_id_to_idx = {
                    cid: idx for idx, cid in enumerate(self._bm25_ids)
                }
                tokenized = [
                    _BM25_TOKEN_PATTERN.findall(t.lower())
                    for t in self._bm25_texts
                ]
                self.bm25 = BM25Okapi(tokenized)

                # Build secondary metadata indexes
                self._source_index = {}
                self._type_index = {}
                for idx, meta in enumerate(self._bm25_metadatas):
                    if isinstance(meta, dict):
                        src = meta.get("source") or meta.get("source_file", "")
                        if src:
                            self._source_index.setdefault(src, []).append(idx)
                        ctype = meta.get("chunk_type", "standalone")
                        self._type_index.setdefault(ctype, []).append(idx)

                elapsed = time.time() - t0
                logger.info(
                    "BM25 index built: %d docs in %.1fs (%d sources, %d chunk types)",
                    len(self._bm25_ids), elapsed,
                    len(self._source_index), len(self._type_index),
                )

                # --- Persist to disk ---
                if persist_enabled:
                    try:
                        self._bm25_cache_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(self._bm25_cache_path, "wb") as f:
                            pickle.dump({
                                "count": current_count,
                                "bm25": self.bm25,
                                "ids": self._bm25_ids,
                                "texts": self._bm25_texts,
                                "metadatas": self._bm25_metadatas,
                                "id_to_idx": self._bm25_id_to_idx,
                                "source_index": self._source_index,
                                "type_index": self._type_index,
                            }, f)
                        logger.info(
                            "BM25 index persisted to %s (%.1f KB)",
                            self._bm25_cache_path,
                            self._bm25_cache_path.stat().st_size / 1024,
                        )
                    except Exception as e:
                        logger.debug("BM25 cache write failed: %s", e)

            except ImportError:
                logger.warning("rank_bm25 not installed, keyword search disabled.")
            self._bm25_built = True

    def _ensure_kg_loaded(self) -> None:
        """Load the knowledge graph from disk if enabled in config."""
        if self._kg_loaded:
            return
        with self._kg_lock:
            if self._kg_loaded:
                return
            try:
                from core.config import load_config as _load_cfg
                kb_cfg = _load_cfg().knowledge
                if kb_cfg.knowledge_graph:
                    from core.knowledge.graph import KnowledgeGraph
                    self._knowledge_graph = KnowledgeGraph.load(
                        path=kb_cfg.graph_path,
                        max_hops=kb_cfg.graph_max_hops,
                    )
                    if self._knowledge_graph:
                        stats = self._knowledge_graph.get_statistics()
                        logger.info(
                            "Knowledge graph active: %d nodes, %d edges",
                            stats["nodes"], stats["edges"],
                        )
                    else:
                        logger.info("Knowledge graph enabled but no graph file found yet")
            except Exception as exc:
                logger.warning("Failed to load knowledge graph: %s", exc)
                self._knowledge_graph = None
            self._kg_loaded = True

    # -- Retrieval -----------------------------------------------------

    def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> List[KnowledgeChunk]:
        # Check retrieval cache (only for unfiltered queries)
        _use_cache = False
        _retrieval_cache = None
        if not filters:
            try:
                from core.config import config as _app_cfg
                if _app_cfg.cache.enable_retrieval_cache:
                    from core.cache import get_retrieval_cache
                    _retrieval_cache = get_retrieval_cache()
                    cached = _retrieval_cache.get(query_text, top_k)
                    if cached is not None:
                        logger.debug("Retrieval cache HIT (%d results)", len(cached))
                        return cached
                    _use_cache = True
            except Exception:
                pass

        self._ensure_models_loaded()
        self._ensure_bm25_built()

        if not query_text.strip():
            return []

        retrieve_all = top_k == -1
        total = self.store.count()

        if self.debug:
            logger.debug(
                "Retrieving for query: '%s...' (top_k=%s)",
                query_text[:50], "ALL" if retrieve_all else top_k,
            )

        # 1. Dense search via ChromaDB HNSW
        assert self.embedder is not None, "Embedder must be loaded before retrieval"
        query_embedding = self.embedder.encode(query_text, normalize_embeddings=True)
        n_dense = total if retrieve_all else min(top_k * 5, total)
        chroma_where = _build_chroma_where(filters)
        chroma_results = self.store.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_dense,
            where=chroma_where,
        )
        dense_ids = chroma_results["ids"][0]
        distances = chroma_results["distances"]
        assert distances is not None, "ChromaDB query must return distances"
        dense_dists = distances[0]
        dense_sims = {cid: 1.0 - dist for cid, dist in zip(dense_ids, dense_dists)}

        # 2. Sparse (BM25) search
        sparse_scores_map: dict = {}
        bm25_all_scores = None
        if self.bm25:
            tokenized_query = _BM25_TOKEN_PATTERN.findall(query_text.lower())
            bm25_all_scores = self.bm25.get_scores(tokenized_query)
            bm25_max = bm25_all_scores.max()
            if bm25_max > 0:
                bm25_all_scores = bm25_all_scores / bm25_max
            bm25_top_k = len(self._bm25_ids) if retrieve_all else min(top_k * 5, len(self._bm25_ids))
            bm25_top_indices = np.argsort(bm25_all_scores)[::-1][:bm25_top_k]
            for idx in bm25_top_indices:
                if bm25_all_scores[idx] > 0:
                    sparse_scores_map[self._bm25_ids[idx]] = float(bm25_all_scores[idx])

        # 3. Hybrid fusion
        all_candidate_ids = set(dense_sims.keys()) | set(sparse_scores_map.keys())
        candidates: List[tuple] = []
        for cid in all_candidate_ids:
            d_score = dense_sims.get(cid, 0.0)
            s_score = sparse_scores_map.get(cid, 0.0)
            if s_score == 0.0 and bm25_all_scores is not None and cid in self._bm25_id_to_idx:
                s_score = float(bm25_all_scores[self._bm25_id_to_idx[cid]])
            hybrid = self.dense_weight * d_score + self.sparse_weight * s_score
            if hybrid >= self.min_score_threshold:
                candidates.append((cid, hybrid))

        if not candidates:
            return []
        candidates.sort(key=lambda x: x[1], reverse=True)

        if self.debug:
            logger.debug("Score range: [%.3f, %.3f]", candidates[-1][1], candidates[0][1])
            logger.debug("%d candidates after filtering", len(candidates))

        if not retrieve_all:
            initial_k = top_k * 3 if self.use_reranker else top_k
            candidates = candidates[:initial_k]

        candidate_items = self._resolve_items(candidates)
        if not candidate_items:
            return []

        # 4. Re-ranking
        results = self._rerank_and_build(candidate_items, query_text, top_k, retrieve_all)

        # Store in retrieval cache
        if _use_cache and _retrieval_cache is not None:
            _retrieval_cache.put(query_text, top_k, results)

        return results

    # -- Multi-query batched retrieval ---------------------------------

    def retrieve_multi_query(
        self,
        feature_text: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
        enable_decomposition: bool = True,
        enable_expansion: bool = True,
        max_sub_queries: int = 4,
    ) -> List[KnowledgeChunk]:
        """Batched multi-query retrieval with query transforms."""
        # Check retrieval cache (only for unfiltered queries)
        _use_cache = False
        _retrieval_cache = None
        if not filters:
            try:
                from core.config import config as _app_cfg
                if _app_cfg.cache.enable_retrieval_cache:
                    from core.cache import get_retrieval_cache
                    _retrieval_cache = get_retrieval_cache()
                    cached = _retrieval_cache.get(feature_text, top_k)
                    if cached is not None:
                        logger.debug("Multi-query retrieval cache HIT (%d results)", len(cached))
                        return cached
                    _use_cache = True
            except Exception:
                pass

        self._ensure_models_loaded()
        self._ensure_bm25_built()

        queries: List[str] = []
        if enable_decomposition and len(feature_text) > 500:
            sub_queries = self.decompose_query(feature_text, max_sub_queries)
            if self.debug:
                logger.debug("Decomposed into %d sub-queries", len(sub_queries))
            queries.extend(sub_queries)
        else:
            queries.append(feature_text)

        if enable_expansion:
            expanded: List[str] = []
            for q in queries:
                exp = self.expand_query(q)
                expanded.append(exp)
                if self.debug and exp != q:
                    logger.debug("Query expanded by %d chars", len(exp) - len(q))
            queries = expanded

        if len(queries) == 1:
            return self.retrieve(queries[0], top_k=top_k, filters=filters)

        queries = [q[:1500] for q in queries]
        retrieve_all = top_k == -1
        total = self.store.count()
        logger.info("Batched multi-query retrieval: %d queries x top_k=%s", len(queries), top_k)
        t0 = time.time()

        # A. Batch dense encode + ChromaDB query
        assert self.embedder is not None, "Embedder must be loaded before retrieval"
        query_embeddings = self.embedder.encode(
            queries, normalize_embeddings=True, batch_size=len(queries),
        )
        n_dense = total if retrieve_all else min(top_k * 5, total)
        chroma_where = _build_chroma_where(filters)
        chroma_results = self.store.query(
            query_embeddings=[emb.tolist() for emb in query_embeddings],
            n_results=n_dense,
            where=chroma_where,
        )

        distances = chroma_results["distances"]
        assert distances is not None, "ChromaDB query must return distances"
        dense_best: dict = {}
        for q_idx in range(len(queries)):
            ids = chroma_results["ids"][q_idx]
            dists = distances[q_idx]
            for cid, dist in zip(ids, dists):
                sim = 1.0 - dist
                if cid not in dense_best or sim > dense_best[cid]:
                    dense_best[cid] = sim

        # B. BM25 max-fusion across queries
        bm25_best = np.zeros(len(self._bm25_ids))
        if self.bm25:
            for q in queries:
                tokens = _BM25_TOKEN_PATTERN.findall(q.lower())
                scores = self.bm25.get_scores(tokens)
                mx = scores.max()
                if mx > 0:
                    scores = scores / mx
                bm25_best = np.maximum(bm25_best, scores)

        sparse_best: dict = {}
        if self.bm25:
            n_sparse = len(self._bm25_ids) if retrieve_all else min(top_k * 5, len(self._bm25_ids))
            top_bm25_idx = np.argsort(bm25_best)[::-1][:n_sparse]
            for idx in top_bm25_idx:
                if bm25_best[idx] > 0:
                    sparse_best[self._bm25_ids[idx]] = float(bm25_best[idx])

        # C. Hybrid fusion
        all_ids = set(dense_best.keys()) | set(sparse_best.keys())
        candidates: List[tuple] = []
        for cid in all_ids:
            d = dense_best.get(cid, 0.0)
            s = sparse_best.get(cid, 0.0)
            if s == 0.0 and cid in self._bm25_id_to_idx:
                s = float(bm25_best[self._bm25_id_to_idx[cid]])
            hybrid = self.dense_weight * d + self.sparse_weight * s
            if hybrid >= self.min_score_threshold:
                candidates.append((cid, hybrid))

        if not candidates:
            return []
        candidates.sort(key=lambda x: x[1], reverse=True)

        if self.debug:
            logger.debug("Fused score range: [%.3f, %.3f]", candidates[-1][1], candidates[0][1])

        if not retrieve_all:
            initial_k = top_k * 3 if self.use_reranker else top_k
            candidates = candidates[:initial_k]

        candidate_items = self._resolve_items(candidates)
        if not candidate_items:
            return []

        logger.info(
            "Batched multi-query done: %d candidates from %d queries in %.2fs",
            len(candidate_items), len(queries), time.time() - t0,
        )
        results = self._rerank_and_build(
            candidate_items, feature_text[:1500], top_k, retrieve_all,
        )

        # Store in retrieval cache
        if _use_cache and _retrieval_cache is not None:
            _retrieval_cache.put(feature_text, top_k, results)

        return results

    # -- Shared helpers ------------------------------------------------

    def _resolve_items(self, candidates: List[tuple]) -> List[dict]:
        items = []
        for cid, score in candidates:
            if cid in self._bm25_id_to_idx:
                idx = self._bm25_id_to_idx[cid]
                items.append({
                    "id": cid,
                    "text": self._bm25_texts[idx],
                    "metadata": self._bm25_metadatas[idx],
                    "score": score,
                })
        return items

    # -- Secondary index lookups --------------------------------------

    def get_chunks_by_source(self, source: str) -> List[KnowledgeChunk]:
        """Fast lookup of all chunks from a specific source file.

        Uses the secondary source index built alongside BM25.
        Falls back to a full scan if the index isn't ready.
        """
        self._ensure_bm25_built()
        indices = self._source_index.get(source, [])
        return [
            KnowledgeChunk(
                id=self._bm25_ids[idx],
                text=self._bm25_texts[idx],
                metadata=self._bm25_metadatas[idx],
                embedding=None,
            )
            for idx in indices
        ]

    def get_source_list(self) -> List[str]:
        """Return all unique source file names in the knowledge base."""
        self._ensure_bm25_built()
        return list(self._source_index.keys())

    def get_chunk_type_counts(self) -> dict[str, int]:
        """Return counts of each chunk type (parent, child, standalone)."""
        self._ensure_bm25_built()
        return {ctype: len(indices) for ctype, indices in self._type_index.items()}

    def _expand_children_to_parents(
        self,
        items: List[dict],
    ) -> List[dict]:
        """
        Replace child chunks with their parent chunks for richer context.

        If a result has ``metadata["chunk_type"] == "child"`` and a
        ``parent_id``, fetch the parent text from the vector store and
        return it instead.  Deduplicates so each parent appears at most once,
        keeping the highest child score.

        Chunks without parent–child metadata pass through unchanged.
        """
        parent_ids_needed: List[str] = []
        parent_score_map: dict[str, float] = {}  # parent_id → best child score

        for item in items:
            meta = item.get("metadata", {})
            if meta.get("chunk_type") == "child" and "parent_id" in meta:
                pid = meta["parent_id"]
                parent_ids_needed.append(pid)
                if pid not in parent_score_map or item["score"] > parent_score_map[pid]:
                    parent_score_map[pid] = item["score"]

        if not parent_ids_needed:
            return items  # no children — nothing to expand

        # Fetch parents from ChromaDB
        unique_parent_ids = list(dict.fromkeys(parent_ids_needed))
        try:
            parent_data = self.store.get_by_ids(
                unique_parent_ids, include=["documents", "metadatas"],
            )
        except Exception as exc:
            logger.warning("Parent expansion failed: %s — returning children", exc)
            return items

        parent_lookup: dict[str, dict] = {}
        if parent_data and parent_data.get("ids"):
            for pid, doc, meta in zip(
                parent_data["ids"],
                parent_data["documents"] or [],
                parent_data["metadatas"] or [],
            ):
                parent_lookup[pid] = {"id": pid, "text": doc, "metadata": meta}

        # Rebuild result list: replace children with parents, pass non-children through
        seen_parents: set[str] = set()
        expanded: List[dict] = []

        for item in items:
            meta = item.get("metadata", {})
            if meta.get("chunk_type") == "child" and "parent_id" in meta:
                pid = meta["parent_id"]
                if pid in seen_parents:
                    continue  # already emitted this parent
                seen_parents.add(pid)
                if pid in parent_lookup:
                    parent_item = dict(parent_lookup[pid])
                    parent_item["score"] = parent_score_map.get(pid, item["score"])
                    expanded.append(parent_item)
                else:
                    # Parent missing from store — fall back to child
                    expanded.append(item)
            else:
                expanded.append(item)

        if self.debug and seen_parents:
            logger.debug(
                "Parent expansion: %d children → %d unique parents",
                len(parent_ids_needed), len(seen_parents),
            )

        return expanded

    def _expand_via_graph(
        self,
        items: List[dict],
        max_expansion: int | None = None,
    ) -> List[dict]:
        """
        Append graph-related chunks that the vector/BM25 search missed.

        Uses the knowledge graph to find structurally connected chunks
        (same source, cross-references, shared entities) within
        ``graph_max_hops`` of the current results.  Up to
        ``max_expansion`` new chunks are appended.

        Chunks already present in *items* are not duplicated.
        """
        self._ensure_kg_loaded()

        if not self._knowledge_graph:
            return items

        if max_expansion is None:
            try:
                from core.config import load_config as _load_cfg
                max_expansion = _load_cfg().knowledge.graph_max_expansion
            except Exception:
                max_expansion = 3

        existing_ids = {item["id"] for item in items}
        seed_ids = list(existing_ids)

        related_ids = self._knowledge_graph.get_related_ids(seed_ids)

        # Filter out IDs already in results
        new_ids = [rid for rid in related_ids if rid not in existing_ids]
        if not new_ids:
            return items

        new_ids = new_ids[:max_expansion]

        # Try resolving from BM25 cache first (fast), fall back to store
        expanded_items: List[dict] = []
        store_fetch_ids: List[str] = []

        for nid in new_ids:
            if nid in self._bm25_id_to_idx:
                idx = self._bm25_id_to_idx[nid]
                expanded_items.append({
                    "id": nid,
                    "text": self._bm25_texts[idx],
                    "metadata": self._bm25_metadatas[idx],
                    "score": 0.0,  # graph-expanded, no hybrid score
                })
            else:
                store_fetch_ids.append(nid)

        # Fetch any remaining from ChromaDB
        if store_fetch_ids:
            try:
                fetched = self.store.get_by_ids(
                    store_fetch_ids, include=["documents", "metadatas"],
                )
                if fetched and fetched.get("ids"):
                    for fid, doc, meta in zip(
                        fetched["ids"],
                        fetched["documents"] or [],
                        fetched["metadatas"] or [],
                    ):
                        expanded_items.append({
                            "id": fid,
                            "text": doc,
                            "metadata": meta,
                            "score": 0.0,
                        })
            except Exception as exc:
                logger.warning("Graph expansion store fetch failed: %s", exc)

        if expanded_items and self.debug:
            logger.debug(
                "Graph expansion: added %d chunks from %d seeds",
                len(expanded_items), len(seed_ids),
            )

        return items + expanded_items

    def _rerank_and_build(
        self,
        candidate_items: List[dict],
        query_text: str,
        top_k: int,
        retrieve_all: bool,
    ) -> List[KnowledgeChunk]:
        if self.use_reranker and self.reranker:
            pairs = [[query_text, item["text"]] for item in candidate_items]
            rerank_scores = self.reranker.predict(pairs)
            if self.debug:
                logger.debug("Re-ranker scores: %s...", rerank_scores[:3])
            scored = sorted(
                zip(candidate_items, rerank_scores),
                key=lambda x: x[1], reverse=True,
            )
            if retrieve_all:
                final = [item for item, _ in scored]
            else:
                final = [item for item, _ in scored[:top_k]]
        else:
            final = candidate_items if retrieve_all else candidate_items[:top_k]

        # Expand child → parent chunks for richer context
        final = self._expand_children_to_parents(final)

        # Expand via knowledge graph for cross-document tracing
        final = self._expand_via_graph(final)

        if self.debug:
            logger.debug("Returning %d results", len(final))

        return [
            KnowledgeChunk(
                id=item["id"],
                text=item["text"],
                metadata=item["metadata"],
                embedding=None,
            )
            for item in final
        ]

    # -- Query Transforms (unchanged) ---------------------------------

    @staticmethod
    def decompose_query(feature_text: str, max_sub_queries: int = 4) -> List[str]:
        if len(feature_text) < 300:
            return [feature_text]
        blocks = re.split(r'\n\s*\n|\n(?=#{1,4}\s)|(?=\d+\.\s)', feature_text)
        blocks = [b.strip() for b in blocks if b.strip() and len(b.strip()) > 30]
        if len(blocks) <= 1:
            return [feature_text]
        blocks.sort(key=len, reverse=True)
        return blocks[:max_sub_queries]

    @staticmethod
    def expand_query(query_text: str) -> str:
        query_lower = query_text.lower()
        expansions: List[str] = []
        for term, synonyms in _DOMAIN_EXPANSIONS.items():
            if term in query_lower:
                for syn in synonyms:
                    if syn.lower() not in query_lower:
                        expansions.append(syn)
        for pattern in _KEY_TERM_PATTERNS:
            for match in pattern.finditer(query_text):
                term = match.group(1).strip()
                if len(term) >= 2 and term.lower() not in query_lower:
                    expansions.append(term)
        if not expansions:
            return query_text
        expansion_str = " ".join(expansions[:15])
        return f"{query_text}\n\nRelated terms: {expansion_str}"


def _build_chroma_where(filters: Optional[dict]) -> Optional[dict]:
    if not filters:
        return None
    if len(filters) == 1:
        k, v = next(iter(filters.items()))
        return {k: v}
    return {"$and": [{k: v} for k, v in filters.items()]}
