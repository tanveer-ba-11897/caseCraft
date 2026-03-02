"""
Advanced hybrid retriever backed by ChromaDB + BM25.

Dense search runs inside ChromaDB's native HNSW index.
Sparse (BM25) search is built lazily from the stored documents.
Re-ranking uses a CrossEncoder model.

Init is near-instant (~0.05 s) because ChromaDB persists on disk and
ML models are loaded in a background thread.
"""

import logging
import re
import threading
import time
from typing import List, Optional
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
        self._bm25_metadatas: List[dict] = []
        self._bm25_id_to_idx: dict = {}
        self._bm25_lock = threading.Lock()
        self._bm25_built = False

        # Background pre-load
        self._preload_thread = threading.Thread(
            target=self._preload_all, daemon=True,
        )
        self._preload_thread.start()

    # -- Lazy initialisers ---------------------------------------------

    def _preload_all(self) -> None:
        """Background thread: load ML models + build BM25 concurrently."""
        self._ensure_models_loaded()
        self._ensure_bm25_built()

    def _ensure_models_loaded(self) -> None:
        if self._models_loaded:
            return
        with self._models_lock:
            if self._models_loaded:
                return
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading embedding model (lazy)...")
                t0 = time.time()
                self.embedder = SentenceTransformer(self._model_name)
                logger.info("Embedding model loaded in %.1fs", time.time() - t0)

                if self.use_reranker:
                    try:
                        from sentence_transformers import CrossEncoder
                        logger.info("Loading re-ranker model (lazy)...")
                        t0 = time.time()
                        self.reranker = CrossEncoder(self._reranker_name)
                        logger.info("Re-ranker loaded in %.1fs", time.time() - t0)
                    except ImportError:
                        logger.warning("CrossEncoder not available, re-ranking disabled.")
                        self.use_reranker = False
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
            try:
                from rank_bm25 import BM25Okapi
                logger.info("Building BM25 index from vector store...")
                t0 = time.time()
                all_data = self.store.get_all(include=["documents", "metadatas"])
                self._bm25_ids = all_data["ids"]
                self._bm25_texts = all_data["documents"]
                self._bm25_metadatas = all_data["metadatas"]
                self._bm25_id_to_idx = {
                    cid: idx for idx, cid in enumerate(self._bm25_ids)
                }
                tokenized = [
                    _BM25_TOKEN_PATTERN.findall(t.lower())
                    for t in self._bm25_texts
                ]
                self.bm25 = BM25Okapi(tokenized)
                logger.info(
                    "BM25 index built: %d docs in %.1fs",
                    len(self._bm25_ids), time.time() - t0,
                )
            except ImportError:
                logger.warning("rank_bm25 not installed, keyword search disabled.")
            self._bm25_built = True

    # -- Retrieval -----------------------------------------------------

    def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> List[KnowledgeChunk]:
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
        query_embedding = self.embedder.encode(query_text, normalize_embeddings=True)
        n_dense = total if retrieve_all else min(top_k * 5, total)
        chroma_where = _build_chroma_where(filters)
        chroma_results = self.store.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_dense,
            where=chroma_where,
        )
        dense_ids = chroma_results["ids"][0]
        dense_dists = chroma_results["distances"][0]
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
        return self._rerank_and_build(candidate_items, query_text, top_k, retrieve_all)

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

        dense_best: dict = {}
        for q_idx in range(len(queries)):
            ids = chroma_results["ids"][q_idx]
            dists = chroma_results["distances"][q_idx]
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
        return self._rerank_and_build(
            candidate_items, feature_text[:1500], top_k, retrieve_all,
        )

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
