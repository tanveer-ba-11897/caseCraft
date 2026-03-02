import json
import logging
import re
import time
from typing import List, Optional, Set
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from core.knowledge.models import KnowledgeChunk

logger = logging.getLogger("casecraft.retriever")

# Precompiled tokenizer for BM25 (strips punctuation, lowercases)
_BM25_TOKEN_PATTERN = re.compile(r'\w+')

# Patterns for extracting key terms from feature text for query expansion
_KEY_TERM_PATTERNS = [
    # Quoted terms
    re.compile(r'"([^"]+)"'),
    re.compile(r"'([^']+)'"),
    # ALL CAPS terms (acronyms, constants)
    re.compile(r'\b([A-Z]{2,})\b'),
]

# Common QA/testing domain synonyms for query expansion
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
    Advanced retriever using Hybrid Search (BM25 + Dense) and Cross-Encoder Re-ranking.
    
    Args:
        index_path: Path to the knowledge index JSON file.
        model_name: SentenceTransformer model for dense embeddings.
        reranker_name: CrossEncoder model for re-ranking.
        use_reranker: Enable/disable re-ranking step.
        dense_weight: Weight for dense (semantic) search scores (0-1).
        sparse_weight: Weight for sparse (BM25) search scores (0-1).
        min_score_threshold: Minimum hybrid score to include a result (-1 to 1).
        debug: Enable verbose logging for retrieval debugging.
    """

    def __init__(
        self,
        index_path: str = "knowledge_base/index.json",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_reranker: bool = True,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        min_score_threshold: float = 0.1,
        debug: bool = False,
    ):
        self.index_path = Path(index_path)
        self.use_reranker = use_reranker
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.min_score_threshold = min_score_threshold
        self.debug = debug

        if not self.index_path.exists():
            raise RetrievalError(
                f"Knowledge index not found at {self.index_path}"
            )

        try:
            # 1. Load Dense Embedder
            logger.info("Loading embedding model...")
            t0 = time.time()
            self.embedder = SentenceTransformer(model_name)
            logger.info("Embedding model loaded in %.1fs", time.time()-t0)
            
            # 2. Load Re-ranker if enabled
            self.reranker = None
            if self.use_reranker:
                try:
                    from sentence_transformers import CrossEncoder
                    logger.info("Loading re-ranker model...")
                    t0 = time.time()
                    self.reranker = CrossEncoder(reranker_name)
                    logger.info("Re-ranker loaded in %.1fs", time.time()-t0)
                except ImportError:
                    logger.warning("CrossEncoder not available, re-ranking disabled.")
                    self.use_reranker = False
            
        except Exception as exc:
            raise RetrievalError(
                "Failed to load embedding/reranking models"
            ) from exc

        try:
            logger.info("Loading knowledge index from %s...", self.index_path)
            t0 = time.time()

            # R3: Verify integrity hash before trusting the index
            from core.knowledge.integrity import verify_hash, IntegrityError
            try:
                verify_hash(self.index_path)
            except IntegrityError as ie:
                # Missing hash file on a legacy/first-run index → warn, don't block
                if "not found" in str(ie):
                    logger.warning("Index integrity check skipped: %s", ie)
                else:
                    # Actual tamper / mismatch → fatal
                    raise RetrievalError(str(ie)) from ie
            except Exception as ve:
                logger.warning("Index integrity check skipped: %s", ve)

            with self.index_path.open("r", encoding="utf-8") as f:
                self.index = json.load(f)
            logger.info("Index loaded: %d chunks in %.1fs", len(self.index), time.time()-t0)
        except RetrievalError:
            raise  # Don't re-wrap our own errors
        except Exception as exc:
            raise RetrievalError(
                "Failed to load knowledge index"
            ) from exc

        # 3. Prepare Dense Index
        logger.info("Building dense index...")
        t0 = time.time()
        self.embeddings = np.array(
            [item["embedding"] for item in self.index],
            dtype=np.float32,
        )
        logger.info("Dense index built in %.1fs (%s)", time.time()-t0, self.embeddings.shape)
        
        # Free embedding data from the in-memory index to halve memory usage
        for item in self.index:
            del item["embedding"]
        
        # 4. Prepare Sparse Index (BM25) with proper tokenization
        self.bm25 = None
        try:
            from rank_bm25 import BM25Okapi
            logger.info("Building BM25 index...")
            t0 = time.time()
            tokenized_corpus = [_BM25_TOKEN_PATTERN.findall(item["text"].lower()) for item in self.index]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info("BM25 index built in %.1fs", time.time()-t0)
            if self.debug:
                logger.debug("BM25 index built with %d documents", len(tokenized_corpus))
        except ImportError:
            logger.warning("rank_bm25 not installed, keyword search disabled.")

    def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> List[KnowledgeChunk]:
        if not query_text.strip():
            return []

        # top_k == -1 means retrieve ALL chunks
        retrieve_all = top_k == -1

        if self.debug:
            logger.debug("Retrieving for query: '%s...' (top_k=%s)", query_text[:50], 'ALL' if retrieve_all else top_k)

        # --- Step 1: Hybrid Retrieval (Dense + Sparse) ---
        
        # A. Dense Search
        query_embedding = self.embedder.encode(query_text, normalize_embeddings=True)
        dense_scores = np.dot(self.embeddings, query_embedding)
        
        # B. Sparse Search (BM25)
        if self.bm25:
            tokenized_query = _BM25_TOKEN_PATTERN.findall(query_text.lower())
            sparse_scores = self.bm25.get_scores(tokenized_query)
            # Normalize sparse scores (simple min-max)
            if sparse_scores.max() > 0:
                sparse_scores = sparse_scores / sparse_scores.max()
        else:
            sparse_scores = np.zeros(len(self.index))

        # C. Combine Scores (using configurable weights)
        hybrid_scores = (self.dense_weight * dense_scores) + (self.sparse_weight * sparse_scores)
        
        if self.debug:
            logger.debug("Score range: [%.3f, %.3f]", hybrid_scores.min(), hybrid_scores.max())
        
        # Determine initial retrieval set (larger than top_k for re-ranking, or all)
        if retrieve_all:
            initial_k = len(self.index)
        else:
            initial_k = top_k * 3 if self.use_reranker else top_k
        ranked_indices = np.argsort(hybrid_scores)[::-1][:initial_k]
        
        candidates = []
        for idx in ranked_indices:
            score = hybrid_scores[idx]
            # Apply minimum score threshold
            if score < self.min_score_threshold:
                if self.debug:
                    logger.debug("Filtered out result with score %.3f (below threshold %.3f)", score, self.min_score_threshold)
                continue
            item = self.index[idx]
            if filters and not all(item["metadata"].get(k) == v for k, v in filters.items()):
                continue
            candidates.append((item, score))
            
        if not candidates:
            return []
        
        if self.debug:
            logger.debug("%d candidates after filtering", len(candidates))

        # --- Step 2: Re-ranking ---
        
        results = []
        if self.use_reranker and self.reranker:
            # Extract items from (item, score) tuples for re-ranking
            items_only = [item for item, _ in candidates]
            
            # Prepare pairs: [query, doc_text]
            pairs = [[query_text, item["text"]] for item in items_only]
            rerank_scores = self.reranker.predict(pairs)
            
            if self.debug:
                logger.debug("Re-ranker scores: %s...", rerank_scores[:3])
            
            # Sort by re-ranker score
            scored_candidates = list(zip(items_only, rerank_scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Take top_k (or all if retrieve_all)
            if retrieve_all:
                final_selection = [item for item, _ in scored_candidates]
            else:
                final_selection = [item for item, _ in scored_candidates[:top_k]]
        else:
            # No re-ranking, just extract items from tuples
            if retrieve_all:
                final_selection = [item for item, _ in candidates]
            else:
                final_selection = [item for item, _ in candidates[:top_k]]

        if self.debug:
            logger.debug("Returning %d results", len(final_selection))

        # Convert to models
        for item in final_selection:
            results.append(
                KnowledgeChunk(
                    id=item["id"],
                    text=item["text"],
                    metadata=item["metadata"],
                    embedding=None,
                )
            )

        return results

    # --- Query Transforms ---

    @staticmethod
    def decompose_query(feature_text: str, max_sub_queries: int = 4) -> List[str]:
        """
        Query Decomposition: Break long feature text into focused sub-queries.
        
        Splits by detected sections/paragraphs so each sub-query targets
        a specific aspect of the feature, improving retrieval precision.
        """
        if len(feature_text) < 300:
            return [feature_text]
        
        # Split by double newlines (paragraphs) or section markers
        blocks = re.split(r'\n\s*\n|\n(?=#{1,4}\s)|(?=\d+\.\s)', feature_text)
        blocks = [b.strip() for b in blocks if b.strip() and len(b.strip()) > 30]
        
        if len(blocks) <= 1:
            return [feature_text]
        
        # Take the most substantial blocks as sub-queries
        blocks.sort(key=len, reverse=True)
        sub_queries = blocks[:max_sub_queries]
        
        return sub_queries

    @staticmethod
    def expand_query(query_text: str) -> str:
        """
        Query Expansion: Enrich the query with domain-relevant synonyms
        and extracted key terms to improve recall.
        
        Adds synonyms for recognized QA/testing domain terms without
        changing the original query structure.
        """
        query_lower = query_text.lower()
        expansions: List[str] = []
        
        # 1. Domain synonym expansion
        for term, synonyms in _DOMAIN_EXPANSIONS.items():
            if term in query_lower:
                # Add synonyms that aren't already in the query
                for syn in synonyms:
                    if syn.lower() not in query_lower:
                        expansions.append(syn)
        
        # 2. Extract quoted terms and acronyms (important entities)
        for pattern in _KEY_TERM_PATTERNS:
            for match in pattern.finditer(query_text):
                term = match.group(1).strip()
                if len(term) >= 2 and term.lower() not in query_lower:
                    expansions.append(term)
        
        if not expansions:
            return query_text
        
        # Append expansion terms (capped to avoid bloating the query)
        expansion_str = " ".join(expansions[:15])
        expanded = f"{query_text}\n\nRelated terms: {expansion_str}"
        
        return expanded

    def retrieve_multi_query(
        self,
        feature_text: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
        enable_decomposition: bool = True,
        enable_expansion: bool = True,
        max_sub_queries: int = 4,
    ) -> List[KnowledgeChunk]:
        """
        Multi-Query Retrieval with Query Transforms.
        
        1. Decomposes the feature text into focused sub-queries.
        2. Expands each sub-query with domain synonyms.
        3. Runs hybrid retrieval for each transformed query.
        4. Merges and deduplicates results by chunk ID, keeping highest scores.
        
        This dramatically improves recall for complex multi-aspect features
        by ensuring each aspect's terminology hits the right KB chunks.
        """
        # Build the set of queries to run
        queries: List[str] = []
        
        if enable_decomposition and len(feature_text) > 500:
            sub_queries = self.decompose_query(feature_text, max_sub_queries)
            if self.debug:
                logger.debug("Decomposed into %d sub-queries", len(sub_queries))
            queries.extend(sub_queries)
        else:
            queries.append(feature_text)
        
        # Expand each query with domain terms
        if enable_expansion:
            expanded_queries = []
            for q in queries:
                expanded = self.expand_query(q)
                expanded_queries.append(expanded)
                if self.debug and expanded != q:
                    extra_chars = len(expanded) - len(q)
                    logger.debug("Query expanded by %d chars", extra_chars)
            queries = expanded_queries
        
        if len(queries) == 1:
            # Single query — use standard retrieve path
            return self.retrieve(queries[0], top_k=top_k, filters=filters)
        
        # Multi-query retrieval: run each query, merge by chunk ID
        logger.info("Multi-query retrieval: %d queries × top_k=%s", len(queries), top_k)
        
        seen_ids: Set[str] = set()
        merged_results: List[KnowledgeChunk] = []
        
        # Per-query top_k: retrieve more per query since we'll deduplicate
        per_query_k = top_k if top_k == -1 else max(top_k, 3)
        
        for i, query in enumerate(queries):
            # Truncate individual sub-queries for embedding model limits
            if len(query) > 1500:
                query = query[:1500]
            
            chunks = self.retrieve(query, top_k=per_query_k, filters=filters)
            
            new_count = 0
            for chunk in chunks:
                if chunk.id not in seen_ids:
                    seen_ids.add(chunk.id)
                    merged_results.append(chunk)
                    new_count += 1
            
            if self.debug:
                logger.debug("Sub-query %d returned %d chunks, %d new", i+1, len(chunks), new_count)
        
        logger.info("Multi-query merged: %d unique chunks from %d queries", len(merged_results), len(queries))
        
        # If not retrieving all, cap at top_k
        if top_k != -1 and len(merged_results) > top_k:
            merged_results = merged_results[:top_k]
        
        return merged_results
