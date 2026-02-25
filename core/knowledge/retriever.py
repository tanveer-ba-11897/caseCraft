import json
import re
from typing import List, Optional
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from core.knowledge.models import KnowledgeChunk

# Precompiled tokenizer for BM25 (strips punctuation, lowercases)
_BM25_TOKEN_PATTERN = re.compile(r'\w+')


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
            self.embedder = SentenceTransformer(model_name)
            
            # 2. Load Re-ranker if enabled
            self.reranker = None
            if self.use_reranker:
                try:
                    from sentence_transformers import CrossEncoder
                    self.reranker = CrossEncoder(reranker_name)
                except ImportError:
                    print("Warning: CrossEncoder not available, re-ranking disabled.")
                    self.use_reranker = False
            
        except Exception as exc:
            raise RetrievalError(
                "Failed to load embedding/reranking models"
            ) from exc

        try:
            with self.index_path.open("r", encoding="utf-8") as f:
                self.index = json.load(f)
        except Exception as exc:
            raise RetrievalError(
                "Failed to load knowledge index"
            ) from exc

        # 3. Prepare Dense Index
        self.embeddings = np.array(
            [item["embedding"] for item in self.index],
            dtype=np.float32,
        )
        
        # Free embedding data from the in-memory index to halve memory usage
        for item in self.index:
            del item["embedding"]
        
        # 4. Prepare Sparse Index (BM25) with proper tokenization
        self.bm25 = None
        try:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [_BM25_TOKEN_PATTERN.findall(item["text"].lower()) for item in self.index]
            self.bm25 = BM25Okapi(tokenized_corpus)
            if self.debug:
                print(f"DEBUG: BM25 index built with {len(tokenized_corpus)} documents")
        except ImportError:
            print("Warning: rank_bm25 not installed, keyword search disabled.")

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
            print(f"DEBUG: Retrieving for query: '{query_text[:50]}...' (top_k={'ALL' if retrieve_all else top_k})")

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
            print(f"DEBUG: Score range: [{hybrid_scores.min():.3f}, {hybrid_scores.max():.3f}]")
        
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
                    print(f"DEBUG: Filtered out result with score {score:.3f} (below threshold {self.min_score_threshold})")
                continue
            item = self.index[idx]
            if filters and not all(item["metadata"].get(k) == v for k, v in filters.items()):
                continue
            candidates.append((item, score))
            
        if not candidates:
            return []
        
        if self.debug:
            print(f"DEBUG: {len(candidates)} candidates after filtering")

        # --- Step 2: Re-ranking ---
        
        results = []
        if self.use_reranker and self.reranker:
            # Extract items from (item, score) tuples for re-ranking
            items_only = [item for item, _ in candidates]
            
            # Prepare pairs: [query, doc_text]
            pairs = [[query_text, item["text"]] for item in items_only]
            rerank_scores = self.reranker.predict(pairs)
            
            if self.debug:
                print(f"DEBUG: Re-ranker scores: {rerank_scores[:3]}...")
            
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
            print(f"DEBUG: Returning {len(final_selection)} results")

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
