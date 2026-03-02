"""
ChromaDB-backed persistent vector store for the CaseCraft knowledge base.

Replaces the flat JSON index with a proper vector database that supports:
- Persistent on-disk storage (no 52 MB JSON reload on every startup)
- Native HNSW approximate nearest-neighbor search
- Built-in metadata filtering
- Efficient batch upsert operations

Usage::

    store = VectorStore()                       # opens / creates DB
    store.add_chunks(embedded_chunks)           # upsert after ingest
    results = store.query([embedding], n=10)    # similarity search
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.types import GetResult, Include, QueryResult
from chromadb.config import Settings as ChromaSettings

from core.knowledge.models import KnowledgeChunk

logger = logging.getLogger("casecraft.vector_store")

DEFAULT_PERSIST_DIR = "knowledge_base/chroma_db"
DEFAULT_COLLECTION = "casecraft_kb"


class VectorStoreError(Exception):
    """Raised when a vector-store operation fails."""
    pass


class VectorStore:
    """
    Persistent ChromaDB collection wrapper.

    Parameters
    ----------
    persist_dir:
        Directory for the on-disk ChromaDB database.
    collection_name:
        Name of the ChromaDB collection to use.
    """

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
    ):
        self.persist_dir = Path(persist_dir)
        self._collection_name = collection_name

        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "Vector store ready: %d chunks in %s",
                self.collection.count(),
                self.persist_dir,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to initialise ChromaDB at {self.persist_dir}"
            ) from exc

    # ── Write Operations ──────────────────────────────────────────

    def add_chunks(
        self,
        chunks: List[KnowledgeChunk],
        batch_size: int = 500,
    ) -> int:
        """Upsert embedded knowledge chunks. Returns count added."""
        if not chunks:
            return 0

        added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.id for c in batch]
            docs = [c.text for c in batch]
            metas = [c.metadata for c in batch]
            embs = [c.embedding for c in batch]

            kwargs: dict = {
                "ids": ids,
                "documents": docs,
                "metadatas": metas,
            }
            # Only include embeddings if every chunk in the batch has one
            if all(e is not None and len(e) > 0 for e in embs):
                kwargs["embeddings"] = embs

            self.collection.upsert(**kwargs)
            added += len(batch)

        return added

    def add_raw_entries(
        self,
        entries: List[dict],
        batch_size: int = 500,
    ) -> int:
        """Upsert raw dict entries (used during JSON→ChromaDB migration)."""
        if not entries:
            return 0

        added = 0
        for i in range(0, len(entries), batch_size):
            batch = entries[i : i + batch_size]
            ids = [e.get("id", f"legacy-{i + j}") for j, e in enumerate(batch)]
            docs = [e["text"] for e in batch]
            metas = [e.get("metadata", {}) for e in batch]
            embs = [e.get("embedding") for e in batch]

            kwargs: dict = {"ids": ids, "documents": docs, "metadatas": metas}
            if all(e is not None and isinstance(e, list) and len(e) > 0 for e in embs):
                kwargs["embeddings"] = embs

            self.collection.upsert(**kwargs)
            added += len(batch)

        return added

    def reset_collection(self) -> None:
        """Delete and recreate the collection (for clean re-ingestion)."""
        self.client.delete_collection(self._collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection reset: %s", self._collection_name)

    # ── Read / Query Operations ───────────────────────────────────

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """
        HNSW similarity search.

        Returns a ChromaDB results dict with keys
        ``ids``, ``documents``, ``metadatas``, ``distances``.
        Each value is a list-of-lists (one inner list per query embedding).
        """
        n = min(n_results, max(self.collection.count(), 1))
        kwargs: dict = {
            "query_embeddings": query_embeddings,
            "n_results": n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    def get_all(
        self,
        include: Optional[Include] = None,
    ) -> GetResult:
        """
        Fetch every document in the collection.

        Typically used once to build the BM25 sparse index.
        """
        if include is None:
            include = ["documents", "metadatas"]
        return self.collection.get(include=include)

    def count(self) -> int:
        """Return number of chunks in the collection."""
        return self.collection.count()


# ── JSON → ChromaDB Migration ────────────────────────────────────


def migrate_json_to_chroma(
    json_path: str | Path = "knowledge_base/index.json",
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> int:
    """
    One-time migration of a flat JSON knowledge index into ChromaDB.

    Parameters
    ----------
    json_path:
        Path to the legacy ``index.json`` file.
    persist_dir:
        ChromaDB persistence directory.
    collection_name:
        Target collection name.

    Returns
    -------
    int
        Number of chunks migrated.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        logger.warning("No legacy index found at %s — nothing to migrate", json_path)
        return 0

    logger.info("Migrating legacy index %s → ChromaDB (%s)...", json_path, persist_dir)
    t0 = time.time()

    with open(json_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    if not isinstance(index, list) or not index:
        logger.warning("Legacy index is empty or invalid — nothing to migrate")
        return 0

    # Filter valid entries (must have text + embedding)
    valid = [
        e for e in index
        if isinstance(e, dict)
        and "text" in e
        and "embedding" in e
        and isinstance(e["embedding"], list)
        and len(e["embedding"]) > 0
    ]
    skipped = len(index) - len(valid)
    if skipped:
        logger.warning("Skipped %d entries without valid embeddings", skipped)

    if not valid:
        return 0

    store = VectorStore(persist_dir=persist_dir, collection_name=collection_name)
    migrated = store.add_raw_entries(valid)

    elapsed = time.time() - t0
    logger.info("Migration complete: %d chunks in %.1fs", migrated, elapsed)
    return migrated
