"""
Shared knowledge-base ingestion pipeline.

This module extracts the common chunk → embed → store pipeline used by
both ``cli/ingest.py`` and ``bridge_server.py`` so the logic lives in
exactly one place.

Storage backend: ChromaDB persistent vector store.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

from core.knowledge.chunker import chunk_document
from core.knowledge.embedder import Embedder
from core.knowledge.models import KnowledgeChunk, RawDocument
from core.knowledge.vector_store import (
    VectorStore,
    DEFAULT_PERSIST_DIR,
    DEFAULT_COLLECTION,
)

logger = logging.getLogger("casecraft.ingest")

# Type alias for an optional progress callback.
# Signature: (stage: str, message: str) -> None
ProgressCallback = Optional[Callable[[str, str], None]]


class IngestResult:
    """Value object returned after a successful ingestion run."""

    __slots__ = ("documents", "chunks", "total_index_size")

    def __init__(self, documents: int, chunks: int, total_index_size: int):
        self.documents = documents
        self.chunks = chunks
        self.total_index_size = total_index_size


def ingest_documents(
    docs: List[RawDocument],
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    *,
    collection_name: str = DEFAULT_COLLECTION,
    kb_chunk_size: int = 1500,
    progress: ProgressCallback = None,
) -> IngestResult:
    """Run the full ingestion pipeline and return an :class:`IngestResult`.

    Parameters
    ----------
    docs:
        Parsed raw documents to ingest.
    persist_dir:
        ChromaDB persistence directory.
    collection_name:
        ChromaDB collection name.
    kb_chunk_size:
        Maximum characters per knowledge-base chunk (passed to the chunker).
    progress:
        Optional ``(stage, message)`` callback for reporting progress.

    Returns
    -------
    IngestResult
        Summary counters for the caller.

    Raises
    ------
    ValueError
        If *docs* is empty or no content can be extracted.
    """
    if not docs:
        raise ValueError("No documents to ingest")

    def _progress(stage: str, msg: str) -> None:
        if progress:
            progress(stage, msg)

    # ── 1. Chunk ──────────────────────────────────────────────────────────
    _progress("chunking", f"Chunking {len(docs)} documents…")
    all_chunks: List[KnowledgeChunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, max_chars=kb_chunk_size))

    if not all_chunks:
        raise ValueError("No content could be extracted from the documents")

    # ── 2. Embed ──────────────────────────────────────────────────────────
    _progress("embedding", f"Embedding {len(all_chunks)} chunks…")
    embedder = Embedder()
    embedder.embed_chunks(all_chunks)

    # ── 3. Store in ChromaDB ──────────────────────────────────────────────
    _progress("saving", "Storing chunks in vector database…")
    store = VectorStore(
        persist_dir=str(persist_dir),
        collection_name=collection_name,
    )
    added = store.add_chunks(all_chunks)
    total = store.count()

    logger.info("Stored %d new chunks (%d total in collection)", added, total)

    return IngestResult(
        documents=len(docs),
        chunks=len(all_chunks),
        total_index_size=total,
    )
