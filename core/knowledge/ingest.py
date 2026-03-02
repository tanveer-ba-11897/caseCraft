"""
Shared knowledge-base ingestion pipeline.

This module extracts the common chunk → embed → merge → write → hash
pipeline used by both ``cli/ingest.py`` and ``bridge_server.py`` so the
logic lives in exactly one place.
"""

from __future__ import annotations

import json
import logging
import os
import platform
from pathlib import Path
from typing import Callable, List, Optional

from core.knowledge.chunker import chunk_document
from core.knowledge.embedder import Embedder
from core.knowledge.models import KnowledgeChunk, RawDocument

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
    index_path: str | Path,
    *,
    kb_chunk_size: int = 1500,
    progress: ProgressCallback = None,
) -> IngestResult:
    """Run the full ingestion pipeline and return an :class:`IngestResult`.

    Parameters
    ----------
    docs:
        Parsed raw documents to ingest.
    index_path:
        Path to the knowledge index JSON file.
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

    index_path = Path(index_path)

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

    # ── 3. Merge with existing index ──────────────────────────────────────
    _progress("saving", "Updating knowledge index…")
    index_path.parent.mkdir(parents=True, exist_ok=True)

    existing: list = []
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            if isinstance(raw_data, list):
                # Accept entries that have at least 'text' and 'metadata'
                existing = [
                    entry for entry in raw_data
                    if isinstance(entry, dict)
                    and "text" in entry
                    and "metadata" in entry
                ]
                skipped = len(raw_data) - len(existing)
                if skipped:
                    logger.warning("Skipped %d invalid entries in existing index", skipped)
            else:
                logger.warning("Invalid index format — starting fresh")
                existing = []
        except (json.JSONDecodeError, OSError) as read_err:
            logger.warning("Could not read existing index, starting fresh: %s", read_err)
            existing = []

    for chunk in all_chunks:
        entry: dict = {
            "text": chunk.text,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding if chunk.embedding is not None else [],
        }
        if chunk.id:
            entry["id"] = chunk.id
        existing.append(entry)

    # ── 4. Write index ────────────────────────────────────────────────────
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False)

    # ── 5. Integrity hash (R3) ────────────────────────────────────────────
    try:
        from core.knowledge.integrity import write_hash
        write_hash(index_path)
    except Exception as hash_err:
        logger.warning("Could not write index integrity hash: %s", hash_err)

    # ── 6. Restrict file permissions (best-effort) ────────────────────────
    try:
        os.chmod(index_path, 0o600)
    except OSError:
        if platform.system() != "Windows":
            logger.warning("Could not restrict index file permissions")

    return IngestResult(
        documents=len(docs),
        chunks=len(all_chunks),
        total_index_size=len(existing),
    )
