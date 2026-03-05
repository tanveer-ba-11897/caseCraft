"""
Shared knowledge-base ingestion pipeline.

This module extracts the common chunk → embed → store → graph pipeline
used by ``cli/ingest.py`` (and the MCP server) so the logic
lives in exactly one place.

Storage backend: ChromaDB persistent vector store.

Supports two chunking strategies:
- **Flat** (default): one tier of chunks.
- **Parent–child**: large parent chunks for context, small child chunks
  for precise retrieval.  Both are stored; the retriever expands children
  back to their parents at query time.

Optional knowledge graph:
- Built from chunk metadata + lightweight entity extraction (no LLM).
- Persisted as JSON alongside ChromaDB.
- Used by the retriever for graph-expanded results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

from core.knowledge.chunker import chunk_document, chunk_document_parent_child
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

    __slots__ = (
        "documents", "chunks", "total_index_size",
        "parent_chunks", "child_chunks",
        "graph_nodes", "graph_edges",
    )

    def __init__(
        self,
        documents: int,
        chunks: int,
        total_index_size: int,
        parent_chunks: int = 0,
        child_chunks: int = 0,
        graph_nodes: int = 0,
        graph_edges: int = 0,
    ):
        self.documents = documents
        self.chunks = chunks
        self.total_index_size = total_index_size
        self.parent_chunks = parent_chunks
        self.child_chunks = child_chunks
        self.graph_nodes = graph_nodes
        self.graph_edges = graph_edges


def ingest_documents(
    docs: List[RawDocument],
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    *,
    collection_name: str = DEFAULT_COLLECTION,
    kb_chunk_size: int = 1500,
    parent_child: bool | None = None,
    child_chunk_size: int | None = None,
    child_overlap: int | None = None,
    progress: ProgressCallback = None,
) -> IngestResult:
    """Run the full ingestion pipeline and return an :class:`IngestResult`.

    Parent–child settings are read from ``casecraft.yaml`` by default.
    Pass explicit values to override the config file.

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
        Also used as parent size in parent–child mode.
    parent_child:
        If ``True``, use two-tier parent–child chunking.  ``None`` = read
        from config file.
    child_chunk_size:
        Maximum characters per child chunk.  ``None`` = read from config.
    child_overlap:
        Character overlap between adjacent children.  ``None`` = read from
        config.
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
    # Resolve parent–child settings from config when not explicitly provided
    from core.config import load_config as _load_cfg
    _kb_cfg = _load_cfg().knowledge
    if parent_child is None:
        parent_child = _kb_cfg.parent_child_chunking
    if child_chunk_size is None:
        child_chunk_size = _kb_cfg.child_chunk_size
    if child_overlap is None:
        child_overlap = _kb_cfg.child_overlap
    def _progress(stage: str, msg: str) -> None:
        if progress:
            progress(stage, msg)

    # ── 1. Chunk ──────────────────────────────────────────────────────────
    _progress("chunking", f"Chunking {len(docs)} documents…")
    all_chunks: List[KnowledgeChunk] = []
    n_parents = 0
    n_children = 0

    if parent_child:
        logger.info(
            "Using parent–child chunking (parent=%d, child=%d, overlap=%d)",
            kb_chunk_size, child_chunk_size, child_overlap,
        )
        for doc in docs:
            parents, children = chunk_document_parent_child(
                doc,
                parent_size=kb_chunk_size,
                child_size=child_chunk_size,
                child_overlap=child_overlap,
            )
            all_chunks.extend(parents)
            all_chunks.extend(children)
            n_parents += len(parents)
            n_children += len(children)
        logger.info(
            "Parent–child chunking: %d parents + %d children = %d total",
            n_parents, n_children, len(all_chunks),
        )
    else:
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

    # ── 4. Build knowledge graph (optional) ───────────────────────────
    n_graph_nodes = 0
    n_graph_edges = 0
    if _kb_cfg.knowledge_graph:
        _progress("graph", "Building knowledge graph…")
        from core.knowledge.graph import KnowledgeGraph

        kg = KnowledgeGraph(
            graph_path=_kb_cfg.graph_path,
            max_hops=_kb_cfg.graph_max_hops,
        )
        kg.build_from_chunks(all_chunks)
        kg.save()
        stats = kg.get_statistics()
        n_graph_nodes = stats["nodes"]
        n_graph_edges = stats["edges"]
        logger.info(
            "Knowledge graph: %d nodes, %d edges, %d components | relations: %s",
            n_graph_nodes,
            n_graph_edges,
            stats["connected_components"],
            stats["relations"],
        )

    return IngestResult(
        documents=len(docs),
        chunks=len(all_chunks),
        total_index_size=total,
        parent_chunks=n_parents,
        child_chunks=n_children,
        graph_nodes=n_graph_nodes,
        graph_edges=n_graph_edges,
    )
