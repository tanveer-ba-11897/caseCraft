import uuid
from typing import List

from core.chunking import (
    KB_HEADING_PATTERNS,
    detect_sections,
    split_by_paragraphs,
    merge_paragraphs,
    recursive_chunk_text,
)
from core.knowledge.models import RawDocument, KnowledgeChunk
from core.config import load_config


_cfg = load_config()
DEFAULT_MAX_CHARS: int = _cfg.knowledge.kb_chunk_size

# Parent–child defaults
PARENT_CHUNK_SIZE: int = 1500   # larger window for rich context
CHILD_CHUNK_SIZE: int = 400     # smaller window for precise retrieval
CHILD_OVERLAP: int = 50


def _chunk_section_aware(text: str, max_chars: int) -> List[str]:
    """
    Section-aware chunking for knowledge base documents.
    
    1. Detects headings/sections in the text.
    2. Splits at section boundaries to keep related content together.
    3. Prepends section heading to each chunk for structural context.
    4. Falls back to recursive paragraph-based splitting if no sections detected.
    """
    sections = detect_sections(text, KB_HEADING_PATTERNS)
    
    if len(sections) < 2:
        # No meaningful structure — fall back to recursive splitting
        return recursive_chunk_text(text, chunk_size=max_chars, overlap=50)
    
    chunks: List[str] = []
    
    # Capture preamble (text before first heading)
    first_pos = sections[0][1]
    preamble = text[:first_pos].strip()
    if preamble:
        chunks.extend(recursive_chunk_text(preamble, chunk_size=max_chars, overlap=50))
    
    # Process each section
    for i, (heading, pos) in enumerate(sections):
        end_pos = sections[i + 1][1] if i + 1 < len(sections) else len(text)
        section_text = text[pos:end_pos].strip()
        
        # Separate heading from body
        lines = section_text.split("\n", 1)
        body = lines[1].strip() if len(lines) > 1 else ""
        
        if not body:
            continue
        
        heading_prefix = f"[{heading}] "
        effective_max = max(max_chars - len(heading_prefix), 200)
        
        if len(body) <= effective_max:
            chunks.append(f"{heading_prefix}{body}")
        else:
            # Recursively split large sections, prepend heading to each
            sub_chunks = recursive_chunk_text(body, chunk_size=effective_max, overlap=50)
            for sub in sub_chunks:
                chunks.append(f"{heading_prefix}{sub}")
    
    return chunks


def chunk_document(
    document: RawDocument,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> List[KnowledgeChunk]:
    """
    Split a RawDocument into semantically meaningful KnowledgeChunks.
    Uses section-aware splitting with recursive sub-splitting to keep
    related content together and prepend section headings for structural context.
    """
    merged_chunks = _chunk_section_aware(document.text, max_chars)

    knowledge_chunks: List[KnowledgeChunk] = []

    for index, chunk_text in enumerate(merged_chunks):
        chunk_id = f"{document.source_name}-{index}-{uuid.uuid4().hex[:8]}"

        metadata = {
            "source_type": document.source_type,
            "source_name": document.source_name,
        }

        knowledge_chunks.append(
            KnowledgeChunk(
                id=chunk_id,
                text=chunk_text.strip(),
                metadata=metadata,
            )
        )

    return knowledge_chunks


# ── Parent–Child Chunking ─────────────────────────────────────────────────


def chunk_document_parent_child(
    document: RawDocument,
    parent_size: int = PARENT_CHUNK_SIZE,
    child_size: int = CHILD_CHUNK_SIZE,
    child_overlap: int = CHILD_OVERLAP,
) -> tuple[List[KnowledgeChunk], List[KnowledgeChunk]]:
    """
    Two-tier parent–child chunking for improved RAG retrieval.

    Produces **parent chunks** (large, context-rich) and **child chunks**
    (small, retrieval-optimised).  Each child stores its ``parent_id`` in
    metadata so the retriever can fetch the parent for richer context.

    Flow::

        Document → section-aware split → PARENT chunks (≤ parent_size)
                                             ↓
                                   recursive split → CHILD chunks (≤ child_size)

    Parameters
    ----------
    document:
        Raw document to chunk.
    parent_size:
        Maximum characters per parent chunk.
    child_size:
        Maximum characters per child chunk.
    child_overlap:
        Overlap between adjacent child chunks within the same parent.

    Returns
    -------
    (parents, children)
        Two lists of :class:`KnowledgeChunk`.  Parents have
        ``metadata["chunk_type"] == "parent"``; children have
        ``metadata["chunk_type"] == "child"`` and ``metadata["parent_id"]``.
    """
    # Step 1: Create parent chunks with section-aware splitting
    parent_texts = _chunk_section_aware(document.text, parent_size)

    parents: List[KnowledgeChunk] = []
    children: List[KnowledgeChunk] = []

    for p_idx, parent_text in enumerate(parent_texts):
        parent_text = parent_text.strip()
        if not parent_text:
            continue

        parent_id = f"{document.source_name}-p{p_idx}-{uuid.uuid4().hex[:8]}"

        parent_meta = {
            "source_type": document.source_type,
            "source_name": document.source_name,
            "chunk_type": "parent",
            "child_count": 0,  # updated below
        }

        parent_chunk = KnowledgeChunk(
            id=parent_id,
            text=parent_text,
            metadata=parent_meta,
        )

        # Step 2: Split each parent into children using recursive chunker
        if len(parent_text) <= child_size:
            # Parent is small enough to be its own child
            child_id = f"{parent_id}-c0"
            child_meta = {
                "source_type": document.source_type,
                "source_name": document.source_name,
                "chunk_type": "child",
                "parent_id": parent_id,
            }
            child_chunk = KnowledgeChunk(
                id=child_id,
                text=parent_text,
                metadata=child_meta,
            )
            children.append(child_chunk)
            parent_chunk.metadata["child_count"] = 1
        else:
            child_texts = recursive_chunk_text(
                parent_text,
                chunk_size=child_size,
                overlap=child_overlap,
            )
            for c_idx, child_text in enumerate(child_texts):
                child_text = child_text.strip()
                if not child_text:
                    continue
                child_id = f"{parent_id}-c{c_idx}"
                child_meta = {
                    "source_type": document.source_type,
                    "source_name": document.source_name,
                    "chunk_type": "child",
                    "parent_id": parent_id,
                }
                children.append(
                    KnowledgeChunk(
                        id=child_id,
                        text=child_text,
                        metadata=child_meta,
                    )
                )
            parent_chunk.metadata["child_count"] = len(
                [c for c in children if c.metadata.get("parent_id") == parent_id]
            )

        parents.append(parent_chunk)

    return parents, children
