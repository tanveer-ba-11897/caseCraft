import uuid
from typing import List

from core.chunking import (
    KB_HEADING_PATTERNS,
    detect_sections,
    split_by_paragraphs,
    merge_paragraphs,
)
from core.knowledge.models import RawDocument, KnowledgeChunk
from core.config import load_config


_cfg = load_config()
DEFAULT_MAX_CHARS: int = _cfg.knowledge.kb_chunk_size


def _chunk_section_aware(text: str, max_chars: int) -> List[str]:
    """
    Section-aware chunking for knowledge base documents.
    
    1. Detects headings/sections in the text.
    2. Splits at section boundaries to keep related content together.
    3. Prepends section heading to each chunk for structural context.
    4. Falls back to paragraph-based merging if no sections detected.
    """
    sections = detect_sections(text, KB_HEADING_PATTERNS)
    
    if len(sections) < 2:
        # No meaningful structure — fall back to paragraph merging
        paragraphs = split_by_paragraphs(text)
        return merge_paragraphs(paragraphs, max_chars)
    
    chunks: List[str] = []
    
    # Capture preamble (text before first heading)
    first_pos = sections[0][1]
    preamble = text[:first_pos].strip()
    if preamble:
        preamble_paras = split_by_paragraphs(preamble)
        chunks.extend(merge_paragraphs(preamble_paras, max_chars))
    
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
            # Split large sections by paragraphs, prepend heading to each
            paras = split_by_paragraphs(body)
            merged = merge_paragraphs(paras, effective_max)
            for sub in merged:
                chunks.append(f"{heading_prefix}{sub}")
    
    return chunks


def chunk_document(
    document: RawDocument,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> List[KnowledgeChunk]:
    """
    Split a RawDocument into semantically meaningful KnowledgeChunks.
    Uses section-aware splitting to keep related content together
    and prepend section headings for structural context.
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
