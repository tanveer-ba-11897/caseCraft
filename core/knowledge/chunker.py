import uuid
from typing import List

from core.knowledge.models import RawDocument, KnowledgeChunk


DEFAULT_MAX_CHARS = 1500  # ~300–500 tokens depending on content


def _split_by_paragraphs(text: str) -> List[str]:
    """
    Split text into logical paragraphs.
    """
    paragraphs = []
    current = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue

        current.append(line)

    if current:
        paragraphs.append(" ".join(current))

    return paragraphs


def _merge_paragraphs(paragraphs: List[str], max_chars: int) -> List[str]:
    """
    Merge paragraphs into chunks that do not exceed max_chars.
    """
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) <= max_chars:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunk_document(
    document: RawDocument,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> List[KnowledgeChunk]:
    """
    Split a RawDocument into semantically meaningful KnowledgeChunks.
    """
    paragraphs = _split_by_paragraphs(document.text)
    merged_chunks = _merge_paragraphs(paragraphs, max_chars)

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
