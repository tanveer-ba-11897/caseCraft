"""
Shared chunking utilities for documents and knowledge base.

This module centralizes heading detection patterns and section-aware
chunking logic to avoid duplication between parser.py and chunker.py.
"""

import re
from typing import List, Tuple

# Patterns for detecting section headings in documents
HEADING_PATTERNS = [
    # Markdown headings: # Heading, ## Heading, ### Heading
    re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE),
    # Numbered sections: 1. Heading, 1.1 Heading, 2.3.1 Heading
    re.compile(r'^(\d+(?:\.\d+)*)\s*[\.:\-]\s*(.+)$', re.MULTILINE),
    # ALL CAPS headings (standalone lines, 3+ chars, no lowercase)
    re.compile(r'^([A-Z][A-Z\s\-&/]{2,})$', re.MULTILINE),
    # Underlined headings (line followed by === or ---)
    re.compile(r'^(.+)\n[=\-]{3,}$', re.MULTILINE),
]

# Additional patterns for web content headings
KB_HEADING_PATTERNS = HEADING_PATTERNS + [
    re.compile(
        r'^(?:Overview|Introduction|Features|Requirements|Setup|Configuration|'
        r'Usage|API|FAQ|Troubleshooting|Notes|Summary)\s*[:.]?\s*$',
        re.MULTILINE | re.IGNORECASE
    ),
]

# Noise patterns to strip during cleaning
NOISE_PATTERNS = [
    re.compile(r'^Page\s+\d+\s*(of\s+\d+)?$', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^\s*Table\s+of\s+Contents\s*$', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^[-_=]{10,}$', re.MULTILINE),
    re.compile(r'^\s*(confidential|proprietary|draft)\s*$', re.MULTILINE | re.IGNORECASE),
]


def detect_sections(text: str, patterns: List[re.Pattern] | None = None) -> List[Tuple[str, int]]:
    """
    Detect section headings and their positions in the text.
    
    Args:
        text: Document text to scan for headings.
        patterns: List of regex patterns to use. Defaults to HEADING_PATTERNS.
        
    Returns:
        List of (heading_text, char_position) sorted by position.
    """
    if patterns is None:
        patterns = HEADING_PATTERNS
        
    sections: List[Tuple[str, int]] = []
    seen_positions: set = set()
    
    for pattern in patterns:
        for match in pattern.finditer(text):
            pos = match.start()
            # Avoid duplicate detections at the same position
            if pos in seen_positions:
                continue
            seen_positions.add(pos)
            
            # Get the full heading text
            heading = match.group(0).strip()
            # Skip very short "headings" that are likely false positives
            if len(heading) < 3:
                continue
            sections.append((heading, pos))
    
    sections.sort(key=lambda x: x[1])
    return sections


def split_by_paragraphs(text: str) -> List[str]:
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


def merge_paragraphs(paragraphs: List[str], max_chars: int) -> List[str]:
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


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100,
) -> List[str]:
    """
    Split text into overlapping chunks.

    Prefers to split at the nearest whitespace before `chunk_size` so chunks
    don't cut words in half. If no whitespace is found in the window (e.g., a
    very long token), falls back to character-based slicing.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Prefer to cut at the last whitespace inside the window
        if end < text_length:
            split_pos = -1
            for sep in (" ", "\n", "\t"):
                pos = text.rfind(sep, start, end)
                if pos > split_pos:
                    split_pos = pos
            if split_pos >= start:
                end = split_pos + 1

        chunk = text[start:end]

        # Skip empty/whitespace-only chunks
        if not chunk.strip():
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]
            if not chunk.strip():
                start = end
                continue

        chunks.append(chunk)

        if end >= text_length:
            break

        # Compute next start with overlap; ensure progress
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


# ── Recursive Character Splitter ──────────────────────────────────────────

# Separator hierarchy from coarsest to finest
_RECURSIVE_SEPARATORS = [
    "\n\n",  # Paragraph break
    "\n",    # Line break
    ". ",    # Sentence end
    "; ",    # Clause separator
    ", ",    # Phrase separator
    " ",     # Word boundary
    "",      # Character (last resort)
]


def recursive_chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100,
    separators: List[str] | None = None,
    _depth: int = 0,
) -> List[str]:
    """
    Recursively split text using progressively finer separators.

    Tries the coarsest separator first (``\\n\\n``).  For any resulting
    piece that still exceeds *chunk_size*, it recurses with the next
    finer separator.  This produces chunks that respect document
    structure (paragraphs > lines > sentences > words) as deeply as
    possible while staying within the size budget.

    Parameters
    ----------
    text:
        Text to split.
    chunk_size:
        Maximum characters per chunk.
    overlap:
        Characters of overlap between adjacent chunks at the same level.
    separators:
        Ordered list of separators to try (coarsest first).
        Defaults to ``_RECURSIVE_SEPARATORS``.

    Returns
    -------
    list[str]
        Chunks that each fit within *chunk_size*.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    if separators is None:
        separators = list(_RECURSIVE_SEPARATORS)

    text = text.strip()
    if not text:
        return []

    # Base case: text already fits
    if len(text) <= chunk_size:
        return [text]

    # Base case: no separators left — hard split at word/char boundary
    if not separators:
        return chunk_text(text, chunk_size, overlap)

    sep = separators[0]
    remaining_seps = separators[1:]

    # Split on current separator
    if sep:
        parts = text.split(sep)
    else:
        # Empty separator = character-level (last resort)
        return chunk_text(text, chunk_size, overlap)

    # Merge small splits back together up to chunk_size
    chunks: List[str] = []
    current = ""
    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # Flush current buffer
            if current:
                chunks.append(current)
            # This part alone may exceed chunk_size → recurse with finer sep
            if len(part) > chunk_size:
                sub_chunks = recursive_chunk_text(
                    part, chunk_size, overlap, remaining_seps, _depth + 1,
                )
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    # Apply overlap between adjacent chunks at this recursion level
    if overlap > 0 and len(chunks) > 1:
        overlapped: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            # Take the tail of the previous chunk as overlap prefix
            tail = prev[-overlap:] if len(prev) > overlap else prev
            merged = tail + sep + chunks[i]
            if len(merged) <= chunk_size:
                overlapped.append(merged)
            else:
                overlapped.append(chunks[i])
        chunks = overlapped

    return [c for c in chunks if c.strip()]


def chunk_by_sections(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100,
    patterns: List[re.Pattern] | None = None,
    prefix_format: str = "[Section: {heading}]\n",
) -> List[str]:
    """
    Section-aware chunking: split at section boundaries first, then apply
    size-based splitting within each section. Each chunk gets its section
    heading prepended as context.
    
    Args:
        text: Document text to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Character overlap between chunks (for non-section fallback).
        patterns: Heading patterns to use. Defaults to HEADING_PATTERNS.
        prefix_format: Format string for heading prefix. Use {heading} placeholder.
        
    Returns:
        List of text chunks with section context.
    """
    sections = detect_sections(text, patterns)
    
    if len(sections) < 2:
        # No meaningful structure detected — fall back to basic chunking
        return chunk_text(text, chunk_size, overlap)
    
    # Split text into section blocks
    section_blocks: List[Tuple[str, str]] = []  # (heading, body)
    for i, (heading, pos) in enumerate(sections):
        end_pos = sections[i + 1][1] if i + 1 < len(sections) else len(text)
        body = text[pos:end_pos].strip()
        
        # Remove the heading line from the body
        body_lines = body.split("\n", 1)
        if len(body_lines) > 1:
            body = body_lines[1].strip()
        else:
            body = ""
        
        if body:
            section_blocks.append((heading, body))
    
    # Also capture text before the first heading (preamble)
    first_heading_pos = sections[0][1]
    preamble = text[:first_heading_pos].strip()
    
    chunks: List[str] = []
    
    if preamble:
        preamble_chunks = chunk_text(preamble, chunk_size, overlap)
        chunks.extend(preamble_chunks)
    
    # Chunk each section, prepending its heading for structural context
    for heading, body in section_blocks:
        heading_prefix = prefix_format.format(heading=heading)
        effective_size = max(chunk_size - len(heading_prefix), 200)
        
        if len(body) <= effective_size:
            chunks.append(f"{heading_prefix}{body}")
        else:
            sub_chunks = chunk_text(body, effective_size, overlap)
            for sub in sub_chunks:
                chunks.append(f"{heading_prefix}{sub}")
    
    return chunks
