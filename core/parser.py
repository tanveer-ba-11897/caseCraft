import logging
from pathlib import Path
from typing import List

from pypdf import PdfReader

from core.chunking import (
    NOISE_PATTERNS,
    chunk_by_sections,
)

logger = logging.getLogger("casecraft.parser")


class DocumentParseError(Exception):
    pass


def _clean_text(text: str) -> str:
    """
    Text normalization with noise removal.
    Strips page numbers, decorative separators, and boilerplate lines.
    """
    text = text.replace("\r", "\n")
    
    # Remove noise patterns using shared patterns
    for pattern in NOISE_PATTERNS:
        text = pattern.sub('', text)
    
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _parse_pdf(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        pages_text: List[str] = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)

        if not pages_text:
            raise DocumentParseError("No extractable text found in PDF")

        return "\n".join(pages_text)

    except Exception as exc:
        raise DocumentParseError(f"Failed to parse PDF: {exc}") from exc


def _parse_text(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8")


def parse_document(
    file_path: str,
    chunk_size: int = 1000,
    overlap: int = 100,
) -> List[str]:
    """
    Parse a document and return cleaned, section-aware chunked text.
    Uses heading detection to split at logical boundaries, preserving
    section context in each chunk.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # SEC-08: Reject files larger than 50MB to prevent memory exhaustion
    MAX_FILE_SIZE_MB = 50
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise DocumentParseError(f"File too large ({file_size_mb:.1f}MB). Maximum allowed: {MAX_FILE_SIZE_MB}MB.")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        raw_text = _parse_pdf(file_path)
    elif suffix in {".txt", ".md"}:
        raw_text = _parse_text(file_path)
    else:
        raise DocumentParseError(f"Unsupported file type: {suffix}")

    cleaned_text = _clean_text(raw_text)

    if not cleaned_text:
        raise DocumentParseError("Document is empty after cleaning")

    return chunk_by_sections(
        cleaned_text,
        chunk_size=chunk_size,
        overlap=overlap,
    )
