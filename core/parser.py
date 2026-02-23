from pathlib import Path
from typing import List

from pypdf import PdfReader


class DocumentParseError(Exception):
    pass


def _clean_text(text: str) -> str:
    """
    Basic text normalization.
    """
    text = text.replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100,
) -> List[str]:
    """
    Split text into overlapping chunks.

    Prefers to split at the nearest whitespace before `chunk_size` so chunks
    don't cut words in half. If no whitespace is found in the window (e.g., a
    very long token), falls back to character-based slicing. Chunks include
    trailing whitespace when possible so boundaries are clear.
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

        # Prefer to cut at the last whitespace inside the window so words aren't
        # split in the middle. If we find a whitespace, include it in the chunk
        # (end = pos + 1) so the chunk ends with whitespace.
        if end < text_length:
            split_pos = -1
            for sep in (" ", "\n", "\t"):
                pos = text.rfind(sep, start, end)
                if pos > split_pos:
                    split_pos = pos
            if split_pos >= start:
                end = split_pos + 1

        chunk = text[start:end]

        # If the chunk is empty or contains only whitespace (which can happen if
        # start==end or whitespace clusters), fall back to a raw slice to make
        # progress and avoid infinite loops.
        if not chunk.strip():
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]
            if not chunk.strip():
                # Nothing useful here; move forward
                start = end
                continue

        chunks.append(chunk)

        # Compute next start with overlap; ensure progress to avoid infinite
        # loops in edge cases
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


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
    Parse a document and return cleaned, chunked text.
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

    return _chunk_text(
        cleaned_text,
        chunk_size=chunk_size,
        overlap=overlap,
    )
