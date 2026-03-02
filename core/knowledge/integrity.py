"""
Knowledge-index integrity verification (R3).

Provides SHA-256 hash computation, persistence, and verification for the
knowledge index file (``knowledge_base/index.json``).  A tampered or
corrupted index will be detected at load time so the retriever can reject
it rather than silently serving poisoned RAG context.

Usage
-----
**After writing the index** (ingest path):

    from core.knowledge.integrity import write_hash
    write_hash(index_path)

**Before reading the index** (retriever path):

    from core.knowledge.integrity import verify_hash, IntegrityError
    verify_hash(index_path)          # raises IntegrityError on mismatch
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger("casecraft.integrity")

# The hash file lives alongside the index with a ``.sha256`` extension.
_HASH_SUFFIX = ".sha256"

# Read in 64 KiB blocks to keep memory usage bounded for large indices.
_READ_BLOCK = 65_536


class IntegrityError(Exception):
    """Raised when the index hash does not match the stored hash."""


def _hash_path(index_path: Path) -> Path:
    """Return the sibling hash-file path for *index_path*."""
    return index_path.with_suffix(index_path.suffix + _HASH_SUFFIX)


def compute_hash(index_path: Path) -> str:
    """Return the hex-encoded SHA-256 digest of *index_path*.

    Parameters
    ----------
    index_path:
        Path to the knowledge index file (e.g. ``knowledge_base/index.json``).

    Returns
    -------
    str
        Lowercase hex SHA-256 digest.

    Raises
    ------
    FileNotFoundError
        If *index_path* does not exist.
    OSError
        On I/O errors.
    """
    h = hashlib.sha256()
    with open(index_path, "rb") as f:
        while True:
            block = f.read(_READ_BLOCK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_hash(index_path: str | Path) -> Path:
    """Compute and persist the SHA-256 hash of the index file.

    The hash is written atomically to a sibling file
    (``<index>.sha256``) so a partial write cannot leave a
    half-written hash.

    Parameters
    ----------
    index_path:
        Path to the knowledge index file.

    Returns
    -------
    Path
        The path to the written hash file.
    """
    index_path = Path(index_path)
    digest = compute_hash(index_path)
    hp = _hash_path(index_path)

    # Atomic write: write to a temp file, then rename.
    tmp = hp.with_suffix(".tmp")
    try:
        tmp.write_text(digest, encoding="utf-8")
        tmp.replace(hp)
    except BaseException:
        # Clean up on failure.
        tmp.unlink(missing_ok=True)
        raise

    logger.info("Wrote index hash %s → %s", digest[:12], hp.name)
    return hp


def verify_hash(index_path: str | Path) -> None:
    """Verify that the index file matches its stored SHA-256 hash.

    Parameters
    ----------
    index_path:
        Path to the knowledge index file.

    Raises
    ------
    IntegrityError
        If the hash file is missing, unreadable, or does not match.
    """
    index_path = Path(index_path)
    hp = _hash_path(index_path)

    if not hp.exists():
        raise IntegrityError(
            f"Hash file not found at {hp}. "
            "Re-run ingestion to regenerate the integrity hash."
        )

    try:
        stored = hp.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise IntegrityError(f"Cannot read hash file {hp}: {exc}") from exc

    if len(stored) != 64 or not all(c in "0123456789abcdef" for c in stored):
        raise IntegrityError(
            f"Hash file {hp} contains an invalid SHA-256 digest."
        )

    actual = compute_hash(index_path)
    if actual != stored:
        raise IntegrityError(
            f"Index integrity check FAILED for {index_path.name}. "
            f"Expected {stored[:12]}…, got {actual[:12]}…. "
            "The index may have been tampered with or corrupted. "
            "Re-run ingestion to rebuild and re-sign the index."
        )

    logger.info("Index integrity verified (%s)", actual[:12])
