from pathlib import Path
from typing import List

from core.parser import parse_document
from core.knowledge.models import RawDocument


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def _detect_source_type(path: Path) -> str:
    """
    Infer document type based on folder or filename.
    """
    lower_path = str(path).lower()

    if "feature" in lower_path:
        return "feature_doc"
    if "rule" in lower_path or "permission" in lower_path:
        return "system_rule"
    if "product" in lower_path or "guide" in lower_path:
        return "product_doc"

    return "unknown"


def load_documents(directory: str) -> List[RawDocument]:
    """
    Load supported documents from a directory and return raw text documents.
    """
    base_path = Path(directory)

    if not base_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    documents: List[RawDocument] = []

    for file_path in base_path.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text_chunks = parse_document(str(file_path))

        if not text_chunks:
            continue

        full_text = "\n\n".join(text_chunks)

        documents.append(
            RawDocument(
                text=full_text,
                source_name=file_path.name,
                source_type=_detect_source_type(file_path),
            )
        )

    return documents
