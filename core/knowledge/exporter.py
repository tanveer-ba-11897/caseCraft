import json
from pathlib import Path
from typing import List

from core.knowledge.models import KnowledgeChunk


def export_knowledge_index(
    chunks: List[KnowledgeChunk],
    output_path: str = "knowledge_base/index.json",
) -> None:
    """
    Export embedded knowledge chunks to a JSON index
    for lightweight runtime retrieval.
    """
    data = []

    for chunk in chunks:
        if chunk.embedding is None:
            continue

        data.append(
            {
                "id": chunk.id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "embedding": chunk.embedding,
            }
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
