from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class RawDocument:
    """
    Represents a raw document loaded from disk before chunking.
    """
    text: str
    source_name: str
    source_type: str


@dataclass
class KnowledgeChunk:
    """
    Represents a single retrievable knowledge unit.
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
