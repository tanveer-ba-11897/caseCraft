from typing import List, Optional

from sentence_transformers import SentenceTransformer

from core.knowledge.models import KnowledgeChunk


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class EmbeddingError(Exception):
    pass


class Embedder:
    """
    Handles embedding generation for knowledge chunks.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, model: Optional[SentenceTransformer] = None):
        try:
            self.model = model if model is not None else SentenceTransformer(model_name)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model: {model_name}"
            ) from exc

    def embed_strings(
        self,
        texts: List[str],
        batch_size: int = 16,
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of strings.
        """
        if not texts:
            return []

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            return embeddings.tolist()
        except Exception as exc:
            raise EmbeddingError("Embedding generation failed") from exc

    def embed_chunks(
        self,
        chunks: List[KnowledgeChunk],
        batch_size: int = 16,
    ) -> List[KnowledgeChunk]:
        """
        Generate embeddings for a list of KnowledgeChunks.
        """
        if not chunks:
            return chunks

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_strings(texts, batch_size=batch_size)

        for chunk, vector in zip(chunks, embeddings):
            chunk.embedding = vector
        
        return chunks
