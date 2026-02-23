from typing import List, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings

from core.knowledge.models import KnowledgeChunk


class KnowledgeStoreError(Exception):
    pass


class KnowledgeStore:
    """
    Local persistent vector store for knowledge chunks using Chroma.
    """

    def __init__(
        self,
        persist_directory: str = "knowledge_base/chroma",
        collection_name: str = "casecraft_knowledge",
    ):
        try:
            persist_path = Path(persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)

            self.client = chromadb.Client(
                Settings(
                    persist_directory=str(persist_path),
                    anonymized_telemetry=False,
                    chroma_db_impl="duckdb+parquet",
                )
            )

            self.collection = self.client.get_or_create_collection(
                name=collection_name
            )

        except Exception as exc:
            raise KnowledgeStoreError(
                "Failed to initialize knowledge store"
            ) from exc

    def add(self, chunks: List[KnowledgeChunk]) -> None:
        """
        Add knowledge chunks to the vector store.
        """
        if not chunks:
            return

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for chunk in chunks:
            if chunk.embedding is None:
                raise KnowledgeStoreError(
                    f"Chunk {chunk.id} has no embedding"
                )

            ids.append(chunk.id)
            documents.append(chunk.text)
            embeddings.append(chunk.embedding)
            metadatas.append(chunk.metadata)

        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            self.client.persist()
        except Exception as exc:
            raise KnowledgeStoreError(
                "Failed to add chunks to store"
            ) from exc

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> List[KnowledgeChunk]:
        """
        Query the vector store and return matching KnowledgeChunks.
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters,
            )
        except Exception as exc:
            raise KnowledgeStoreError(
                "Vector store query failed"
            ) from exc

        ids = results.get("ids")
        documents = results.get("documents")
        metadatas = results.get("metadatas")

        if not ids or not ids[0]:
            return []

        chunks: List[KnowledgeChunk] = []

        for idx, chunk_id in enumerate(ids[0]):
            text = ""
            metadata = {}

            if documents and documents[0]:
                text = documents[0][idx]

            if metadatas and metadatas[0]:
                metadata = metadatas[0][idx]

            chunks.append(
                KnowledgeChunk(
                    id=chunk_id,
                    text=text,
                    metadata=metadata,
                    embedding=None,
                )
            )

        return chunks
