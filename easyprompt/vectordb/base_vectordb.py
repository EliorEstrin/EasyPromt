"""Base class for vector database adapters."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class BaseVectorDB(ABC):
    """Abstract base class for vector database adapters."""

    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector database connection."""
        pass

    @abstractmethod
    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents with embeddings to the database.

        Args:
            documents: List of documents with format:
                {
                    "id": str,
                    "content": str,
                    "embedding": np.ndarray,
                    "metadata": Dict[str, Any]
                }
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            filters: Optional metadata filters

        Returns:
            List of documents with similarity scores
        """
        pass

    @abstractmethod
    async def delete_by_id(self, document_id: str) -> bool:
        """Delete a document by ID."""
        pass

    @abstractmethod
    async def delete_by_metadata(self, metadata_filter: Dict[str, Any]) -> int:
        """Delete documents matching metadata filter.

        Returns:
            Number of documents deleted
        """
        pass

    @abstractmethod
    async def update_document(
        self,
        document_id: str,
        content: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update a document."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Get total number of documents in the database."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all documents from the database."""
        pass

    @abstractmethod
    async def get_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the database connection."""
        pass

    # Helper methods that can be overridden by implementations

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize an embedding vector."""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def _compute_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)

    def _filter_by_threshold(
        self, results: List[Dict[str, Any]], threshold: float
    ) -> List[Dict[str, Any]]:
        """Filter results by similarity threshold."""
        return [
            result for result in results
            if result.get("similarity", 0.0) >= threshold
        ]

    def _apply_metadata_filters(
        self, documents: List[Dict[str, Any]], filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply metadata filters to documents."""
        if not filters:
            return documents

        filtered_docs = []
        for doc in documents:
            metadata = doc.get("metadata", {})

            # Check if all filter conditions are met
            match = True
            for key, value in filters.items():
                if key not in metadata or metadata[key] != value:
                    match = False
                    break

            if match:
                filtered_docs.append(doc)

        return filtered_docs

    async def health_check(self) -> bool:
        """Check if the database is healthy and responsive."""
        try:
            # Simple health check - try to get count
            await self.count()
            return True
        except Exception:
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            total_docs = await self.count()
            return {
                "total_documents": total_docs,
                "status": "healthy" if await self.health_check() else "unhealthy"
            }
        except Exception as e:
            return {
                "total_documents": 0,
                "status": "error",
                "error": str(e)
            }