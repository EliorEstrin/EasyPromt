"""Pinecone vector database adapter."""

import logging
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from .base_vectordb import BaseVectorDB

# Optional import for Pinecone
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    pinecone = None
    PINECONE_AVAILABLE = False

logger = logging.getLogger(__name__)


class PineconeAdapter(BaseVectorDB):
    """Pinecone vector database adapter."""

    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str = "easyprompt-index",
        dimension: int = 384,
        **kwargs
    ):
        if not PINECONE_AVAILABLE:
            raise ImportError(
                "pinecone-client is required for Pinecone adapter. "
                "Install it with: pip install pinecone-client"
            )
        super().__init__(**kwargs)
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.index = None

    async def initialize(self) -> None:
        """Initialize the Pinecone index."""
        try:
            # Initialize Pinecone
            pinecone.init(api_key=self.api_key, environment=self.environment)

            # Check if index exists
            existing_indexes = pinecone.list_indexes()

            if self.index_name not in existing_indexes:
                # Create index
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    pinecone.create_index,
                    self.index_name,
                    self.dimension,
                    "cosine"  # similarity metric
                )
                logger.info(f"Created new Pinecone index: {self.index_name}")
            else:
                logger.info(f"Connected to existing Pinecone index: {self.index_name}")

            # Connect to index
            self.index = pinecone.Index(self.index_name)

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise

    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to Pinecone."""
        if not self.index:
            await self.initialize()

        try:
            # Prepare vectors for Pinecone
            vectors = []
            for doc in documents:
                vector = {
                    "id": doc["id"],
                    "values": doc["embedding"].tolist(),
                    "metadata": {
                        **doc["metadata"],
                        "content": doc["content"]  # Store content in metadata
                    }
                }
                vectors.append(vector)

            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.index.upsert,
                    batch
                )

            logger.debug(f"Added {len(documents)} documents to Pinecone")

        except Exception as e:
            logger.error(f"Failed to add documents to Pinecone: {e}")
            raise

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents in Pinecone."""
        if not self.index:
            await self.initialize()

        try:
            # Query Pinecone
            query_response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.index.query,
                query_embedding.tolist(),
                top_k,
                True,  # include_metadata
                filters  # filter
            )

            # Process results
            processed_results = []
            for match in query_response.matches:
                similarity = match.score

                if similarity >= threshold:
                    metadata = match.metadata or {}
                    content = metadata.pop("content", "")

                    processed_results.append({
                        "id": match.id,
                        "content": content,
                        "metadata": metadata,
                        "similarity": similarity
                    })

            return processed_results

        except Exception as e:
            logger.error(f"Failed to search Pinecone: {e}")
            raise

    async def delete_by_id(self, document_id: str) -> bool:
        """Delete a document by ID."""
        if not self.index:
            await self.initialize()

        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.index.delete,
                [document_id]
            )
            logger.debug(f"Deleted document {document_id} from Pinecone")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    async def delete_by_metadata(self, metadata_filter: Dict[str, Any]) -> int:
        """Delete documents matching metadata filter."""
        if not self.index:
            await self.initialize()

        try:
            # Pinecone supports delete by filter
            delete_response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.index.delete,
                None,  # ids
                metadata_filter  # filter
            )

            # Pinecone doesn't return count of deleted documents
            # We'll return 0 as we can't determine the actual count
            logger.debug("Deleted documents matching filter from Pinecone")
            return 0  # Pinecone doesn't provide deleted count

        except Exception as e:
            logger.error(f"Failed to delete documents by metadata: {e}")
            return 0

    async def update_document(
        self,
        document_id: str,
        content: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update a document in Pinecone."""
        if not self.index:
            await self.initialize()

        try:
            # Get existing document
            existing_doc = await self.get_by_id(document_id)
            if not existing_doc:
                return False

            # Prepare updated vector
            updated_metadata = {**existing_doc.get("metadata", {}), **(metadata or {})}
            if content is not None:
                updated_metadata["content"] = content

            vector = {
                "id": document_id,
                "values": embedding.tolist() if embedding is not None else None,
                "metadata": updated_metadata
            }

            # Upsert (update or insert)
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.index.upsert,
                [vector]
            )

            logger.debug(f"Updated document {document_id} in Pinecone")
            return True

        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            return False

    async def count(self) -> int:
        """Get total number of documents."""
        if not self.index:
            await self.initialize()

        try:
            # Pinecone doesn't have a direct count method
            # We'll use describe_index_stats for approximate count
            stats = await asyncio.get_event_loop().run_in_executor(
                None,
                self.index.describe_index_stats
            )
            return stats.total_vector_count

        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0

    async def clear(self) -> None:
        """Clear all documents from the index."""
        if not self.index:
            await self.initialize()

        try:
            # Delete all vectors in the index
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.index.delete,
                delete_all=True
            )
            logger.info("Cleared Pinecone index")

        except Exception as e:
            logger.error(f"Failed to clear Pinecone index: {e}")
            raise

    async def get_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        if not self.index:
            await self.initialize()

        try:
            # Pinecone doesn't have a direct get by ID method
            # We'll use fetch method
            fetch_response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.index.fetch,
                [document_id]
            )

            if document_id in fetch_response.vectors:
                vector = fetch_response.vectors[document_id]
                metadata = vector.metadata or {}
                content = metadata.pop("content", "")

                return {
                    "id": document_id,
                    "content": content,
                    "metadata": metadata
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None

    async def close(self) -> None:
        """Close the Pinecone connection."""
        try:
            # Pinecone doesn't require explicit closing
            logger.info("Pinecone connection closed")
        except Exception as e:
            logger.warning(f"Error closing Pinecone connection: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone statistics."""
        try:
            if not self.index:
                await self.initialize()

            stats = await asyncio.get_event_loop().run_in_executor(
                None,
                self.index.describe_index_stats
            )

            return {
                "database_type": "pinecone",
                "index_name": self.index_name,
                "environment": self.environment,
                "dimension": self.dimension,
                "total_documents": stats.total_vector_count,
                "status": "healthy"
            }

        except Exception as e:
            return {
                "database_type": "pinecone",
                "status": "error",
                "error": str(e)
            }