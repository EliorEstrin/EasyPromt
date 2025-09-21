"""ChromaDB vector database adapter."""

import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from .base_vectordb import BaseVectorDB

# Optional import for ChromaDB
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None
    ChromaSettings = None
    CHROMADB_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChromaDBAdapter(BaseVectorDB):
    """ChromaDB vector database adapter."""

    def __init__(self, db_path: str = "./data/chroma.db", collection_name: str = "easyprompt", **kwargs):
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb is required for ChromaDB adapter. "
                "Install it with: pip install chromadb"
            )
        super().__init__(**kwargs)
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None

    async def initialize(self) -> None:
        """Initialize the ChromaDB client and collection."""
        try:
            # Ensure the directory exists
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)

            # Initialize ChromaDB client
            chroma_settings = ChromaSettings(
                persist_directory=str(db_dir),
                anonymized_telemetry=False
            )

            self.client = chromadb.Client(chroma_settings)

            # Get or create collection
            try:
                self.collection = self.client.get_collection(self.collection_name)
                logger.info(f"Connected to existing ChromaDB collection: {self.collection_name}")
            except ValueError:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "EasyPrompt document embeddings"}
                )
                logger.info(f"Created new ChromaDB collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to ChromaDB."""
        if not self.collection:
            await self.initialize()

        try:
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents_content = []

            for doc in documents:
                ids.append(doc["id"])
                embeddings.append(doc["embedding"].tolist())
                metadatas.append(doc["metadata"])
                documents_content.append(doc["content"])

            # Add to collection in a thread to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.collection.add,
                embeddings,
                metadatas,
                documents_content,
                ids
            )

            logger.debug(f"Added {len(documents)} documents to ChromaDB")

        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents in ChromaDB."""
        if not self.collection:
            await self.initialize()

        try:
            # Convert query embedding to list
            query_embedding_list = query_embedding.tolist()

            # Prepare where clause for filtering
            where_clause = filters if filters else None

            # Query the collection
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                self.collection.query,
                [query_embedding_list],
                top_k,
                where_clause
            )

            # Process results
            processed_results = []
            if results["ids"] and len(results["ids"]) > 0:
                ids = results["ids"][0]
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0] if "distances" in results else [0] * len(ids)

                for i, (doc_id, content, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    # Convert distance to similarity (ChromaDB returns L2 distance)
                    similarity = 1.0 / (1.0 + distance)

                    if similarity >= threshold:
                        processed_results.append({
                            "id": doc_id,
                            "content": content,
                            "metadata": metadata or {},
                            "similarity": similarity,
                            "distance": distance
                        })

            return processed_results

        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {e}")
            raise

    async def delete_by_id(self, document_id: str) -> bool:
        """Delete a document by ID."""
        if not self.collection:
            await self.initialize()

        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.collection.delete,
                [document_id]
            )
            logger.debug(f"Deleted document {document_id} from ChromaDB")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    async def delete_by_metadata(self, metadata_filter: Dict[str, Any]) -> int:
        """Delete documents matching metadata filter."""
        if not self.collection:
            await self.initialize()

        try:
            # First, find documents matching the filter
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                self.collection.get,
                None,  # ids
                metadata_filter  # where
            )

            if results["ids"]:
                # Delete the found documents
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.collection.delete,
                    results["ids"]
                )
                deleted_count = len(results["ids"])
                logger.debug(f"Deleted {deleted_count} documents matching filter")
                return deleted_count

            return 0

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
        """Update a document in ChromaDB."""
        if not self.collection:
            await self.initialize()

        try:
            # ChromaDB doesn't have direct update, so we delete and re-add
            existing_doc = await self.get_by_id(document_id)
            if not existing_doc:
                return False

            # Prepare updated document
            updated_doc = {
                "id": document_id,
                "content": content if content is not None else existing_doc["content"],
                "embedding": embedding if embedding is not None else existing_doc.get("embedding"),
                "metadata": {**existing_doc.get("metadata", {}), **(metadata or {})}
            }

            # Delete old document
            await self.delete_by_id(document_id)

            # Add updated document
            await self.add_documents([updated_doc])

            logger.debug(f"Updated document {document_id} in ChromaDB")
            return True

        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            return False

    async def count(self) -> int:
        """Get total number of documents."""
        if not self.collection:
            await self.initialize()

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.collection.count
            )
            return result

        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0

    async def clear(self) -> None:
        """Clear all documents from the collection."""
        if not self.collection:
            await self.initialize()

        try:
            # Delete the collection and recreate it
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.delete_collection,
                self.collection_name
            )

            self.collection = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.create_collection,
                self.collection_name
            )

            logger.info("Cleared ChromaDB collection")

        except Exception as e:
            logger.error(f"Failed to clear ChromaDB collection: {e}")
            raise

    async def get_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        if not self.collection:
            await self.initialize()

        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                self.collection.get,
                [document_id]
            )

            if results["ids"] and len(results["ids"]) > 0:
                return {
                    "id": results["ids"][0],
                    "content": results["documents"][0],
                    "metadata": results["metadatas"][0] or {}
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None

    async def close(self) -> None:
        """Close the ChromaDB connection."""
        try:
            if self.client:
                # ChromaDB doesn't require explicit closing
                logger.info("ChromaDB connection closed")
        except Exception as e:
            logger.warning(f"Error closing ChromaDB connection: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics."""
        try:
            stats = await super().get_stats()
            stats.update({
                "database_type": "chromadb",
                "db_path": self.db_path,
                "collection_name": self.collection_name
            })
            return stats
        except Exception as e:
            return {
                "database_type": "chromadb",
                "status": "error",
                "error": str(e)
            }