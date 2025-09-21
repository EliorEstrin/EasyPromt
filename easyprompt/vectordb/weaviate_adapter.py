"""Weaviate vector database adapter."""

import logging
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from .base_vectordb import BaseVectorDB

# Optional import for Weaviate
try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    weaviate = None
    WEAVIATE_AVAILABLE = False

logger = logging.getLogger(__name__)


class WeaviateAdapter(BaseVectorDB):
    """Weaviate vector database adapter."""

    def __init__(
        self,
        url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        class_name: str = "EasyPromptDocument",
        **kwargs
    ):
        if not WEAVIATE_AVAILABLE:
            raise ImportError(
                "weaviate-client is required for Weaviate adapter. "
                "Install it with: pip install weaviate-client"
            )
        super().__init__(**kwargs)
        self.url = url
        self.api_key = api_key
        self.class_name = class_name
        self.client = None

    async def initialize(self) -> None:
        """Initialize the Weaviate client and schema."""
        try:
            # Initialize Weaviate client
            auth_config = None
            if self.api_key:
                auth_config = weaviate.AuthApiKey(api_key=self.api_key)

            self.client = weaviate.Client(
                url=self.url,
                auth_client_secret=auth_config
            )

            # Check if class exists, if not create it
            schema = self.client.schema.get()
            class_exists = any(cls["class"] == self.class_name for cls in schema.get("classes", []))

            if not class_exists:
                await self._create_schema()
                logger.info(f"Created Weaviate class: {self.class_name}")
            else:
                logger.info(f"Connected to existing Weaviate class: {self.class_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Weaviate: {e}")
            raise

    async def _create_schema(self) -> None:
        """Create the schema for document storage."""
        schema = {
            "class": self.class_name,
            "description": "EasyPrompt document embeddings",
            "vectorizer": "none",  # We provide our own vectors
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Document content"
                },
                {
                    "name": "file_path",
                    "dataType": ["string"],
                    "description": "Source file path"
                },
                {
                    "name": "section",
                    "dataType": ["string"],
                    "description": "Document section"
                },
                {
                    "name": "start_index",
                    "dataType": ["int"],
                    "description": "Start index in original document"
                },
                {
                    "name": "end_index",
                    "dataType": ["int"],
                    "description": "End index in original document"
                },
                {
                    "name": "metadata_json",
                    "dataType": ["text"],
                    "description": "Additional metadata as JSON"
                }
            ]
        }

        await asyncio.get_event_loop().run_in_executor(
            None,
            self.client.schema.create_class,
            schema
        )

    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to Weaviate."""
        if not self.client:
            await self.initialize()

        try:
            # Add documents in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                await self._add_batch(batch)

            logger.debug(f"Added {len(documents)} documents to Weaviate")

        except Exception as e:
            logger.error(f"Failed to add documents to Weaviate: {e}")
            raise

    async def _add_batch(self, documents: List[Dict[str, Any]]) -> None:
        """Add a batch of documents to Weaviate."""
        with self.client.batch as batch:
            for doc in documents:
                metadata = doc["metadata"]

                # Prepare properties for Weaviate
                properties = {
                    "content": doc["content"],
                    "file_path": metadata.get("file_path", ""),
                    "section": metadata.get("section", ""),
                    "start_index": metadata.get("start_index", 0),
                    "end_index": metadata.get("end_index", 0),
                    "metadata_json": str(metadata)  # Store as string for complex metadata
                }

                batch.add_data_object(
                    data_object=properties,
                    class_name=self.class_name,
                    uuid=doc["id"],
                    vector=doc["embedding"].tolist()
                )

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents in Weaviate."""
        if not self.client:
            await self.initialize()

        try:
            # Build query
            query = (
                self.client.query
                .get(self.class_name, ["content", "file_path", "section", "start_index", "end_index", "metadata_json"])
                .with_near_vector({"vector": query_embedding.tolist()})
                .with_limit(top_k)
                .with_additional(["certainty", "distance", "id"])
            )

            # Add where filter if provided
            if filters:
                where_filter = self._build_where_filter(filters)
                if where_filter:
                    query = query.with_where(where_filter)

            # Execute query
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                query.do
            )

            # Process results
            processed_results = []
            if "data" in result and "Get" in result["data"] and self.class_name in result["data"]["Get"]:
                for item in result["data"]["Get"][self.class_name]:
                    certainty = item["_additional"]["certainty"]
                    distance = item["_additional"]["distance"]

                    # Convert certainty to similarity (Weaviate uses certainty 0-1)
                    similarity = certainty

                    if similarity >= threshold:
                        # Parse metadata
                        import json
                        try:
                            metadata = json.loads(item.get("metadata_json", "{}"))
                        except:
                            metadata = {}

                        processed_results.append({
                            "id": item["_additional"]["id"],
                            "content": item["content"],
                            "metadata": {
                                "file_path": item.get("file_path", ""),
                                "section": item.get("section", ""),
                                "start_index": item.get("start_index", 0),
                                "end_index": item.get("end_index", 0),
                                **metadata
                            },
                            "similarity": similarity,
                            "distance": distance
                        })

            return processed_results

        except Exception as e:
            logger.error(f"Failed to search Weaviate: {e}")
            raise

    def _build_where_filter(self, filters: Dict[str, Any]) -> Optional[Dict]:
        """Build Weaviate where filter from metadata filters."""
        if not filters:
            return None

        # Simple implementation for string filters
        conditions = []
        for key, value in filters.items():
            if key in ["file_path", "section"]:
                conditions.append({
                    "path": [key],
                    "operator": "Equal",
                    "valueString": str(value)
                })

        if conditions:
            if len(conditions) == 1:
                return conditions[0]
            else:
                return {
                    "operator": "And",
                    "operands": conditions
                }

        return None

    async def delete_by_id(self, document_id: str) -> bool:
        """Delete a document by ID."""
        if not self.client:
            await self.initialize()

        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.data_object.delete,
                document_id
            )
            logger.debug(f"Deleted document {document_id} from Weaviate")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    async def delete_by_metadata(self, metadata_filter: Dict[str, Any]) -> int:
        """Delete documents matching metadata filter."""
        if not self.client:
            await self.initialize()

        try:
            # First find matching documents
            where_filter = self._build_where_filter(metadata_filter)
            if not where_filter:
                return 0

            query = (
                self.client.query
                .get(self.class_name, [])
                .with_where(where_filter)
                .with_additional(["id"])
            )

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                query.do
            )

            # Delete found documents
            deleted_count = 0
            if "data" in result and "Get" in result["data"] and self.class_name in result["data"]["Get"]:
                for item in result["data"]["Get"][self.class_name]:
                    doc_id = item["_additional"]["id"]
                    if await self.delete_by_id(doc_id):
                        deleted_count += 1

            logger.debug(f"Deleted {deleted_count} documents matching filter")
            return deleted_count

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
        """Update a document in Weaviate."""
        if not self.client:
            await self.initialize()

        try:
            # Get existing document
            existing_doc = await self.get_by_id(document_id)
            if not existing_doc:
                return False

            # Prepare updated properties
            updated_metadata = {**existing_doc.get("metadata", {}), **(metadata or {})}
            properties = {
                "content": content if content is not None else existing_doc["content"],
                "file_path": updated_metadata.get("file_path", ""),
                "section": updated_metadata.get("section", ""),
                "start_index": updated_metadata.get("start_index", 0),
                "end_index": updated_metadata.get("end_index", 0),
                "metadata_json": str(updated_metadata)
            }

            # Update properties
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.data_object.update,
                properties,
                self.class_name,
                document_id
            )

            # Update vector if provided
            if embedding is not None:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.client.data_object.replace,
                    properties,
                    self.class_name,
                    document_id,
                    embedding.tolist()
                )

            logger.debug(f"Updated document {document_id} in Weaviate")
            return True

        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            return False

    async def count(self) -> int:
        """Get total number of documents."""
        if not self.client:
            await self.initialize()

        try:
            # Use aggregate query to get count
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.query.aggregate(self.class_name).with_meta_count().do
            )

            if "data" in result and "Aggregate" in result["data"] and self.class_name in result["data"]["Aggregate"]:
                return result["data"]["Aggregate"][self.class_name][0]["meta"]["count"]

            return 0

        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0

    async def clear(self) -> None:
        """Clear all documents from the class."""
        if not self.client:
            await self.initialize()

        try:
            # Delete the class and recreate it
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.schema.delete_class,
                self.class_name
            )

            await self._create_schema()
            logger.info("Cleared Weaviate class")

        except Exception as e:
            logger.error(f"Failed to clear Weaviate class: {e}")
            raise

    async def get_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        if not self.client:
            await self.initialize()

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.data_object.get_by_id,
                document_id
            )

            if result:
                properties = result.get("properties", {})

                # Parse metadata
                import json
                try:
                    metadata = json.loads(properties.get("metadata_json", "{}"))
                except:
                    metadata = {}

                return {
                    "id": document_id,
                    "content": properties.get("content", ""),
                    "metadata": {
                        "file_path": properties.get("file_path", ""),
                        "section": properties.get("section", ""),
                        "start_index": properties.get("start_index", 0),
                        "end_index": properties.get("end_index", 0),
                        **metadata
                    }
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None

    async def close(self) -> None:
        """Close the Weaviate connection."""
        try:
            # Weaviate client doesn't require explicit closing
            logger.info("Weaviate connection closed")
        except Exception as e:
            logger.warning(f"Error closing Weaviate connection: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get Weaviate statistics."""
        try:
            stats = await super().get_stats()
            stats.update({
                "database_type": "weaviate",
                "url": self.url,
                "class_name": self.class_name
            })
            return stats
        except Exception as e:
            return {
                "database_type": "weaviate",
                "status": "error",
                "error": str(e)
            }