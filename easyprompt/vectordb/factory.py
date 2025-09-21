"""Factory for creating vector database instances."""

import logging
from typing import Optional
from ..config import Settings
from .base_vectordb import BaseVectorDB
from .chromadb_adapter import ChromaDBAdapter
from .pinecone_adapter import PineconeAdapter
from .weaviate_adapter import WeaviateAdapter

logger = logging.getLogger(__name__)


def get_vector_db(settings: Settings) -> BaseVectorDB:
    """Create and return a vector database instance based on settings."""
    db_type = settings.vector_db_type.lower()

    if db_type == "chromadb":
        return ChromaDBAdapter(
            db_path=settings.vector_db_url,
            collection_name="easyprompt"
        )

    elif db_type == "pinecone":
        if not settings.pinecone_api_key:
            raise ValueError("Pinecone API key is required for Pinecone database")
        if not settings.pinecone_environment:
            raise ValueError("Pinecone environment is required for Pinecone database")

        return PineconeAdapter(
            api_key=settings.pinecone_api_key,
            environment=settings.pinecone_environment,
            index_name=settings.pinecone_index_name,
            dimension=settings.embedding_dimension
        )

    elif db_type == "weaviate":
        return WeaviateAdapter(
            url=settings.weaviate_url,
            api_key=settings.weaviate_api_key,
            class_name="EasyPromptDocument"
        )

    else:
        raise ValueError(
            f"Unsupported vector database type: {db_type}. "
            f"Supported types: chromadb, pinecone, weaviate"
        )


async def test_vector_db_connection(settings: Settings) -> bool:
    """Test the vector database connection."""
    try:
        db = get_vector_db(settings)
        await db.initialize()
        health_check = await db.health_check()
        await db.close()
        return health_check
    except Exception as e:
        logger.error(f"Vector database connection test failed: {e}")
        return False


async def get_vector_db_info(settings: Settings) -> dict:
    """Get information about the configured vector database."""
    try:
        db = get_vector_db(settings)
        await db.initialize()
        stats = await db.get_stats()
        await db.close()
        return stats
    except Exception as e:
        logger.error(f"Failed to get vector database info: {e}")
        return {
            "database_type": settings.vector_db_type,
            "status": "error",
            "error": str(e)
        }