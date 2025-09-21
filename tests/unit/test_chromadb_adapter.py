"""Tests for ChromaDB adapter."""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

from easyprompt.vectordb.chromadb_adapter import ChromaDBAdapter


class TestChromaDBAdapter:
    """Test ChromaDBAdapter class."""

    @pytest.fixture
    def adapter(self, temp_dir):
        """Create a ChromaDBAdapter instance."""
        return ChromaDBAdapter(
            db_path=str(temp_dir / "test_chroma.db"),
            collection_name="test_collection"
        )

    @pytest.mark.asyncio
    async def test_initialize_new_collection(self, adapter):
        """Test initializing with a new collection."""
        mock_client = Mock()
        mock_collection = Mock()

        with patch('easyprompt.vectordb.chromadb_adapter.chromadb.Client') as mock_chroma:
            mock_chroma.return_value = mock_client
            mock_client.get_collection.side_effect = ValueError("Collection not found")
            mock_client.create_collection.return_value = mock_collection

            await adapter.initialize()

            assert adapter.client == mock_client
            assert adapter.collection == mock_collection
            mock_client.create_collection.assert_called_once_with(
                name="test_collection",
                metadata={"description": "EasyPrompt document embeddings"}
            )

    @pytest.mark.asyncio
    async def test_initialize_existing_collection(self, adapter):
        """Test initializing with an existing collection."""
        mock_client = Mock()
        mock_collection = Mock()

        with patch('easyprompt.vectordb.chromadb_adapter.chromadb.Client') as mock_chroma:
            mock_chroma.return_value = mock_client
            mock_client.get_collection.return_value = mock_collection

            await adapter.initialize()

            assert adapter.client == mock_client
            assert adapter.collection == mock_collection
            mock_client.get_collection.assert_called_once_with("test_collection")

    @pytest.mark.asyncio
    async def test_add_documents(self, adapter, sample_embeddings):
        """Test adding documents to ChromaDB."""
        documents = [
            {
                "id": "doc1",
                "content": "test content 1",
                "embedding": sample_embeddings[0],
                "metadata": {"source": "test1.md"}
            },
            {
                "id": "doc2",
                "content": "test content 2",
                "embedding": sample_embeddings[1],
                "metadata": {"source": "test2.md"}
            }
        ]

        mock_collection = Mock()
        adapter.collection = mock_collection

        await adapter.add_documents(documents)

        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]

        assert len(call_args["ids"]) == 2
        assert call_args["ids"] == ["doc1", "doc2"]
        assert call_args["documents"] == ["test content 1", "test content 2"]
        assert len(call_args["embeddings"]) == 2
        assert len(call_args["metadatas"]) == 2

    @pytest.mark.asyncio
    async def test_search(self, adapter, sample_embeddings):
        """Test searching in ChromaDB."""
        query_embedding = sample_embeddings[0]

        mock_collection = Mock()
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["content 1", "content 2"]],
            "metadatas": [[{"source": "test1.md"}, {"source": "test2.md"}]],
            "distances": [[0.1, 0.3]]
        }
        adapter.collection = mock_collection

        results = await adapter.search(query_embedding, top_k=2, threshold=0.5)

        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[0]["content"] == "content 1"
        assert results[0]["similarity"] > results[1]["similarity"]  # Closer distance = higher similarity

    @pytest.mark.asyncio
    async def test_search_with_threshold_filtering(self, adapter, sample_embeddings):
        """Test search with similarity threshold filtering."""
        query_embedding = sample_embeddings[0]

        mock_collection = Mock()
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["content 1", "content 2"]],
            "metadatas": [[{"source": "test1.md"}, {"source": "test2.md"}]],
            "distances": [[0.1, 2.0]]  # Second result has high distance (low similarity)
        }
        adapter.collection = mock_collection

        results = await adapter.search(query_embedding, top_k=2, threshold=0.8)

        # Should filter out the second result with low similarity
        assert len(results) == 1
        assert results[0]["id"] == "doc1"

    @pytest.mark.asyncio
    async def test_delete_by_id(self, adapter):
        """Test deleting a document by ID."""
        mock_collection = Mock()
        adapter.collection = mock_collection

        result = await adapter.delete_by_id("doc1")

        assert result is True
        mock_collection.delete.assert_called_once_with(["doc1"])

    @pytest.mark.asyncio
    async def test_delete_by_metadata(self, adapter):
        """Test deleting documents by metadata filter."""
        mock_collection = Mock()
        mock_collection.get.return_value = {"ids": ["doc1", "doc2"]}
        adapter.collection = mock_collection

        deleted_count = await adapter.delete_by_metadata({"source": "test.md"})

        assert deleted_count == 2
        mock_collection.get.assert_called_once_with(None, {"source": "test.md"})
        mock_collection.delete.assert_called_once_with(["doc1", "doc2"])

    @pytest.mark.asyncio
    async def test_update_document(self, adapter, sample_embeddings):
        """Test updating a document."""
        # Mock getting existing document
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "ids": ["doc1"],
            "documents": ["old content"],
            "metadatas": [{"old_meta": "value"}]
        }
        adapter.collection = mock_collection

        # Mock delete and add operations
        with patch.object(adapter, 'delete_by_id', return_value=True) as mock_delete, \
             patch.object(adapter, 'add_documents') as mock_add:

            result = await adapter.update_document(
                "doc1",
                content="new content",
                embedding=sample_embeddings[0],
                metadata={"new_meta": "value"}
            )

            assert result is True
            mock_delete.assert_called_once_with("doc1")
            mock_add.assert_called_once()

    @pytest.mark.asyncio
    async def test_count(self, adapter):
        """Test getting document count."""
        mock_collection = Mock()
        mock_collection.count.return_value = 42
        adapter.collection = mock_collection

        count = await adapter.count()

        assert count == 42
        mock_collection.count.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear(self, adapter):
        """Test clearing the collection."""
        mock_client = Mock()
        mock_new_collection = Mock()
        mock_client.delete_collection.return_value = None
        mock_client.create_collection.return_value = mock_new_collection

        adapter.client = mock_client
        adapter.collection_name = "test_collection"

        await adapter.clear()

        mock_client.delete_collection.assert_called_once_with("test_collection")
        mock_client.create_collection.assert_called_once_with("test_collection")
        assert adapter.collection == mock_new_collection

    @pytest.mark.asyncio
    async def test_get_by_id(self, adapter):
        """Test getting a document by ID."""
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "ids": ["doc1"],
            "documents": ["test content"],
            "metadatas": [{"source": "test.md"}]
        }
        adapter.collection = mock_collection

        document = await adapter.get_by_id("doc1")

        assert document is not None
        assert document["id"] == "doc1"
        assert document["content"] == "test content"
        assert document["metadata"]["source"] == "test.md"

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, adapter):
        """Test getting a document that doesn't exist."""
        mock_collection = Mock()
        mock_collection.get.return_value = {"ids": []}
        adapter.collection = mock_collection

        document = await adapter.get_by_id("nonexistent")

        assert document is None

    @pytest.mark.asyncio
    async def test_close(self, adapter):
        """Test closing the adapter."""
        adapter.client = Mock()

        await adapter.close()

        # ChromaDB doesn't require explicit closing, so this should not raise

    @pytest.mark.asyncio
    async def test_get_stats(self, adapter):
        """Test getting adapter statistics."""
        mock_collection = Mock()
        mock_collection.count.return_value = 100
        adapter.collection = mock_collection

        stats = await adapter.get_stats()

        assert stats["database_type"] == "chromadb"
        assert stats["total_documents"] == 100
        assert stats["status"] == "healthy"
        assert "db_path" in stats
        assert "collection_name" in stats

    @pytest.mark.asyncio
    async def test_initialize_failure(self, adapter):
        """Test initialization failure handling."""
        with patch('easyprompt.vectordb.chromadb_adapter.chromadb.Client') as mock_chroma:
            mock_chroma.side_effect = Exception("ChromaDB initialization failed")

            with pytest.raises(Exception, match="ChromaDB initialization failed"):
                await adapter.initialize()

    @pytest.mark.asyncio
    async def test_search_empty_results(self, adapter, sample_embeddings):
        """Test search with no results."""
        query_embedding = sample_embeddings[0]

        mock_collection = Mock()
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        adapter.collection = mock_collection

        results = await adapter.search(query_embedding, top_k=5)

        assert len(results) == 0