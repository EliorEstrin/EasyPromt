"""Integration tests for document indexer."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from easyprompt.indexer.indexer import DocumentIndexer


class TestDocumentIndexerIntegration:
    """Test DocumentIndexer integration."""

    @pytest.fixture
    def indexer(self, sample_settings):
        """Create a DocumentIndexer instance."""
        return DocumentIndexer(sample_settings)

    @pytest.mark.asyncio
    async def test_index_documentation_full_workflow(self, indexer, sample_markdown_files, mock_vector_db):
        """Test the complete documentation indexing workflow."""
        with patch('easyprompt.vectordb.factory.get_vector_db', return_value=mock_vector_db), \
             patch('easyprompt.indexer.embedding_generator.EmbeddingGenerator') as mock_emb_gen:

            # Setup embedding generator mock
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.initialize = AsyncMock()
            mock_embedding_gen.generate_embeddings = AsyncMock(return_value=[
                [0.1, 0.2, 0.3, 0.4],  # Mock embeddings
                [0.5, 0.6, 0.7, 0.8]
            ])
            mock_emb_gen.return_value = mock_embedding_gen

            # Set paths to our sample files
            indexer.settings.readme_path = str(sample_markdown_files["readme"])
            indexer.settings.docs_path = str(sample_markdown_files["docs_dir"])

            await indexer.initialize()

            # Index documentation
            stats = await indexer.index_documentation()

            # Verify results
            assert stats["documents"] >= 2  # README + API doc
            assert stats["chunks"] > 0

            # Verify vector database interactions
            mock_vector_db.initialize.assert_called_once()
            mock_vector_db.add_documents.assert_called()

    @pytest.mark.asyncio
    async def test_index_documentation_force_rebuild(self, indexer, sample_markdown_files, mock_vector_db):
        """Test force rebuilding the index."""
        with patch('easyprompt.vectordb.factory.get_vector_db', return_value=mock_vector_db), \
             patch('easyprompt.indexer.embedding_generator.EmbeddingGenerator') as mock_emb_gen:

            # Setup mocks
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.initialize = AsyncMock()
            mock_embedding_gen.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
            mock_emb_gen.return_value = mock_embedding_gen

            indexer.settings.readme_path = str(sample_markdown_files["readme"])

            await indexer.initialize()

            # Index with force rebuild
            await indexer.index_documentation(force_rebuild=True)

            # Should clear existing index
            mock_vector_db.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_index_specific_paths(self, indexer, sample_markdown_files, mock_vector_db):
        """Test indexing specific paths."""
        with patch('easyprompt.vectordb.factory.get_vector_db', return_value=mock_vector_db), \
             patch('easyprompt.indexer.embedding_generator.EmbeddingGenerator') as mock_emb_gen:

            # Setup mocks
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.initialize = AsyncMock()
            mock_embedding_gen.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
            mock_emb_gen.return_value = mock_embedding_gen

            await indexer.initialize()

            # Index specific path
            specific_path = str(sample_markdown_files["readme"])
            stats = await indexer.index_documentation(paths=[specific_path])

            assert stats["documents"] >= 1
            assert stats["chunks"] > 0

    @pytest.mark.asyncio
    async def test_update_document(self, indexer, sample_markdown_files, mock_vector_db):
        """Test updating a single document."""
        with patch('easyprompt.vectordb.factory.get_vector_db', return_value=mock_vector_db), \
             patch('easyprompt.indexer.embedding_generator.EmbeddingGenerator') as mock_emb_gen:

            # Setup mocks
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.initialize = AsyncMock()
            mock_embedding_gen.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
            mock_emb_gen.return_value = mock_embedding_gen

            await indexer.initialize()

            # Update document
            file_path = str(sample_markdown_files["readme"])
            result = await indexer.update_document(file_path)

            assert result is True
            # Should delete existing chunks for the file
            mock_vector_db.delete_by_metadata.assert_called_with({"file_path": file_path})
            # Should add new chunks
            mock_vector_db.add_documents.assert_called()

    @pytest.mark.asyncio
    async def test_update_nonexistent_document(self, indexer, mock_vector_db):
        """Test updating a document that doesn't exist."""
        with patch('easyprompt.vectordb.factory.get_vector_db', return_value=mock_vector_db), \
             patch('easyprompt.indexer.embedding_generator.EmbeddingGenerator') as mock_emb_gen:

            # Setup mocks
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.initialize = AsyncMock()
            mock_emb_gen.return_value = mock_embedding_gen

            await indexer.initialize()

            # Try to update non-existent file
            result = await indexer.update_document("/nonexistent/file.md")

            assert result is False

    @pytest.mark.asyncio
    async def test_get_index_stats(self, indexer, mock_vector_db):
        """Test getting index statistics."""
        with patch('easyprompt.vectordb.factory.get_vector_db', return_value=mock_vector_db):
            mock_vector_db.count.return_value = 42

            await indexer.initialize()
            stats = await indexer.get_index_stats()

            assert stats["total_chunks"] == 42
            mock_vector_db.count.assert_called_once()

    @pytest.mark.asyncio
    async def test_index_no_documents(self, indexer, mock_vector_db):
        """Test indexing when no documents are found."""
        with patch('easyprompt.vectordb.factory.get_vector_db', return_value=mock_vector_db), \
             patch('easyprompt.indexer.embedding_generator.EmbeddingGenerator') as mock_emb_gen:

            # Setup mocks
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.initialize = AsyncMock()
            mock_emb_gen.return_value = mock_embedding_gen

            # Set invalid paths
            indexer.settings.readme_path = "/nonexistent/readme.md"
            indexer.settings.docs_path = "/nonexistent/docs"

            await indexer.initialize()
            stats = await indexer.index_documentation()

            assert stats["documents"] == 0
            assert stats["chunks"] == 0

    @pytest.mark.asyncio
    async def test_chunk_processing_and_merging(self, indexer, sample_markdown_files, mock_vector_db):
        """Test that chunks are properly processed and merged."""
        with patch('easyprompt.vectordb.factory.get_vector_db', return_value=mock_vector_db), \
             patch('easyprompt.indexer.embedding_generator.EmbeddingGenerator') as mock_emb_gen:

            # Setup mocks
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.initialize = AsyncMock()
            # Return more embeddings than we expect to ensure we handle the list properly
            mock_embedding_gen.generate_embeddings = AsyncMock(return_value=[
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2]
            ])
            mock_emb_gen.return_value = mock_embedding_gen

            indexer.settings.readme_path = str(sample_markdown_files["readme"])

            await indexer.initialize()
            stats = await indexer.index_documentation()

            # Verify that documents were added to vector DB
            assert mock_vector_db.add_documents.called
            call_args = mock_vector_db.add_documents.call_args[0][0]

            # Check that documents have required fields
            for doc in call_args:
                assert "id" in doc
                assert "content" in doc
                assert "embedding" in doc
                assert "metadata" in doc
                assert "file_path" in doc["metadata"]

    @pytest.mark.asyncio
    async def test_error_handling_during_indexing(self, indexer, sample_markdown_files, mock_vector_db):
        """Test error handling during the indexing process."""
        with patch('easyprompt.vectordb.factory.get_vector_db', return_value=mock_vector_db), \
             patch('easyprompt.indexer.embedding_generator.EmbeddingGenerator') as mock_emb_gen:

            # Setup mocks with an error
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.initialize = AsyncMock()
            mock_embedding_gen.generate_embeddings = AsyncMock(side_effect=Exception("Embedding error"))
            mock_emb_gen.return_value = mock_embedding_gen

            indexer.settings.readme_path = str(sample_markdown_files["readme"])

            await indexer.initialize()

            # Should handle the error gracefully
            with pytest.raises(Exception, match="Embedding error"):
                await indexer.index_documentation()

    @pytest.mark.asyncio
    async def test_cleanup(self, indexer, mock_vector_db):
        """Test cleanup of resources."""
        with patch('easyprompt.vectordb.factory.get_vector_db', return_value=mock_vector_db):
            await indexer.initialize()
            await indexer.cleanup()

            mock_vector_db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_default_paths(self, indexer):
        """Test getting default paths for indexing."""
        indexer.settings.readme_path = "/path/to/README.md"
        indexer.settings.docs_path = "/path/to/docs;extra1.md;extra2.md"

        paths = indexer._get_default_paths()

        assert "/path/to/README.md" in paths
        assert "/path/to/docs" in paths
        assert "extra1.md" in paths
        assert "extra2.md" in paths

    @pytest.mark.asyncio
    async def test_store_chunks_with_embeddings(self, indexer, sample_chunks, sample_embeddings, mock_vector_db):
        """Test storing chunks with embeddings in vector database."""
        with patch('easyprompt.vectordb.factory.get_vector_db', return_value=mock_vector_db):
            await indexer.initialize()

            # Convert TextChunk objects to the format expected by the method
            chunks = []
            for i, chunk_data in enumerate(sample_chunks):
                from easyprompt.indexer.text_chunker import TextChunk
                chunk = TextChunk(
                    content=chunk_data["content"],
                    start_index=chunk_data["start_index"],
                    end_index=chunk_data["end_index"],
                    section=chunk_data["section"],
                    file_path=chunk_data["file_path"],
                    chunk_id=chunk_data["chunk_id"],
                    metadata=chunk_data["metadata"]
                )
                chunks.append(chunk)

            await indexer._store_chunks_with_embeddings(chunks, sample_embeddings)

            # Verify that documents were added to vector DB
            mock_vector_db.add_documents.assert_called_once()
            call_args = mock_vector_db.add_documents.call_args[0][0]

            assert len(call_args) == len(chunks)
            for i, doc in enumerate(call_args):
                assert doc["id"] == chunks[i].chunk_id
                assert doc["content"] == chunks[i].content
                assert "metadata" in doc