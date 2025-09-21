"""Tests for embedding generator."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from easyprompt.indexer.embedding_generator import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test EmbeddingGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create an EmbeddingGenerator instance."""
        return EmbeddingGenerator("sentence-transformers/all-MiniLM-L6-v2")

    @pytest.mark.asyncio
    async def test_initialize(self, generator):
        """Test initializing the embedding generator."""
        with patch('easyprompt.indexer.embedding_generator.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_st.return_value = mock_model

            await generator.initialize()

            assert generator.model == mock_model
            mock_st.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")

    @pytest.mark.asyncio
    async def test_generate_embedding(self, generator):
        """Test generating a single embedding."""
        with patch('easyprompt.indexer.embedding_generator.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_embedding = np.array([0.1, 0.2, 0.3])
            mock_model.encode.return_value = [mock_embedding]
            mock_st.return_value = mock_model

            await generator.initialize()
            embedding = await generator.generate_embedding("test text")

            assert isinstance(embedding, np.ndarray)
            np.testing.assert_array_equal(embedding, mock_embedding)

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self, generator):
        """Test generating multiple embeddings."""
        texts = ["text 1", "text 2", "text 3"]

        with patch('easyprompt.indexer.embedding_generator.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
            mock_model.encode.return_value = mock_embeddings
            mock_st.return_value = mock_model

            await generator.initialize()
            embeddings = await generator.generate_embeddings(texts)

            assert len(embeddings) == 3
            assert all(isinstance(emb, np.ndarray) for emb in embeddings)

    def test_preprocess_text(self, generator):
        """Test text preprocessing."""
        # Test whitespace normalization
        text_with_spaces = "  This   has    excessive   spaces  "
        processed = generator._preprocess_text(text_with_spaces)
        assert processed == "This has excessive spaces"

        # Test long text truncation
        long_text = " ".join(["word"] * 600)  # 600 words
        processed = generator._preprocess_text(long_text)
        words = processed.split()
        assert len(words) <= 500

    @pytest.mark.asyncio
    async def test_get_embedding_dimension(self, generator):
        """Test getting embedding dimension."""
        with patch('easyprompt.indexer.embedding_generator.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_embedding = np.array([0.1, 0.2, 0.3, 0.4])  # 4 dimensions
            mock_model.encode.return_value = [mock_embedding]
            mock_st.return_value = mock_model

            await generator.initialize()
            dimension = await generator.get_embedding_dimension()

            assert dimension == 4

    def test_compute_similarity(self, generator):
        """Test computing similarity between embeddings."""
        # Test identical vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        similarity = generator.compute_similarity(vec1, vec2)
        assert similarity == pytest.approx(1.0, abs=1e-6)

        # Test orthogonal vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = generator.compute_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0, abs=1e-6)

        # Test zero vectors
        vec1 = np.array([0.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        similarity = generator.compute_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_find_most_similar(self, generator):
        """Test finding most similar embeddings."""
        query_embedding = np.array([1.0, 0.0, 0.0])
        embeddings = [
            np.array([1.0, 0.0, 0.0]),  # identical
            np.array([0.5, 0.5, 0.0]),  # somewhat similar
            np.array([0.0, 1.0, 0.0]),  # orthogonal
            np.array([-1.0, 0.0, 0.0]) # opposite
        ]

        results = generator.find_most_similar(query_embedding, embeddings, top_k=3)

        assert len(results) == 3
        # Results should be sorted by similarity (descending)
        assert results[0][1] > results[1][1] > results[2][1]
        # First result should be the identical vector
        assert results[0][0] == 0  # index of identical vector

    @pytest.mark.asyncio
    async def test_generate_query_embedding(self, generator):
        """Test generating query embedding with preprocessing."""
        with patch('easyprompt.indexer.embedding_generator.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_embedding = np.array([0.1, 0.2, 0.3])
            mock_model.encode.return_value = [mock_embedding]
            mock_st.return_value = mock_model

            await generator.initialize()
            embedding = await generator.generate_query_embedding("How do I list files?")

            assert isinstance(embedding, np.ndarray)
            # Should have called preprocess_query
            mock_model.encode.assert_called_once()

    def test_preprocess_query(self, generator):
        """Test query preprocessing."""
        # Test stop word removal
        query = "How do I want to list files please help me"
        processed = generator._preprocess_query(query)

        # Should remove some stop words but keep meaningful words
        assert "list" in processed
        assert "files" in processed
        # Stop words should be reduced
        assert len(processed.split()) <= len(query.split())

        # Test query that becomes too short after filtering
        short_query = "how do you"
        processed = generator._preprocess_query(short_query)
        # Should return original if filtered too much
        assert processed == short_query

    @pytest.mark.asyncio
    async def test_batch_generate_with_metadata(self, generator):
        """Test batch generation with metadata."""
        texts_with_metadata = [
            {"content": "text 1", "metadata": {"source": "doc1"}},
            {"content": "text 2", "metadata": {"source": "doc2"}}
        ]

        with patch('easyprompt.indexer.embedding_generator.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
            mock_model.encode.return_value = mock_embeddings
            mock_st.return_value = mock_model

            await generator.initialize()
            results = await generator.batch_generate_with_metadata(texts_with_metadata)

            assert len(results) == 2
            assert "embedding" in results[0]
            assert "metadata" in results[0]
            assert results[0]["metadata"]["source"] == "doc1"

    @pytest.mark.asyncio
    async def test_initialize_failure(self, generator):
        """Test initialization failure handling."""
        with patch('easyprompt.indexer.embedding_generator.SentenceTransformer') as mock_st:
            mock_st.side_effect = Exception("Model loading failed")

            with pytest.raises(Exception, match="Model loading failed"):
                await generator.initialize()

    @pytest.mark.asyncio
    async def test_generate_embedding_without_init(self, generator):
        """Test generating embedding without initialization."""
        with patch('easyprompt.indexer.embedding_generator.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_embedding = np.array([0.1, 0.2, 0.3])
            mock_model.encode.return_value = [mock_embedding]
            mock_st.return_value = mock_model

            # Should auto-initialize
            embedding = await generator.generate_embedding("test")

            assert generator.model is not None
            assert isinstance(embedding, np.ndarray)