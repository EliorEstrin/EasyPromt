"""Integration tests for query processor."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from easyprompt.query.query_processor import QueryProcessor, QueryResult
from easyprompt.config import Settings


class TestQueryProcessor:
    """Test QueryProcessor integration."""

    @pytest.fixture
    def query_processor(self, sample_settings):
        """Create a QueryProcessor instance."""
        return QueryProcessor(sample_settings)

    @pytest.mark.asyncio
    async def test_process_query_full_workflow(self, query_processor, mock_vector_db, mock_llm_provider):
        """Test the complete query processing workflow."""
        # Mock dependencies
        with patch('easyprompt.query.context_retriever.get_vector_db', return_value=mock_vector_db), \
             patch('easyprompt.indexer.embedding_generator.EmbeddingGenerator') as mock_emb_gen, \
             patch('easyprompt.llm.provider_factory.ProviderFactory') as mock_factory:

            # Setup mocks
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.initialize = AsyncMock()
            mock_embedding_gen.generate_query_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
            mock_emb_gen.return_value = mock_embedding_gen

            mock_provider_factory = AsyncMock()
            mock_provider_factory.get_provider = AsyncMock(return_value=mock_llm_provider)
            mock_factory.return_value = mock_provider_factory

            # Initialize and process query
            await query_processor.initialize()
            result = await query_processor.process_query(
                "list all files",
                include_explanation=True
            )

            # Verify result
            assert isinstance(result, QueryResult)
            assert result.success is True
            assert result.command == "test-cli list"
            assert result.explanation == "This command lists all items"
            assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_process_query_with_alternatives(self, query_processor, mock_vector_db, mock_llm_provider):
        """Test processing query with alternatives."""
        with patch('easyprompt.query.context_retriever.get_vector_db', return_value=mock_vector_db), \
             patch('easyprompt.indexer.embedding_generator.EmbeddingGenerator') as mock_emb_gen, \
             patch('easyprompt.llm.provider_factory.ProviderFactory') as mock_factory:

            # Setup mocks
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.initialize = AsyncMock()
            mock_embedding_gen.generate_query_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
            mock_emb_gen.return_value = mock_embedding_gen

            mock_provider_factory = AsyncMock()
            mock_provider_factory.get_provider = AsyncMock(return_value=mock_llm_provider)
            mock_factory.return_value = mock_provider_factory

            # Mock generate_alternative_commands
            mock_llm_provider.generate_alternative_commands = AsyncMock(return_value=[
                ("test-cli list --all", "List all files including hidden"),
                ("test-cli ls", "Short form of list command")
            ])

            await query_processor.initialize()
            result = await query_processor.process_query_with_alternatives(
                "list files",
                num_alternatives=2
            )

            assert result["success"] is True
            assert "alternatives" in result
            assert len(result["alternatives"]) == 2

    @pytest.mark.asyncio
    async def test_refine_query_result(self, query_processor, mock_vector_db, mock_llm_provider):
        """Test refining a query result."""
        with patch('easyprompt.query.context_retriever.get_vector_db', return_value=mock_vector_db), \
             patch('easyprompt.indexer.embedding_generator.EmbeddingGenerator') as mock_emb_gen, \
             patch('easyprompt.llm.provider_factory.ProviderFactory') as mock_factory:

            # Setup mocks
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.initialize = AsyncMock()
            mock_embedding_gen.generate_query_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
            mock_emb_gen.return_value = mock_embedding_gen

            mock_provider_factory = AsyncMock()
            mock_provider_factory.get_provider = AsyncMock(return_value=mock_llm_provider)
            mock_factory.return_value = mock_provider_factory

            # Mock command refinement
            mock_llm_provider.refine_command = AsyncMock(return_value=(
                "test-cli list --verbose",
                "Added verbose flag for detailed output"
            ))

            # Create original result
            original_result = QueryResult(
                query="list files",
                command="test-cli list",
                explanation="Lists files",
                success=True
            )

            await query_processor.initialize()
            refined_result = await query_processor.refine_query_result(
                original_result,
                "make it more verbose"
            )

            assert refined_result.success is True
            assert refined_result.command == "test-cli list --verbose"
            assert refined_result.metadata["is_refinement"] is True

    @pytest.mark.asyncio
    async def test_find_command_examples(self, query_processor, mock_vector_db):
        """Test finding command examples."""
        with patch('easyprompt.query.context_retriever.get_vector_db', return_value=mock_vector_db), \
             patch('easyprompt.indexer.embedding_generator.EmbeddingGenerator') as mock_emb_gen:

            # Setup mocks
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.initialize = AsyncMock()
            mock_embedding_gen.generate_query_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
            mock_emb_gen.return_value = mock_embedding_gen

            # Mock command examples search
            mock_vector_db.search.return_value = [
                {
                    "id": "example1",
                    "content": "Use `test-cli list` to list files",
                    "metadata": {"has_commands": "true"},
                    "similarity": 0.9
                }
            ]

            await query_processor.initialize()
            examples = await query_processor.find_command_examples("list files")

            assert len(examples) >= 1
            assert "test-cli list" in examples[0]["content"]

    @pytest.mark.asyncio
    async def test_search_documentation(self, query_processor, mock_vector_db):
        """Test searching documentation."""
        with patch('easyprompt.query.context_retriever.get_vector_db', return_value=mock_vector_db), \
             patch('easyprompt.indexer.embedding_generator.EmbeddingGenerator') as mock_emb_gen:

            # Setup mocks
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.initialize = AsyncMock()
            mock_embedding_gen.generate_query_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
            mock_emb_gen.return_value = mock_embedding_gen

            await query_processor.initialize()
            results = await query_processor.search_documentation("file operations", top_k=5)

            assert len(results) >= 1
            mock_vector_db.search.assert_called()

    @pytest.mark.asyncio
    async def test_get_system_status(self, query_processor, mock_vector_db, mock_llm_provider):
        """Test getting system status."""
        with patch('easyprompt.query.context_retriever.get_vector_db', return_value=mock_vector_db), \
             patch('easyprompt.indexer.embedding_generator.EmbeddingGenerator') as mock_emb_gen, \
             patch('easyprompt.llm.provider_factory.ProviderFactory') as mock_factory:

            # Setup mocks
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.initialize = AsyncMock()
            mock_emb_gen.return_value = mock_embedding_gen

            mock_provider_factory = AsyncMock()
            mock_provider_factory.get_provider = AsyncMock(return_value=mock_llm_provider)
            mock_factory.return_value = mock_provider_factory

            await query_processor.initialize()
            status = await query_processor.get_system_status()

            assert status["initialized"] is True
            assert "cli_tool" in status
            assert "vector_db_status" in status
            assert "llm_provider" in status

    @pytest.mark.asyncio
    async def test_validate_command(self, query_processor):
        """Test command validation."""
        # Mock command generator
        mock_command_gen = Mock()
        mock_command_gen._validate_command = Mock(return_value=True)
        mock_command_gen._check_command_safety = Mock(return_value=True)
        mock_command_gen._classify_command = Mock(return_value="read")

        query_processor.command_generator = mock_command_gen

        validation = await query_processor.validate_command("ls -la")

        assert validation["is_valid"] is True
        assert validation["is_safe"] is True
        assert validation["command_type"] == "read"
        assert "recommendations" in validation

    @pytest.mark.asyncio
    async def test_process_query_error_handling(self, query_processor):
        """Test error handling in query processing."""
        # Mock context retriever to raise an exception
        mock_context_retriever = AsyncMock()
        mock_context_retriever.initialize = AsyncMock()
        mock_context_retriever.retrieve_context = AsyncMock(side_effect=Exception("Database error"))

        query_processor.context_retriever = mock_context_retriever

        result = await query_processor.process_query("test query")

        assert result.success is False
        assert "Database error" in result.error
        assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_close(self, query_processor):
        """Test closing the query processor."""
        # Mock components
        mock_context_retriever = AsyncMock()
        mock_command_generator = AsyncMock()

        query_processor.context_retriever = mock_context_retriever
        query_processor.command_generator = mock_command_generator

        await query_processor.close()

        mock_context_retriever.close.assert_called_once()
        mock_command_generator.close.assert_called_once()

    def test_get_command_recommendations(self, query_processor):
        """Test getting command recommendations."""
        # Test safe command
        recommendations = query_processor._get_command_recommendations(
            "ls -la", is_safe=True, command_type="read"
        )

        # Should have some recommendations due to settings
        assert len(recommendations) > 0

        # Test dangerous command
        recommendations = query_processor._get_command_recommendations(
            "rm -rf /", is_safe=False, command_type="delete"
        )

        # Should have warnings
        dangerous_warnings = [r for r in recommendations if "dangerous" in r.lower() or "⚠️" in r]
        assert len(dangerous_warnings) > 0

    @pytest.mark.asyncio
    async def test_process_query_unclear_request(self, query_processor, mock_vector_db, mock_llm_provider):
        """Test processing query that results in unclear request."""
        with patch('easyprompt.query.context_retriever.get_vector_db', return_value=mock_vector_db), \
             patch('easyprompt.indexer.embedding_generator.EmbeddingGenerator') as mock_emb_gen, \
             patch('easyprompt.llm.provider_factory.ProviderFactory') as mock_factory:

            # Setup mocks
            mock_embedding_gen = AsyncMock()
            mock_embedding_gen.initialize = AsyncMock()
            mock_embedding_gen.generate_query_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
            mock_emb_gen.return_value = mock_embedding_gen

            # Mock LLM to return unclear request
            mock_llm_provider.generate_command_with_explanation = AsyncMock(return_value=(
                "UNCLEAR_REQUEST",
                "The request could not be understood"
            ))

            mock_provider_factory = AsyncMock()
            mock_provider_factory.get_provider = AsyncMock(return_value=mock_llm_provider)
            mock_factory.return_value = mock_provider_factory

            await query_processor.initialize()
            result = await query_processor.process_query("gibberish query xyz")

            assert result.success is True  # Still successful even if unclear
            assert result.command == "UNCLEAR_REQUEST"