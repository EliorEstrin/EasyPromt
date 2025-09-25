"""Minimal E2E tests that don't require real API keys."""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from easyprompt.config import Settings
from easyprompt.indexer import DocumentIndexer
from easyprompt.query import QueryProcessor


class TestMinimalE2EFlow:
    """Test minimal end-to-end flow without real LLM calls."""

    @pytest.fixture
    def temp_docs(self):
        """Create temporary docs for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create sample docs
            (temp_path / "guide.md").write_text("""
# User Guide

## Installation

Install with:
```bash
pip install myapp
```

## Quick Start

Run the app:
```bash
myapp run --debug
```
""")

            yield temp_path

    @pytest.fixture
    def minimal_settings(self, temp_docs):
        """Create minimal settings for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            return Settings(
                vector_db_type="chromadb",
                vector_db_url=str(Path(temp_dir) / "test.db"),
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                embedding_dimension=384,
                docs_path=str(temp_docs),
                openai_api_key="test-key",  # Mock key for testing
                max_context_length=500,
                top_k_results=2,
            )

    @pytest.mark.asyncio
    async def test_indexing_flow_success(self, minimal_settings):
        """Test that indexing completes successfully."""
        indexer = DocumentIndexer(minimal_settings)
        await indexer.initialize()

        # Index documents
        stats = await indexer.index_documentation(force_rebuild=True)

        # Verify indexing worked
        assert stats["documents"] >= 1
        assert stats["chunks"] >= 1

        # Verify we can get stats
        index_stats = await indexer.get_index_stats()
        assert index_stats["total_chunks"] >= 1

        await indexer.cleanup()

    @pytest.mark.asyncio
    async def test_search_flow_success(self, minimal_settings):
        """Test that search works after indexing."""
        # First index
        indexer = DocumentIndexer(minimal_settings)
        await indexer.initialize()
        await indexer.index_documentation(force_rebuild=True)
        await indexer.cleanup()

        # Then search
        processor = QueryProcessor(minimal_settings)
        await processor.initialize()

        # Search for content
        results = await processor.search_documentation("installation", top_k=2)

        assert len(results) > 0
        assert any("install" in result["content"].lower() for result in results)

        await processor.close()

    @pytest.mark.asyncio
    async def test_system_status_success(self, minimal_settings):
        """Test system status check works."""
        processor = QueryProcessor(minimal_settings)
        await processor.initialize()

        status = await processor.get_system_status()

        assert status["initialized"] is True
        assert "embedding_model" in status
        assert "vector_db_type" in status

        await processor.close()

    @pytest.mark.asyncio
    async def test_command_validation_success(self, minimal_settings):
        """Test command validation works."""
        processor = QueryProcessor(minimal_settings)
        await processor.initialize()

        # Test basic validation
        result = await processor.validate_command("ls -la")
        assert "is_valid" in result
        assert "is_safe" in result
        assert "command_type" in result

        await processor.close()

    @pytest.mark.asyncio
    async def test_qa_flow_with_mock_llm(self, minimal_settings):
        """Test Q&A flow with mocked LLM response."""
        # First index
        indexer = DocumentIndexer(minimal_settings)
        await indexer.initialize()
        await indexer.index_documentation(force_rebuild=True)
        await indexer.cleanup()

        # Mock the LLM response
        with patch('easyprompt.llm.openai_provider.OpenAIProvider.chat_completion') as mock_llm:
            mock_llm.return_value = AsyncMock()
            mock_llm.return_value.content = "To install the application, use: pip install myapp"

            processor = QueryProcessor(minimal_settings)
            await processor.initialize()

            # Test Q&A
            result = await processor.process_qa_query("How do I install the app?")

            assert result["success"] is True
            assert "pip install" in result["answer"]
            assert len(result["context_used"]) > 0

            await processor.close()

    @pytest.mark.asyncio
    async def test_command_generation_with_mock_llm(self, minimal_settings):
        """Test command generation with mocked LLM."""
        # First index
        indexer = DocumentIndexer(minimal_settings)
        await indexer.initialize()
        await indexer.index_documentation(force_rebuild=True)
        await indexer.cleanup()

        # Mock the LLM response
        with patch('easyprompt.llm.openai_provider.OpenAIProvider.generate_command') as mock_llm:
            from easyprompt.llm.base_provider import LLMResponse
            mock_llm.return_value = LLMResponse(
                content="myapp run --debug",
                model="gpt-3.5-turbo"
            )

            processor = QueryProcessor(minimal_settings)
            await processor.initialize()

            # Test command generation
            result = await processor.process_query("run the app in debug mode")

            assert result.success is True
            assert "myapp run" in result.command
            assert "debug" in result.command

            await processor.close()

    @pytest.mark.asyncio
    async def test_end_to_end_cleanup(self, minimal_settings):
        """Test that cleanup works properly."""
        # Test full cycle
        indexer = DocumentIndexer(minimal_settings)
        await indexer.initialize()
        await indexer.index_documentation(force_rebuild=True)

        processor = QueryProcessor(minimal_settings)
        await processor.initialize()

        # Verify everything is working
        status = await processor.get_system_status()
        assert status["initialized"] is True

        # Cleanup
        await processor.close()
        await indexer.cleanup()

        # Test passes if no exceptions