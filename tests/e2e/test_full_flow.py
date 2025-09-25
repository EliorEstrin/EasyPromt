"""End-to-end tests for full EasyPrompt flow."""

import os
import pytest
import tempfile
from pathlib import Path

from easyprompt.config import Settings
from easyprompt.indexer import DocumentIndexer
from easyprompt.query import QueryProcessor


class TestE2EFlow:
    """Test complete end-to-end flow."""

    @pytest.fixture
    def temp_docs(self):
        """Create temporary docs for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create sample docs
            (temp_path / "getting_started.md").write_text("""
# Getting Started

This is a sample documentation for testing.

## Installation

To install the application:
```bash
pip install myapp
```

## Usage

Run the application with:
```bash
myapp start --port 8080
```

## Configuration

Set environment variables:
```bash
export APP_ENV=production
export APP_DEBUG=false
```
""")

            (temp_path / "api.md").write_text("""
# API Reference

## Authentication

Use API keys for authentication:
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" api.example.com
```

## Endpoints

- GET /status - Check application status
- POST /data - Submit data
""")

            yield temp_path

    @pytest.fixture
    def e2e_settings(self, temp_docs):
        """Create settings for E2E testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            return Settings(
                vector_db_type="chromadb",
                vector_db_url=str(Path(temp_dir) / "test.db"),
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                embedding_dimension=384,
                docs_path=str(temp_docs),
                openai_api_key=os.getenv("OPENAI_API_KEY", "test-key"),
                max_context_length=1000,
                top_k_results=3,
                similarity_threshold=0.3,
            )

    @pytest.mark.asyncio
    async def test_complete_indexing_and_query_flow(self, e2e_settings):
        """Test complete flow: index docs -> query -> get results."""

        # Step 1: Index documents
        indexer = DocumentIndexer(e2e_settings)
        await indexer.initialize()

        stats = await indexer.index_documentation(force_rebuild=True)

        assert stats["documents"] > 0
        assert stats["chunks"] > 0

        # Step 2: Query the indexed documents
        processor = QueryProcessor(e2e_settings)
        await processor.initialize()

        # Test Q&A query
        qa_result = await processor.process_qa_query("How do I install the application?")

        assert qa_result["success"] is True
        assert "pip install" in qa_result["answer"].lower()
        assert len(qa_result["context_used"]) > 0

        # Test command generation query
        cmd_result = await processor.process_query("start the application on port 8080")

        assert cmd_result.success is True
        assert "8080" in cmd_result.command

        # Cleanup
        await processor.close()
        await indexer.cleanup()

    @pytest.mark.asyncio
    async def test_system_status_check(self, e2e_settings):
        """Test system status after initialization."""
        processor = QueryProcessor(e2e_settings)
        await processor.initialize()

        status = await processor.get_system_status()

        assert status["initialized"] is True
        assert status["embedding_model"] == e2e_settings.embedding_model
        assert status["vector_db_type"] == e2e_settings.vector_db_type

        await processor.close()

    @pytest.mark.asyncio
    async def test_command_validation(self, e2e_settings):
        """Test command validation functionality."""
        processor = QueryProcessor(e2e_settings)
        await processor.initialize()

        # Test safe command
        safe_result = await processor.validate_command("ls -la")
        assert safe_result["is_valid"] is True
        assert safe_result["is_safe"] is True

        # Test dangerous command
        dangerous_result = await processor.validate_command("rm -rf /")
        assert dangerous_result["is_valid"] is True
        assert dangerous_result["is_safe"] is False

        await processor.close()

    @pytest.mark.asyncio
    async def test_search_without_llm(self, e2e_settings):
        """Test document search without LLM processing."""
        # Index first
        indexer = DocumentIndexer(e2e_settings)
        await indexer.initialize()
        await indexer.index_documentation(force_rebuild=True)
        await indexer.cleanup()

        # Search
        processor = QueryProcessor(e2e_settings)
        await processor.initialize()

        results = await processor.search_documentation("authentication", top_k=2)

        assert len(results) > 0
        # Should find API docs that mention authentication
        assert any("authentication" in result["content"].lower() for result in results)

        await processor.close()

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "").startswith("test"),
        reason="Requires real OpenAI API key for full E2E test"
    )
    @pytest.mark.asyncio
    async def test_real_llm_integration(self, e2e_settings):
        """Test with real LLM (only if API key available)."""
        # Index documents
        indexer = DocumentIndexer(e2e_settings)
        await indexer.initialize()
        await indexer.index_documentation(force_rebuild=True)
        await indexer.cleanup()

        # Query with real LLM
        processor = QueryProcessor(e2e_settings)
        await processor.initialize()

        qa_result = await processor.process_qa_query("What commands can I use to start the application?")

        assert qa_result["success"] is True
        assert len(qa_result["answer"]) > 10  # Should have meaningful response
        assert "myapp start" in qa_result["answer"] or "8080" in qa_result["answer"]

        await processor.close()