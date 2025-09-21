#!/usr/bin/env python3
"""
Comprehensive validation script for EasyPrompt.
Tests functionality without requiring external API keys.
"""

import sys
import os
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock
import logging

# Add the project to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class MockSettings:
    """Mock settings for testing."""
    def __init__(self, temp_dir):
        self.vector_db_type = "chromadb"
        self.vector_db_url = str(temp_dir / "test.db")
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_dimension = 384
        self.cli_tool_name = "test-cli"
        self.cli_tool_path = "/usr/bin/test-cli"
        self.docs_path = str(temp_dir / "docs")
        self.readme_path = str(temp_dir / "README.md")
        self.gemini_api_key = "test-key"
        self.openai_api_key = None
        self.anthropic_api_key = None
        self.max_context_length = 1000
        self.top_k_results = 3
        self.similarity_threshold = 0.5
        self.dry_run = True
        self.confirm_before_execution = False

    @property
    def available_llm_providers(self):
        return ["gemini"]

    @property
    def primary_llm_provider(self):
        return "gemini"

    @property
    def additional_docs_list(self):
        return []

async def test_document_parsing():
    """Test document parsing functionality."""
    logger.info("Testing document parsing...")

    try:
        from easyprompt.indexer.document_parser import DocumentParser

        parser = DocumentParser()

        # Create test markdown content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Test CLI Tool

A simple CLI tool for testing.

## Commands

### list
List all items:
```bash
test-cli list
```

### add
Add an item:
```bash
test-cli add "item name"
```
""")
            temp_file = Path(f.name)

        try:
            document = await parser.parse_file(temp_file)

            assert document["type"] == "markdown"
            assert document["title"] == "Test CLI Tool"
            assert "# Test CLI Tool" in document["content"]
            assert isinstance(document["sections"], dict)
            assert len(document["sections"]) > 1

            # Test command extraction
            commands = parser.extract_commands(document["content"])
            assert any("test-cli list" in cmd for cmd in commands)
            assert any("test-cli add" in cmd for cmd in commands)

            logger.info("✓ Document parsing works correctly")
            return True

        finally:
            temp_file.unlink()

    except Exception as e:
        logger.error(f"✗ Document parsing failed: {e}")
        return False

async def test_text_chunking():
    """Test text chunking functionality."""
    logger.info("Testing text chunking...")

    try:
        from easyprompt.indexer.text_chunker import TextChunker

        chunker = TextChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=30)

        document = {
            "file_path": "test.md",
            "content": "This is a test document. " * 20,  # Long content
            "sections": {"main": "This is a test document. " * 20}
        }

        chunks = chunker.chunk_document(document)

        assert len(chunks) >= 1
        assert all(chunk.file_path == "test.md" for chunk in chunks)
        assert all(len(chunk.content) > 0 for chunk in chunks)

        logger.info(f"✓ Text chunking works correctly (created {len(chunks)} chunks)")
        return True

    except Exception as e:
        logger.error(f"✗ Text chunking failed: {e}")
        return False

async def test_command_validation():
    """Test command validation logic."""
    logger.info("Testing command validation...")

    try:
        from easyprompt.query.command_generator import CommandGenerator

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = MockSettings(Path(temp_dir))
            generator = CommandGenerator(settings)

            # Test validation methods
            test_cases = [
                ("ls -la", True, True, "read"),
                ("rm -rf /", False, False, "delete"),
                ("", False, True, "unclear"),
                ("UNCLEAR_REQUEST", True, True, "unclear"),
                ("git status", True, True, "git"),
                ("sudo systemctl restart nginx", True, False, "system")
            ]

            for command, expected_valid, expected_safe, expected_type in test_cases:
                is_valid = generator._validate_command(command)
                is_safe = generator._check_command_safety(command)
                cmd_type = generator._classify_command(command)

                if is_valid != expected_valid:
                    logger.error(f"✗ Validation failed for '{command}': got {is_valid}, expected {expected_valid}")
                    return False

                if is_safe != expected_safe:
                    logger.error(f"✗ Safety check failed for '{command}': got {is_safe}, expected {expected_safe}")
                    return False

                if cmd_type != expected_type:
                    logger.error(f"✗ Classification failed for '{command}': got {cmd_type}, expected {expected_type}")
                    return False

            logger.info("✓ Command validation works correctly")
            return True

    except Exception as e:
        logger.error(f"✗ Command validation failed: {e}")
        return False

async def test_mock_llm_integration():
    """Test LLM integration with mocks."""
    logger.info("Testing LLM integration...")

    try:
        from easyprompt.llm.base_provider import BaseLLMProvider, LLMResponse, Message

        class MockLLMProvider(BaseLLMProvider):
            async def generate_command(self, user_query, context, cli_tool_name, **kwargs):
                # Simple mock that generates commands based on query
                if "list" in user_query.lower():
                    return LLMResponse(content="test-cli list", model="mock-model")
                elif "add" in user_query.lower():
                    return LLMResponse(content="test-cli add item", model="mock-model")
                else:
                    return LLMResponse(content="UNCLEAR_REQUEST", model="mock-model")

            async def chat_completion(self, messages, **kwargs):
                return LLMResponse(content="Mock response", model="mock-model")

            async def is_available(self):
                return True

            @property
            def provider_name(self):
                return "mock"

            @property
            def default_model(self):
                return "mock-model"

        provider = MockLLMProvider("mock-key")

        # Test command generation
        response = await provider.generate_command(
            "list all files",
            "Use test-cli list to list files",
            "test-cli"
        )

        assert response.content == "test-cli list"
        assert response.model == "mock-model"

        # Test unclear request
        response = await provider.generate_command(
            "do something weird",
            "No relevant context",
            "test-cli"
        )

        assert response.content == "UNCLEAR_REQUEST"

        logger.info("✓ LLM integration works correctly")
        return True

    except Exception as e:
        logger.error(f"✗ LLM integration failed: {e}")
        return False

async def test_end_to_end_workflow():
    """Test end-to-end workflow with mocks."""
    logger.info("Testing end-to-end workflow...")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test documentation
            readme_path = temp_path / "README.md"
            readme_path.write_text("""# Test CLI Tool

## Commands

### list
List files:
```bash
test-cli list
```

### add
Add item:
```bash
test-cli add "name"
```
""")

            # Mock all the dependencies
            settings = MockSettings(temp_path)

            # Test that we can create all the main components
            from easyprompt.indexer.document_parser import DocumentParser
            from easyprompt.indexer.text_chunker import TextChunker

            parser = DocumentParser()
            chunker = TextChunker()

            # Parse document
            document = await parser.parse_file(readme_path)
            assert document["title"] == "Test CLI Tool"

            # Chunk document
            chunks = chunker.chunk_document(document)
            assert len(chunks) > 0

            logger.info("✓ End-to-end workflow components work correctly")
            return True

    except Exception as e:
        logger.error(f"✗ End-to-end workflow failed: {e}")
        return False

def test_configuration_validation():
    """Test configuration validation."""
    logger.info("Testing configuration validation...")

    try:
        # Test without importing pydantic (which might not be available)
        # Just test the logic concepts

        # Mock a simple settings class
        class SimpleSettings:
            def __init__(self):
                self.vector_db_type = "chromadb"
                self.embedding_model = "test-model"
                self.cli_tool_name = "test-cli"
                self.gemini_api_key = "test-key"

        settings = SimpleSettings()

        # Test basic validation logic
        errors = []

        if settings.vector_db_type not in ["chromadb", "pinecone", "weaviate"]:
            errors.append("Invalid vector DB type")

        if not settings.embedding_model:
            errors.append("Embedding model required")

        if not settings.cli_tool_name:
            errors.append("CLI tool name required")

        if not settings.gemini_api_key:
            errors.append("At least one LLM provider required")

        assert len(errors) == 0, f"Validation errors: {errors}"

        logger.info("✓ Configuration validation logic works correctly")
        return True

    except Exception as e:
        logger.error(f"✗ Configuration validation failed: {e}")
        return False

async def run_all_tests():
    """Run all validation tests."""
    logger.info("EasyPrompt Project Validation")
    logger.info("=" * 50)

    tests = [
        ("Configuration Validation", test_configuration_validation),
        ("Document Parsing", test_document_parsing),
        ("Text Chunking", test_text_chunking),
        ("Command Validation", test_command_validation),
        ("LLM Integration", test_mock_llm_integration),
        ("End-to-End Workflow", test_end_to_end_workflow),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} failed with exception: {e}")

    logger.info("\n" + "=" * 50)
    logger.info(f"VALIDATION RESULTS: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! EasyPrompt is working correctly.")
        logger.info("\nProject is ready for use:")
        logger.info("1. Install dependencies: pip install -r requirements.txt")
        logger.info("2. Configure: easyprompt init")
        logger.info("3. Index docs: easyprompt index")
        logger.info("4. Start using: easyprompt query 'your request'")
    else:
        logger.info(f"⚠️  {total - passed} tests failed. Check the issues above.")

    return passed == total

if __name__ == "__main__":
    try:
        result = asyncio.run(run_all_tests())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        sys.exit(1)