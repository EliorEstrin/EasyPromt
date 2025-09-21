"""Test configuration and fixtures."""

import asyncio
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock
import numpy as np

from easyprompt.config import Settings


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_settings(temp_dir):
    """Create sample settings for testing."""
    return Settings(
        vector_db_type="chromadb",
        vector_db_url=str(temp_dir / "test.db"),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        cli_tool_name="test-cli",
        cli_tool_path="/usr/bin/test-cli",
        docs_path=str(temp_dir / "docs"),
        readme_path=str(temp_dir / "README.md"),
        gemini_api_key="test-gemini-key",
        max_context_length=1000,
        top_k_results=3,
        similarity_threshold=0.5,
        dry_run=True,
        confirm_before_execution=False
    )


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "file_path": "README.md",
            "content": "# Test CLI\n\nThis is a test CLI tool.\n\n## Commands\n\n### list\nList all items:\n```bash\ntest-cli list\n```",
            "sections": {
                "introduction": "# Test CLI\n\nThis is a test CLI tool.",
                "commands": "## Commands\n\n### list\nList all items:\n```bash\ntest-cli list\n```"
            },
            "type": "markdown",
            "title": "Test CLI"
        },
        {
            "file_path": "docs/advanced.md",
            "content": "# Advanced Usage\n\n## Filtering\nFilter items:\n```bash\ntest-cli list --filter name=value\n```",
            "sections": {
                "advanced_usage": "# Advanced Usage",
                "filtering": "## Filtering\nFilter items:\n```bash\ntest-cli list --filter name=value\n```"
            },
            "type": "markdown",
            "title": "Advanced Usage"
        }
    ]


@pytest.fixture
def sample_chunks():
    """Sample text chunks for testing."""
    return [
        {
            "content": "List all items using the test-cli list command",
            "start_index": 0,
            "end_index": 50,
            "section": "commands",
            "file_path": "README.md",
            "chunk_id": "README.md:commands:0",
            "metadata": {
                "title": "Test CLI",
                "type": "markdown",
                "has_commands": "true"
            }
        },
        {
            "content": "Filter items using test-cli list --filter name=value",
            "start_index": 0,
            "end_index": 55,
            "section": "filtering",
            "file_path": "docs/advanced.md",
            "chunk_id": "docs/advanced.md:filtering:0",
            "metadata": {
                "title": "Advanced Usage",
                "type": "markdown",
                "has_commands": "true"
            }
        }
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    np.random.seed(42)  # For reproducible tests
    return [
        np.random.rand(384).astype(np.float32),
        np.random.rand(384).astype(np.float32)
    ]


@pytest.fixture
def mock_llm_response():
    """Mock LLM response."""
    from easyprompt.llm.base_provider import LLMResponse
    return LLMResponse(
        content="test-cli list",
        model="test-model",
        usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        metadata={"provider": "test"}
    )


@pytest.fixture
def mock_vector_db():
    """Mock vector database."""
    mock_db = AsyncMock()
    mock_db.initialize = AsyncMock()
    mock_db.add_documents = AsyncMock()
    mock_db.search = AsyncMock(return_value=[
        {
            "id": "test:chunk:0",
            "content": "test content",
            "metadata": {"file_path": "test.md", "section": "main"},
            "similarity": 0.8
        }
    ])
    mock_db.count = AsyncMock(return_value=10)
    mock_db.close = AsyncMock()
    return mock_db


@pytest.fixture
def mock_embedding_generator():
    """Mock embedding generator."""
    mock_gen = AsyncMock()
    mock_gen.initialize = AsyncMock()
    mock_gen.generate_embedding = AsyncMock(return_value=np.random.rand(384))
    mock_gen.generate_embeddings = AsyncMock(return_value=[np.random.rand(384), np.random.rand(384)])
    mock_gen.generate_query_embedding = AsyncMock(return_value=np.random.rand(384))
    return mock_gen


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider."""
    from easyprompt.llm.base_provider import LLMResponse

    mock_provider = AsyncMock()
    mock_provider.provider_name = "test"
    mock_provider.default_model = "test-model"
    mock_provider.initialize = AsyncMock()
    mock_provider.generate_command = AsyncMock(return_value=LLMResponse(
        content="test-cli list",
        model="test-model"
    ))
    mock_provider.generate_command_with_explanation = AsyncMock(return_value=(
        "test-cli list",
        "This command lists all items"
    ))
    mock_provider.is_available = AsyncMock(return_value=True)
    return mock_provider


@pytest.fixture
def sample_markdown_files(temp_dir):
    """Create sample markdown files for testing."""
    docs_dir = temp_dir / "docs"
    docs_dir.mkdir()

    # README.md
    readme = temp_dir / "README.md"
    readme.write_text("""# Test CLI Tool

A simple CLI tool for testing.

## Installation

```bash
pip install test-cli
```

## Usage

### Basic Commands

List all items:
```bash
test-cli list
```

Add an item:
```bash
test-cli add "item name"
```

### Advanced Commands

Filter items:
```bash
test-cli list --filter status=active
```
""")

    # docs/api.md
    api_doc = docs_dir / "api.md"
    api_doc.write_text("""# API Reference

## Commands

### list
List items with optional filtering.

```bash
test-cli list [OPTIONS]
```

Options:
- `--filter`: Filter by key=value pairs
- `--format`: Output format (json, table)

### add
Add a new item.

```bash
test-cli add <name> [OPTIONS]
```

Options:
- `--description`: Item description
- `--tags`: Comma-separated tags
""")

    return {
        "readme": readme,
        "api_doc": api_doc,
        "docs_dir": docs_dir
    }


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_subprocess_run():
    """Mock subprocess.run for command execution tests."""
    from unittest.mock import patch
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Command executed successfully"
        mock_run.return_value.stderr = ""
        yield mock_run