# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical First Steps

**ALWAYS start with validation**: Run `python3 validate_core_logic.py` before making any changes to ensure the core system is working.

**Key workflow for development**:
1. `source ./activate.sh` - Activate development environment
2. `python3 validate_core_logic.py` - Ensure system works
3. Make your changes
4. `make format && make lint` - Code quality checks
5. `pytest tests/` - Run tests
6. Test with CLI commands (`easyprompt init`, `easyprompt chat`, etc.)

## Project Overview

**EasyPrompt** is a Natural Language to CLI Command Interface that uses Retrieval-Augmented Generation (RAG) architecture to translate natural language queries into precise CLI commands. The system indexes documentation offline and uses vector similarity search with LLM providers for real-time command generation.

### Key Features
- 🎯 Natural Language Processing: Convert plain English to CLI commands
- 📚 RAG Architecture: Uses documentation for accurate command generation
- 🔌 Multiple LLM Providers: Gemini, OpenAI, and Anthropic support
- 💾 Flexible Vector Databases: ChromaDB, Pinecone, and Weaviate
- 🛡️ Safety Features: Command validation and confirmation prompts
- 💬 Interactive Chat: Real-time conversation interface
- ⚡ Fast & Efficient: Optimized for quick response times

## Essential Development Commands

### Setup and Environment
```bash
./setup.sh                       # Complete project setup (asks about existing venv)
./setup.sh --force               # Force recreate virtual environment
source ./activate.sh             # Activate development environment
pip install -e .                 # Install with basic dependencies
pip install -e ".[full]"         # Install with ALL dependencies (heavy)
pip install -e ".[dev]"          # Install with dev dependencies
pip install -e ".[minimal]"      # Install with minimal dependencies only
```

### Dependency Management
The project has **three-tier dependency system**:
- **Basic install**: Core functionality with LLM providers only
- **Minimal install**: Same as basic (explicitly defined)
- **Full install**: Includes heavy vector processing dependencies
- **Dev install**: Development tools for testing and code quality

For development with minimal dependencies:
```bash
pip install -e .                 # Basic: no vector DBs, no sentence-transformers
pip install -e ".[minimal]"      # Explicit minimal install
```

For full functionality:
```bash
pip install -e ".[full]"         # Full: includes ChromaDB, Pinecone, sentence-transformers
```

For development with all tools:
```bash
pip install -e ".[dev]"          # Dev: includes pytest, black, mypy, flake8, pre-commit
```

### Testing and Validation
```bash
python3 validate_core_logic.py   # Core algorithm validation (run first!)
python3 test_setup.py           # Validate setup completeness
pytest tests/ -v                # Run full test suite
pytest tests/unit/ -v           # Run unit tests only
pytest --cov=easyprompt tests/  # Run tests with coverage
```

### Code Quality
```bash
make format                     # Format code with black and isort
make lint                       # Run flake8 and mypy
make format-check              # Check formatting without changes
make test-cov                   # Run tests with coverage report
make run-tests                  # Run comprehensive test suite (format-check + lint + test)
black easyprompt/ tests/       # Format code directly
mypy easyprompt/               # Type checking
flake8 easyprompt/ tests/      # Linting
isort easyprompt/ tests/       # Import sorting
```

### CLI Usage and Testing
```bash
easyprompt init                 # Interactive configuration setup
easyprompt index ./example_docs # Index documentation
easyprompt query "text"         # Single query mode
easyprompt chat                 # Interactive chat mode
easyprompt status               # System status check
```

## Project Architecture

### Two-Phase RAG System

1. **Indexing Phase (Offline)**
   - Document parsing from README.md, docs/, and specified files
   - Text chunking with overlap for context preservation
   - Vectorization using sentence-transformers
   - Storage in chosen vector database

2. **Query Phase (Real-time)**
   - Query processing and vectorization
   - Context retrieval using similarity search
   - Command generation via LLM with retrieved context
   - Validation and optional execution

### Core Components

```
easyprompt/
├── config/          # Configuration management and validation
├── indexer/         # Document processing and vectorization
├── vectordb/        # Vector database adapters (ChromaDB, Pinecone, Weaviate)
├── llm/             # LLM provider integrations (Gemini, OpenAI, Anthropic)
├── query/           # Query processing and command generation
├── cli/             # Command-line interface and commands
└── utils/           # Utilities and helpers
```

## Key Files and Responsibilities

### Configuration System
- `easyprompt/config/settings.py` - Centralized configuration using Pydantic
- `easyprompt/config/validators.py` - Configuration validation logic
- `.env.example` - Environment variable template

### Document Processing
- `easyprompt/indexer/indexer.py` - Main indexing orchestrator
- `easyprompt/indexer/document_parser.py` - Markdown and text parsing
- `easyprompt/indexer/text_chunker.py` - Smart text chunking with overlap
- `easyprompt/indexer/embedding_generator.py` - Vector embedding generation

### Vector Databases
- `easyprompt/vectordb/base_vectordb.py` - Abstract base class
- `easyprompt/vectordb/chromadb_adapter.py` - ChromaDB implementation (default)
- `easyprompt/vectordb/pinecone_adapter.py` - Pinecone cloud implementation
- `easyprompt/vectordb/weaviate_adapter.py` - Weaviate implementation
- `easyprompt/vectordb/factory.py` - Factory pattern for database selection

### LLM Providers
- `easyprompt/llm/base_provider.py` - Abstract provider interface
- `easyprompt/llm/gemini_provider.py` - Google Gemini integration
- `easyprompt/llm/openai_provider.py` - OpenAI GPT integration
- `easyprompt/llm/anthropic_provider.py` - Anthropic Claude integration
- `easyprompt/llm/provider_factory.py` - Factory for provider selection

### Query Processing
- `easyprompt/query/query_processor.py` - **Main orchestrator** - handles complete query workflow
- `easyprompt/query/context_retriever.py` - Vector similarity search and context retrieval
- `easyprompt/query/command_generator.py` - LLM-based command generation

### CLI Interface
- `easyprompt/cli/main.py` - Main CLI entry point using Typer
- `easyprompt/cli/commands.py` - Core commands including interactive ChatCommand
- `easyprompt/cli/init_command.py` - Interactive configuration setup
- `easyprompt/cli/search_command.py` - Documentation search without command generation
- `easyprompt/cli/validate_command.py` - Command validation utilities
- `easyprompt/cli/examples_command.py` - Example command finder

## Common Development Tasks

### Adding New Vector Database Support
1. Create new adapter in `easyprompt/vectordb/` inheriting from `BaseVectorDB`
2. Implement required methods: `store_embeddings`, `search_similar`, `delete_collection`
3. Add factory registration in `easyprompt/vectordb/factory.py`
4. Update configuration validation in `easyprompt/config/validators.py`

### Adding New LLM Provider
1. Create provider in `easyprompt/llm/` inheriting from `BaseLLMProvider`
2. Implement `generate_command` method with proper error handling
3. Add factory registration in `easyprompt/llm/provider_factory.py`
4. Update configuration validation for new API key requirements

### Adding New CLI Command
1. Create command function or class in `easyprompt/cli/`
2. Register in `easyprompt/cli/main.py` using `@app.command()` decorator
3. Follow existing patterns for error handling and rich output
4. Add corresponding tests in `tests/unit/test_cli_commands.py`

## Configuration Management

### Environment Variables (via .env file)
```bash
# Vector Database
VECTOR_DB_TYPE=chromadb|pinecone|weaviate
VECTOR_DB_URL=./data/chroma.db

# LLM Provider (at least one required)
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# CLI Tool Configuration
CLI_TOOL_NAME=kubectl
CLI_TOOL_PATH=/usr/local/bin/kubectl

# Documentation Paths
DOCS_PATH=./docs
README_PATH=./README.md

# Optional Tuning
MAX_CONTEXT_LENGTH=4000
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7
DRY_RUN=false
CONFIRM_BEFORE_EXECUTION=true
```

### Settings System
- Uses Pydantic for type-safe configuration
- Automatic validation with clear error messages
- Environment variable loading with defaults
- Validator classes for complex validation logic

## Testing Framework

### Validation Scripts (No External Dependencies)
- `validate_core_logic.py` - Core algorithm validation (6 test cases)
- `test_basic_functionality.py` - Basic functionality verification
- `test_setup.py` - Setup completeness validation

### Full Test Suite
```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v

# Run with coverage
pytest --cov=easyprompt tests/
```

### Test Structure
```
tests/
├── conftest.py                    # Pytest configuration and fixtures
├── unit/                         # Unit tests for individual components
│   ├── test_config.py
│   ├── test_document_parser.py
│   ├── test_text_chunker.py
│   ├── test_embedding_generator.py
│   ├── test_chromadb_adapter.py
│   ├── test_llm_providers.py
│   └── test_cli_commands.py
└── integration/                  # Integration tests
    ├── test_query_processor.py
    └── test_document_indexer.py
```

## Safety and Validation

### Command Safety Features
- Pattern-based dangerous command detection
- User confirmation prompts for destructive operations
- Dry-run mode for command preview
- Command syntax validation
- Execution timeouts to prevent runaway processes

### Safety Classifications
Commands are classified as:
- `safe` - Read-only operations, status checks
- `moderate` - Configuration changes, non-destructive operations
- `dangerous` - Destructive operations, system modifications

## Additional CLI Commands
```bash
easyprompt search "text"         # Search without command generation
easyprompt validate "command"    # Validate command safety
easyprompt examples "text"       # Find related command examples
```

## Error Handling Patterns

### Configuration Errors
- Use `ConfigValidator` for validation with clear error messages
- Fail fast during startup for missing critical configuration
- Provide helpful hints for common configuration issues

### Runtime Errors
- Use tenacity for retries with exponential backoff
- Log errors with appropriate levels (INFO, WARNING, ERROR)
- Graceful degradation when possible (e.g., fallback providers)

### CLI Error Handling
- Use Rich console for formatted error output
- Provide actionable error messages with suggestions
- Use appropriate exit codes (0 for success, 1 for errors)

## Integration Points

### Vector Database Integration
All vector databases implement the same interface:
```python
async def store_embeddings(self, texts: List[str], embeddings: List[List[float]], metadata: List[Dict] = None)
async def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]
async def delete_collection(self, collection_name: str)
```

### LLM Provider Integration
All providers implement:
```python
async def generate_command(self, query: str, context: str, cli_tool: str) -> str
```

## Troubleshooting Guide

### Common Issues

1. **Import Errors**
   - Ensure virtual environment is activated: `source ./activate.sh`
   - Reinstall in development mode: `pip install -e .`

2. **Configuration Issues**
   - Run `easyprompt init` for interactive setup
   - Check `.env` file exists and has required keys
   - Validate with `python3 validate_core_logic.py`

3. **Vector Database Issues**
   - For ChromaDB: Check data directory permissions
   - For Pinecone: Verify API key and environment
   - For Weaviate: Ensure service is running

4. **LLM Provider Issues**
   - Verify API keys are set correctly
   - Check network connectivity
   - Try different provider as fallback

5. **CLI Command Not Found**
   - Check package installation: `pip list | grep easyprompt`
   - Ensure virtual environment is activated
   - Reinstall if necessary: `pip install -e .`

### Clean Reinstall
```bash
# Remove virtual environment and recreate
rm -rf venv
./setup.sh
source ./activate.sh
```

## Development Workflow

### For New Features
1. Understand the component you're modifying
2. Check existing patterns and conventions
3. Add appropriate tests
4. Run validation: `python3 validate_core_logic.py`
5. Test manually with CLI commands
6. Update documentation if needed

### For Bug Fixes
1. Reproduce the issue
2. Add test case that fails
3. Fix the issue
4. Ensure test passes
5. Run full validation suite

### Code Style Guidelines
- Follow existing patterns in the codebase
- Use type hints consistently
- Add docstrings for public APIs
- Keep functions focused and modular
- Use factory patterns for pluggable components

## Key Design Patterns

### Factory Pattern
Used for vector databases and LLM providers to enable easy extension and configuration-based selection.

### Dependency Injection
Settings and configuration passed down through the call stack to enable testing and flexibility.

### Async/Await
Used throughout for I/O operations to enable better performance and responsiveness.

### Rich Integration
Consistent use of Rich console for formatted output, progress bars, and error display.

## Performance Considerations

### Vector Search Optimization
- Tune `SIMILARITY_THRESHOLD` based on your documentation quality
- Adjust `TOP_K_RESULTS` to balance context relevance vs. token usage
- Consider chunking strategy for large documents

### LLM Usage Optimization
- Keep context within `MAX_CONTEXT_LENGTH` to manage costs
- Use caching where appropriate
- Implement fallback providers for reliability

### Memory Management
- Vector databases handle large embedding storage
- Text chunking prevents memory issues with large documents
- Streaming responses where possible

## Python 3.12 Compatibility

The project is **fully compatible with Python 3.12**. Previous setup issues have been resolved:

### Setup Issue Resolution
- **Fixed dependency conflicts**: Updated requirements.txt to use `>=` instead of `==` for better compatibility
- **Optional imports**: Heavy dependencies (sentence-transformers, chromadb, etc.) are now optional
- **Two-tier system**: Basic installation works without heavy ML dependencies
- **Graceful failures**: Missing dependencies show helpful error messages
- **Safe venv handling**: Setup script no longer auto-removes existing virtual environments

### Quick Setup for Python 3.12
```bash
./setup.sh                       # Standard setup (asks about existing venv)
# OR for clean start:
./setup.sh --force               # Force recreate venv from scratch
# OR manual setup:
python3 -m venv venv             # Create environment
source venv/bin/activate         # Activate
pip install --upgrade pip setuptools wheel  # Upgrade build tools
pip install -e .                 # Basic install
python3 validate_core_logic.py   # Verify installation
easyprompt --help               # Test CLI
```

## Key Points for Development

1. **Project location**: Repository root is `/home/elior/EasyPromt_repo`
2. **Always validate first**: Run `python3 validate_core_logic.py` before any changes
3. **Two-tier dependencies**: Basic install works without heavy ML libraries
4. **Main entry point**: `easyprompt/query/query_processor.py` orchestrates the entire workflow
5. **Factory patterns**: Used for vector databases (`vectordb/factory.py`) and LLM providers (`llm/provider_factory.py`)
6. **Configuration**: All settings managed through Pydantic in `config/settings.py`
7. **Testing approach**: Use validation scripts first, then pytest for comprehensive testing
8. **CLI testing**: Interactive commands (`easyprompt chat`) are best for manual testing

The codebase uses async/await throughout, Rich for CLI output, and maintains strict type safety with mypy.

## Git Workflow

Current git status shows modified file: `easyprompt/cli/init_command.py`. The main branch is `main`.

When making changes:
1. Always commit changes with descriptive messages
2. Use `git status` and `git diff` to review changes before committing
3. Test thoroughly before pushing to main branch