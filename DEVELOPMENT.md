# EasyPrompt Development Guide

## Quick Developer Setup

For developers with no context who want to get started immediately:

```bash
# Clone and setup everything
git clone <repository-url>
cd easyprompt
./setup.sh

# Activate development environment
source ./activate.sh

# You're ready to go!
easyprompt --help
```

## Development Environment

### What setup.sh does:
1. ✅ Creates isolated Python virtual environment in `./venv`
2. ✅ Installs all dependencies from `requirements.txt`
3. ✅ Installs EasyPrompt in development mode (`pip install -e .`)
4. ✅ Creates example documentation for testing
5. ✅ Runs validation tests to ensure everything works
6. ✅ Creates `./activate.sh` for easy environment activation

### Daily Development Workflow

```bash
# Activate environment (do this once per terminal session)
source ./activate.sh

# Make your changes to the code
# Test your changes
python3 validate_core_logic.py
python3 test_setup.py

# Try the CLI
easyprompt query "test my changes"
easyprompt chat

# Run tests (when pytest is available)
pytest tests/
```

## Project Structure

```
easyprompt/
├── setup.sh                   # 🚀 One-command setup script
├── activate.sh                # Environment activation
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Package configuration
├── QUICKSTART.md              # User quick start guide
├── DEVELOPMENT.md             # This file
├── easyprompt/                # Main package
│   ├── config/                # Configuration management
│   ├── indexer/               # Document processing
│   ├── vectordb/              # Vector database adapters
│   ├── llm/                   # LLM provider integrations
│   ├── query/                 # Query processing
│   └── cli/                   # Command-line interface
├── tests/                     # Test suite
├── example_docs/              # Example documentation (created by setup)
├── venv/                      # Virtual environment (created by setup)
└── data/                      # Data directory (created by setup)
```

## Key Files for Developers

### Core Components
- `easyprompt/config/settings.py` - Configuration management
- `easyprompt/indexer/indexer.py` - Document indexing orchestrator
- `easyprompt/query/query_processor.py` - Main query processing logic
- `easyprompt/cli/main.py` - CLI entry point

### Testing
- `validate_core_logic.py` - Core algorithm validation (no dependencies)
- `test_setup.py` - Setup validation
- `tests/` - Full test suite (requires pytest)

### Configuration
- `.env.example` - Environment variable template
- `easyprompt/config/` - Configuration system

## Development Commands

```bash
# Validate core logic (no external dependencies)
python3 validate_core_logic.py

# Test setup completeness
python3 test_setup.py

# Test basic functionality
python3 test_basic_functionality.py

# Use the CLI
easyprompt init          # Interactive setup
easyprompt index         # Index documentation
easyprompt query "..."   # Single query
easyprompt chat          # Interactive mode
easyprompt status        # System status
```

## Testing Your Changes

### 1. Core Logic Tests (Always Run)
```bash
python3 validate_core_logic.py
```
Tests fundamental algorithms without external dependencies.

### 2. Setup Validation (After Changes)
```bash
python3 test_setup.py
```
Ensures the development environment is correctly configured.

### 3. Manual Testing
```bash
# Test CLI functionality
easyprompt query "list files"
easyprompt chat

# Test with example docs
easyprompt index ./example_docs
easyprompt query "show me all commands"
```

### 4. Full Test Suite (Optional)
```bash
# If you have pytest installed
pytest tests/ -v
```

## Making Changes

### Adding New Features
1. Create your feature in the appropriate module
2. Add tests in `tests/`
3. Update documentation if needed
4. Run validation: `python3 validate_core_logic.py`
5. Test manually: `easyprompt query "test your feature"`

### Debugging
- Use `easyprompt status` to check system health
- Check `easyprompt --help` for available commands
- Enable verbose logging in configuration

### Code Style
- Follow existing patterns in the codebase
- Use type hints where possible
- Keep functions focused and modular
- Add docstrings for public APIs

## Environment Variables

The system uses `.env` files for configuration. Key variables:

```bash
# Vector Database
VECTOR_DB_TYPE=chromadb
VECTOR_DB_URL=./data/chroma.db

# LLM Provider (need at least one)
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# CLI Tool
CLI_TOOL_NAME=kubectl
CLI_TOOL_PATH=/usr/local/bin/kubectl

# Documentation
DOCS_PATH=./docs
README_PATH=./README.md
```

## Troubleshooting

### Setup Issues
```bash
# Clean reinstall
rm -rf venv
./setup.sh
```

### Import Errors
```bash
# Ensure you're in the virtual environment
source ./activate.sh

# Reinstall in development mode
pip install -e .
```

### CLI Not Found
```bash
# Check if package is installed
pip list | grep easyprompt

# Reinstall if needed
pip install -e .
```

## Contributing

1. Make your changes
2. Run validation tests
3. Test manually with the CLI
4. Update documentation if needed
5. Submit your changes

## Getting Help

- Check existing issues in the repository
- Read the full documentation in `README.md`
- Look at the architecture in `ARCHITECTURE.md`
- Run `easyprompt --help` for CLI help