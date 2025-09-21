# EasyPrompt - Natural Language to CLI Command Interface

EasyPrompt is a powerful tool that translates natural language queries into precise CLI commands using Retrieval-Augmented Generation (RAG) architecture. Simply describe what you want to do in plain language, and EasyPrompt will generate the appropriate command for your CLI tool.

## Features

- 🎯 **Natural Language Processing**: Convert plain English to CLI commands
- 📚 **RAG Architecture**: Uses your documentation for accurate command generation
- 🔌 **Multiple LLM Providers**: Support for Gemini, OpenAI, and Anthropic
- 💾 **Flexible Vector Databases**: ChromaDB, Pinecone, and Weaviate support
- 🛡️ **Safety Features**: Command validation and confirmation prompts
- 💬 **Interactive Chat**: Real-time conversation with your CLI tool
- 🔍 **Smart Search**: Find relevant documentation and examples
- ⚡ **Fast & Efficient**: Optimized for quick response times

## Quick Start

### 🚀 One-Command Setup (Ubuntu/Linux)

```bash
# Clone the repository
git clone <repository-url>
cd easyprompt

# Run the setup script - it handles everything!
./setup.sh
```

**That's it!** The setup script will:
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Install EasyPrompt package
- ✅ Run validation tests
- ✅ Create example documentation

### After Setup

1. **Activate environment:**
   ```bash
   source ./activate.sh
   ```

2. **Initialize configuration:**
   ```bash
   easyprompt init
   ```

3. **Index documentation and start using:**
   ```bash
   easyprompt index ./example_docs
   easyprompt query "list all items"
   easyprompt chat  # Interactive mode
   ```

📖 **See [QUICKSTART.md](QUICKSTART.md) for detailed setup guide**

> **For Developers**: See [DEVELOPMENT.md](DEVELOPMENT.md) for development setup and workflow

## Usage Examples

### Single Query Mode
```bash
# Generate a command
easyprompt query "list all running pods in the default namespace"

# Execute immediately
easyprompt query "scale deployment myapp to 5 replicas" --execute

# Get alternatives
easyprompt query "backup the database" --alternatives 3
```

### Interactive Chat Mode
```bash
easyprompt chat
```
```
Query: how do I check the status of my services?
Command: kubectl get services
Explanation: This command lists all services in the current namespace...
Execute this command? [y/N]: y
```

### Documentation Search
```bash
# Search without generating commands
easyprompt search "deployment strategies"

# Find command examples
easyprompt examples "rolling updates"
```

### Command Validation
```bash
# Validate a command before running
easyprompt validate "kubectl delete namespace production"
```

## Configuration

EasyPrompt uses environment variables for configuration. Create a `.env` file or use `easyprompt init`:

```env
# Vector Database
VECTOR_DB_TYPE=chromadb
VECTOR_DB_URL=./data/chroma.db

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# CLI Tool
CLI_TOOL_NAME=kubectl
CLI_TOOL_PATH=/usr/local/bin/kubectl

# Documentation
DOCS_PATH=./docs
README_PATH=./README.md

# LLM Provider (choose one)
GEMINI_API_KEY=your_api_key
# OPENAI_API_KEY=your_api_key
# ANTHROPIC_API_KEY=your_api_key

# Optional Settings
MAX_CONTEXT_LENGTH=4000
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7
DRY_RUN=false
CONFIRM_BEFORE_EXECUTION=true
```

## Architecture

EasyPrompt uses a two-phase RAG architecture:

### 1. Indexing Phase (Offline)
- **Document Parsing**: Extracts content from README.md, docs/, and other specified files
- **Text Chunking**: Splits documents into manageable, contextual chunks
- **Vectorization**: Converts text to embeddings using sentence-transformers
- **Storage**: Stores embeddings in your chosen vector database

### 2. Query Phase (Real-time)
- **Query Processing**: Converts user queries to vector embeddings
- **Context Retrieval**: Finds most relevant documentation chunks
- **Command Generation**: Uses LLM with retrieved context to generate commands
- **Validation & Execution**: Validates commands and optionally executes them

## Supported Integrations

### Vector Databases
- **ChromaDB**: Local, file-based vector database (default)
- **Pinecone**: Cloud-hosted vector database with high performance
- **Weaviate**: Open-source vector database with GraphQL API

### LLM Providers
- **Google Gemini**: Fast and efficient for command generation
- **OpenAI GPT**: Powerful language understanding and generation
- **Anthropic Claude**: Advanced reasoning and safety features

### CLI Tools
EasyPrompt can work with any CLI tool. Popular integrations include:
- Kubernetes (`kubectl`)
- Docker (`docker`)
- Git (`git`)
- AWS CLI (`aws`)
- Terraform (`terraform`)
- And many more!

## Commands Reference

### Core Commands

- `easyprompt init` - Initialize configuration
- `easyprompt index [paths]` - Index documentation files
- `easyprompt query <text>` - Generate command from natural language
- `easyprompt chat` - Start interactive session
- `easyprompt status` - Show system status

### Utility Commands

- `easyprompt search <text>` - Search documentation
- `easyprompt examples <text>` - Find command examples
- `easyprompt validate <command>` - Validate a command

### Options

- `--execute, -e` - Execute generated commands immediately
- `--dry-run` - Show commands without executing
- `--alternatives <n>` - Generate alternative commands
- `--provider <name>` - Choose specific LLM provider
- `--verbose, -v` - Detailed output

## Safety Features

EasyPrompt includes several safety mechanisms:

- **Command Validation**: Checks for syntax and dangerous patterns
- **Confirmation Prompts**: Optional user confirmation before execution
- **Dry Run Mode**: Preview commands without executing
- **Safety Classifications**: Identifies potentially dangerous operations
- **Execution Timeouts**: Prevents runaway commands

## Development

### Project Structure
```
easyprompt/
├── config/          # Configuration management
├── indexer/         # Document parsing and indexing
├── vectordb/        # Vector database adapters
├── llm/             # LLM provider integrations
├── query/           # Query processing and command generation
├── cli/             # Command-line interface
└── utils/           # Utilities and helpers
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=easyprompt

# Run specific tests
pytest tests/test_query_processor.py
```

## Troubleshooting

### Common Issues

**"No LLM provider configured"**
- Ensure you have at least one API key set (GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY)

**"Vector database connection failed"**
- For ChromaDB: Check if the data directory is writable
- For Pinecone: Verify API key and environment settings
- For Weaviate: Ensure the service is running and accessible

**"No documents found"**
- Run `easyprompt index` to index your documentation
- Check that your DOCS_PATH and README_PATH point to existing files

**Command generation is inaccurate**
- Index more relevant documentation
- Adjust SIMILARITY_THRESHOLD for better context retrieval
- Try different LLM providers

### Getting Help

- Check the [documentation](./ARCHITECTURE.md) for detailed architecture info
- Search existing [issues](https://github.com/your-org/easyprompt/issues)
- Create a new issue with detailed information

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Typer](https://typer.tiangolo.com/) for CLI
- Powered by [sentence-transformers](https://www.sbert.net/) for embeddings
- Uses [Rich](https://rich.readthedocs.io/) for beautiful terminal output