# EasyPrompt - Natural Language to CLI Command Interface

## Project Overview

EasyPrompt is a standalone application that serves as a natural language interface for CLI tools. Users can describe tasks in plain language, and the application translates these requests into precise CLI commands and executes them.

## Core Architecture: RAG (Retrieval-Augmented Generation)

The system operates in two distinct phases:

### Phase 1: Data Indexing (Offline Process)
- **Data Ingestion**: Parse documentation files (README.md, docs/*.md, etc.)
- **Text Processing**: Clean and chunk documentation content
- **Vectorization**: Convert text chunks into embeddings using a specified model
- **Storage**: Store embeddings in a vector database with metadata

### Phase 2: Command Generation (Real-time Process)
- **Query Processing**: Convert user's natural language input to vector embedding
- **Context Retrieval**: Query vector database for relevant documentation
- **LLM Integration**: Combine context with user query for command generation
- **Execution**: Run the generated CLI command and return results

## System Components

### 1. Data Indexing Module (`indexer/`)
- **DocumentParser**: Extracts content from markdown files
- **TextChunker**: Splits documents into manageable chunks
- **EmbeddingGenerator**: Creates vector embeddings
- **VectorStore**: Manages database operations

### 2. Query Processing Module (`query/`)
- **QueryEmbedder**: Converts user queries to vectors
- **ContextRetriever**: Searches vector database for relevant content
- **CommandGenerator**: Interfaces with LLM to generate commands

### 3. LLM Provider Adapters (`llm/`)
- **BaseProvider**: Abstract interface for LLM providers
- **GeminiProvider**: Google Gemini integration
- **OpenAIProvider**: OpenAI GPT integration
- **AnthropicProvider**: Claude integration
- **ProviderFactory**: Selects provider based on environment variables

### 4. Vector Database Adapters (`vectordb/`)
- **BaseVectorDB**: Abstract interface for vector databases
- **ChromaDBAdapter**: ChromaDB integration
- **PineconeAdapter**: Pinecone integration
- **WeaviateAdapter**: Weaviate integration

### 5. Configuration Management (`config/`)
- **EnvironmentLoader**: Loads and validates environment variables
- **ConfigValidator**: Ensures required configurations are present
- **ModelRegistry**: Manages embedding model configurations

## Environment Configuration

The application is configured entirely through environment variables:

### Required Variables
```env
# Vector Database
VECTOR_DB_TYPE=chromadb|pinecone|weaviate
VECTOR_DB_URL=your_database_url
VECTOR_DB_API_KEY=your_api_key (if required)

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Target CLI Tool
CLI_TOOL_NAME=your_cli_tool
CLI_TOOL_PATH=/path/to/cli/tool
```

### LLM Provider (Choose One)
```env
# Google Gemini
GEMINI_API_KEY=your_gemini_api_key

# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### Optional Variables
```env
# Documentation Paths
DOCS_PATH=./docs
README_PATH=./README.md
ADDITIONAL_DOCS=cloud.md,api.md

# Performance Tuning
MAX_CONTEXT_LENGTH=4000
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7

# Execution Settings
DRY_RUN=false
CONFIRM_BEFORE_EXECUTION=true
```

## Application Modes

### 1. Indexing Mode
```bash
easyprompt index --source /path/to/docs --rebuild
```
- Processes documentation files
- Generates and stores embeddings
- Updates vector database

### 2. Interactive Mode
```bash
easyprompt chat
```
- Starts interactive session
- Processes natural language queries
- Generates and executes commands

### 3. Single Query Mode
```bash
easyprompt query "create a new deployment with 3 replicas"
```
- Processes single query
- Returns generated command
- Optionally executes command

## Technology Stack

### Core Dependencies
- **Python 3.9+**: Main programming language
- **FastAPI**: Web framework for API endpoints
- **Typer**: CLI framework
- **Pydantic**: Data validation and settings management

### Vector Processing
- **sentence-transformers**: Embedding generation
- **numpy**: Vector operations
- **faiss-cpu**: Fast similarity search (fallback)

### Database Options
- **ChromaDB**: Local vector database
- **Pinecone**: Cloud vector database
- **Weaviate**: Open-source vector database

### LLM Integration
- **google-generativeai**: Gemini integration
- **openai**: OpenAI GPT integration
- **anthropic**: Claude integration

## Security Considerations

### API Key Management
- All API keys stored in environment variables
- No hardcoded credentials in source code
- Support for external secret management systems

### Command Execution Safety
- Command validation before execution
- Dry-run mode for testing
- User confirmation for destructive operations
- Execution logging and audit trail

### Data Privacy
- Local processing of sensitive documentation
- Optional cloud vector database support
- Configurable data retention policies

## Deployment Options

### 1. Standalone Application
- Self-contained Python application
- Local vector database (ChromaDB)
- Suitable for individual developers

### 2. Team Deployment
- Shared vector database
- Centralized documentation indexing
- Multiple user support

### 3. Enterprise Integration
- API-first architecture
- SSO integration capabilities
- Audit logging and compliance features

## Development Workflow

### Setup
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure environment variables
4. Index documentation: `easyprompt index`
5. Start interactive session: `easyprompt chat`

### Testing
- Unit tests for each module
- Integration tests for end-to-end workflows
- Mock LLM responses for consistent testing
- Vector database testing with sample data

### Continuous Integration
- Automated testing on multiple Python versions
- Code quality checks (flake8, black, mypy)
- Security scanning for dependencies
- Documentation generation and validation

## Future Enhancements

### Advanced Features
- Multi-modal input support (voice, images)
- Command history and learning
- Custom command templates
- Integration with popular CLI tools

### Performance Optimizations
- Caching layer for frequent queries
- Parallel processing for large document sets
- Incremental indexing for document updates
- Query optimization and result ranking improvements

### Enterprise Features
- Role-based access control
- Command approval workflows
- Integration with ticketing systems
- Advanced analytics and reporting