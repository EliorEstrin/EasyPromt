"""Application settings and configuration management."""

from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Vector Database Settings
    vector_db_type: str = Field(default="chromadb", description="Type of vector database")
    vector_db_url: str = Field(default="./data/chroma.db", description="Vector database URL")
    vector_db_api_key: Optional[str] = Field(default=None, description="Vector database API key")

    # Embedding Settings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name"
    )
    embedding_dimension: int = Field(default=384, description="Embedding vector dimension")

    # Project Settings
    project_name: str = Field(default="", description="Name of the project")
    project_domain: str = Field(default="", description="Domain/category of the project")

    # Legacy CLI Tool (kept for compatibility)
    cli_tool_name: str = Field(default="bash", description="CLI tool name for command generation")

    # Documentation Settings
    docs_path: str = Field(default="./docs", description="Path to documentation directory(s) - semicolon separated for multiple")
    supported_file_types: str = Field(default="md,txt,pdf", description="Supported file extensions (comma-separated)")

    # Index Settings
    chunk_size: int = Field(default=1000, description="Size of text chunks for indexing")
    chunk_overlap: int = Field(default=200, description="Overlap between consecutive chunks")
    chunking_strategy: str = Field(default="recursive", description="Text chunking strategy")
    index_storage_path: str = Field(default="./data/index", description="Path to store index data")
    chunked_docs_path: str = Field(default="./data/chunked_docs", description="Path to store chunked documents")
    enable_metadata_extraction: bool = Field(default=True, description="Extract metadata from documents")
    min_chunk_size: int = Field(default=100, description="Minimum size for a valid chunk")

    # LLM Provider API Keys
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")

    # Performance Settings
    max_context_length: int = Field(default=4000, description="Maximum context length for LLM")
    top_k_results: int = Field(default=5, description="Number of top results to retrieve")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity threshold")

    # Execution Settings
    dry_run: bool = Field(default=False, description="Enable dry run mode")
    confirm_before_execution: bool = Field(default=True, description="Confirm before executing commands")
    log_level: str = Field(default="INFO", description="Logging level")

    # Optional: Pinecone Settings
    pinecone_api_key: Optional[str] = Field(default=None, description="Pinecone API key")
    pinecone_environment: Optional[str] = Field(default=None, description="Pinecone environment")
    pinecone_index_name: str = Field(default="easyprompt-index", description="Pinecone index name")

    # Optional: Weaviate Settings
    weaviate_url: str = Field(default="http://localhost:8080", description="Weaviate URL")
    weaviate_api_key: Optional[str] = Field(default=None, description="Weaviate API key")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def supported_file_types_list(self) -> List[str]:
        """Get supported file types as a list."""
        if not self.supported_file_types:
            return []
        return [ext.strip() for ext in self.supported_file_types.split(",") if ext.strip()]

    @property
    def docs_path_list(self) -> List[str]:
        """Get documentation paths as a list."""
        if not self.docs_path:
            return []
        return [path.strip() for path in self.docs_path.split(";") if path.strip()]

    @property
    def available_llm_providers(self) -> List[str]:
        """Get list of available LLM providers based on configured API keys."""
        providers = []
        if self.gemini_api_key:
            providers.append("gemini")
        if self.openai_api_key:
            providers.append("openai")
        if self.anthropic_api_key:
            providers.append("anthropic")
        return providers

    @property
    def primary_llm_provider(self) -> Optional[str]:
        """Get the primary LLM provider (first available)."""
        providers = self.available_llm_providers
        return providers[0] if providers else None