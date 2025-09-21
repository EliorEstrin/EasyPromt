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

    # CLI Tool Settings
    cli_tool_name: str = Field(default="", description="Name of the CLI tool")
    cli_tool_path: str = Field(default="", description="Path to the CLI tool executable")

    # Documentation Paths
    docs_path: str = Field(default="./docs", description="Path to documentation directory")
    readme_path: str = Field(default="./README.md", description="Path to README file")
    additional_docs: str = Field(default="", description="Comma-separated list of additional docs")

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
    def additional_docs_list(self) -> List[str]:
        """Get additional docs as a list."""
        if not self.additional_docs:
            return []
        return [doc.strip() for doc in self.additional_docs.split(",") if doc.strip()]

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