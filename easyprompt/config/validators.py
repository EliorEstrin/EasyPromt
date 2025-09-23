"""Configuration validation utilities."""

import os
from pathlib import Path
from typing import List, Optional
from .settings import Settings


class ConfigValidator:
    """Validates application configuration."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.errors: List[str] = []

    def validate_all(self) -> bool:
        """Validate all configuration settings."""
        self.errors.clear()

        self._validate_vector_db()
        self._validate_embedding_model()
        self._validate_project_setup()
        self._validate_documentation_paths()
        self._validate_llm_providers()

        return len(self.errors) == 0

    def _validate_vector_db(self) -> None:
        """Validate vector database configuration."""
        supported_dbs = ["chromadb", "pinecone", "weaviate"]

        if self.settings.vector_db_type not in supported_dbs:
            self.errors.append(
                f"Unsupported vector database type: {self.settings.vector_db_type}. "
                f"Supported types: {', '.join(supported_dbs)}"
            )

        if self.settings.vector_db_type == "pinecone":
            if not self.settings.pinecone_api_key:
                self.errors.append("Pinecone API key is required when using Pinecone")
            if not self.settings.pinecone_environment:
                self.errors.append("Pinecone environment is required when using Pinecone")

    def _validate_embedding_model(self) -> None:
        """Validate embedding model configuration."""
        if not self.settings.embedding_model:
            self.errors.append("Embedding model must be specified")

        if self.settings.embedding_dimension <= 0:
            self.errors.append("Embedding dimension must be positive")

    def _validate_project_setup(self) -> None:
        """Validate project setup configuration."""
        # Project name and domain are optional
        pass

    def _validate_documentation_paths(self) -> None:
        """Validate documentation paths."""
        # Validate each documentation path
        for path in self.settings.docs_path_list:
            if path and not Path(path).exists():
                self.errors.append(f"Documentation path does not exist: {path}")

        # Validate supported file types
        if not self.settings.supported_file_types:
            self.errors.append("At least one supported file type must be specified")

    def _validate_llm_providers(self) -> None:
        """Validate LLM provider configuration."""
        if not self.settings.available_llm_providers:
            self.errors.append(
                "At least one LLM provider API key must be configured "
                "(GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY)"
            )

    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors."""
        return self.errors.copy()

    def create_missing_directories(self) -> None:
        """Create missing directories that can be created."""
        directories = [
            self.settings.docs_path,
        ]

        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_environment() -> Optional[str]:
        """Validate basic environment setup."""
        # Check if .env file exists
        env_file = Path(".env")
        if not env_file.exists():
            example_file = Path(".env.example")
            if example_file.exists():
                return (
                    "No .env file found. Please copy .env.example to .env "
                    "and configure your settings."
                )
            else:
                return "No .env file found. Please create one with your configuration."

        return None