"""Tests for configuration module."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from easyprompt.config import Settings, ConfigValidator


class TestSettings:
    """Test Settings class."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()

        assert settings.vector_db_type == "chromadb"
        assert settings.vector_db_url == "./data/chroma.db"
        assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert settings.embedding_dimension == 384
        assert settings.max_context_length == 4000
        assert settings.top_k_results == 5
        assert settings.similarity_threshold == 0.7
        assert settings.dry_run is False
        assert settings.confirm_before_execution is True

    def test_settings_from_env(self):
        """Test settings loaded from environment variables."""
        env_vars = {
            "VECTOR_DB_TYPE": "pinecone",
            "VECTOR_DB_URL": "custom-url",
            "CLI_TOOL_NAME": "kubectl",
            "GEMINI_API_KEY": "test-key",
            "TOP_K_RESULTS": "10",
            "DRY_RUN": "true"
        }

        with patch.dict('os.environ', env_vars):
            settings = Settings()

            assert settings.vector_db_type == "pinecone"
            assert settings.vector_db_url == "custom-url"
            assert settings.cli_tool_name == "kubectl"
            assert settings.gemini_api_key == "test-key"
            assert settings.top_k_results == 10
            assert settings.dry_run is True

    def test_additional_docs_list(self):
        """Test additional_docs_list property."""
        settings = Settings(additional_docs="doc1.md,doc2.md, doc3.md")

        docs_list = settings.additional_docs_list
        assert docs_list == ["doc1.md", "doc2.md", "doc3.md"]

    def test_additional_docs_list_empty(self):
        """Test additional_docs_list with empty string."""
        settings = Settings(additional_docs="")

        docs_list = settings.additional_docs_list
        assert docs_list == []

    def test_available_llm_providers(self):
        """Test available_llm_providers property."""
        settings = Settings(
            gemini_api_key="key1",
            openai_api_key="key2",
            anthropic_api_key=None
        )

        providers = settings.available_llm_providers
        assert "gemini" in providers
        assert "openai" in providers
        assert "anthropic" not in providers

    def test_primary_llm_provider(self):
        """Test primary_llm_provider property."""
        settings = Settings(gemini_api_key="key1")
        assert settings.primary_llm_provider == "gemini"

        settings = Settings()
        assert settings.primary_llm_provider is None


class TestConfigValidator:
    """Test ConfigValidator class."""

    def test_validate_all_success(self, sample_settings, temp_dir):
        """Test successful validation."""
        # Create required files
        (temp_dir / "docs").mkdir()
        (temp_dir / "README.md").touch()

        validator = ConfigValidator(sample_settings)
        assert validator.validate_all() is True
        assert len(validator.get_validation_errors()) == 0

    def test_validate_vector_db_unsupported_type(self, sample_settings):
        """Test validation with unsupported vector DB type."""
        sample_settings.vector_db_type = "unsupported"

        validator = ConfigValidator(sample_settings)
        assert validator.validate_all() is False

        errors = validator.get_validation_errors()
        assert any("Unsupported vector database type" in error for error in errors)

    def test_validate_vector_db_pinecone_missing_config(self, sample_settings):
        """Test Pinecone validation with missing configuration."""
        sample_settings.vector_db_type = "pinecone"
        sample_settings.pinecone_api_key = None

        validator = ConfigValidator(sample_settings)
        assert validator.validate_all() is False

        errors = validator.get_validation_errors()
        assert any("Pinecone API key is required" in error for error in errors)

    def test_validate_embedding_model_empty(self, sample_settings):
        """Test validation with empty embedding model."""
        sample_settings.embedding_model = ""

        validator = ConfigValidator(sample_settings)
        assert validator.validate_all() is False

        errors = validator.get_validation_errors()
        assert any("Embedding model must be specified" in error for error in errors)

    def test_validate_embedding_dimension_invalid(self, sample_settings):
        """Test validation with invalid embedding dimension."""
        sample_settings.embedding_dimension = -1

        validator = ConfigValidator(sample_settings)
        assert validator.validate_all() is False

        errors = validator.get_validation_errors()
        assert any("Embedding dimension must be positive" in error for error in errors)

    def test_validate_cli_tool_empty_name(self, sample_settings):
        """Test validation with empty CLI tool name."""
        sample_settings.cli_tool_name = ""

        validator = ConfigValidator(sample_settings)
        assert validator.validate_all() is False

        errors = validator.get_validation_errors()
        assert any("CLI tool name must be specified" in error for error in errors)

    def test_validate_cli_tool_invalid_path(self, sample_settings):
        """Test validation with invalid CLI tool path."""
        sample_settings.cli_tool_path = "/nonexistent/path"

        validator = ConfigValidator(sample_settings)
        assert validator.validate_all() is False

        errors = validator.get_validation_errors()
        assert any("CLI tool path does not exist" in error for error in errors)

    def test_validate_documentation_paths_missing(self, sample_settings):
        """Test validation with missing documentation paths."""
        sample_settings.docs_path = "/nonexistent/docs"
        sample_settings.readme_path = "/nonexistent/readme.md"

        validator = ConfigValidator(sample_settings)
        assert validator.validate_all() is False

        errors = validator.get_validation_errors()
        assert any("docs_path does not exist" in error for error in errors)
        assert any("readme_path does not exist" in error for error in errors)

    def test_validate_llm_providers_none_configured(self, sample_settings):
        """Test validation with no LLM providers configured."""
        sample_settings.gemini_api_key = None
        sample_settings.openai_api_key = None
        sample_settings.anthropic_api_key = None

        validator = ConfigValidator(sample_settings)
        assert validator.validate_all() is False

        errors = validator.get_validation_errors()
        assert any("at least one LLM provider API key must be configured" in error for error in errors)

    def test_create_missing_directories(self, sample_settings, temp_dir):
        """Test creating missing directories."""
        missing_dir = temp_dir / "missing_docs"
        sample_settings.docs_path = str(missing_dir)

        validator = ConfigValidator(sample_settings)
        validator.create_missing_directories()

        assert missing_dir.exists()

    def test_validate_environment_no_env_file(self):
        """Test environment validation with no .env file."""
        with patch('pathlib.Path.exists', return_value=False):
            result = ConfigValidator.validate_environment()
            assert result is not None
            assert ".env file" in result

    def test_validate_environment_with_example(self):
        """Test environment validation with .env.example but no .env."""
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.side_effect = lambda: mock_exists.call_count > 1  # .env.example exists

            result = ConfigValidator.validate_environment()
            assert result is not None
            assert "copy .env.example" in result

    def test_validate_environment_success(self):
        """Test successful environment validation."""
        with patch('pathlib.Path.exists', return_value=True):
            result = ConfigValidator.validate_environment()
            assert result is None