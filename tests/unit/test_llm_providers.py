"""Tests for LLM providers."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from easyprompt.llm.base_provider import BaseLLMProvider, LLMResponse, Message
from easyprompt.llm.gemini_provider import GeminiProvider
from easyprompt.llm.openai_provider import OpenAIProvider
from easyprompt.llm.anthropic_provider import AnthropicProvider
from easyprompt.llm.provider_factory import ProviderFactory


class TestBaseLLMProvider:
    """Test BaseLLMProvider abstract class."""

    class ConcreteProvider(BaseLLMProvider):
        """Concrete implementation for testing."""

        async def generate_command(self, user_query, context, cli_tool_name, **kwargs):
            return LLMResponse(content="test command", model="test-model")

        async def chat_completion(self, messages, **kwargs):
            return LLMResponse(content="test response", model="test-model")

        async def is_available(self):
            return True

        @property
        def provider_name(self):
            return "test"

        @property
        def default_model(self):
            return "test-model"

    @pytest.fixture
    def provider(self):
        """Create a concrete provider instance."""
        return self.ConcreteProvider("test-api-key")

    def test_create_command_generation_prompt(self, provider):
        """Test creating command generation prompt."""
        messages = provider.create_command_generation_prompt(
            user_query="list files",
            context="Use ls command to list files",
            cli_tool_name="bash"
        )

        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert "bash" in messages[0].content
        assert "list files" in messages[1].content

    def test_create_explanation_prompt(self, provider):
        """Test creating explanation prompt."""
        messages = provider.create_explanation_prompt(
            command="ls -la",
            user_query="list files",
            context="List files command"
        )

        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert "ls -la" in messages[1].content

    @pytest.mark.asyncio
    async def test_generate_command_with_explanation(self, provider):
        """Test generating command with explanation."""
        with patch.object(provider, 'generate_command') as mock_gen_cmd, \
             patch.object(provider, 'chat_completion') as mock_chat:

            mock_gen_cmd.return_value = LLMResponse(content="ls -la", model="test")
            mock_chat.return_value = LLMResponse(content="Lists files with details", model="test")

            command, explanation = await provider.generate_command_with_explanation(
                user_query="list files",
                context="Use ls command",
                cli_tool_name="bash"
            )

            assert command == "ls -la"
            assert explanation == "Lists files with details"

    def test_clean_command_response(self, provider):
        """Test cleaning command responses."""
        test_cases = [
            ("$ ls -la", "ls -la"),
            ("> git status", "git status"),
            ("```bash\nls -la\n```", "ls -la"),
            ("`git status`", "git status"),
            ("  ls -la  ", "ls -la"),
            ("ls\n-la", "ls -la")
        ]

        for input_cmd, expected in test_cases:
            cleaned = provider._clean_command_response(input_cmd)
            assert cleaned == expected

    @pytest.mark.asyncio
    async def test_validate_response(self, provider):
        """Test response validation."""
        assert await provider.validate_response("ls -la", "bash") is True
        assert await provider.validate_response("UNCLEAR_REQUEST", "bash") is True
        assert await provider.validate_response("", "bash") is False


class TestGeminiProvider:
    """Test GeminiProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a GeminiProvider instance."""
        return GeminiProvider("test-api-key")

    @pytest.mark.asyncio
    async def test_initialize(self, provider):
        """Test initializing Gemini provider."""
        with patch('easyprompt.llm.gemini_provider.genai') as mock_genai:
            mock_model = Mock()
            mock_genai.GenerativeModel.return_value = mock_model

            await provider.initialize()

            mock_genai.configure.assert_called_once_with(api_key="test-api-key")
            assert provider.client == mock_model

    @pytest.mark.asyncio
    async def test_generate_command(self, provider):
        """Test generating command with Gemini."""
        mock_response = Mock()
        mock_response.text = "ls -la"

        with patch.object(provider, '_generate_with_retry', return_value=mock_response):
            await provider.initialize()

            response = await provider.generate_command(
                user_query="list files",
                context="Use ls command",
                cli_tool_name="bash"
            )

            assert response.content == "ls -la"
            assert response.model == "gemini-pro"

    @pytest.mark.asyncio
    async def test_is_available(self, provider):
        """Test checking if Gemini is available."""
        mock_response = Mock()
        mock_response.text = "Hello"

        with patch.object(provider, 'initialize'), \
             patch.object(provider.client, 'generate_content', return_value=mock_response):
            provider.client = Mock()

            available = await provider.is_available()

            assert available is True

    def test_messages_to_prompt(self, provider):
        """Test converting messages to Gemini prompt format."""
        messages = [
            Message(role="system", content="You are a helpful assistant"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there")
        ]

        prompt = provider._messages_to_prompt(messages)

        assert "System: You are a helpful assistant" in prompt
        assert "User: Hello" in prompt
        assert "Assistant: Hi there" in prompt


class TestOpenAIProvider:
    """Test OpenAIProvider class."""

    @pytest.fixture
    def provider(self):
        """Create an OpenAIProvider instance."""
        return OpenAIProvider("test-api-key")

    @pytest.mark.asyncio
    async def test_initialize(self, provider):
        """Test initializing OpenAI provider."""
        with patch('easyprompt.llm.openai_provider.openai.AsyncOpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            await provider.initialize()

            mock_openai.assert_called_once_with(api_key="test-api-key")
            assert provider.client == mock_client

    @pytest.mark.asyncio
    async def test_generate_command(self, provider):
        """Test generating command with OpenAI."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "ls -la"
        mock_response.model = "gpt-3.5-turbo"

        with patch.object(provider, '_generate_with_retry', return_value=mock_response):
            await provider.initialize()

            response = await provider.generate_command(
                user_query="list files",
                context="Use ls command",
                cli_tool_name="bash"
            )

            assert response.content == "ls -la"
            assert response.model == "gpt-3.5-turbo"

    def test_messages_to_openai_format(self, provider):
        """Test converting messages to OpenAI format."""
        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hello")
        ]

        openai_messages = provider._messages_to_openai_format(messages)

        assert len(openai_messages) == 2
        assert openai_messages[0]["role"] == "system"
        assert openai_messages[0]["content"] == "You are helpful"
        assert openai_messages[1]["role"] == "user"
        assert openai_messages[1]["content"] == "Hello"

    def test_get_available_models(self, provider):
        """Test getting available models."""
        models = provider.get_available_models()

        assert isinstance(models, list)
        assert "gpt-3.5-turbo" in models
        assert "gpt-4" in models


class TestAnthropicProvider:
    """Test AnthropicProvider class."""

    @pytest.fixture
    def provider(self):
        """Create an AnthropicProvider instance."""
        return AnthropicProvider("test-api-key")

    @pytest.mark.asyncio
    async def test_initialize(self, provider):
        """Test initializing Anthropic provider."""
        with patch('easyprompt.llm.anthropic_provider.anthropic.AsyncAnthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            await provider.initialize()

            mock_anthropic.assert_called_once_with(api_key="test-api-key")
            assert provider.client == mock_client

    def test_messages_to_anthropic_format(self, provider):
        """Test converting messages to Anthropic format."""
        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi")
        ]

        system_message, anthropic_messages = provider._messages_to_anthropic_format(messages)

        assert system_message == "You are helpful"
        assert len(anthropic_messages) == 2
        assert anthropic_messages[0]["role"] == "user"
        assert anthropic_messages[1]["role"] == "assistant"

    def test_get_available_models(self, provider):
        """Test getting available models."""
        models = provider.get_available_models()

        assert isinstance(models, list)
        assert "claude-3-sonnet-20240229" in models
        assert "claude-3-opus-20240229" in models


class TestProviderFactory:
    """Test ProviderFactory class."""

    @pytest.fixture
    def factory(self, sample_settings):
        """Create a ProviderFactory instance."""
        return ProviderFactory(sample_settings)

    @pytest.mark.asyncio
    async def test_get_provider_gemini(self, factory):
        """Test getting Gemini provider."""
        with patch('easyprompt.llm.gemini_provider.GeminiProvider') as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider_class.return_value = mock_provider

            provider = await factory.get_provider("gemini")

            assert provider == mock_provider
            mock_provider.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_provider_default(self, factory):
        """Test getting default provider."""
        with patch('easyprompt.llm.gemini_provider.GeminiProvider') as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider_class.return_value = mock_provider

            provider = await factory.get_provider()  # No provider specified

            assert provider == mock_provider

    @pytest.mark.asyncio
    async def test_get_provider_unknown(self, factory):
        """Test getting unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider: unknown"):
            await factory.get_provider("unknown")

    @pytest.mark.asyncio
    async def test_get_provider_no_api_key(self, factory):
        """Test getting provider without API key."""
        factory.settings.gemini_api_key = None

        with pytest.raises(ValueError, match="Gemini API key not configured"):
            await factory.get_provider("gemini")

    @pytest.mark.asyncio
    async def test_test_provider(self, factory):
        """Test testing a provider."""
        mock_provider = AsyncMock()
        mock_provider.is_available.return_value = True

        with patch.object(factory, 'get_provider', return_value=mock_provider):
            result = await factory.test_provider("gemini")

            assert result is True
            mock_provider.is_available.assert_called_once()

    @pytest.mark.asyncio
    async def test_test_all_providers(self, factory):
        """Test testing all providers."""
        with patch.object(factory, 'test_provider') as mock_test:
            mock_test.return_value = True

            results = await factory.test_all_providers()

            assert "gemini" in results
            assert results["gemini"] is True

    @pytest.mark.asyncio
    async def test_get_best_provider(self, factory):
        """Test getting the best available provider."""
        mock_provider = AsyncMock()

        with patch.object(factory, 'test_provider', return_value=True), \
             patch.object(factory, 'get_provider', return_value=mock_provider):

            best_provider = await factory.get_best_provider()

            assert best_provider == mock_provider

    def test_get_provider_info(self, factory):
        """Test getting provider info."""
        info = factory.get_provider_info("gemini")

        assert info["name"] == "gemini"
        assert "default_model" in info
        assert "configured" in info

    @pytest.mark.asyncio
    async def test_close_all(self, factory):
        """Test closing all providers."""
        mock_provider = Mock()
        mock_provider.close = AsyncMock()
        factory._providers["test"] = mock_provider

        await factory.close_all()

        # Should clear cache
        assert len(factory._providers) == 0