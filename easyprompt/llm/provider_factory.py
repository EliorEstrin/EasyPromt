"""Factory for creating LLM providers."""

import logging
from typing import Optional, List
from ..config import Settings
from .base_provider import BaseLLMProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider

logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory for creating and managing LLM providers."""

    PROVIDER_CLASSES = {
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }

    def __init__(self, settings: Settings):
        self.settings = settings
        self._providers = {}

    async def get_provider(self, provider_name: Optional[str] = None) -> BaseLLMProvider:
        """Get an LLM provider instance."""
        if provider_name is None:
            provider_name = self.settings.primary_llm_provider

        if not provider_name:
            raise ValueError("No LLM provider configured. Please set at least one API key.")

        if provider_name not in self.PROVIDER_CLASSES:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available providers: {list(self.PROVIDER_CLASSES.keys())}"
            )

        # Return cached provider if available
        if provider_name in self._providers:
            return self._providers[provider_name]

        # Create new provider
        provider = await self._create_provider(provider_name)
        self._providers[provider_name] = provider
        return provider

    async def _create_provider(self, provider_name: str) -> BaseLLMProvider:
        """Create a new provider instance."""
        provider_class = self.PROVIDER_CLASSES[provider_name]

        if provider_name == "gemini":
            if not self.settings.gemini_api_key:
                raise ValueError("Gemini API key not configured")
            provider = provider_class(
                api_key=self.settings.gemini_api_key,
                model="gemini-pro"
            )

        elif provider_name == "openai":
            if not self.settings.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            provider = provider_class(
                api_key=self.settings.openai_api_key,
                model="gpt-3.5-turbo"
            )

        elif provider_name == "anthropic":
            if not self.settings.anthropic_api_key:
                raise ValueError("Anthropic API key not configured")
            provider = provider_class(
                api_key=self.settings.anthropic_api_key,
                model="claude-3-sonnet-20240229"
            )

        else:
            raise ValueError(f"Unknown provider: {provider_name}")

        # Initialize the provider
        await provider.initialize()
        logger.info(f"Created and initialized {provider_name} provider")

        return provider

    async def get_available_providers(self) -> List[str]:
        """Get list of available providers based on configuration."""
        return self.settings.available_llm_providers

    async def test_provider(self, provider_name: str) -> bool:
        """Test if a provider is working correctly."""
        try:
            provider = await self.get_provider(provider_name)
            return await provider.is_available()
        except Exception as e:
            logger.error(f"Failed to test provider {provider_name}: {e}")
            return False

    async def test_all_providers(self) -> dict[str, bool]:
        """Test all configured providers."""
        results = {}
        available_providers = await self.get_available_providers()

        for provider_name in available_providers:
            results[provider_name] = await self.test_provider(provider_name)

        return results

    async def get_best_provider(self) -> Optional[BaseLLMProvider]:
        """Get the best available provider based on testing and preferences."""
        # Priority order (can be made configurable)
        priority_order = ["anthropic", "openai", "gemini"]

        available_providers = await self.get_available_providers()

        # Try providers in priority order
        for provider_name in priority_order:
            if provider_name in available_providers:
                if await self.test_provider(provider_name):
                    return await self.get_provider(provider_name)

        # Fallback to any working provider
        for provider_name in available_providers:
            if await self.test_provider(provider_name):
                return await self.get_provider(provider_name)

        return None

    def clear_cache(self) -> None:
        """Clear the provider cache."""
        self._providers.clear()
        logger.info("Provider cache cleared")

    async def close_all(self) -> None:
        """Close all provider connections."""
        for provider in self._providers.values():
            try:
                if hasattr(provider, 'close'):
                    await provider.close()
            except Exception as e:
                logger.warning(f"Error closing provider {provider.provider_name}: {e}")

        self.clear_cache()

    def get_provider_info(self, provider_name: str) -> dict:
        """Get information about a provider."""
        if provider_name not in self.PROVIDER_CLASSES:
            raise ValueError(f"Unknown provider: {provider_name}")

        provider_class = self.PROVIDER_CLASSES[provider_name]
        temp_provider = provider_class("dummy_key")

        return {
            "name": provider_name,
            "class": provider_class.__name__,
            "default_model": temp_provider.default_model,
            "available_models": getattr(temp_provider, 'get_available_models', lambda: [])(),
            "configured": provider_name in self.settings.available_llm_providers,
        }