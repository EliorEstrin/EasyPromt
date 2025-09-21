"""Anthropic Claude LLM provider."""

import logging
from typing import List, Optional
import anthropic
from .base_provider import BaseLLMProvider, LLMResponse, Message

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229", **kwargs):
        super().__init__(api_key, **kwargs)
        self.model_name = model
        self.client = None

    async def initialize(self) -> None:
        """Initialize the Anthropic client."""
        try:
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
            logger.info(f"Initialized Anthropic provider with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic provider: {e}")
            raise

    async def generate_command(
        self,
        user_query: str,
        context: str,
        cli_tool_name: str,
        **kwargs
    ) -> LLMResponse:
        """Generate a CLI command using Claude."""
        if self.client is None:
            await self.initialize()

        messages = self.create_command_generation_prompt(
            user_query, context, cli_tool_name
        )

        try:
            # Convert to Anthropic format
            system_message, user_messages = self._messages_to_anthropic_format(messages)

            # Generate response
            response = await self._generate_with_retry(
                system_message, user_messages, **kwargs
            )

            # Extract content
            content = response.content[0].text

            # Clean the command
            command = self._clean_command_response(content)

            return LLMResponse(
                content=command,
                model=response.model,
                usage=self._extract_usage(response),
                metadata={"provider": "anthropic"}
            )

        except Exception as e:
            logger.error(f"Error generating command with Anthropic: {e}")
            raise

    async def chat_completion(
        self,
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """Generate a chat completion using Claude."""
        if self.client is None:
            await self.initialize()

        try:
            # Convert to Anthropic format
            system_message, user_messages = self._messages_to_anthropic_format(messages)

            # Generate response
            response = await self._generate_with_retry(
                system_message, user_messages, **kwargs
            )

            # Extract content
            content = response.content[0].text

            return LLMResponse(
                content=content,
                model=response.model,
                usage=self._extract_usage(response),
                metadata={"provider": "anthropic"}
            )

        except Exception as e:
            logger.error(f"Error with Anthropic chat completion: {e}")
            raise

    async def _generate_with_retry(
        self,
        system_message: str,
        messages: List[dict],
        max_retries: int = 3,
        **kwargs
    ):
        """Generate with retry logic."""
        import asyncio

        for attempt in range(max_retries):
            try:
                response = await self.client.messages.create(
                    model=self.model_name,
                    system=system_message,
                    messages=messages,
                    temperature=kwargs.get("temperature", 0.1),
                    max_tokens=kwargs.get("max_tokens", 1024),
                    top_p=kwargs.get("top_p", 0.8),
                )
                return response

            except anthropic.RateLimitError as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}")
                await asyncio.sleep(wait_time)

            except anthropic.APIError as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Anthropic API error on attempt {attempt + 1}: {e}")
                await asyncio.sleep(1)

            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Anthropic request attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2 ** attempt)

    def _messages_to_anthropic_format(self, messages: List[Message]) -> tuple[str, List[dict]]:
        """Convert messages to Anthropic format."""
        system_message = ""
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        return system_message, anthropic_messages

    def _extract_usage(self, response) -> Optional[dict]:
        """Extract usage information from Anthropic response."""
        try:
            if hasattr(response, 'usage') and response.usage:
                return {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
        except Exception:
            pass
        return None

    async def is_available(self) -> bool:
        """Check if Anthropic provider is available."""
        try:
            if self.client is None:
                await self.initialize()

            # Test with a simple request
            response = await self.client.messages.create(
                model=self.model_name,
                system="You are a helpful assistant.",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return bool(response.content[0].text)

        except Exception as e:
            logger.error(f"Anthropic availability check failed: {e}")
            return False

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "anthropic"

    @property
    def default_model(self) -> str:
        """Get the default model."""
        return "claude-3-sonnet-20240229"

    def set_model(self, model_name: str) -> None:
        """Set the model to use."""
        self.model_name = model_name

    def get_available_models(self) -> List[str]:
        """Get list of available Anthropic models."""
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
        ]