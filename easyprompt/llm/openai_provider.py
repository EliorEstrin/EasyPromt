"""OpenAI LLM provider."""

import logging
from typing import List, Optional
import openai
from .base_provider import BaseLLMProvider, LLMResponse, Message

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", **kwargs):
        super().__init__(api_key, **kwargs)
        self.model_name = model
        self.client = None

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        try:
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI provider with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
            raise

    async def generate_command(
        self,
        user_query: str,
        context: str,
        cli_tool_name: str,
        **kwargs
    ) -> LLMResponse:
        """Generate a CLI command using OpenAI."""
        if self.client is None:
            await self.initialize()

        messages = self.create_command_generation_prompt(
            user_query, context, cli_tool_name
        )

        try:
            # Convert to OpenAI format
            openai_messages = self._messages_to_openai_format(messages)

            # Generate response
            response = await self._generate_with_retry(openai_messages, **kwargs)

            # Extract content
            content = response.choices[0].message.content

            # Clean the command
            command = self._clean_command_response(content)

            return LLMResponse(
                content=command,
                model=response.model,
                usage=self._extract_usage(response),
                metadata={"provider": "openai"}
            )

        except Exception as e:
            logger.error(f"Error generating command with OpenAI: {e}")
            raise

    async def chat_completion(
        self,
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """Generate a chat completion using OpenAI."""
        if self.client is None:
            await self.initialize()

        try:
            # Convert to OpenAI format
            openai_messages = self._messages_to_openai_format(messages)

            # Generate response
            response = await self._generate_with_retry(openai_messages, **kwargs)

            # Extract content
            content = response.choices[0].message.content

            return LLMResponse(
                content=content,
                model=response.model,
                usage=self._extract_usage(response),
                metadata={"provider": "openai"}
            )

        except Exception as e:
            logger.error(f"Error with OpenAI chat completion: {e}")
            raise

    async def _generate_with_retry(self, messages: List[dict], max_retries: int = 3, **kwargs):
        """Generate with retry logic."""
        import asyncio

        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=kwargs.get("temperature", 0.1),
                    max_tokens=kwargs.get("max_tokens", 1024),
                    top_p=kwargs.get("top_p", 0.8),
                    frequency_penalty=kwargs.get("frequency_penalty", 0),
                    presence_penalty=kwargs.get("presence_penalty", 0),
                )
                return response

            except openai.RateLimitError as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}")
                await asyncio.sleep(wait_time)

            except openai.APIError as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"OpenAI API error on attempt {attempt + 1}: {e}")
                await asyncio.sleep(1)

            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"OpenAI request attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2 ** attempt)

    def _messages_to_openai_format(self, messages: List[Message]) -> List[dict]:
        """Convert messages to OpenAI format."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

    def _extract_usage(self, response) -> Optional[dict]:
        """Extract usage information from OpenAI response."""
        try:
            if hasattr(response, 'usage') and response.usage:
                return {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
        except Exception:
            pass
        return None

    async def is_available(self) -> bool:
        """Check if OpenAI provider is available."""
        try:
            if self.client is None:
                await self.initialize()

            # Test with a simple request
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return bool(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"OpenAI availability check failed: {e}")
            return False

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "openai"

    @property
    def default_model(self) -> str:
        """Get the default model."""
        return "gpt-3.5-turbo"

    def set_model(self, model_name: str) -> None:
        """Set the model to use."""
        self.model_name = model_name

    def get_available_models(self) -> List[str]:
        """Get list of available OpenAI models."""
        return [
            "gpt-4",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        ]