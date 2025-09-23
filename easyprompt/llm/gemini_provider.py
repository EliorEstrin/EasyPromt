"""Google Gemini LLM provider."""

import logging
from typing import List, Optional
import google.generativeai as genai
from .base_provider import BaseLLMProvider, LLMResponse, Message

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider."""

    def __init__(self, api_key: str, model: str = "gemini-pro", **kwargs):
        super().__init__(api_key, **kwargs)
        self.model_name = model
        self.client = None

    async def initialize(self) -> None:
        """Initialize the Gemini client."""
        try:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)
            logger.info(f"Initialized Gemini provider with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini provider: {e}")
            raise

    async def generate_command(
        self,
        user_query: str,
        context: str,
        **kwargs
    ) -> LLMResponse:
        """Generate a CLI command using Gemini."""
        if self.client is None:
            await self.initialize()

        messages = self.create_command_generation_prompt(
            user_query, context
        )

        try:
            # Convert messages to Gemini format
            prompt = self._messages_to_prompt(messages)

            # Generate response
            response = await self._generate_with_retry(prompt, **kwargs)

            # Clean the command
            command = self._clean_command_response(response.text)

            return LLMResponse(
                content=command,
                model=self.model_name,
                usage=self._extract_usage(response),
                metadata={"provider": "gemini"}
            )

        except Exception as e:
            logger.error(f"Error generating command with Gemini: {e}")
            raise

    async def chat_completion(
        self,
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """Generate a chat completion using Gemini."""
        if self.client is None:
            await self.initialize()

        try:
            # Convert messages to Gemini format
            prompt = self._messages_to_prompt(messages)

            # Generate response
            response = await self._generate_with_retry(prompt, **kwargs)

            return LLMResponse(
                content=response.text,
                model=self.model_name,
                usage=self._extract_usage(response),
                metadata={"provider": "gemini"}
            )

        except Exception as e:
            logger.error(f"Error with Gemini chat completion: {e}")
            raise

    async def _generate_with_retry(self, prompt: str, max_retries: int = 3, **kwargs):
        """Generate with retry logic."""
        import asyncio

        for attempt in range(max_retries):
            try:
                # Configure generation parameters
                generation_config = {
                    "temperature": kwargs.get("temperature", 0.1),
                    "top_p": kwargs.get("top_p", 0.8),
                    "top_k": kwargs.get("top_k", 40),
                    "max_output_tokens": kwargs.get("max_tokens", 1024),
                }

                # Generate content
                response = self.client.generate_content(
                    prompt,
                    generation_config=generation_config
                )

                return response

            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Gemini API attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    def _messages_to_prompt(self, messages: List[Message]) -> str:
        """Convert messages to a single prompt for Gemini."""
        prompt_parts = []

        for message in messages:
            if message.role == "system":
                prompt_parts.append(f"System: {message.content}")
            elif message.role == "user":
                prompt_parts.append(f"User: {message.content}")
            elif message.role == "assistant":
                prompt_parts.append(f"Assistant: {message.content}")

        return "\n\n".join(prompt_parts)

    def _extract_usage(self, response) -> Optional[dict]:
        """Extract usage information from Gemini response."""
        try:
            if hasattr(response, 'usage_metadata'):
                return {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
                }
        except Exception:
            pass
        return None

    async def is_available(self) -> bool:
        """Check if Gemini provider is available."""
        try:
            if self.client is None:
                await self.initialize()

            # Test with a simple prompt
            test_response = self.client.generate_content("Hello")
            return bool(test_response.text)

        except Exception as e:
            logger.error(f"Gemini availability check failed: {e}")
            return False

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "gemini"

    @property
    def default_model(self) -> str:
        """Get the default model."""
        return "gemini-pro"

    def set_model(self, model_name: str) -> None:
        """Set the model to use."""
        self.model_name = model_name
        if self.client:
            self.client = genai.GenerativeModel(model_name)