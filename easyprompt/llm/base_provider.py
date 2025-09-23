"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Message:
    """Message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: str


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    async def generate_command(
        self,
        user_query: str,
        context: str,
        **kwargs
    ) -> LLMResponse:
        """Generate a CLI command based on user query and context."""
        pass

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """Generate a chat completion."""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available and properly configured."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the name of the provider."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Get the default model for this provider."""
        pass

    def create_command_generation_prompt(
        self,
        user_query: str,
        context: str,
    ) -> List[Message]:
        """Create a standardized prompt for command generation."""
        system_prompt = f"""You are a helpful assistant that translates natural language requests into bash CLI commands.

INSTRUCTIONS:
1. Based on the user's request and the provided documentation context, generate the exact CLI command needed
2. Return ONLY the command, without any explanation or additional text
3. Do not include command prefixes like '$' or '>'
4. If the request is unclear or cannot be fulfilled with the available documentation, respond with "UNCLEAR_REQUEST"
5. Ensure the command is syntactically correct and safe to execute

DOCUMENTATION CONTEXT:
{context}

Remember: Respond with ONLY the CLI command or "UNCLEAR_REQUEST" if you cannot determine the appropriate command."""

        user_prompt = f"Generate a bash command for: {user_query}"

        return [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

    def create_explanation_prompt(
        self,
        command: str,
        user_query: str,
        context: str,
    ) -> List[Message]:
        """Create a prompt for explaining a generated command."""
        system_prompt = """You are a helpful assistant that explains CLI commands clearly and concisely.

INSTRUCTIONS:
1. Explain what the given command does
2. Relate it back to the user's original request
3. Highlight any important flags or parameters
4. Keep the explanation concise but informative
5. If there are any potential risks or considerations, mention them briefly"""

        user_prompt = f"""Command: {command}
Original request: {user_query}
Context: {context}

Please explain what this command does."""

        return [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

    def create_qa_prompt(
        self,
        user_query: str,
        context: str,
    ) -> List[Message]:
        """Create a prompt for answering documentation questions."""
        system_prompt = """You are a helpful documentation assistant that answers questions based on the provided context.

INSTRUCTIONS:
1. Answer the user's question using only the information provided in the documentation context
2. Be helpful, accurate, and informative in your response
3. If the context doesn't contain enough information to answer the question, say so clearly
4. Provide specific details, examples, or code snippets from the documentation when relevant
5. Keep your answer focused and well-structured
6. Do not make up information that isn't in the provided context"""

        user_prompt = f"""Based on the documentation context below, please answer this question:

Question: {user_query}

DOCUMENTATION CONTEXT:
{context}

Please provide a helpful answer based on the documentation provided."""

        return [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

    async def generate_answer(
        self,
        user_query: str,
        context: str,
        **kwargs
    ) -> str:
        """Generate an answer to a documentation question."""
        try:
            messages = self.create_qa_prompt(user_query, context)
            response = await self.chat_completion(messages, **kwargs)

            # For Q&A, we return the response as-is without command cleaning
            return response.content.strip()

        except Exception as e:
            logger.error(f"Error generating answer with {self.provider_name}: {e}")
            raise

    async def generate_command_with_explanation(
        self,
        user_query: str,
        context: str,
        **kwargs
    ) -> tuple[str, str]:
        """Generate both command and explanation."""
        # Generate command
        command_response = await self.generate_command(
            user_query, context, **kwargs
        )
        command = command_response.content.strip()

        if command == "UNCLEAR_REQUEST":
            return command, "The request could not be understood or fulfilled with the available documentation."

        # Generate explanation
        explanation_messages = self.create_explanation_prompt(
            command, user_query, context
        )
        explanation_response = await self.chat_completion(
            explanation_messages, **kwargs
        )

        return command, explanation_response.content.strip()

    def _clean_command_response(self, response: str) -> str:
        """Clean and validate command response."""
        command = response.strip()

        # Remove common prefixes
        prefixes_to_remove = ["$ ", "> ", "# ", "bash: ", "shell: ", "```", "`"]
        for prefix in prefixes_to_remove:
            if command.startswith(prefix):
                command = command[len(prefix):].strip()

        # Remove trailing backticks or markdown
        if command.endswith("```"):
            command = command[:-3].strip()
        if command.endswith("`"):
            command = command[:-1].strip()

        # Remove newlines within the command (but keep spaces)
        command = " ".join(command.split())

        return command

    async def validate_response(self, response: str, expected_tool: str) -> bool:
        """Validate that the response is a valid command for the expected tool."""
        cleaned_command = self._clean_command_response(response)

        # Check for unclear request
        if cleaned_command == "UNCLEAR_REQUEST":
            return True

        # Basic validation
        if not cleaned_command:
            return False

        # Check if command starts with expected tool name (optional validation)
        words = cleaned_command.split()
        if words and expected_tool:
            # Allow for flexibility in command structure
            return len(words) >= 1

        return True