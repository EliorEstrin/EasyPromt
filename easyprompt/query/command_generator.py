"""Command generation using LLM providers."""

import logging
import re
from typing import Optional, Tuple, Dict, Any, List
from ..config import Settings
from ..llm import ProviderFactory, BaseLLMProvider

logger = logging.getLogger(__name__)


class CommandGenerator:
    """Generates CLI commands using LLM providers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.provider_factory = ProviderFactory(settings)
        self.provider: Optional[BaseLLMProvider] = None

    async def initialize(self, provider_name: Optional[str] = None) -> None:
        """Initialize the command generator with an LLM provider."""
        try:
            self.provider = await self.provider_factory.get_provider(provider_name)
            logger.info(f"Command generator initialized with {self.provider.provider_name}")
        except Exception as e:
            logger.error(f"Failed to initialize command generator: {e}")
            raise

    async def generate_command(
        self,
        user_query: str,
        context: str,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a command for the user query."""
        if not self.provider:
            await self.initialize()

        try:
            response = await self.provider.generate_command(
                user_query=user_query,
                context=context,
                **kwargs
            )

            command = response.content
            metadata = {
                "provider": self.provider.provider_name,
                "model": response.model,
                "usage": response.usage,
                "is_valid": self._validate_command(command),
                "is_safe": self._check_command_safety(command),
                "command_type": self._classify_command(command)
            }

            logger.debug(f"Generated command: {command}")
            return command, metadata

        except Exception as e:
            logger.error(f"Failed to generate command: {e}")
            raise

    async def generate_command_with_explanation(
        self,
        user_query: str,
        context: str,
        **kwargs
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Generate a command with explanation."""
        if not self.provider:
            await self.initialize()

        try:
            command, explanation = await self.provider.generate_command_with_explanation(
                user_query=user_query,
                context=context,
                **kwargs
            )

            metadata = {
                "provider": self.provider.provider_name,
                "is_valid": self._validate_command(command),
                "is_safe": self._check_command_safety(command),
                "command_type": self._classify_command(command)
            }

            return command, explanation, metadata

        except Exception as e:
            logger.error(f"Failed to generate command with explanation: {e}")
            raise

    async def refine_command(
        self,
        original_command: str,
        user_feedback: str,
        context: str
    ) -> Tuple[str, str]:
        """Refine a command based on user feedback."""
        if not self.provider:
            await self.initialize()

        try:
            refine_prompt = f"""
Original command: {original_command}
User feedback: {user_feedback}
Context: {context}

Please provide a refined version of the command that addresses the user's feedback.
Return only the refined command, without explanation.
"""

            messages = [
                {"role": "system", "content": "You are a helpful CLI command assistant. Refine commands based on user feedback."},
                {"role": "user", "content": refine_prompt}
            ]

            response = await self.provider.chat_completion(messages)
            refined_command = self.provider._clean_command_response(response.content)

            # Generate explanation for the refined command
            explanation_prompt = f"""
Original command: {original_command}
Refined command: {refined_command}
User feedback: {user_feedback}

Explain what changes were made and why.
"""

            explanation_messages = [
                {"role": "system", "content": "Explain the changes made to the CLI command."},
                {"role": "user", "content": explanation_prompt}
            ]

            explanation_response = await self.provider.chat_completion(explanation_messages)

            return refined_command, explanation_response.content

        except Exception as e:
            logger.error(f"Failed to refine command: {e}")
            raise

    async def generate_alternative_commands(
        self,
        user_query: str,
        context: str,
        num_alternatives: int = 3
    ) -> List[Tuple[str, str]]:
        """Generate alternative commands for the same query."""
        if not self.provider:
            await self.initialize()

        alternatives = []

        for i in range(num_alternatives):
            try:
                # Use different temperature/creativity for each alternative
                temperature = 0.1 + (i * 0.3)  # 0.1, 0.4, 0.7

                command, explanation, _ = await self.generate_command_with_explanation(
                    user_query=user_query,
                    context=context,
                    temperature=temperature
                )

                # Avoid duplicate commands
                if not any(alt[0] == command for alt in alternatives):
                    alternatives.append((command, explanation))

            except Exception as e:
                logger.warning(f"Failed to generate alternative {i+1}: {e}")

        return alternatives

    def _validate_command(self, command: str) -> bool:
        """Validate that the command looks reasonable."""
        if not command or command.strip() == "":
            return False

        if command == "UNCLEAR_REQUEST":
            return True

        # Basic validation checks
        command = command.strip()

        # Check for obviously invalid patterns
        invalid_patterns = [
            r"^[{}]+$",  # Only braces
            r"^[()]+$",  # Only parentheses
            r"^[<>]+$",  # Only angle brackets
            r"^\s*$",    # Only whitespace
        ]

        for pattern in invalid_patterns:
            if re.match(pattern, command):
                return False

        # Should be reasonable length
        if len(command) > 1000:
            return False

        return True

    def _check_command_safety(self, command: str) -> bool:
        """Check if a command appears safe to execute."""
        if not command or command == "UNCLEAR_REQUEST":
            return True

        command_lower = command.lower()

        # Dangerous patterns
        dangerous_patterns = [
            r"\brm\s+-rf\s+/",     # rm -rf /
            r"\bmv\s+.+\s+/dev/null",  # Move to /dev/null
            r"\bdd\s+if=",         # dd command (can be dangerous)
            r"\bformat\b",         # format command
            r">\s*/dev/sd[a-z]",   # Writing to disk devices
            r"\bsudo\s+rm",       # sudo rm
            r"[:;|&]\s*rm\s+-rf", # Chained dangerous rm
            r"\bchmod\s+777",     # Overly permissive chmod
            r"\bchown\s+.*:\s*\/", # Changing ownership of root directories
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, command_lower):
                return False

        # Check for suspicious keywords
        suspicious_keywords = [
            "password", "passwd", "secret", "token", "key",
            "/etc/passwd", "/etc/shadow", "~/.ssh",
            "curl.*|.*sh", "wget.*|.*sh"  # Pipe to shell
        ]

        for keyword in suspicious_keywords:
            if keyword in command_lower:
                return False

        return True

    def _classify_command(self, command: str) -> str:
        """Classify the type of command."""
        if not command or command == "UNCLEAR_REQUEST":
            return "unclear"

        command_lower = command.lower()

        # Classification patterns
        classifications = {
            "read": [r"\bls\b", r"\bcat\b", r"\bhead\b", r"\btail\b", r"\bgrep\b", r"\bfind\b"],
            "write": [r"\btouch\b", r"\becho\b", r"\bmkdir\b", r"\bcp\b", r"\bmv\b"],
            "delete": [r"\brm\b", r"\brmdir\b"],
            "network": [r"\bcurl\b", r"\bwget\b", r"\bping\b", r"\bssh\b"],
            "process": [r"\bps\b", r"\btop\b", r"\bkill\b", r"\bjobs\b"],
            "git": [r"\bgit\b"],
            "package": [r"\bapt\b", r"\byum\b", r"\bnpm\b", r"\bpip\b", r"\bcargo\b"],
            "system": [r"\bsudo\b", r"\bsystemctl\b", r"\bservice\b"],
        }

        for cmd_type, patterns in classifications.items():
            for pattern in patterns:
                if re.search(pattern, command_lower):
                    return cmd_type

        return "other"

    async def test_provider_availability(self) -> Dict[str, bool]:
        """Test availability of all configured providers."""
        return await self.provider_factory.test_all_providers()

    async def switch_provider(self, provider_name: str) -> bool:
        """Switch to a different LLM provider."""
        try:
            self.provider = await self.provider_factory.get_provider(provider_name)
            logger.info(f"Switched to {provider_name} provider")
            return True
        except Exception as e:
            logger.error(f"Failed to switch to {provider_name}: {e}")
            return False

    async def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider."""
        if not self.provider:
            return {"status": "not_initialized"}

        return {
            "provider": self.provider.provider_name,
            "model": self.provider.default_model,
            "status": "ready"
        }

    async def generate_answer(
        self,
        user_query: str,
        context: str
    ) -> str:
        """Generate an answer to a documentation question."""
        if not self.provider:
            await self.initialize()

        try:
            answer = await self.provider.generate_answer(user_query, context)
            logger.info(f"Generated answer for documentation query")
            return answer

        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise

    async def close(self) -> None:
        """Close the command generator."""
        if self.provider_factory:
            await self.provider_factory.close_all()