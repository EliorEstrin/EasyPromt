"""Main query processor that orchestrates the query handling workflow."""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from ..config import Settings
from .context_retriever import ContextRetriever
from .command_generator import CommandGenerator

logger = logging.getLogger(__name__)


class QueryResult:
    """Result of query processing."""

    def __init__(
        self,
        query: str,
        command: str,
        explanation: str = "",
        context_summary: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        processing_time: float = 0.0,
        success: bool = True,
        error: str = ""
    ):
        self.query = query
        self.command = command
        self.explanation = explanation
        self.context_summary = context_summary or {}
        self.metadata = metadata or {}
        self.processing_time = processing_time
        self.success = success
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "command": self.command,
            "explanation": self.explanation,
            "context_summary": self.context_summary,
            "metadata": self.metadata,
            "processing_time": self.processing_time,
            "success": self.success,
            "error": self.error
        }


class QueryProcessor:
    """Main query processor that orchestrates the entire query handling workflow."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.context_retriever = ContextRetriever(settings)
        self.command_generator = CommandGenerator(settings)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        logger.info("Initializing query processor...")

        try:
            await self.context_retriever.initialize()
            await self.command_generator.initialize()
            self._initialized = True
            logger.info("Query processor initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize query processor: {e}")
            raise

    async def process_query(
        self,
        query: str,
        include_explanation: bool = True,
        context_filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> QueryResult:
        """Process a user query and generate a command."""
        start_time = time.time()

        if not self._initialized:
            await self.initialize()

        try:
            # Step 1: Retrieve relevant context
            logger.debug(f"Processing query: {query}")
            context_results = await self.context_retriever.retrieve_context(
                query=query,
                filters=context_filters
            )

            # Step 2: Format context for LLM
            formatted_context = self.context_retriever.format_context_for_llm(context_results)

            # Step 3: Generate command
            if include_explanation:
                command, explanation, cmd_metadata = await self.command_generator.generate_command_with_explanation(
                    user_query=query,
                    context=formatted_context,
                    **kwargs
                )
            else:
                command, cmd_metadata = await self.command_generator.generate_command(
                    user_query=query,
                    context=formatted_context,
                    **kwargs
                )
                explanation = ""

            # Step 4: Get context summary
            context_summary = self.context_retriever.get_context_summary(context_results)

            # Step 5: Prepare metadata
            processing_time = time.time() - start_time
            metadata = {
                **cmd_metadata,
                "context_chunks_used": len(context_results),
                "query_length": len(query),
                "context_length": len(formatted_context)
            }

            return QueryResult(
                query=query,
                command=command,
                explanation=explanation,
                context_summary=context_summary,
                metadata=metadata,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Query processing failed: {e}")

            return QueryResult(
                query=query,
                command="",
                explanation="",
                processing_time=processing_time,
                success=False,
                error=str(e)
            )

    async def process_query_with_alternatives(
        self,
        query: str,
        num_alternatives: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """Process query and generate alternative commands."""
        try:
            # Get main result
            main_result = await self.process_query(query, **kwargs)

            if not main_result.success:
                return main_result.to_dict()

            # Get context for alternatives
            context_results = await self.context_retriever.retrieve_context(query)
            formatted_context = self.context_retriever.format_context_for_llm(context_results)

            # Generate alternatives
            alternatives = await self.command_generator.generate_alternative_commands(
                user_query=query,
                context=formatted_context,
                num_alternatives=num_alternatives
            )

            result_dict = main_result.to_dict()
            result_dict["alternatives"] = [
                {"command": cmd, "explanation": exp}
                for cmd, exp in alternatives
            ]

            return result_dict

        except Exception as e:
            logger.error(f"Failed to generate alternatives: {e}")
            return {
                "query": query,
                "success": False,
                "error": str(e)
            }

    async def refine_query_result(
        self,
        original_result: QueryResult,
        user_feedback: str
    ) -> QueryResult:
        """Refine a previous query result based on user feedback."""
        start_time = time.time()

        try:
            # Get context again (could be cached in future)
            context_results = await self.context_retriever.retrieve_context(
                original_result.query
            )
            formatted_context = self.context_retriever.format_context_for_llm(context_results)

            # Refine the command
            refined_command, explanation = await self.command_generator.refine_command(
                original_command=original_result.command,
                user_feedback=user_feedback,
                context=formatted_context
            )

            # Prepare metadata
            processing_time = time.time() - start_time
            metadata = {
                "is_refinement": True,
                "original_command": original_result.command,
                "refinement_feedback": user_feedback,
                "provider": self.command_generator.provider.provider_name if self.command_generator.provider else "unknown"
            }

            return QueryResult(
                query=original_result.query,
                command=refined_command,
                explanation=explanation,
                context_summary=original_result.context_summary,
                metadata=metadata,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Query refinement failed: {e}")

            return QueryResult(
                query=original_result.query,
                command="",
                explanation="",
                processing_time=processing_time,
                success=False,
                error=str(e)
            )

    async def find_command_examples(self, query: str) -> List[Dict[str, Any]]:
        """Find command examples related to the query."""
        try:
            return await self.context_retriever.find_command_examples(query)
        except Exception as e:
            logger.error(f"Failed to find command examples: {e}")
            return []

    async def search_documentation(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search documentation without generating commands."""
        try:
            return await self.context_retriever.retrieve_context(
                query=query,
                top_k=top_k
            )
        except Exception as e:
            logger.error(f"Documentation search failed: {e}")
            return []

    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status and health information."""
        try:
            status = {
                "initialized": self._initialized,
                "embedding_model": self.settings.embedding_model,
                "vector_db_type": self.settings.vector_db_type,
            }

            if self._initialized:
                # Test vector database
                try:
                    test_results = await self.context_retriever.retrieve_context(
                        "test query", top_k=1
                    )
                    status["vector_db_status"] = "healthy"
                    status["total_documents"] = await self.context_retriever.vector_db.count()
                except Exception as e:
                    status["vector_db_status"] = f"error: {e}"

                # Test LLM provider
                try:
                    provider_info = await self.command_generator.get_provider_info()
                    status["llm_provider"] = provider_info
                except Exception as e:
                    status["llm_provider"] = {"status": f"error: {e}"}

            return status

        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                "initialized": False,
                "error": str(e)
            }

    async def validate_command(self, command: str) -> Dict[str, Any]:
        """Validate a command without executing it."""
        try:
            is_valid = self.command_generator._validate_command(command)
            is_safe = self.command_generator._check_command_safety(command)
            command_type = self.command_generator._classify_command(command)

            return {
                "command": command,
                "is_valid": is_valid,
                "is_safe": is_safe,
                "command_type": command_type,
                "recommendations": self._get_command_recommendations(command, is_safe, command_type)
            }

        except Exception as e:
            logger.error(f"Command validation failed: {e}")
            return {
                "command": command,
                "error": str(e)
            }

    def _get_command_recommendations(
        self, command: str, is_safe: bool, command_type: str
    ) -> List[str]:
        """Get recommendations for command execution."""
        recommendations = []

        if not is_safe:
            recommendations.append("⚠️  This command may be dangerous. Review carefully before execution.")

        if command_type == "delete":
            recommendations.append("🗑️  This command will delete files/directories. Consider backing up first.")

        if command_type == "system" or "sudo" in command.lower():
            recommendations.append("🔒 This command requires elevated privileges. Ensure you understand its effects.")

        if command_type == "network":
            recommendations.append("🌐 This command will make network requests. Verify the destination is trusted.")

        if self.settings.dry_run:
            recommendations.append("🔍 Dry run mode is enabled. Command will not be executed.")

        if self.settings.confirm_before_execution:
            recommendations.append("✋ Confirmation is enabled. You will be prompted before execution.")

        return recommendations

    async def process_qa_query(
        self,
        query: str,
        context_filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a Q&A query and generate an answer from documentation."""
        start_time = time.time()
        if not self._initialized:
            await self.initialize()

        try:
            # Retrieve relevant context
            context_chunks = await self.context_retriever.retrieve_context(
                query,
                filters=context_filters
            )

            if not context_chunks:
                return {
                    "success": False,
                    "answer": "No relevant documentation found for your question.",
                    "context_used": [],
                    "processing_time": time.time() - start_time
                }

            # Generate answer using Q&A mode
            context_text = "\n\n".join([chunk["content"] for chunk in context_chunks])
            answer = await self.command_generator.generate_answer(query, context_text)

            # Extract unique files used
            files_used = list(set([chunk["metadata"].get("file_path", "unknown") for chunk in context_chunks]))

            # Calculate processing time
            processing_time = time.time() - start_time

            return {
                "success": True,
                "answer": answer,
                "context_used": context_chunks,
                "files_used": files_used,
                "processing_time": processing_time
            }

        except Exception as e:
            logger.error(f"Error processing Q&A query: {e}")
            return {
                "success": False,
                "answer": f"Error processing question: {str(e)}",
                "context_used": [],
                "processing_time": time.time() - start_time
            }

    async def close(self) -> None:
        """Close the query processor and cleanup resources."""
        try:
            await self.context_retriever.close()
            await self.command_generator.close()
            logger.info("Query processor closed")
        except Exception as e:
            logger.warning(f"Error closing query processor: {e}")