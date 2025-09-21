"""Context retrieval from vector database."""

import logging
from typing import List, Dict, Any, Optional
from ..config import Settings
from ..vectordb import get_vector_db, BaseVectorDB
from ..indexer import EmbeddingGenerator

logger = logging.getLogger(__name__)


class ContextRetriever:
    """Retrieves relevant context from the vector database."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.vector_db: Optional[BaseVectorDB] = None
        self.embedding_generator: Optional[EmbeddingGenerator] = None

    async def initialize(self) -> None:
        """Initialize the context retriever."""
        # Initialize vector database
        self.vector_db = get_vector_db(self.settings)
        await self.vector_db.initialize()

        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(self.settings.embedding_model)
        await self.embedding_generator.initialize()

        logger.info("Context retriever initialized")

    async def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context for a query."""
        if not self.vector_db or not self.embedding_generator:
            await self.initialize()

        # Use settings defaults if not provided
        top_k = top_k or self.settings.top_k_results
        threshold = threshold or self.settings.similarity_threshold

        try:
            # Generate query embedding
            query_embedding = await self.embedding_generator.generate_query_embedding(query)

            # Search vector database
            results = await self.vector_db.search(
                query_embedding=query_embedding,
                top_k=top_k,
                threshold=threshold,
                filters=filters
            )

            logger.debug(f"Retrieved {len(results)} context chunks for query: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            raise

    async def retrieve_context_with_ranking(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        boost_factors: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve context with custom ranking."""
        results = await self.retrieve_context(query, top_k, threshold)

        if boost_factors:
            # Apply boost factors based on metadata
            for result in results:
                metadata = result.get("metadata", {})
                boost = 1.0

                for key, factor in boost_factors.items():
                    if metadata.get(key):
                        boost *= factor

                result["boosted_similarity"] = result.get("similarity", 0) * boost

            # Re-sort by boosted similarity
            results.sort(key=lambda x: x.get("boosted_similarity", x.get("similarity", 0)), reverse=True)

        return results

    async def retrieve_context_by_file(
        self,
        query: str,
        file_path: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve context specifically from a file."""
        filters = {"file_path": file_path}
        return await self.retrieve_context(query, top_k, threshold, filters)

    async def retrieve_context_by_section(
        self,
        query: str,
        section: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve context specifically from a section."""
        filters = {"section": section}
        return await self.retrieve_context(query, top_k, threshold, filters)

    def format_context_for_llm(
        self,
        context_results: List[Dict[str, Any]],
        max_context_length: Optional[int] = None
    ) -> str:
        """Format context results for LLM consumption."""
        max_length = max_context_length or self.settings.max_context_length

        if not context_results:
            return "No relevant documentation found."

        formatted_parts = []
        current_length = 0

        for i, result in enumerate(context_results):
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            similarity = result.get("similarity", 0)

            # Create section header
            file_path = metadata.get("file_path", "unknown")
            section = metadata.get("section", "main")
            header = f"## Source {i+1}: {file_path} - {section} (similarity: {similarity:.3f})\n"

            # Check if adding this would exceed length limit
            section_content = header + content + "\n\n"
            if current_length + len(section_content) > max_length and formatted_parts:
                break

            formatted_parts.append(section_content)
            current_length += len(section_content)

        if not formatted_parts:
            # If even the first result is too long, truncate it
            if context_results:
                result = context_results[0]
                content = result.get("content", "")
                metadata = result.get("metadata", {})
                file_path = metadata.get("file_path", "unknown")
                section = metadata.get("section", "main")

                header = f"## Source 1: {file_path} - {section}\n"
                available_length = max_length - len(header) - 10  # Leave some buffer

                if len(content) > available_length:
                    content = content[:available_length] + "..."

                return header + content

        return "".join(formatted_parts).strip()

    def get_context_summary(self, context_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get a summary of the retrieved context."""
        if not context_results:
            return {
                "total_chunks": 0,
                "files": [],
                "sections": [],
                "avg_similarity": 0.0
            }

        files = set()
        sections = set()
        similarities = []

        for result in context_results:
            metadata = result.get("metadata", {})
            files.add(metadata.get("file_path", "unknown"))
            sections.add(metadata.get("section", "main"))
            similarities.append(result.get("similarity", 0.0))

        return {
            "total_chunks": len(context_results),
            "files": list(files),
            "sections": list(sections),
            "avg_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
            "min_similarity": min(similarities) if similarities else 0.0,
            "max_similarity": max(similarities) if similarities else 0.0
        }

    async def find_command_examples(self, query: str) -> List[Dict[str, Any]]:
        """Find command examples related to the query."""
        # Use special boost factors for command-related content
        boost_factors = {
            "has_commands": 2.0,
            "has_code": 1.5,
            "type": 1.2  # Boost if type indicates code/commands
        }

        results = await self.retrieve_context_with_ranking(
            query=query,
            top_k=self.settings.top_k_results * 2,  # Get more results for filtering
            boost_factors=boost_factors
        )

        # Filter for results that likely contain commands
        command_results = []
        for result in results:
            content = result.get("content", "").lower()
            metadata = result.get("metadata", {})

            # Check for command indicators
            has_command_indicators = any([
                metadata.get("has_commands") == "true",
                "`" in result.get("content", ""),
                "```" in result.get("content", ""),
                "$" in content,
                any(cmd in content for cmd in ["run", "execute", "command", "cli"])
            ])

            if has_command_indicators:
                command_results.append(result)

            # Limit results
            if len(command_results) >= self.settings.top_k_results:
                break

        return command_results

    async def close(self) -> None:
        """Close the context retriever."""
        if self.vector_db:
            await self.vector_db.close()
        if self.embedding_generator:
            # Embedding generator cleanup is handled in __del__
            pass