"""Main document indexer that orchestrates the indexing process."""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional
from ..config import Settings
from ..vectordb import get_vector_db
from .document_parser import DocumentParser
from .text_chunker import TextChunker, TextChunk
from .embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """Main class for indexing documents and storing embeddings."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.parser = DocumentParser()
        self.chunker = TextChunker(
            chunk_size=1000, chunk_overlap=200, min_chunk_size=100
        )
        self.embedding_generator = EmbeddingGenerator(settings.embedding_model)
        self.vector_db = None

    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing document indexer...")

        # Initialize embedding generator
        await self.embedding_generator.initialize()

        # Initialize vector database
        self.vector_db = get_vector_db(self.settings)
        await self.vector_db.initialize()

        logger.info("Document indexer initialized successfully")

    async def index_documentation(
        self, force_rebuild: bool = False, paths: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """Index all documentation files."""
        if self.vector_db is None:
            await self.initialize()

        logger.info("Starting documentation indexing...")

        # Determine paths to index
        if paths is None:
            paths = self._get_default_paths()

        # Check if we need to rebuild
        if force_rebuild:
            logger.info("Force rebuild requested, clearing existing index...")
            await self.vector_db.clear()

        # Parse all documents
        all_documents = []
        for path_str in paths:
            path = Path(path_str)
            if path.exists():
                if path.is_file():
                    try:
                        doc = await self.parser.parse_file(path)
                        all_documents.append(doc)
                        logger.info(f"Parsed file: {path}")
                    except Exception as e:
                        logger.warning(f"Failed to parse {path}: {e}")
                elif path.is_dir():
                    try:
                        docs = await self.parser.parse_directory(path)
                        all_documents.extend(docs)
                        logger.info(f"Parsed directory: {path} ({len(docs)} files)")
                    except Exception as e:
                        logger.warning(f"Failed to parse directory {path}: {e}")
            else:
                logger.warning(f"Path does not exist: {path}")

        if not all_documents:
            logger.warning("No documents found to index")
            return {"documents": 0, "chunks": 0}

        # Chunk all documents
        all_chunks = []
        for doc in all_documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {len(all_documents)} documents")

        # Merge small chunks
        all_chunks = self.chunker.merge_small_chunks(all_chunks)
        logger.info(f"After merging: {len(all_chunks)} chunks")

        # Generate embeddings
        chunk_contents = [chunk.content for chunk in all_chunks]
        embeddings = await self.embedding_generator.generate_embeddings(chunk_contents)

        # Store in vector database
        await self._store_chunks_with_embeddings(all_chunks, embeddings)

        logger.info(f"Successfully indexed {len(all_chunks)} chunks")

        return {"documents": len(all_documents), "chunks": len(all_chunks)}

    async def _store_chunks_with_embeddings(
        self, chunks: List[TextChunk], embeddings: List
    ) -> None:
        """Store chunks with their embeddings in the vector database."""
        documents = []

        for chunk, embedding in zip(chunks, embeddings):
            document = {
                "id": chunk.chunk_id,
                "content": chunk.content,
                "embedding": embedding,
                "metadata": {
                    "file_path": chunk.file_path,
                    "section": chunk.section,
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    **chunk.metadata,
                },
            }
            documents.append(document)

        # Store in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            await self.vector_db.add_documents(batch)
            logger.debug(f"Stored batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")

    def _get_default_paths(self) -> List[str]:
        """Get default paths to index based on settings."""
        paths = []

        # Add README
        if self.settings.readme_path:
            paths.append(self.settings.readme_path)

        # Add docs directory
        if self.settings.docs_path:
            paths.append(self.settings.docs_path)

        # Add additional docs from docs_path_list
        for doc in self.settings.docs_path_list:
            if doc != self.settings.docs_path:  # Avoid duplicates
                paths.append(doc)

        return [path for path in paths if path]

    async def update_document(self, file_path: str) -> bool:
        """Update a single document in the index."""
        if self.vector_db is None:
            await self.initialize()

        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"File does not exist: {file_path}")
                return False

            # Remove existing chunks for this file
            await self.vector_db.delete_by_metadata({"file_path": file_path})

            # Parse and re-index the file
            doc = await self.parser.parse_file(path)
            chunks = self.chunker.chunk_document(doc)
            chunks = self.chunker.merge_small_chunks(chunks)

            if chunks:
                chunk_contents = [chunk.content for chunk in chunks]
                embeddings = await self.embedding_generator.generate_embeddings(
                    chunk_contents
                )
                await self._store_chunks_with_embeddings(chunks, embeddings)

            logger.info(f"Updated document: {file_path} ({len(chunks)} chunks)")
            return True

        except Exception as e:
            logger.error(f"Failed to update document {file_path}: {e}")
            return False

    async def get_index_stats(self) -> Dict[str, int]:
        """Get statistics about the current index."""
        if self.vector_db is None:
            await self.initialize()

        try:
            total_docs = await self.vector_db.count()
            return {"total_chunks": total_docs}
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {"total_chunks": 0}

    async def search_similar_content(
        self, query: str, top_k: int = 5, threshold: float = 0.0
    ) -> List[Dict]:
        """Search for content similar to the query."""
        if self.vector_db is None:
            await self.initialize()

        # Generate query embedding
        query_embedding = await self.embedding_generator.generate_query_embedding(query)

        # Search vector database
        results = await self.vector_db.search(
            query_embedding, top_k=top_k, threshold=threshold
        )

        return results

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.vector_db:
            await self.vector_db.close()
        # Embedding generator cleanup is handled in __del__