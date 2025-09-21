"""Embedding generation using various models."""

import numpy as np
from typing import List, Dict, Optional, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# Optional import for sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for text using various models."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for embedding generation. "
                "Install it with: pip install sentence-transformers"
            )
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def initialize(self) -> None:
        """Initialize the embedding model."""
        try:
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self.executor, self._load_model, self.model_name
            )
            logger.info(f"Initialized embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def _load_model(self, model_name: str) -> SentenceTransformer:
        """Load the sentence transformer model."""
        return SentenceTransformer(model_name)

    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if self.model is None:
            await self.initialize()

        # Preprocess text
        processed_text = self._preprocess_text(text)

        # Generate embedding in thread pool
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            self.executor, self._encode_text, processed_text
        )

        return embedding

    async def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        if self.model is None:
            await self.initialize()

        # Preprocess all texts
        processed_texts = [self._preprocess_text(text) for text in texts]

        # Generate embeddings in batches
        batch_size = 32
        embeddings = []

        for i in range(0, len(processed_texts), batch_size):
            batch = processed_texts[i : i + batch_size]
            loop = asyncio.get_event_loop()
            batch_embeddings = await loop.run_in_executor(
                self.executor, self._encode_batch, batch
            )
            embeddings.extend(batch_embeddings)

        return embeddings

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode single text to embedding."""
        return self.model.encode([text])[0]

    def _encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Encode batch of texts to embeddings."""
        embeddings = self.model.encode(texts)
        return [embedding for embedding in embeddings]

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding generation."""
        # Remove excessive whitespace
        text = " ".join(text.split())

        # Truncate if too long (most models have token limits)
        max_length = 500  # Approximate word limit
        words = text.split()
        if len(words) > max_length:
            text = " ".join(words[:max_length])

        return text

    async def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if self.model is None:
            await self.initialize()

        # Generate a test embedding to get dimension
        test_embedding = await self.generate_embedding("test")
        return len(test_embedding)

    def compute_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)

    def find_most_similar(
        self, query_embedding: np.ndarray, embeddings: List[np.ndarray], top_k: int = 5
    ) -> List[tuple]:
        """Find most similar embeddings to query."""
        similarities = []

        for i, embedding in enumerate(embeddings):
            similarity = self.compute_similarity(query_embedding, embedding)
            similarities.append((i, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    async def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a user query with query-specific preprocessing."""
        # Enhanced preprocessing for queries
        processed_query = self._preprocess_query(query)
        return await self.generate_embedding(processed_query)

    def _preprocess_query(self, query: str) -> str:
        """Preprocess user query before embedding generation."""
        # Convert to lowercase for better matching
        query = query.lower().strip()

        # Remove common stop words that might not be meaningful for CLI commands
        stop_words = {
            "how",
            "do",
            "i",
            "can",
            "you",
            "please",
            "help",
            "me",
            "want",
            "need",
            "to",
        }

        words = query.split()
        filtered_words = [word for word in words if word not in stop_words]

        # If we filtered too much, keep original
        if len(filtered_words) < len(words) * 0.3:
            return query

        return " ".join(filtered_words)

    async def batch_generate_with_metadata(
        self, texts_with_metadata: List[Dict[str, str]]
    ) -> List[Dict[str, Union[str, np.ndarray]]]:
        """Generate embeddings for texts with metadata."""
        texts = [item["content"] for item in texts_with_metadata]
        embeddings = await self.generate_embeddings(texts)

        results = []
        for i, (text_data, embedding) in enumerate(zip(texts_with_metadata, embeddings)):
            result = text_data.copy()
            result["embedding"] = embedding
            results.append(result)

        return results

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)