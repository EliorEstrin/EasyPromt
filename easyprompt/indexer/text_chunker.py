"""Text chunking utilities for splitting documents into manageable pieces."""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""

    content: str
    start_index: int
    end_index: int
    section: str
    file_path: str
    chunk_id: str
    metadata: Dict[str, str]


class TextChunker:
    """Splits text into chunks suitable for embedding and retrieval."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, document: Dict[str, str]) -> List[TextChunk]:
        """Chunk a document into smaller pieces."""
        chunks = []
        file_path = document["file_path"]

        # Process each section separately
        sections = document.get("sections", {"main": document["content"]})

        for section_name, section_content in sections.items():
            if not section_content.strip():
                continue

            section_chunks = self._chunk_text(
                section_content, section_name, file_path, document
            )
            chunks.extend(section_chunks)

        return chunks

    def _chunk_text(
        self, text: str, section: str, file_path: str, document: Dict[str, str]
    ) -> List[TextChunk]:
        """Split text into chunks with overlap."""
        if len(text) <= self.chunk_size:
            # Text is small enough to be a single chunk
            return [
                TextChunk(
                    content=text,
                    start_index=0,
                    end_index=len(text),
                    section=section,
                    file_path=file_path,
                    chunk_id=f"{file_path}:{section}:0",
                    metadata=self._extract_metadata(text, document),
                )
            ]

        chunks = []
        # Try to split on natural boundaries first
        sentences = self._split_into_sentences(text)

        current_chunk = ""
        current_start = 0
        chunk_index = 0

        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(
                    TextChunk(
                        content=current_chunk.strip(),
                        start_index=current_start,
                        end_index=current_start + len(current_chunk),
                        section=section,
                        file_path=file_path,
                        chunk_id=f"{file_path}:{section}:{chunk_index}",
                        metadata=self._extract_metadata(current_chunk, document),
                    )
                )

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + sentence
                current_start = current_start + len(current_chunk) - len(overlap_text)
                chunk_index += 1
            else:
                current_chunk += sentence

        # Add the last chunk if it has enough content
        if len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(
                TextChunk(
                    content=current_chunk.strip(),
                    start_index=current_start,
                    end_index=current_start + len(current_chunk),
                    section=section,
                    file_path=file_path,
                    chunk_id=f"{file_path}:{section}:{chunk_index}",
                    metadata=self._extract_metadata(current_chunk, document),
                )
            )

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving structure."""
        # Handle markdown headers and lists specially
        lines = text.split("\n")
        sentences = []

        current_sentence = ""

        for line in lines:
            line = line.strip()

            # Keep headers, lists, and code blocks as separate units
            if (
                line.startswith("#")
                or line.startswith("-")
                or line.startswith("*")
                or line.startswith("```")
                or line.startswith(">")
            ):
                if current_sentence:
                    sentences.append(current_sentence + "\n")
                    current_sentence = ""
                sentences.append(line + "\n")
            else:
                # Split on sentence boundaries
                line_sentences = re.split(r"(?<=[.!?])\s+", line)
                for i, sentence in enumerate(line_sentences):
                    if i == len(line_sentences) - 1:
                        # Last sentence in line, might continue
                        current_sentence += sentence + " "
                    else:
                        # Complete sentence
                        sentences.append(current_sentence + sentence + " ")
                        current_sentence = ""

        if current_sentence:
            sentences.append(current_sentence)

        return [s for s in sentences if s.strip()]

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk."""
        if len(text) <= self.chunk_overlap:
            return text

        # Try to find a natural break point within overlap range
        overlap_text = text[-self.chunk_overlap :]

        # Find the last sentence boundary
        sentences = re.split(r"(?<=[.!?])\s+", overlap_text)
        if len(sentences) > 1:
            # Keep complete sentences
            return " ".join(sentences[1:])

        # If no sentence boundary, use word boundary
        words = overlap_text.split()
        if len(words) > 1:
            return " ".join(words[len(words) // 2 :])

        return overlap_text

    def _extract_metadata(self, text: str, document: Dict[str, str]) -> Dict[str, str]:
        """Extract metadata from text chunk."""
        metadata = {
            "title": document.get("title", ""),
            "type": document.get("type", "text"),
            "word_count": str(len(text.split())),
            "char_count": str(len(text)),
        }

        # Extract additional features
        if "```" in text:
            metadata["has_code"] = "true"

        if re.search(r"^\s*[-*+]\s", text, re.MULTILINE):
            metadata["has_list"] = "true"

        if re.search(r"^#+\s", text, re.MULTILINE):
            metadata["has_headers"] = "true"

        # Extract command-like patterns
        command_patterns = [
            r"`[^`]+`",  # Inline code
            r"\$\s+\w+",  # Shell prompt
            r"npm\s+\w+",  # NPM commands
            r"git\s+\w+",  # Git commands
        ]

        for pattern in command_patterns:
            if re.search(pattern, text):
                metadata["has_commands"] = "true"
                break

        return metadata

    def merge_small_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Merge chunks that are too small with adjacent chunks."""
        if not chunks:
            return chunks

        merged_chunks = []
        i = 0

        while i < len(chunks):
            current_chunk = chunks[i]

            # If chunk is too small, try to merge with next chunk
            if (
                len(current_chunk.content) < self.min_chunk_size
                and i + 1 < len(chunks)
                and current_chunk.section == chunks[i + 1].section
            ):
                next_chunk = chunks[i + 1]
                merged_content = current_chunk.content + "\n" + next_chunk.content

                # Only merge if combined size is reasonable
                if len(merged_content) <= self.chunk_size * 1.5:
                    merged_chunk = TextChunk(
                        content=merged_content,
                        start_index=current_chunk.start_index,
                        end_index=next_chunk.end_index,
                        section=current_chunk.section,
                        file_path=current_chunk.file_path,
                        chunk_id=f"{current_chunk.chunk_id}_merged",
                        metadata={**current_chunk.metadata, **next_chunk.metadata},
                    )
                    merged_chunks.append(merged_chunk)
                    i += 2  # Skip next chunk as it's been merged
                    continue

            merged_chunks.append(current_chunk)
            i += 1

        return merged_chunks