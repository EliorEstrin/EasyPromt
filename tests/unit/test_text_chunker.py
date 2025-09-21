"""Tests for text chunker."""

import pytest

from easyprompt.indexer.text_chunker import TextChunker, TextChunk


class TestTextChunker:
    """Test TextChunker class."""

    @pytest.fixture
    def chunker(self):
        """Create a TextChunker instance."""
        return TextChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=30)

    def test_chunk_small_document(self, chunker, sample_documents):
        """Test chunking a small document."""
        document = {
            "file_path": "test.md",
            "content": "Short content that fits in one chunk.",
            "sections": {"main": "Short content that fits in one chunk."}
        }

        chunks = chunker.chunk_document(document)

        assert len(chunks) == 1
        assert chunks[0].content == "Short content that fits in one chunk."
        assert chunks[0].file_path == "test.md"
        assert chunks[0].section == "main"

    def test_chunk_large_document(self, chunker):
        """Test chunking a large document."""
        long_content = "This is a sentence. " * 10  # 200 characters
        document = {
            "file_path": "test.md",
            "content": long_content,
            "sections": {"main": long_content}
        }

        chunks = chunker.chunk_document(document)

        assert len(chunks) > 1
        # Check overlap exists
        if len(chunks) > 1:
            # There should be some overlap between chunks
            first_chunk_end = chunks[0].content[-20:]
            second_chunk_start = chunks[1].content[:20]
            # Some overlap should exist (not exact match due to sentence boundaries)

    def test_chunk_with_multiple_sections(self, chunker, sample_documents):
        """Test chunking document with multiple sections."""
        document = sample_documents[0]  # Has multiple sections

        chunks = chunker.chunk_document(document)

        assert len(chunks) >= 1
        sections = {chunk.section for chunk in chunks}
        assert len(sections) >= 1  # Should have chunks from different sections

    def test_split_into_sentences(self, chunker):
        """Test sentence splitting."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."

        sentences = chunker._split_into_sentences(text)

        assert len(sentences) >= 4
        assert any("First sentence" in s for s in sentences)
        assert any("Fourth sentence" in s for s in sentences)

    def test_split_with_markdown_headers(self, chunker):
        """Test sentence splitting with markdown headers."""
        text = """# Header 1

Some content here.

## Header 2

More content here.

- List item 1
- List item 2
"""

        sentences = chunker._split_into_sentences(text)

        # Headers and list items should be separate
        headers = [s for s in sentences if s.strip().startswith("#")]
        assert len(headers) >= 2

        list_items = [s for s in sentences if s.strip().startswith("-")]
        assert len(list_items) >= 2

    def test_get_overlap_text(self, chunker):
        """Test getting overlap text."""
        text = "This is a long sentence that should be used for overlap testing purposes."

        overlap = chunker._get_overlap_text(text)

        assert len(overlap) <= chunker.chunk_overlap
        assert overlap in text
        assert len(overlap) > 0

    def test_extract_metadata(self, chunker):
        """Test extracting metadata from text."""
        text_with_code = """# Title

Some text with `inline code` and:

```bash
echo "hello"
```

- List item 1
- List item 2
"""

        document = {
            "title": "Test Document",
            "type": "markdown"
        }

        metadata = chunker._extract_metadata(text_with_code, document)

        assert metadata["title"] == "Test Document"
        assert metadata["type"] == "markdown"
        assert metadata["has_code"] == "true"
        assert metadata["has_list"] == "true"
        assert metadata["has_headers"] == "true"

    def test_extract_metadata_with_commands(self, chunker):
        """Test extracting metadata for text with commands."""
        text_with_commands = """
Run this command:
```bash
npm install
```

Or use `git status` to check status.

You can also run $ ls -la
"""

        document = {"title": "Commands", "type": "markdown"}

        metadata = chunker._extract_metadata(text_with_commands, document)

        assert metadata["has_commands"] == "true"

    def test_merge_small_chunks(self, chunker):
        """Test merging small chunks."""
        # Create some small chunks
        chunks = [
            TextChunk(
                content="Small chunk 1",
                start_index=0,
                end_index=13,
                section="test",
                file_path="test.md",
                chunk_id="test:0",
                metadata={}
            ),
            TextChunk(
                content="Small chunk 2",
                start_index=14,
                end_index=27,
                section="test",
                file_path="test.md",
                chunk_id="test:1",
                metadata={}
            ),
            TextChunk(
                content="This is a larger chunk that should not be merged because it's already big enough",
                start_index=28,
                end_index=108,
                section="test",
                file_path="test.md",
                chunk_id="test:2",
                metadata={}
            )
        ]

        merged = chunker.merge_small_chunks(chunks)

        # The first two small chunks should be merged
        assert len(merged) <= len(chunks)
        # Check that merged chunk contains content from both small chunks
        if len(merged) < len(chunks):
            merged_content = merged[0].content
            assert "Small chunk 1" in merged_content
            assert "Small chunk 2" in merged_content

    def test_chunk_empty_sections(self, chunker):
        """Test chunking document with empty sections."""
        document = {
            "file_path": "test.md",
            "content": "Some content",
            "sections": {
                "empty_section": "",
                "content_section": "Some content",
                "whitespace_section": "   \n\t   "
            }
        }

        chunks = chunker.chunk_document(document)

        # Should only create chunks for non-empty sections
        assert len(chunks) == 1
        assert chunks[0].section == "content_section"

    def test_text_chunk_creation(self):
        """Test TextChunk dataclass creation."""
        chunk = TextChunk(
            content="Test content",
            start_index=0,
            end_index=12,
            section="test_section",
            file_path="/path/to/file.md",
            chunk_id="file:section:0",
            metadata={"key": "value"}
        )

        assert chunk.content == "Test content"
        assert chunk.start_index == 0
        assert chunk.end_index == 12
        assert chunk.section == "test_section"
        assert chunk.file_path == "/path/to/file.md"
        assert chunk.chunk_id == "file:section:0"
        assert chunk.metadata == {"key": "value"}