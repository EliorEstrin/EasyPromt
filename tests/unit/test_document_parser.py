"""Tests for document parser."""

import pytest
from pathlib import Path

from easyprompt.indexer.document_parser import DocumentParser


class TestDocumentParser:
    """Test DocumentParser class."""

    @pytest.fixture
    def parser(self):
        """Create a DocumentParser instance."""
        return DocumentParser()

    @pytest.mark.asyncio
    async def test_parse_markdown_file(self, parser, sample_markdown_files):
        """Test parsing a markdown file."""
        readme_path = sample_markdown_files["readme"]

        document = await parser.parse_file(readme_path)

        assert document["file_path"] == str(readme_path)
        assert document["type"] == "markdown"
        assert document["title"] == "Test CLI Tool"
        assert "# Test CLI Tool" in document["content"]
        assert isinstance(document["sections"], dict)
        assert len(document["sections"]) > 1

    @pytest.mark.asyncio
    async def test_parse_directory(self, parser, sample_markdown_files):
        """Test parsing a directory of files."""
        docs_dir = sample_markdown_files["docs_dir"].parent

        documents = await parser.parse_directory(docs_dir)

        assert len(documents) >= 2  # README.md and docs/api.md
        file_paths = [doc["file_path"] for doc in documents]

        # Check that both files were parsed
        readme_found = any("README.md" in path for path in file_paths)
        api_found = any("api.md" in path for path in file_paths)

        assert readme_found
        assert api_found

    @pytest.mark.asyncio
    async def test_parse_unsupported_file_type(self, parser, temp_dir):
        """Test parsing unsupported file type."""
        unsupported_file = temp_dir / "test.xyz"
        unsupported_file.write_text("unsupported content")

        with pytest.raises(ValueError, match="Unsupported file type"):
            await parser.parse_file(unsupported_file)

    def test_extract_markdown_sections(self, parser):
        """Test extracting sections from markdown content."""
        content = """# Main Title

Some introduction text.

## Section 1

Content for section 1.

### Subsection 1.1

Subsection content.

## Section 2

Content for section 2.
"""

        sections = parser._extract_markdown_sections(content)

        assert "main_title" in sections
        assert "section_1" in sections
        assert "subsection_11" in sections
        assert "section_2" in sections

    def test_extract_title(self, parser):
        """Test extracting title from content."""
        content_with_h1 = "# Main Title\n\nSome content"
        assert parser._extract_title(content_with_h1) == "Main Title"

        content_without_h1 = "Some content without title"
        title = parser._extract_title(content_without_h1)
        assert title == "Some content without title"

        content_long = "A" * 100
        title = parser._extract_title(content_long)
        assert len(title) <= 53  # 50 + "..."

    def test_normalize_section_name(self, parser):
        """Test normalizing section names."""
        assert parser._normalize_section_name("Section Title") == "section_title"
        assert parser._normalize_section_name("Section-With-Dashes") == "section_with_dashes"
        assert parser._normalize_section_name("Section With Special!@# Characters") == "section_with_special_characters"
        assert parser._normalize_section_name("  Leading and Trailing  ") == "leading_and_trailing"

    def test_extract_code_blocks(self, parser):
        """Test extracting code blocks from markdown."""
        content = """# Title

Some text.

```bash
echo "hello world"
ls -la
```

More text.

```python
print("hello")
```

```
no language specified
```
"""

        code_blocks = parser.extract_code_blocks(content)

        assert len(code_blocks) == 3

        bash_block = next((block for block in code_blocks if block[0] == "bash"), None)
        assert bash_block is not None
        assert "echo" in bash_block[1]
        assert "ls -la" in bash_block[1]

        python_block = next((block for block in code_blocks if block[0] == "python"), None)
        assert python_block is not None
        assert "print" in python_block[1]

        text_block = next((block for block in code_blocks if block[0] == "text"), None)
        assert text_block is not None

    def test_extract_commands(self, parser):
        """Test extracting CLI commands from content."""
        content = """# CLI Tool

To list files:
```bash
$ ls -la
> find . -name "*.py"
```

You can also run `git status` to check status.

Or use `kubectl get pods` to list pods.
"""

        commands = parser.extract_commands(content)

        assert "ls -la" in commands
        assert "find . -name \"*.py\"" in commands
        # Note: inline commands might also be extracted depending on implementation

    @pytest.mark.asyncio
    async def test_parse_plain_text_file(self, parser, temp_dir):
        """Test parsing a plain text file."""
        txt_file = temp_dir / "test.txt"
        content = "This is plain text content.\nWith multiple lines."
        txt_file.write_text(content)

        document = await parser.parse_file(txt_file)

        assert document["file_path"] == str(txt_file)
        assert document["type"] == "text"
        assert document["content"] == content
        assert document["title"] == "test"  # filename without extension
        assert "main" in document["sections"]

    @pytest.mark.asyncio
    async def test_parse_empty_file(self, parser, temp_dir):
        """Test parsing an empty file."""
        empty_file = temp_dir / "empty.md"
        empty_file.write_text("")

        document = await parser.parse_file(empty_file)

        assert document["file_path"] == str(empty_file)
        assert document["content"] == ""
        assert document["title"] == "Untitled Document"

    @pytest.mark.asyncio
    async def test_parse_file_not_found(self, parser, temp_dir):
        """Test parsing a file that doesn't exist."""
        nonexistent_file = temp_dir / "nonexistent.md"

        with pytest.raises(FileNotFoundError):
            await parser.parse_file(nonexistent_file)