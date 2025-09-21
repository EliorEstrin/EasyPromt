"""Document parsing utilities for extracting content from various file formats."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import aiofiles


class DocumentParser:
    """Parses and extracts content from documentation files."""

    def __init__(self):
        self.supported_extensions = {".md", ".txt", ".rst"}

    async def parse_file(self, file_path: Path) -> Dict[str, str]:
        """Parse a single file and extract structured content."""
        if file_path.suffix not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        async with aiofiles.open(file_path, "r", encoding="utf-8") as file:
            content = await file.read()

        if file_path.suffix == ".md":
            return self._parse_markdown(content, str(file_path))
        else:
            return self._parse_plain_text(content, str(file_path))

    async def parse_directory(self, directory_path: Path) -> List[Dict[str, str]]:
        """Parse all supported files in a directory."""
        documents = []

        for file_path in directory_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in self.supported_extensions:
                try:
                    doc = await self.parse_file(file_path)
                    documents.append(doc)
                except Exception as e:
                    print(f"Warning: Failed to parse {file_path}: {e}")

        return documents

    def _parse_markdown(self, content: str, file_path: str) -> Dict[str, str]:
        """Parse markdown content and extract sections."""
        sections = self._extract_markdown_sections(content)

        return {
            "file_path": file_path,
            "content": content,
            "sections": sections,
            "type": "markdown",
            "title": self._extract_title(content),
        }

    def _parse_plain_text(self, content: str, file_path: str) -> Dict[str, str]:
        """Parse plain text content."""
        return {
            "file_path": file_path,
            "content": content,
            "sections": {"main": content},
            "type": "text",
            "title": Path(file_path).stem,
        }

    def _extract_markdown_sections(self, content: str) -> Dict[str, str]:
        """Extract sections from markdown content based on headers."""
        sections = {}
        current_section = "introduction"
        current_content = []

        lines = content.split("\n")

        for line in lines:
            # Check for headers
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if header_match:
                # Save previous section
                if current_content:
                    sections[current_section] = "\n".join(current_content).strip()

                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = self._normalize_section_name(title)
                current_content = [line]
            else:
                current_content.append(line)

        # Save the last section
        if current_content:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    def _extract_title(self, content: str) -> str:
        """Extract the main title from content."""
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
            elif line and not line.startswith("#"):
                # If no H1 header found, use first non-empty line
                return line[:50] + "..." if len(line) > 50 else line

        return "Untitled Document"

    def _normalize_section_name(self, title: str) -> str:
        """Normalize section names for consistent indexing."""
        # Remove special characters and convert to lowercase
        normalized = re.sub(r"[^\w\s-]", "", title.lower())
        # Replace spaces and hyphens with underscores
        normalized = re.sub(r"[\s-]+", "_", normalized)
        # Remove leading/trailing underscores
        return normalized.strip("_")

    def extract_code_blocks(self, content: str) -> List[Tuple[str, str]]:
        """Extract code blocks from markdown content."""
        code_blocks = []
        pattern = r"```(\w+)?\n(.*?)\n```"

        for match in re.finditer(pattern, content, re.DOTALL):
            language = match.group(1) or "text"
            code = match.group(2).strip()
            code_blocks.append((language, code))

        return code_blocks

    def extract_commands(self, content: str) -> List[str]:
        """Extract CLI commands from content."""
        commands = []

        # Look for code blocks that might contain commands
        code_blocks = self.extract_code_blocks(content)
        for language, code in code_blocks:
            if language in ["bash", "shell", "sh", "console", "terminal"]:
                # Extract individual commands
                lines = code.split("\n")
                for line in lines:
                    line = line.strip()
                    # Remove prompt indicators
                    if line.startswith("$ "):
                        line = line[2:]
                    elif line.startswith("> "):
                        line = line[2:]

                    if line and not line.startswith("#"):
                        commands.append(line)

        # Also look for inline code that might be commands
        inline_code_pattern = r"`([^`]+)`"
        for match in re.finditer(inline_code_pattern, content):
            code = match.group(1).strip()
            # Simple heuristic: if it looks like a command
            if (
                " " in code
                and not code.startswith("http")
                and len(code.split()) >= 2
                and len(code) < 100
            ):
                commands.append(code)

        return commands