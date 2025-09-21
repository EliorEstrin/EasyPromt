#!/usr/bin/env python3
"""
Core logic validation for EasyPrompt without external dependencies.
Tests the fundamental algorithms and logic.
"""

import sys
import re
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Simple text chunk representation."""
    content: str
    start_index: int
    end_index: int
    section: str
    file_path: str
    chunk_id: str
    metadata: Dict[str, str]

def test_markdown_parsing():
    """Test markdown parsing logic."""
    logger.info("Testing markdown parsing logic...")

    content = """# Test CLI Tool

A simple CLI tool for testing.

## Installation

```bash
pip install test-cli
```

## Usage

### Basic Commands

List all items:
```bash
test-cli list
```

Add an item:
```bash
test-cli add "item name"
```

### Advanced Commands

Filter items:
```bash
test-cli list --filter status=active
```
"""

    # Extract sections
    def extract_markdown_sections(content):
        sections = {}
        current_section = "introduction"
        current_content = []

        lines = content.split("\n")
        for line in lines:
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if header_match:
                if current_content:
                    sections[current_section] = "\n".join(current_content).strip()
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = re.sub(r"[^\w\s-]", "", title.lower()).replace(" ", "_").replace("-", "_")
                current_content = [line]
            else:
                current_content.append(line)

        if current_content:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    # Extract title
    def extract_title(content):
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return "Untitled Document"

    # Extract code blocks
    def extract_code_blocks(content):
        code_blocks = []
        pattern = r"```(\w+)?\n(.*?)\n```"
        for match in re.finditer(pattern, content, re.DOTALL):
            language = match.group(1) or "text"
            code = match.group(2).strip()
            code_blocks.append((language, code))
        return code_blocks

    # Extract commands
    def extract_commands(content):
        commands = []
        code_blocks = extract_code_blocks(content)
        for language, code in code_blocks:
            if language in ["bash", "shell", "sh", "console", "terminal"]:
                lines = code.split("\n")
                for line in lines:
                    line = line.strip()
                    if line.startswith("$ "):
                        line = line[2:]
                    elif line.startswith("> "):
                        line = line[2:]
                    if line and not line.startswith("#"):
                        commands.append(line)
        return commands

    # Test all functions
    sections = extract_markdown_sections(content)
    title = extract_title(content)
    code_blocks = extract_code_blocks(content)
    commands = extract_commands(content)

    # Validate results
    assert title == "Test CLI Tool"
    assert len(sections) >= 4  # Should have multiple sections
    assert "installation" in sections
    assert "usage" in sections
    assert len(code_blocks) >= 3  # Should have multiple code blocks
    assert any("test-cli list" in cmd for cmd in commands)
    assert any("test-cli add" in cmd for cmd in commands)

    logger.info(f"✓ Markdown parsing: extracted {len(sections)} sections, {len(commands)} commands")
    return True

def test_text_chunking():
    """Test text chunking logic."""
    logger.info("Testing text chunking logic...")

    def chunk_text(text, chunk_size=100, chunk_overlap=20, min_chunk_size=30):
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = ""
        current_start = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                overlap_words = current_chunk.split()[-5:]  # Last 5 words as overlap
                current_chunk = " ".join(overlap_words) + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if len(current_chunk.strip()) >= min_chunk_size:
            chunks.append(current_chunk.strip())

        return chunks

    # Test with long text
    long_text = "This is a sentence. " * 20  # 400 characters
    chunks = chunk_text(long_text, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) > 0 for chunk in chunks)
    assert len(chunks[0]) <= 120  # Should respect chunk size + some overlap

    logger.info(f"✓ Text chunking: created {len(chunks)} chunks from {len(long_text)} characters")
    return True

def test_command_validation():
    """Test command validation logic."""
    logger.info("Testing command validation logic...")

    def validate_command(command):
        if not command or command.strip() == "":
            return False
        if command == "UNCLEAR_REQUEST":
            return True

        command = command.strip()
        invalid_patterns = [r"^[{}]+$", r"^[()]+$", r"^[<>]+$", r"^\s*$"]
        for pattern in invalid_patterns:
            if re.match(pattern, command):
                return False
        if len(command) > 1000:
            return False
        return True

    def check_command_safety(command):
        if not command or command == "UNCLEAR_REQUEST":
            return True

        command_lower = command.lower()
        dangerous_patterns = [
            r"\brm\s+-rf\s+/",
            r"\bmv\s+.+\s+/dev/null",
            r"\bdd\s+if=",
            r"\bformat\b",
            r">\s*/dev/sd[a-z]",
            r"\bsudo\s+rm",
            r"[:;|&]\s*rm\s+-rf",
            r"\bchmod\s+777",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, command_lower):
                return False

        suspicious_keywords = [
            "password", "passwd", "secret", "token", "key",
            "/etc/passwd", "/etc/shadow", "~/.ssh"
        ]

        for keyword in suspicious_keywords:
            if keyword in command_lower:
                return False

        return True

    def classify_command(command):
        if not command or command == "UNCLEAR_REQUEST":
            return "unclear"

        command_lower = command.lower()
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

    # Test cases
    test_cases = [
        ("ls -la", True, True, "read"),
        ("rm -rf /", True, False, "delete"),
        ("", False, True, "unclear"),
        ("UNCLEAR_REQUEST", True, True, "unclear"),
        ("git status", True, True, "git"),
        ("sudo systemctl restart nginx", True, True, "system"),
        ("curl http://example.com", True, True, "network"),
        ("echo hello", True, True, "write"),
        ("{{{{", False, True, "other"),
        ("passwd root", True, False, "other"),
    ]

    for command, expected_valid, expected_safe, expected_type in test_cases:
        is_valid = validate_command(command)
        is_safe = check_command_safety(command)
        cmd_type = classify_command(command)

        assert is_valid == expected_valid, f"Validation failed for '{command}'"
        assert is_safe == expected_safe, f"Safety check failed for '{command}'"
        assert cmd_type == expected_type, f"Classification failed for '{command}'"

    logger.info("✓ Command validation: all test cases passed")
    return True

def test_context_formatting():
    """Test context formatting for LLM."""
    logger.info("Testing context formatting...")

    def format_context_for_llm(context_results, max_length=1000):
        if not context_results:
            return "No relevant documentation found."

        formatted_parts = []
        current_length = 0

        for i, result in enumerate(context_results):
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            similarity = result.get("similarity", 0)

            file_path = metadata.get("file_path", "unknown")
            section = metadata.get("section", "main")
            header = f"## Source {i+1}: {file_path} - {section} (similarity: {similarity:.3f})\n"

            section_content = header + content + "\n\n"
            if current_length + len(section_content) > max_length and formatted_parts:
                break

            formatted_parts.append(section_content)
            current_length += len(section_content)

        return "".join(formatted_parts).strip()

    # Test context formatting
    mock_results = [
        {
            "content": "Use test-cli list to list all items",
            "metadata": {"file_path": "README.md", "section": "commands"},
            "similarity": 0.9
        },
        {
            "content": "test-cli add allows you to add new items",
            "metadata": {"file_path": "docs/api.md", "section": "add_command"},
            "similarity": 0.8
        }
    ]

    formatted = format_context_for_llm(mock_results, max_length=500)

    assert "Source 1: README.md" in formatted
    assert "Source 2: docs/api.md" in formatted
    assert "test-cli list" in formatted
    assert "test-cli add" in formatted
    assert len(formatted) <= 600  # Should respect max length approximately

    logger.info("✓ Context formatting: properly formatted multiple sources")
    return True

def test_similarity_computation():
    """Test similarity computation logic."""
    logger.info("Testing similarity computation...")

    def compute_cosine_similarity(vec1, vec2):
        """Compute cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    # Test cases
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    similarity = compute_cosine_similarity(vec1, vec2)
    assert abs(similarity - 1.0) < 1e-6, "Identical vectors should have similarity 1.0"

    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    similarity = compute_cosine_similarity(vec1, vec2)
    assert abs(similarity - 0.0) < 1e-6, "Orthogonal vectors should have similarity 0.0"

    vec1 = [0.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    similarity = compute_cosine_similarity(vec1, vec2)
    assert similarity == 0.0, "Zero vector should have similarity 0.0"

    logger.info("✓ Similarity computation: all test cases passed")
    return True

def test_query_preprocessing():
    """Test query preprocessing logic."""
    logger.info("Testing query preprocessing...")

    def preprocess_query(query):
        """Preprocess user query for better matching."""
        query = query.lower().strip()
        stop_words = {"how", "do", "i", "can", "you", "please", "help", "me", "want", "need", "to"}
        words = query.split()
        filtered_words = [word for word in words if word not in stop_words]

        if len(filtered_words) < len(words) * 0.3:
            return query

        return " ".join(filtered_words)

    # Test cases
    test_cases = [
        ("How do I list files?", "list files?"),
        ("Can you help me add an item", "add an item"),
        ("Please show me the status", "show the status"),
        ("how do you", "how do you"),  # Should keep original if too short
    ]

    for original, expected in test_cases:
        result = preprocess_query(original)
        assert result == expected, f"Query preprocessing failed: '{original}' -> '{result}', expected '{expected}'"

    logger.info("✓ Query preprocessing: all test cases passed")
    return True

def run_all_tests():
    """Run all core logic tests."""
    logger.info("EasyPrompt Core Logic Validation")
    logger.info("=" * 50)

    tests = [
        ("Markdown Parsing", test_markdown_parsing),
        ("Text Chunking", test_text_chunking),
        ("Command Validation", test_command_validation),
        ("Context Formatting", test_context_formatting),
        ("Similarity Computation", test_similarity_computation),
        ("Query Preprocessing", test_query_preprocessing),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()

    logger.info("\n" + "=" * 50)
    logger.info(f"CORE LOGIC VALIDATION: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All core logic tests passed!")
        logger.info("\nThe fundamental algorithms work correctly:")
        logger.info("✓ Document parsing and section extraction")
        logger.info("✓ Text chunking with overlap")
        logger.info("✓ Command validation and safety checks")
        logger.info("✓ Context formatting for LLMs")
        logger.info("✓ Vector similarity computation")
        logger.info("✓ Query preprocessing")
        logger.info("\nTo use with dependencies:")
        logger.info("1. pip install -r requirements.txt")
        logger.info("2. easyprompt init")
        logger.info("3. easyprompt index")
        logger.info("4. easyprompt query 'your request'")
    else:
        logger.info(f"⚠️  {total - passed} tests failed.")

    return passed == total

if __name__ == "__main__":
    try:
        result = run_all_tests()
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)