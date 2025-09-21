#!/usr/bin/env python3
"""
Test script to validate EasyPrompt setup.
Run this after setup.sh to ensure everything works.
"""

import sys
import subprocess
import importlib
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")

    required_packages = [
        'easyprompt',
        'easyprompt.config',
        'easyprompt.indexer',
        'easyprompt.vectordb',
        'easyprompt.llm',
        'easyprompt.query',
        'easyprompt.cli'
    ]

    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package} - {e}")
            return False

    return True

def test_cli_command():
    """Test that the CLI command is available."""
    print("\nTesting CLI command...")

    try:
        result = subprocess.run(
            ['easyprompt', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            print("✓ easyprompt command available")
            return True
        else:
            print(f"✗ easyprompt command failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("✗ easyprompt command timed out")
        return False
    except FileNotFoundError:
        print("✗ easyprompt command not found")
        return False

def test_example_docs():
    """Test that example documentation was created."""
    print("\nTesting example documentation...")

    example_dir = Path("example_docs")
    required_files = [
        "README.md",
        "advanced.md"
    ]

    if not example_dir.exists():
        print("✗ example_docs directory not found")
        return False

    for file_name in required_files:
        file_path = example_dir / file_name
        if file_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} not found")
            return False

    return True

def test_project_structure():
    """Test that project structure is correct."""
    print("\nTesting project structure...")

    required_paths = [
        "easyprompt",
        "tests",
        "requirements.txt",
        "pyproject.toml",
        ".env.example",
        "setup.sh",
        "activate.sh",
        "QUICKSTART.md"
    ]

    for path_name in required_paths:
        path = Path(path_name)
        if path.exists():
            print(f"✓ {path}")
        else:
            print(f"✗ {path} not found")
            return False

    return True

def test_virtual_environment():
    """Test that we're running in a virtual environment."""
    print("\nTesting virtual environment...")

    # Check if we're in a virtual environment
    in_venv = (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )

    if in_venv:
        print(f"✓ Running in virtual environment: {sys.prefix}")
        return True
    else:
        print("⚠ Not running in virtual environment (this is okay if installed system-wide)")
        return True  # Don't fail for this

def main():
    """Run all tests."""
    print("EasyPrompt Setup Validation")
    print("=" * 40)

    tests = [
        ("Project Structure", test_project_structure),
        ("Virtual Environment", test_virtual_environment),
        ("Package Imports", test_imports),
        ("CLI Command", test_cli_command),
        ("Example Documentation", test_example_docs),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")

    print("\n" + "=" * 40)
    print(f"SETUP VALIDATION: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 Setup is complete and working!")
        print("\nNext steps:")
        print("1. source ./activate.sh")
        print("2. easyprompt init")
        print("3. easyprompt index ./example_docs")
        print("4. easyprompt chat")
    else:
        print("⚠ Some tests failed. Check the output above.")
        print("Try running ./setup.sh again or check the QUICKSTART.md guide.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)