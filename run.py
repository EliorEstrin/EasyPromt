#!/usr/bin/env python3
"""
EasyPrompt Development Runner

Quick development runner for EasyPrompt that provides easy access to common commands
without needing to remember the full setup process.

Usage: python run.py [command] [args...]

This script handles environment activation automatically and provides shortcuts
for common development tasks.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
VENV_PATH = PROJECT_ROOT / "venv"
ACTIVATE_SCRIPT = PROJECT_ROOT / "activate.sh"


def is_venv_activated():
    """Check if virtual environment is activated."""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)


def run_command(cmd_list, check_venv=True):
    """Run a command, optionally checking for virtual environment."""
    if check_venv and not is_venv_activated():
        print("⚠️  Virtual environment not activated!")
        print("Run: source ./activate.sh")
        print("Then try again, or run: python run.py setup")
        return 1

    try:
        return subprocess.run(cmd_list).returncode
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd_list[0]}")
        print("Make sure the virtual environment is activated and dependencies are installed.")
        return 1


def show_help():
    """Show available commands."""
    print("🚀 EasyPrompt Development Runner")
    print("=" * 40)
    print()
    print("SETUP COMMANDS:")
    print("  setup          - Run initial project setup (./setup.sh)")
    print("  activate       - Show activation command")
    print("  install        - Install package in development mode")
    print("  clean          - Clean build artifacts")
    print()
    print("DEVELOPMENT COMMANDS:")
    print("  test           - Run core validation tests")
    print("  lint           - Run code linting")
    print("  format         - Format code with black")
    print()
    print("EASYPROMPT COMMANDS:")
    print("  init           - Initialize EasyPrompt configuration")
    print("  chat           - Start interactive chat mode")
    print("  query <text>   - Single query mode")
    print("  index <path>   - Index documentation")
    print("  status         - Show system status")
    print("  help           - Show EasyPrompt help")
    print()
    print("EXAMPLES:")
    print('  python run.py setup')
    print('  python run.py chat')
    print('  python run.py query "list all files"')
    print('  python run.py test')
    print()
    print("For detailed documentation, see README.md and CLAUDE.md")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        show_help()
        return 0

    cmd = sys.argv[1].lower()
    args = sys.argv[2:]

    # Setup commands (don't require venv)
    if cmd == "setup":
        print("🔧 Running project setup...")
        return subprocess.run(["./setup.sh"]).returncode

    elif cmd == "activate":
        print("To activate the development environment, run:")
        print("  source ./activate.sh")
        return 0

    elif cmd == "install":
        print("📦 Installing package in development mode...")
        return run_command(["pip", "install", "-e", "."])

    elif cmd == "clean":
        print("🧹 Cleaning build artifacts...")
        return run_command(["make", "clean"], check_venv=False)

    # Development commands (require venv)
    elif cmd == "test":
        print("🧪 Running validation tests...")
        return run_command(["python3", "validate_core_logic.py"])

    elif cmd == "lint":
        print("🔍 Running linting...")
        return run_command(["make", "lint"])

    elif cmd == "format":
        print("🎨 Formatting code...")
        return run_command(["make", "format"])

    # EasyPrompt commands (require venv and installation)
    elif cmd == "init":
        print("⚙️  Initializing EasyPrompt configuration...")
        return run_command(["easyprompt", "init"])

    elif cmd == "chat":
        print("💬 Starting interactive chat mode...")
        return run_command(["easyprompt", "chat"])

    elif cmd == "query":
        if not args:
            print("❌ Query text required. Usage: python run.py query \"your question\"")
            return 1
        print(f"🔍 Processing query: {' '.join(args)}")
        return run_command(["easyprompt", "query"] + args)

    elif cmd == "index":
        if not args:
            print("❌ Path required. Usage: python run.py index <path>")
            return 1
        print(f"📚 Indexing documentation: {args[0]}")
        return run_command(["easyprompt", "index"] + args)

    elif cmd == "status":
        print("📊 Checking system status...")
        return run_command(["easyprompt", "status"])

    elif cmd in ["help", "--help", "-h"]:
        if args:
            # Pass through to easyprompt help
            return run_command(["easyprompt", "--help"])
        else:
            show_help()
            return 0

    else:
        print(f"❌ Unknown command: {cmd}")
        print("Run 'python run.py' to see available commands.")
        return 1


if __name__ == "__main__":
    sys.exit(main())