#!/bin/bash
# Activate EasyPrompt development environment

source "/Users/dvirpa/Documents/EasyPromt/venv/bin/activate"
export PYTHONPATH="/Users/dvirpa/Documents/EasyPromt:$PYTHONPATH"

echo "🚀 EasyPrompt development environment activated!"
echo "Virtual environment: /Users/dvirpa/Documents/EasyPromt/venv"
echo "Project directory: /Users/dvirpa/Documents/EasyPromt"
echo ""
echo "Available commands:"
echo "  easyprompt --help     # Show help"
echo "  easyprompt init       # Initialize configuration"
echo "  easyprompt index      # Index documentation"
echo "  easyprompt chat       # Start interactive mode"
echo "  easyprompt status     # Show system status"
echo ""
echo "To deactivate: deactivate"
