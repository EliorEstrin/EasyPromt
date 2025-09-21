#!/bin/bash
# Activate EasyPrompt development environment

source "/home/elior/EasyPromt_repo/venv/bin/activate"
export PYTHONPATH="/home/elior/EasyPromt_repo:$PYTHONPATH"

echo "🚀 EasyPrompt development environment activated!"
echo "Virtual environment: /home/elior/EasyPromt_repo/venv"
echo "Project directory: /home/elior/EasyPromt_repo"
echo ""
echo "Available commands:"
echo "  easyprompt --help     # Show help"
echo "  easyprompt init       # Initialize configuration"
echo "  easyprompt index      # Index documentation"
echo "  easyprompt chat       # Start interactive mode"
echo "  easyprompt status     # Show system status"
echo ""
echo "To deactivate: deactivate"
