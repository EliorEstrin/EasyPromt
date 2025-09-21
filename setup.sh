#!/bin/bash

# EasyPrompt Quick Setup Script
# This script sets up the entire project from scratch for developers with no context
#
# Usage:
#   ./setup.sh              # Setup with existing venv check
#   ./setup.sh --force       # Force recreate virtual environment

set -e  # Exit on any error

# Parse command line arguments
FORCE_RECREATE=false

for arg in "$@"; do
    case $arg in
        --force)
            FORCE_RECREATE=true
            shift
            ;;
        --help|-h)
            echo "EasyPrompt Setup Script"
            echo ""
            echo "Usage:"
            echo "  ./setup.sh        # Setup (asks about existing venv)"
            echo "  ./setup.sh --force # Force recreate virtual environment"
            echo "  ./setup.sh --help  # Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${CYAN}$1${NC}"
}

# Header
clear
print_header "==============================================="
print_header "🚀 EasyPrompt - Quick Setup Script"
print_header "==============================================="
echo ""
print_status "This script will set up EasyPrompt from scratch:"
echo "  ✓ Create and activate virtual environment"
echo "  ✓ Install all dependencies"
echo "  ✓ Install EasyPrompt package"
echo "  ✓ Run validation tests"
echo "  ✓ Initialize configuration (optional)"
echo ""

# Check if Python is available
print_status "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.9+ and try again."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_success "Found Python $PYTHON_VERSION"

# Check Python version (require 3.9+)
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    print_success "Python version is compatible (3.9+)"
else
    print_error "Python 3.9+ is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
print_status "Project directory: $PROJECT_DIR"

# Create virtual environment
VENV_DIR="$PROJECT_DIR/venv"
print_status "Setting up virtual environment at $VENV_DIR..."

if [ -d "$VENV_DIR" ]; then
    if [ "$FORCE_RECREATE" = true ]; then
        print_status "Removing existing virtual environment (--force)..."
        rm -rf "$VENV_DIR"
        python3 -m venv "$VENV_DIR"
        print_success "Virtual environment recreated"
    else
        print_warning "Virtual environment already exists at $VENV_DIR"
        read -p "Use existing environment? [Y/n]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            print_status "Removing existing virtual environment..."
            rm -rf "$VENV_DIR"
            python3 -m venv "$VENV_DIR"
            print_success "Virtual environment recreated"
        else
            print_status "Using existing virtual environment"
        fi
    fi
else
    python3 -m venv "$VENV_DIR"
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
print_success "Virtual environment activated"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_status "Installing dependencies from requirements.txt..."
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip install -r "$PROJECT_DIR/requirements.txt"
    print_success "Dependencies installed"
else
    print_error "requirements.txt not found!"
    exit 1
fi

# Install EasyPrompt package in development mode
print_status "Installing EasyPrompt package in development mode..."
cd "$PROJECT_DIR"
pip install -e .
print_success "EasyPrompt package installed"

# Run validation tests
print_status "Running validation tests..."
if python3 validate_core_logic.py; then
    print_success "Core logic validation passed!"
else
    print_error "Validation tests failed. Check the output above."
    exit 1
fi

# Run basic functionality test
print_status "Running basic functionality test..."
if python3 test_basic_functionality.py; then
    print_success "Basic functionality test passed!"
else
    print_warning "Basic functionality test had some issues, but core logic works."
fi

# Run setup validation test
print_status "Running setup validation test..."
if python3 test_setup.py; then
    print_success "Setup validation passed!"
else
    print_warning "Setup validation had some issues. Check output above."
fi

# Create data directory
print_status "Creating data directory..."
mkdir -p "$PROJECT_DIR/data"
print_success "Data directory created"

# Create example documentation
print_status "Creating example documentation..."
mkdir -p "$PROJECT_DIR/example_docs"

cat > "$PROJECT_DIR/example_docs/README.md" << 'EOF'
# Example CLI Tool

This is an example CLI tool for testing EasyPrompt.

## Installation

```bash
pip install example-cli
```

## Commands

### list
List all items:
```bash
example-cli list
```

Options:
- `--format`: Output format (json, table)
- `--filter`: Filter by criteria

### add
Add a new item:
```bash
example-cli add "item name"
```

Options:
- `--description`: Item description
- `--tags`: Comma-separated tags

### delete
Delete an item:
```bash
example-cli delete <item-id>
```

### status
Check system status:
```bash
example-cli status
```
EOF

cat > "$PROJECT_DIR/example_docs/advanced.md" << 'EOF'
# Advanced Usage

## Batch Operations

Process multiple items:
```bash
example-cli batch --input file.json
```

## Configuration

Set configuration:
```bash
example-cli config set key=value
```

View configuration:
```bash
example-cli config show
```

## Troubleshooting

Check logs:
```bash
example-cli logs --tail 100
```

Reset configuration:
```bash
example-cli reset --confirm
```
EOF

print_success "Example documentation created"

# Create activation script
print_status "Creating activation script..."
cat > "$PROJECT_DIR/activate.sh" << EOF
#!/bin/bash
# Activate EasyPrompt development environment

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_DIR:\$PYTHONPATH"

echo "🚀 EasyPrompt development environment activated!"
echo "Virtual environment: $VENV_DIR"
echo "Project directory: $PROJECT_DIR"
echo ""
echo "Available commands:"
echo "  easyprompt --help     # Show help"
echo "  easyprompt init       # Initialize configuration"
echo "  easyprompt index      # Index documentation"
echo "  easyprompt chat       # Start interactive mode"
echo "  easyprompt status     # Show system status"
echo ""
echo "To deactivate: deactivate"
EOF

chmod +x "$PROJECT_DIR/activate.sh"
print_success "Activation script created: ./activate.sh"

# Summary
print_header ""
print_header "==============================================="
print_header "🎉 Setup Complete!"
print_header "==============================================="
echo ""
print_success "EasyPrompt has been successfully set up!"
echo ""
print_status "What was installed:"
echo "  ✓ Virtual environment in ./venv"
echo "  ✓ All Python dependencies"
echo "  ✓ EasyPrompt package in development mode"
echo "  ✓ Example documentation in ./example_docs"
echo "  ✓ Validation tests passed"
echo ""
print_status "Next steps:"
echo ""
echo "1. Activate the environment:"
echo "   source ./activate.sh"
echo ""
echo "2. Initialize EasyPrompt configuration:"
echo "   easyprompt init"
echo ""
echo "3. Index the example documentation:"
echo "   easyprompt index ./example_docs"
echo ""
echo "4. Try some queries:"
echo "   easyprompt query \"list all items\""
echo "   easyprompt chat"
echo ""
echo "5. Check system status:"
echo "   easyprompt status"
echo ""
print_status "For help at any time:"
echo "   easyprompt --help"
echo ""
print_warning "Note: You'll need to configure API keys for LLM providers (Gemini/OpenAI/Anthropic)"
print_warning "during the 'easyprompt init' step to use the full functionality."
echo ""
print_success "Happy coding! 🚀"