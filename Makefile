# EasyPrompt Makefile

.PHONY: help install install-dev test lint format clean docs run-tests

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install the package"
	@echo "  install-dev - Install with development dependencies"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  clean       - Clean build artifacts"
	@echo "  docs        - Generate documentation"
	@echo "  run-tests   - Run comprehensive test suite"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=easyprompt --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 easyprompt tests
	mypy easyprompt

format:
	black easyprompt tests
	isort easyprompt tests

format-check:
	black --check easyprompt tests
	isort --check-only easyprompt tests

# Development
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Documentation
docs:
	@echo "Documentation available in ARCHITECTURE.md and README.md"

# Example commands
example-init:
	python -m easyprompt.cli.main init

example-index:
	python -m easyprompt.cli.main index

example-query:
	python -m easyprompt.cli.main query "list all files in current directory"

example-chat:
	python -m easyprompt.cli.main chat

# Docker commands (if using Docker)
docker-build:
	docker build -t easyprompt .

docker-run:
	docker run -it --rm -v $(PWD):/app easyprompt

# Comprehensive test suite
run-tests: format-check lint test

# Build and release
build:
	python -m build

release: clean build
	python -m twine upload dist/*

# Check all systems
check-all: install-dev run-tests
	@echo "All checks passed!"