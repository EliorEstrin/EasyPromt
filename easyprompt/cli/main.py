"""Main CLI application entry point."""

import typer
import logging
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.logging import RichHandler

# Lazy import of config - only load when needed

# Initialize console for rich output
console = Console()

# Create main Typer app
app = typer.Typer(
    name="easyprompt",
    help="🤖 Unified Interactive RAG Assistant - Configure, index, and query documentation in one session",
    add_completion=False,
    rich_markup_mode="rich"
)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


def quick_config_check() -> bool:
    """Quick check if .env exists without heavy validation."""
    return Path(".env").exists()


def validate_config(settings):
    """Validate configuration settings."""
    from ..config import ConfigValidator
    
    # Basic environment validation
    env_error = ConfigValidator.validate_environment()
    if env_error:
        console.print(f"[red]Configuration Error:[/red] {env_error}")
        console.print("Run 'easyprompt init' to set up your configuration.")
        raise typer.Exit(1)
    
    # Validate settings
    validator = ConfigValidator(settings)
    try:
        validator.validate_all()
    except Exception as e:
        console.print(f"[red]Configuration Error:[/red] {e}")
        console.print("Run 'easyprompt init' to fix your configuration.")
        raise typer.Exit(1)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Set the logging level",
        case_sensitive=False
    ),
    config_file: str = typer.Option(
        ".env",
        "--config",
        help="Path to configuration file"
    )
):
    """EasyPrompt - Unified Interactive RAG Assistant."""
    # Only start interactive session if no command is specified
    if ctx.invoked_subcommand is None:
        setup_logging(log_level)
        # Lazy import only when actually launching interactive session
        from .interactive_session import launch_interactive_session
        launch_interactive_session()


@app.command()
def init():
    """Initialize EasyPrompt configuration."""
    from .init_command import init_configuration
    init_configuration()


@app.command()
def chat(
    provider: Optional[str] = typer.Option(
        None, 
        "--provider", 
        help="LLM provider to use (gemini, openai, anthropic)"
    )
):
    """Start interactive chat session."""
    setup_logging()
    
    # Lazy imports
    from ..config import Settings
    from .commands import ChatCommand
    
    try:
        settings = Settings()
        validate_config(settings)
        
        command = ChatCommand(settings, console)
        command.run(provider)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def query(
    query_text: str,
    execute: bool = typer.Option(
        False, 
        "--execute", 
        "-e", 
        help="Execute generated command immediately"
    ),
    dry_run: bool = typer.Option(
        False, 
        "--dry-run", 
        help="Show command without executing"
    ),
    alternatives: int = typer.Option(
        1, 
        "--alternatives", 
        help="Number of alternative commands to generate"
    ),
    explain: bool = typer.Option(
        True, 
        "--explain/--no-explain", 
        help="Include explanation of the command"
    ),
    provider: Optional[str] = typer.Option(
        None, 
        "--provider", 
        help="LLM provider to use (gemini, openai, anthropic)"
    )
):
    """Generate CLI command from natural language query."""
    setup_logging()
    
    # Lazy imports
    from ..config import Settings
    from .commands import QueryCommand
    
    try:
        settings = Settings()
        validate_config(settings)
        
        command = QueryCommand(settings, console)
        command.run(query_text, execute, dry_run, alternatives, explain, provider)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def index(
    paths: Optional[List[str]] = typer.Argument(
        None, 
        help="Paths to index (directories or files)"
    ),
    rebuild: bool = typer.Option(
        False, 
        "--rebuild", 
        help="Force rebuild of the index"
    ),
    verbose: bool = typer.Option(
        False, 
        "--verbose", 
        "-v", 
        help="Show detailed progress"
    )
):
    """Index documentation files."""
    setup_logging()
    
    # Lazy imports
    from ..config import Settings
    from .commands import IndexCommand
    
    try:
        settings = Settings()
        validate_config(settings)
        
        command = IndexCommand(settings, console)
        command.run(paths, rebuild, verbose)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def search(
    query_text: str,
    limit: int = typer.Option(
        5, 
        "--limit", 
        "-l", 
        help="Maximum number of results"
    )
):
    """Search documentation without generating commands."""
    setup_logging()
    
    # Lazy imports
    from ..config import Settings
    from .search_command import search_documentation
    
    try:
        settings = Settings()
        validate_config(settings)
        
        import asyncio
        asyncio.run(search_documentation(settings, console, query_text, limit))
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def validate(
    command: str
):
    """Validate a command without executing it."""
    setup_logging()
    
    # Lazy imports
    from ..config import Settings
    from .validate_command import validate_command
    
    try:
        settings = Settings()
        validate_config(settings)
        
        import asyncio
        asyncio.run(validate_command(settings, console, command))
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def status():
    """Show system status and configuration."""
    setup_logging()
    
    # Lazy imports
    from ..config import Settings
    from .commands import StatusCommand
    
    try:
        settings = Settings()
        validate_config(settings)
        
        command = StatusCommand(settings, console)
        command.run()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def examples(
    query_text: str,
    limit: int = typer.Option(
        5, 
        "--limit", 
        "-l", 
        help="Maximum number of examples"
    )
):
    """Find command examples in documentation."""
    setup_logging()
    
    # Lazy imports
    from ..config import Settings
    from .examples_command import find_examples
    
    try:
        settings = Settings()
        validate_config(settings)
        
        import asyncio
        asyncio.run(find_examples(settings, console, query_text, limit))
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)




if __name__ == "__main__":
    app()