"""Main CLI application entry point."""

import typer
import logging
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler

from ..config import Settings, ConfigValidator
from .commands import IndexCommand, QueryCommand, ChatCommand, StatusCommand

# Initialize console for rich output
console = Console()

# Create main Typer app
app = typer.Typer(
    name="easyprompt",
    help="Natural Language to CLI Command Interface",
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


def validate_config() -> Settings:
    """Validate configuration and return settings."""
    # Check for .env file
    env_error = ConfigValidator.validate_environment()
    if env_error:
        console.print(f"[red]Configuration Error:[/red] {env_error}")
        raise typer.Exit(1)

    # Load settings
    try:
        settings = Settings()
    except Exception as e:
        console.print(f"[red]Failed to load settings:[/red] {e}")
        raise typer.Exit(1)

    # Validate settings
    validator = ConfigValidator(settings)
    if not validator.validate_all():
        console.print("[red]Configuration validation failed:[/red]")
        for error in validator.get_validation_errors():
            console.print(f"  • {error}")
        raise typer.Exit(1)

    return settings


@app.callback()
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
    """EasyPrompt - Natural Language to CLI Command Interface."""
    setup_logging(log_level)

    # Store settings in context for subcommands
    if ctx.invoked_subcommand != "init":
        settings = validate_config()
        ctx.obj = settings


@app.command()
def init(
    force: bool = typer.Option(False, "--force", help="Overwrite existing configuration")
):
    """Initialize EasyPrompt configuration."""
    from .init_command import init_configuration
    init_configuration(force)


@app.command()
def index(
    ctx: typer.Context,
    paths: list[str] = typer.Argument(None, help="Paths to index (files or directories)"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild the entire index"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Index documentation files."""
    settings = ctx.obj
    command = IndexCommand(settings, console)
    command.run(paths, rebuild, verbose)


@app.command()
def query(
    ctx: typer.Context,
    query_text: str = typer.Argument(..., help="Natural language query"),
    execute: bool = typer.Option(False, "--execute", "-e", help="Execute the generated command"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show command without executing"),
    alternatives: int = typer.Option(0, "--alternatives", "-a", help="Show alternative commands"),
    explain: bool = typer.Option(True, "--explain/--no-explain", help="Include explanation"),
    provider: str = typer.Option(None, "--provider", help="LLM provider to use"),
):
    """Generate CLI command from natural language query."""
    settings = ctx.obj
    command = QueryCommand(settings, console)
    command.run(query_text, execute, dry_run, alternatives, explain, provider)


@app.command()
def chat(
    ctx: typer.Context,
    provider: str = typer.Option(None, "--provider", help="LLM provider to use"),
):
    """Start interactive chat session."""
    settings = ctx.obj
    command = ChatCommand(settings, console)
    command.run(provider)


@app.command()
def status(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed status")
):
    """Show system status and configuration."""
    settings = ctx.obj
    command = StatusCommand(settings, console)
    command.run(verbose)


@app.command()
def search(
    ctx: typer.Context,
    query_text: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of results to show"),
):
    """Search documentation without generating commands."""
    from .search_command import search_documentation
    settings = ctx.obj
    search_documentation(settings, console, query_text, limit)


@app.command()
def validate(
    ctx: typer.Context,
    command_text: str = typer.Argument(..., help="Command to validate")
):
    """Validate a command without executing it."""
    from .validate_command import validate_command
    settings = ctx.obj
    validate_command(settings, console, command_text)


@app.command()
def examples(
    ctx: typer.Context,
    query_text: str = typer.Argument(..., help="Query to find examples for"),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of examples to show"),
):
    """Find command examples related to a query."""
    from .examples_command import find_examples
    settings = ctx.obj
    find_examples(settings, console, query_text, limit)


if __name__ == "__main__":
    app()