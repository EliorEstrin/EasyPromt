"""Main CLI application entry point."""

import typer
import logging
from pathlib import Path
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




if __name__ == "__main__":
    app()