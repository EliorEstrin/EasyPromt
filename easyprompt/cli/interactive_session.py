"""Unified interactive session for EasyPrompt."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

# Lazy imports - only load when needed
# from ..config import Settings, ConfigValidator
# from .init_command import init_configuration, load_existing_env_file
# from .commands import IndexCommand, QueryCommand, ChatCommand, StatusCommand

console = Console()


def launch_interactive_session():
    """Launch the unified interactive session with fast startup."""
    console.clear()
    show_welcome()

    # Fast startup - just check if .env exists, defer validation until needed
    env_file = Path(".env")
    if not env_file.exists():
        console.print("[yellow]⚠️  No configuration found. Let's set up EasyPrompt first![/yellow]")
        console.print()
        if Confirm.ask("Start configuration setup?", default=True):
            # Lazy import only when needed
            from .init_command import init_configuration
            init_configuration()
            console.print("\n[green]✅ Configuration complete! Loading EasyPrompt...[/green]")
            console.input("\n[dim]Press Enter to continue...[/dim]")
        else:
            console.print("[yellow]Configuration required to use EasyPrompt. Exiting...[/yellow]")
            return

    # Main interactive loop with lazy loading
    while True:
        try:
            console.clear()
            show_header()
            show_status_overview_fast()

            choice = show_main_menu()

            if choice == "1":
                handle_configuration()
            elif choice == "2":
                handle_indexing()
            elif choice == "3":
                handle_query()
            elif choice == "4":
                handle_chat()
            elif choice == "5":
                handle_search()
            elif choice == "6":
                handle_status()
            elif choice == "q":
                console.print("[yellow]Goodbye! 👋[/yellow]")
                break

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye! 👋[/yellow]")
            break


def show_welcome():
    """Show welcome message."""
    welcome_panel = Panel(
        "[bold blue]🤖 Welcome to EasyPrompt![/bold blue]\n\n"
        "Your unified RAG assistant for documentation Q&A.\n"
        "Configure, index, and query all from one interactive session.",
        title="EasyPrompt Interactive Session",
        style="blue"
    )
    console.print(welcome_panel)
    console.print()


def show_header():
    """Show session header."""
    console.print("[bold blue]🤖 EasyPrompt Interactive Session[/bold blue]")
    console.print()


def show_status_overview_fast():
    """Show quick status overview with fast startup."""
    env_file = Path(".env")
    if not env_file.exists():
        console.print(f"📊 [bold]Quick Status:[/bold] Vector DB ❌ | Embedding ❌ | LLM ❌ | Docs ❌")
        console.print()
        return

    # Fast config check without heavy imports
    config = {}
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    except:
        pass

    # Quick status indicators
    vector_db_status = "✅" if config.get("VECTOR_DB_TYPE") else "❌"
    embedding_status = "✅" if config.get("EMBEDDING_MODEL") else "❌"
    llm_status = "✅" if any(config.get(k) for k in ["OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"]) else "❌"
    docs_status = "✅" if config.get("DOCS_PATH") else "❌"

    console.print(f"📊 [bold]Quick Status:[/bold] Vector DB {vector_db_status} | Embedding {embedding_status} | LLM {llm_status} | Docs {docs_status}")
    console.print()


def show_main_menu() -> str:
    """Show main menu and get user choice."""
    menu_options = [
        ("1", "⚙️  Configuration", "Set up or modify your EasyPrompt configuration"),
        ("2", "📚 Index Documentation", "Index your documentation files for RAG search"),
        ("3", "❓ Ask Question", "Ask a single question about your documentation"),
        ("4", "💬 Chat Session", "Start an interactive chat with your documentation"),
        ("5", "🔍 Search Documents", "Search your documentation without AI"),
        ("6", "📊 System Status", "View detailed system status and configuration"),
        ("q", "🚪 Exit", "Exit EasyPrompt")
    ]

    console.print("[bold yellow]🎯 What would you like to do?[/bold yellow]")
    for key, title, desc in menu_options:
        console.print(f"  {key}. [bold cyan]{title}[/bold cyan] - [dim]{desc}[/dim]")
    console.print()

    try:
        choice = Prompt.ask(
            "[bold]Choose an option",
            choices=[opt[0] for opt in menu_options],
            default="1"
        )
        return choice
    except KeyboardInterrupt:
        raise


def handle_configuration():
    """Handle configuration setup/modification."""
    console.print("[blue]📝 Opening configuration...[/blue]")
    # Lazy import only when needed
    from .init_command import init_configuration
    init_configuration()


def handle_indexing():
    """Handle documentation indexing with lazy loading."""
    # Lazy load settings and commands
    try:
        from ..config import Settings
        settings = Settings()
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        console.print("Please configure EasyPrompt first (option 1)")
        console.input("\n[dim]Press Enter to continue...[/dim]")
        return

    console.clear()
    console.print(Panel(
        "[bold]📚 Index Documentation[/bold]",
        style="green"
    ))

    console.print("This will index your documentation files for RAG search.")
    console.print(f"Documentation path: [cyan]{settings.docs_path}[/cyan]")
    console.print()

    rebuild = Confirm.ask("Rebuild entire index (recommended for first time)?", default=True)
    verbose = Confirm.ask("Show detailed progress?", default=False)

    if Confirm.ask("Start indexing?", default=True):
        try:
            from .commands import IndexCommand
            command = IndexCommand(settings, console)
            command.run(paths=None, rebuild=rebuild, verbose=verbose)
            console.print("[green]✅ Indexing completed![/green]")
        except Exception as e:
            console.print(f"[red]❌ Indexing failed: {e}[/red]")

    console.input("\n[dim]Press Enter to continue...[/dim]")


def handle_query():
    """Handle single query with lazy loading."""
    # Lazy load settings
    try:
        from ..config import Settings
        settings = Settings()
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        console.print("Please configure EasyPrompt first (option 1)")
        console.input("\n[dim]Press Enter to continue...[/dim]")
        return

    console.clear()
    console.print(Panel(
        "[bold]❓ Ask a Question[/bold]",
        style="yellow"
    ))

    console.print("Ask any question about your documentation.")
    console.print()

    try:
        query_text = console.input("[cyan]Your question:[/cyan] ").strip()

        if not query_text:
            console.print("[yellow]No question provided.[/yellow]")
            console.input("\n[dim]Press Enter to continue...[/dim]")
            return

        try:
            from .commands import QueryCommand
            command = QueryCommand(settings, console)
            command.run(
                query_text=query_text,
                execute=False,
                dry_run=False,
                alternatives=0,
                explain=True,
                provider=None
            )
        except Exception as e:
            console.print(f"[red]❌ Query failed: {e}[/red]")

        console.input("\n[dim]Press Enter to continue...[/dim]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Question cancelled.[/yellow]")
        console.input("\n[dim]Press Enter to continue...[/dim]")


def handle_chat():
    """Handle interactive chat session with lazy loading."""
    # Lazy load settings
    try:
        from ..config import Settings
        settings = Settings()
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        console.print("Please configure EasyPrompt first (option 1)")
        console.input("\n[dim]Press Enter to continue...[/dim]")
        return

    console.print("[blue]💬 Starting chat session...[/blue]")
    console.print("[dim]You'll return to the main menu when the chat session ends.[/dim]")
    console.input("\n[dim]Press Enter to continue...[/dim]")

    try:
        from .commands import ChatCommand
        command = ChatCommand(settings, console)
        command.run(provider=None)
    except Exception as e:
        console.print(f"[red]❌ Chat failed: {e}[/red]")
        console.input("\n[dim]Press Enter to continue...[/dim]")


def handle_search():
    """Handle document search with lazy loading."""
    # Lazy load settings
    try:
        from ..config import Settings
        settings = Settings()
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        console.print("Please configure EasyPrompt first (option 1)")
        console.input("\n[dim]Press Enter to continue...[/dim]")
        return

    console.clear()
    console.print(Panel(
        "[bold]🔍 Search Documents[/bold]",
        style="cyan"
    ))

    console.print("Search your documentation files directly (no AI processing).")
    console.print()

    try:
        query_text = console.input("[cyan]Search query:[/cyan] ").strip()

        if not query_text:
            console.print("[yellow]No search query provided.[/yellow]")
            console.input("\n[dim]Press Enter to continue...[/dim]")
            return

        try:
            from .search_command import search_documentation
            search_documentation(settings, console, query_text, limit=10)
        except Exception as e:
            console.print(f"[red]❌ Search failed: {e}[/red]")

        console.input("\n[dim]Press Enter to continue...[/dim]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Search cancelled.[/yellow]")
        console.input("\n[dim]Press Enter to continue...[/dim]")


def handle_status():
    """Handle system status display with lazy loading."""
    # Lazy load settings
    try:
        from ..config import Settings
        settings = Settings()
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        console.print("Please configure EasyPrompt first (option 1)")
        console.input("\n[dim]Press Enter to continue...[/dim]")
        return

    console.clear()
    try:
        from .commands import StatusCommand
        command = StatusCommand(settings, console)
        command.run(verbose=True)
    except Exception as e:
        console.print(f"[red]❌ Status check failed: {e}[/red]")

    console.input("\n[dim]Press Enter to continue...[/dim]")