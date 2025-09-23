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
                handle_search()
            elif choice == "5":
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
        ("4", "🔍 Search Documents", "Search your documentation without AI"),
        ("5", "📊 System Status", "View detailed system status and configuration"),
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

    console.print("[bold blue]📖 What is indexing?[/bold blue]")
    console.print("• Processes your documentation files into searchable chunks")
    console.print("• Creates vector embeddings for semantic search")
    console.print("• Stores data in your chosen vector database for fast retrieval")

    console.print(f"\n[bold green]📁 Current Configuration:[/bold green]")
    console.print(f"• Documentation path: [cyan]{settings.docs_path}[/cyan]")
    console.print(f"• Supported file types: [cyan]{settings.supported_file_types}[/cyan]")
    console.print(f"• Chunking strategy: [cyan]{settings.chunking_strategy}[/cyan]")
    console.print(f"• Chunk size: [cyan]{settings.chunk_size}[/cyan] characters")
    console.print(f"• Vector database: [cyan]{settings.vector_db_type}[/cyan]")
    console.print(f"• Embedding model: [cyan]{settings.embedding_model.split('/')[-1]}[/cyan]")

    # Count files that will be processed
    from pathlib import Path
    import os

    try:
        doc_path = Path(settings.docs_path)
        if doc_path.exists():
            file_extensions = [f".{ext.strip()}" for ext in settings.supported_file_types.split(",")]
            files_to_index = []
            for ext in file_extensions:
                files_to_index.extend(list(doc_path.glob(f"*{ext}")))

            console.print(f"\n[bold yellow]📊 Files to Process:[/bold yellow]")
            console.print(f"• Found [yellow]{len(files_to_index)}[/yellow] files in documentation directory")

            # Show first few files as examples
            if files_to_index:
                console.print(f"• Examples: {', '.join([f.name for f in files_to_index[:3]])}{'...' if len(files_to_index) > 3 else ''}")
        else:
            console.print(f"\n[red]⚠️  Documentation path does not exist: {settings.docs_path}[/red]")
    except Exception as e:
        console.print(f"\n[yellow]⚠️  Could not scan documentation path: {e}[/yellow]")

    console.print(f"\n[bold cyan]🔄 What will happen:[/bold cyan]")
    console.print("1. [dim]Scan documentation directory for supported files[/dim]")
    console.print("2. [dim]Parse and extract text content from each file[/dim]")
    console.print("3. [dim]Split text into chunks using your chunking strategy[/dim]")
    console.print("4. [dim]Generate vector embeddings for each chunk[/dim]")
    console.print("5. [dim]Store embeddings in your vector database[/dim]")
    console.print("6. [dim]Create metadata index for fast retrieval[/dim]")

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