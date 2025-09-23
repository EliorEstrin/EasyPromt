"""Initialize command for setting up EasyPrompt configuration."""

import shutil
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.markdown import Markdown
from typing import Dict, List, Any

console = Console()


def safe_prompt_ask(*args, **kwargs):
    """Wrapper for Prompt.ask that handles Ctrl+C gracefully by returning None."""
    try:
        return Prompt.ask(*args, **kwargs)
    except KeyboardInterrupt:
        return None


def safe_confirm_ask(*args, **kwargs):
    """Wrapper for Confirm.ask that handles Ctrl+C gracefully by returning False."""
    try:
        return Confirm.ask(*args, **kwargs)
    except KeyboardInterrupt:
        return False


def safe_input(prompt: str):
    """Wrapper for console.input that handles Ctrl+C gracefully by returning None."""
    try:
        return console.input(prompt)
    except KeyboardInterrupt:
        return None


def load_existing_env_file(env_file: Path) -> Dict[str, str]:
    """Load existing .env file and return configuration as dictionary."""
    config = {}
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    except Exception as e:
        console.print(f"[yellow]Warning: Could not read .env file: {e}[/yellow]")
    return config


def is_config_usable(config: Dict[str, str]) -> bool:
    """Check if the existing configuration has minimum required settings to be usable."""
    # Check for at least one valid LLM provider (not placeholder)
    has_llm = any(
        config.get(key) and not config[key].startswith("your_") and len(config[key]) > 10
        for key in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    )

    # Check for CLI tool name
    has_cli_tool = config.get("CLI_TOOL_NAME") and len(config["CLI_TOOL_NAME"]) > 0

    # Check for vector DB type
    has_vector_db = config.get("VECTOR_DB_TYPE") and config["VECTOR_DB_TYPE"] in ["chromadb", "pinecone", "weaviate"]

    return has_llm and has_cli_tool and has_vector_db


def show_existing_config_status(config: Dict[str, str]):
    """Show status of existing configuration in a user-friendly way."""
    console.print(Panel(
        "[bold green]✅ Configuration Ready[/bold green]",
        style="green"
    ))

    if not config:
        console.print("[yellow]⚠️  .env file exists but appears to be empty or invalid[/yellow]")
        console.print("You can run '[cyan]easyprompt init --force[/cyan]' to recreate it")
        return

    # Analyze configuration
    vector_db = config.get("VECTOR_DB_TYPE", "Not configured")
    cli_tool = config.get("CLI_TOOL_NAME", "Not configured")

    # Count configured LLM providers
    llm_providers = []
    for key in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        if config.get(key) and not config[key].startswith("your_"):
            provider_name = key.replace("_API_KEY", "").title()
            llm_providers.append(provider_name)

    # Show summary
    console.print(f"\n[bold]📋 Configuration Summary:[/bold]")
    console.print(f"  🗄️  Vector Database: [cyan]{vector_db}[/cyan]")
    console.print(f"  ⚙️  CLI Tool: [cyan]{cli_tool}[/cyan]")

    if llm_providers:
        console.print(f"  🤖 LLM Providers: [green]{', '.join(llm_providers)}[/green]")
    else:
        console.print("  🤖 LLM Providers: [yellow]None configured (check for placeholder values)[/yellow]")

    # Show next steps
    console.print(f"\n[bold]🚀 Next Steps:[/bold]")

    if not llm_providers and any(config.get(k, "").startswith("your_") for k in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]):
        console.print("  1. Edit .env file and replace placeholder API keys with real ones")
        console.print("  2. Run: [cyan]easyprompt index[/cyan] to index your documentation")
        console.print("  3. Try: [cyan]easyprompt query \"your request\"[/cyan]")
    elif cli_tool == "Not configured":
        console.print("  1. Edit .env file and set CLI_TOOL_NAME")
        console.print("  2. Run: [cyan]easyprompt index[/cyan] to index your documentation")
        console.print("  3. Try: [cyan]easyprompt query \"your request\"[/cyan]")
    else:
        console.print("  1. Run: [cyan]easyprompt index[/cyan] to index your documentation")
        console.print("  2. Try: [cyan]easyprompt query \"your request\"[/cyan]")
        console.print("  3. Start interactive mode: [cyan]easyprompt chat[/cyan]")

    console.print(f"\n[dim]To reconfigure completely, use: easyprompt init --force[/dim]")


def start_interactive_configuration(existing_config: Dict[str, str], env_file: Path):
    """Start interactive configuration mode with optional existing settings pre-loaded."""
    # Convert existing config to the format expected by the interactive mode
    config = {}

    # Load existing values into config dict
    for key, value in existing_config.items():
        if value and not value.startswith("#"):
            config[key] = value

    if config:
        console.print(f"\n[bold blue]🔄 Current settings loaded - you can modify any section![/bold blue]")
        # Skip welcome screen for existing users - they already know the system
    else:
        console.print(f"\n[bold blue]🎯 Let's configure EasyPrompt step by step![/bold blue]")
        # Show welcome with exploration options for new users only
        welcome_result = show_welcome()
        if welcome_result is None:
            # User pressed Ctrl+C during welcome, exit gracefully
            console.print("[yellow]Configuration cancelled[/yellow]")
            return

    # Main configuration loop
    while True:
        try:
            console.clear()
            show_header()
            show_current_config(config)

            choice = show_main_menu()

            if choice == "1":
                result = configure_vector_database()
                if result:
                    config.update(result)
                    auto_save_config(config, env_file)
            elif choice == "2":
                result = configure_embedding_model()
                if result:
                    config.update(result)
                    auto_save_config(config, env_file)
            elif choice == "3":
                result = configure_llm_providers()
                if result:
                    config.update(result)
                    auto_save_config(config, env_file)
            elif choice == "4":
                result = configure_project_setup()
                if result:
                    config.update(result)
                    auto_save_config(config, env_file)
            elif choice == "5":
                result = configure_documentation_files()
                if result:
                    config.update(result)
                    auto_save_config(config, env_file)
            elif choice == "6":
                result = configure_index_settings()
                if result:
                    config.update(result)
                    auto_save_config(config, env_file)
            elif choice == "7":
                result = configure_advanced_settings()
                if result:
                    config.update(result)
                    auto_save_config(config, env_file)
            elif choice == "8":
                show_configuration_overview(config)
            elif choice == "9":
                show_help_and_tips()
            elif choice == "q":
                console.print("[yellow]Exiting configuration...[/yellow]")
                return

            # Only pause for configuration actions, not info screens
            if choice in ["1", "2", "3", "4", "5", "6", "7"]:
                pass  # These sections handle their own pausing
            elif choice in ["8", "9"]:
                try:
                    console.input("\n[dim]Press Enter to continue...[/dim]")
                except KeyboardInterrupt:
                    # Ctrl+C just goes back to main menu
                    continue

        except KeyboardInterrupt:
            # Ctrl+C in main menu - exit immediately (auto-save already happened)
            console.print("\n[yellow]Exiting configuration...[/yellow]")
            return


def init_configuration(force: bool = False):
    """Initialize EasyPrompt configuration with ultra-interactive interface."""
    env_file = Path(".env")

    # Load existing configuration if available
    existing_config = {}
    if env_file.exists():
        existing_config = load_existing_env_file(env_file)
        if existing_config:
            console.print("[green]✅ Loading existing configuration[/green]")
        else:
            console.print("[yellow]📄 Found empty .env file - starting fresh[/yellow]")
    else:
        console.print("[blue]📝 Creating new configuration[/blue]")

    # Always start interactive configuration (with existing values pre-loaded if available)
    start_interactive_configuration(existing_config, env_file)


def show_welcome():
    """Show enhanced welcome message."""
    welcome_text = """
# 🚀 EasyPrompt Interactive Configuration

Welcome to the **ultra-interactive** configuration setup! This tool will help you:

- **Explore** all available options with detailed explanations
- **Browse** different configurations and see examples
- **Understand** what each setting does and why you might want it
- **Configure** your setup step by step with full control

You can navigate through different sections, get help at any time, and see your
configuration build up as you go!
    """

    console.print(Panel(
        Markdown(welcome_text),
        title="🎯 Interactive Setup",
        border_style="cyan",
        padding=(1, 2)
    ))

    result = safe_input("\n[bold cyan]Press Enter to start exploring...[/bold cyan]")
    if result is None:
        # Ctrl+C pressed, return None to indicate cancellation
        return None
    return True


def show_header():
    """Show the main header."""
    console.print(Panel(
        "[bold blue]🔧 EasyPrompt Configuration Builder[/bold blue]",
        style="blue",
        padding=(0, 2)
    ))


def get_config_status(key: str, value: str) -> str:
    """Get meaningful status description for configuration values."""
    if not value:
        return "[red]Not set[/red]"

    # API Key status checks
    if "API_KEY" in key:
        if value.startswith("your_"):
            return "[yellow]Placeholder set[/yellow]"
        elif len(value) < 10:
            return "[red]Invalid key[/red]"
        else:
            # Test the key if possible
            provider = key.replace("_API_KEY", "").lower()
            if test_provider_key_quick(provider, value):
                return "[green]Key verified[/green]"
            else:
                return "[yellow]Key not tested[/yellow]"

    # Vector DB specific checks
    elif key == "VECTOR_DB_TYPE":
        if value in ["chromadb", "pinecone", "weaviate"]:
            return "[green]Valid type[/green]"
        else:
            return "[red]Invalid type[/red]"

    elif key == "VECTOR_DB_URL":
        if value.startswith("http"):
            return "[green]URL configured[/green]"
        else:
            # Check if local path exists or can be created
            from pathlib import Path
            try:
                path = Path(value)
                if path.exists():
                    return "[green]Path exists[/green]"
                elif path.parent.exists():
                    return "[green]Path ready[/green]"
                else:
                    return "[yellow]Path will be created[/yellow]"
            except:
                return "[yellow]Path configured[/yellow]"

    # Embedding model checks
    elif key == "EMBEDDING_MODEL":
        if value.startswith("sentence-transformers/"):
            return "[green]Model configured[/green]"
        else:
            return "[yellow]Custom model[/yellow]"

    # Project setup checks
    elif key == "PROJECT_NAME":
        if value:
            return "[green]Project name set[/green]"
        else:
            return "[red]Not configured[/red]"

    elif key == "PROJECT_DOMAIN":
        if value:
            return "[green]Domain set[/green]"
        else:
            return "[yellow]Optional[/yellow]"

    # Documentation checks
    elif key == "DOCS_PATH":
        from pathlib import Path
        # Handle multiple directories separated by semicolon
        paths = value.split(";") if ";" in value else [value]

        total_files = 0
        existing_dirs = 0

        for path in paths:
            path_obj = Path(path.strip())
            if path_obj.exists() and path_obj.is_dir():
                existing_dirs += 1
                # Count supported files (only in immediate directory, not recursive)
                md_files = len(list(path_obj.glob("*.md"))) + len(list(path_obj.glob("*.markdown")))
                txt_files = len(list(path_obj.glob("*.txt")))
                pdf_files = len(list(path_obj.glob("*.pdf")))
                total_files += md_files + txt_files + pdf_files

        if existing_dirs == len(paths) and total_files > 0:
            return f"[green]{total_files} files found[/green]"
        elif existing_dirs == len(paths) and total_files == 0:
            return f"[yellow]{existing_dirs} dir(s), no files[/yellow]"
        elif existing_dirs > 0:
            return f"[yellow]{existing_dirs}/{len(paths)} dirs, {total_files} files[/yellow]"
        else:
            return "[yellow]Dirs will be created[/yellow]"

    elif key == "SUPPORTED_FILE_TYPES":
        types = [t.strip() for t in value.split(",") if t.strip()] if value else []
        if len(types) > 0:
            # Show the actual extensions
            ext_display = ", ".join(f".{t}" for t in types)
            return f"[green]{ext_display}[/green]"
        else:
            return "[red]No file types[/red]"

    # Advanced settings
    elif key in ["MAX_CONTEXT_LENGTH", "TOP_K_RESULTS"]:
        try:
            num = int(value)
            if num > 0:
                return "[green]Valid number[/green]"
            else:
                return "[red]Invalid value[/red]"
        except:
            return "[red]Not a number[/red]"

    elif key == "SIMILARITY_THRESHOLD":
        try:
            num = float(value)
            if 0.0 <= num <= 1.0:
                return "[green]Valid threshold[/green]"
            else:
                return "[red]Out of range[/red]"
        except:
            return "[red]Invalid number[/red]"

    elif key in ["DRY_RUN", "CONFIRM_BEFORE_EXECUTION"]:
        if value.lower() in ["true", "false"]:
            return "[green]Boolean set[/green]"
        else:
            return "[red]Invalid boolean[/red]"

    # Default for any other settings
    else:
        return "[green]Configured[/green]"


def test_provider_key_quick(provider: str, api_key: str) -> bool:
    """Quick test of provider key without detailed output."""
    import logging

    # Suppress HTTP request logs during testing
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("google.generativeai").setLevel(logging.WARNING)

    try:
        if provider == "gemini":
            return test_gemini_key_quick(api_key)
        elif provider == "openai":
            return test_openai_key_quick(api_key)
        elif provider == "anthropic":
            return test_anthropic_key_quick(api_key)
        else:
            return False
    except:
        return False


def test_gemini_key_quick(api_key: str) -> bool:
    """Quick test of Gemini key."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        models = list(genai.list_models())
        return len(models) > 0
    except:
        return False


def test_openai_key_quick(api_key: str) -> bool:
    """Quick test of OpenAI key."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        models = client.models.list()
        return len(models.data) > 0
    except:
        return False


def test_anthropic_key_quick(api_key: str) -> bool:
    """Quick test of Anthropic key."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        # Try a minimal request
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1,
            messages=[{"role": "user", "content": "Hi"}]
        )
        return True
    except:
        return False


def show_current_config(config: Dict[str, Any]):
    """Show current configuration status."""
    if not config:
        console.print("[dim]No configuration set yet. Start by choosing a section below![/dim]\n")
        return

    table = Table(title="📋 Current Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    table.add_column("Status", justify="left", style="white")

    # Group settings by category
    categories = {
        "Vector DB": ["VECTOR_DB_TYPE", "VECTOR_DB_URL", "PINECONE_API_KEY", "WEAVIATE_URL"],
        "Embedding Model": ["EMBEDDING_MODEL"],
        "LLM Providers": ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        "Project": ["PROJECT_NAME", "PROJECT_DOMAIN"],
        "Documentation": ["DOCS_PATH", "SUPPORTED_FILE_TYPES"],
        "Index Settings": ["CHUNKING_STRATEGY", "CHUNK_SIZE", "CHUNK_OVERLAP", "INDEX_STORAGE_PATH", "CHUNKED_DOCS_PATH"],
        "Advanced": ["MAX_CONTEXT_LENGTH", "TOP_K_RESULTS", "SIMILARITY_THRESHOLD"]
    }

    for category, keys in categories.items():
        category_items = [(k, v) for k, v in config.items() if k in keys]
        if category_items:
            table.add_row(f"[bold]{category}[/bold]", "", "")
            for key, value in category_items:
                display_value = "***" if "API_KEY" in key else str(value)
                status = get_config_status(key, value)
                table.add_row(f"  {key}", display_value, status)

    console.print(table)
    console.print()


def show_main_menu() -> str:
    """Show main menu and get user choice."""
    console.print("[bold yellow]📂 Configuration Sections[/bold yellow]")

    menu_options = [
        ("1", "🗄️  Vector Database", "Choose and configure your vector database"),
        ("2", "🔤 Embedding Model", "Configure text embedding model for RAG"),
        ("3", "🤖 LLM Providers", "Configure AI language model providers"),
        ("4", "📋 Project Setup", "Configure your project or domain name"),
        ("5", "📚 Documentation Files", "Configure documentation file sources"),
        ("6", "⚙️  Index Settings", "Configure chunking strategy and indexing options"),
        ("7", "🎛️  Advanced Settings", "Performance and behavior tuning"),
        ("8", "👀 Preview Configuration", "See complete configuration overview"),
        ("9", "❓ Help & Tips", "Get help and see examples"),
        ("q", "🚪 Exit", "Exit configuration")
    ]

    for key, title, desc in menu_options:
        console.print(f"  {key}. [bold cyan]{title}[/bold cyan] - [dim]{desc}[/dim]")

    console.print()
    try:
        choice = Prompt.ask(
            "[bold]Choose section",
            choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "q"],
            default="1"
        )
        return choice
    except KeyboardInterrupt:
        # Let the KeyboardInterrupt bubble up to the main loop
        raise


def configure_vector_database() -> Dict[str, Any]:
    """Configure vector database with full exploration."""
    console.clear()
    console.print(Panel(
        "[bold]🗄️ Vector Database Configuration[/bold]",
        style="blue"
    ))

    # Show detailed comparison
    show_vector_db_comparison()

    console.print("\n[bold yellow]🗄️ Vector Database Options:[/bold yellow]")
    console.print("  [cyan]compare[/cyan]   - Show detailed comparison")
    console.print("  [cyan]chromadb[/cyan]  - Local database (recommended)")
    console.print("  [cyan]pinecone[/cyan]  - Cloud database")
    console.print("  [cyan]weaviate[/cyan]  - Self-hosted/cloud database")
    console.print("  [cyan]q[/cyan]         - 🚪 Back to main configuration menu")

    while True:
        try:
            choice = Prompt.ask(
                "\nChoose action",
                choices=["compare", "chromadb", "pinecone", "weaviate", "q"],
                default="chromadb"
            )
        except KeyboardInterrupt:
            # Ctrl+C goes back to main configuration menu
            return {}

        if choice == "compare":
            show_vector_db_comparison()
        elif choice == "chromadb":
            return configure_chromadb()
        elif choice == "pinecone":
            return configure_pinecone()
        elif choice == "weaviate":
            return configure_weaviate()
        elif choice == "q":
            return {}


def show_vector_db_comparison():
    """Show detailed vector database comparison."""
    table = Table(title="🗄️ Vector Database Comparison", show_header=True)
    table.add_column("Database", style="cyan", no_wrap=True)
    table.add_column("Type", style="yellow")
    table.add_column("Pros", style="green")
    table.add_column("Cons", style="red")
    table.add_column("Best For", style="blue")

    table.add_row(
        "ChromaDB",
        "Local",
        "• Free\n• Easy setup\n• No API keys\n• Good for dev",
        "• Local only\n• Limited scale",
        "Development\nPrototyping"
    )

    table.add_row(
        "Pinecone",
        "Cloud",
        "• High performance\n• Scalable\n• Managed service",
        "• Requires API key\n• Cost for usage",
        "Production\nHigh scale"
    )

    table.add_row(
        "Weaviate",
        "Hybrid",
        "• Open source\n• GraphQL API\n• Flexible",
        "• More complex\n• Requires hosting",
        "Custom solutions\nAdvanced features"
    )

    console.print(table)


def configure_chromadb() -> Dict[str, Any]:
    """Configure ChromaDB with detailed options."""
    config = {"VECTOR_DB_TYPE": "chromadb"}

    console.print("\n[bold green]✅ ChromaDB Selected[/bold green]")
    console.print("ChromaDB is a local vector database - perfect for development!")

    # Show path options
    console.print("\n[bold]Database Storage Options:[/bold]")
    console.print("1. [cyan]./data/chroma.db[/cyan] (default, in project)")
    console.print("2. [cyan]~/.easyprompt/chroma.db[/cyan] (user home)")
    console.print("3. [cyan]Custom path[/cyan]")

    try:
        path_choice = Prompt.ask("Choose path option", choices=["1", "2", "3"], default="1")

        if path_choice == "1":
            config["VECTOR_DB_URL"] = "./data/chroma.db"
        elif path_choice == "2":
            config["VECTOR_DB_URL"] = "~/.easyprompt/chroma.db"
        else:
            custom_path = Prompt.ask("Enter custom path", default="./data/chroma.db")
            config["VECTOR_DB_URL"] = custom_path
    except KeyboardInterrupt:
        # Ctrl+C goes back to vector database menu
        return {}

    console.print(f"\n[green]✅ ChromaDB will store data at: {config['VECTOR_DB_URL']}[/green]")

    return config


def configure_pinecone() -> Dict[str, Any]:
    """Configure Pinecone with detailed setup."""
    config = {"VECTOR_DB_TYPE": "pinecone"}

    console.print("\n[bold blue]🌲 Pinecone Configuration[/bold blue]")
    console.print("Pinecone is a managed vector database service. You'll need:")
    console.print("• API Key from https://app.pinecone.io/")
    console.print("• Environment name (from your Pinecone console)")
    console.print("• Index name (will be created if it doesn't exist)")

    try:
        if not Confirm.ask("\nDo you have a Pinecone account and API key?"):
            console.print("[yellow]ℹ️  Visit https://app.pinecone.io/ to create an account first[/yellow]")
            return {}

        config["PINECONE_API_KEY"] = Prompt.ask("Pinecone API key", password=True)
        config["PINECONE_ENVIRONMENT"] = Prompt.ask("Pinecone environment (e.g., 'us-east1-aws')")
        config["PINECONE_INDEX_NAME"] = Prompt.ask("Index name", default="easyprompt-index")
    except KeyboardInterrupt:
        # Ctrl+C goes back to vector database menu
        return {}

    console.print("\n[green]✅ Pinecone configured![/green]")
    return config


def configure_weaviate() -> Dict[str, Any]:
    """Configure Weaviate with detailed setup."""
    config = {"VECTOR_DB_TYPE": "weaviate"}

    console.print("\n[bold purple]🕸️ Weaviate Configuration[/bold purple]")
    console.print("Weaviate options:")
    console.print("1. [cyan]Local Weaviate[/cyan] (Docker required)")
    console.print("2. [cyan]Weaviate Cloud[/cyan] (WCS)")
    console.print("3. [cyan]Custom Weaviate instance[/cyan]")

    try:
        setup_choice = Prompt.ask("Choose setup", choices=["1", "2", "3"], default="1")

        if setup_choice == "1":
            config["WEAVIATE_URL"] = "http://localhost:8080"
            console.print("\n[yellow]ℹ️  Make sure to run Weaviate locally:[/yellow]")
            console.print("docker run -p 8080:8080 semitechnologies/weaviate:latest")
        elif setup_choice == "2":
            config["WEAVIATE_URL"] = Prompt.ask("WCS cluster URL")
            config["WEAVIATE_API_KEY"] = Prompt.ask("WCS API key", password=True)
        else:
            config["WEAVIATE_URL"] = Prompt.ask("Custom Weaviate URL", default="http://localhost:8080")
    except KeyboardInterrupt:
        # Ctrl+C goes back to vector database menu
        return {}

    console.print("\n[green]✅ Weaviate configured![/green]")
    return config


def configure_embedding_model() -> Dict[str, Any]:
    """Configure embedding model with detailed options and explanations."""
    console.clear()
    console.print(Panel(
        "[bold]🔤 Embedding Model Configuration[/bold]",
        style="blue"
    ))

    console.print("[bold blue]What is this?[/bold blue]")
    console.print("• The embedding model converts text into numerical vectors for similarity search")
    console.print("• This is the core of the RAG system - it determines how well documents are matched to queries")
    console.print("• Different models have trade-offs between speed, accuracy, and memory usage")
    console.print("• Better embeddings = more accurate CLI command generation")

    console.print("\n[bold green]🏠 Local & Private:[/bold green]")
    console.print("• All models run locally on your machine - no API calls for embeddings")
    console.print("• Your documentation never leaves your system during embedding")
    console.print("• Models downloaded once from HuggingFace, then cached locally")
    console.print("• Works offline after initial download")

    show_embedding_model_comparison()

    console.print("\n[bold yellow]Model Options:[/bold yellow]")
    console.print("1. [cyan]all-MiniLM-L6-v2[/cyan] - Recommended (fast, good quality, 22MB)")
    console.print("2. [cyan]all-mpnet-base-v2[/cyan] - High accuracy (slower, excellent quality, 420MB)")
    console.print("3. [cyan]all-MiniLM-L12-v2[/cyan] - Balanced (good speed, very good quality, 120MB)")
    console.print("4. [cyan]paraphrase-multilingual[/cyan] - Multiple languages (slow, 970MB)")
    console.print("5. [cyan]Custom model[/cyan] - Specify any HuggingFace model")

    console.print("  q. 🚪 Back - Return to main configuration menu")

    try:
        choice = Prompt.ask(
            "\nChoose model",
            choices=["1", "2", "3", "4", "5", "q"],
            default="1"
        )
    except KeyboardInterrupt:
        # Ctrl+C goes back to main configuration menu
        return {}

    if choice == "1":
        console.print("[green]✅ Using all-MiniLM-L6-v2 (recommended)[/green]")
        return {"EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"}
    elif choice == "2":
        console.print("[green]✅ Using all-mpnet-base-v2 (high accuracy)[/green]")
        return {"EMBEDDING_MODEL": "sentence-transformers/all-mpnet-base-v2"}
    elif choice == "3":
        console.print("[green]✅ Using all-MiniLM-L12-v2 (balanced)[/green]")
        return {"EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L12-v2"}
    elif choice == "4":
        console.print("[green]✅ Using multilingual model[/green]")
        return {"EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}
    elif choice == "5":
        return configure_custom_embedding()
    elif choice == "q":
        return {}
    else:
        return {}


def show_embedding_model_comparison():
    """Show simplified embedding model comparison."""
    table = Table(title="🔤 Embedding Model Comparison", show_header=True)
    table.add_column("Model", style="cyan")
    table.add_column("Speed", style="yellow")
    table.add_column("Quality", style="green")
    table.add_column("Size", style="red")
    table.add_column("Best For", style="blue")

    table.add_row(
        "all-MiniLM-L6-v2",
        "⚡⚡⚡ Fastest",
        "⭐⭐⭐ Good",
        "22MB",
        "General use, fast search"
    )

    table.add_row(
        "all-mpnet-base-v2",
        "⚡⚡ Fast",
        "⭐⭐⭐⭐ Excellent",
        "420MB",
        "High accuracy, production"
    )

    table.add_row(
        "all-MiniLM-L12-v2",
        "⚡⚡ Fast",
        "⭐⭐⭐⭐ Very Good",
        "120MB",
        "Balanced performance"
    )

    table.add_row(
        "paraphrase-multilingual",
        "⚡ Slower",
        "⭐⭐⭐ Good",
        "970MB",
        "Multiple languages"
    )

    console.print(table)

    console.print("\n[bold blue]💡 Key Info:[/bold blue]")
    console.print("• All models run locally (private, no API calls)")
    console.print("• Downloaded once from HuggingFace, then cached locally")
    console.print("• Compatible with any LLM provider (OpenAI, Claude, Gemini)")


def configure_recommended_embedding() -> Dict[str, Any]:
    """Configure recommended embedding model."""
    console.print("\n[bold green]✅ Recommended: all-MiniLM-L6-v2[/bold green]")
    console.print("This is the best balance of speed, accuracy, and memory usage for CLI documentation.")

    console.print("\n[bold]Model Details:[/bold]")
    console.print("• [bold]Provider:[/bold] Microsoft Research")
    console.print("• [bold]Type:[/bold] Local model (🏠 runs on your machine)")
    console.print("• [bold]Speed:[/bold] Very fast (⚡⚡⚡) - ~0.1s per document")
    console.print("• [bold]Accuracy:[/bold] Good (⭐⭐⭐) - excellent for CLI docs")
    console.print("• [bold]Size:[/bold] Only 22MB download")
    console.print("• [bold]Languages:[/bold] English optimized")
    console.print("• [bold]Best for:[/bold] CLI commands, technical documentation, quick searches")

    console.print("\n[bold cyan]📊 What to expect:[/bold cyan]")
    console.print("• [bold]First use:[/bold] ~30s download from HuggingFace")
    console.print("• [bold]Storage:[/bold] Cached in ~/.cache/huggingface/")
    console.print("• [bold]Performance:[/bold] Near-instant embedding after download")
    console.print("• [bold]Privacy:[/bold] No data sent to external servers")

    if Confirm.ask("\nUse recommended model?", default=True):
        console.print("[green]✅ Using sentence-transformers/all-MiniLM-L6-v2[/green]")
        console.print("[dim]Model will download automatically on first use[/dim]")
        return {"EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"}

    return {}


def configure_fast_embedding() -> Dict[str, Any]:
    """Configure fastest embedding model."""
    console.print("\n[bold yellow]⚡ Fastest: all-MiniLM-L6-v2[/bold yellow]")
    console.print("Optimized for speed with minimal memory usage.")

    console.print("\n[bold]Why choose fast?[/bold]")
    console.print("• Instant search responses")
    console.print("• Low memory usage (22MB)")
    console.print("• Good enough accuracy for most CLI tools")
    console.print("• Works well on low-resource systems")

    if Confirm.ask("\nUse fastest model?", default=True):
        console.print("[green]✅ Using fastest model: all-MiniLM-L6-v2[/green]")
        return {"EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"}

    return {}


def configure_multilingual_embedding() -> Dict[str, Any]:
    """Configure multilingual embedding model."""
    console.print("\n[bold purple]🌍 Multilingual: paraphrase-multilingual-MiniLM-L12-v2[/bold purple]")
    console.print("Supports multiple languages for international documentation.")

    console.print("\n[bold]Model Details:[/bold]")
    console.print("• [bold]Provider:[/bold] SentenceTransformers Team")
    console.print("• [bold]Type:[/bold] Local model (🏠 runs on your machine)")
    console.print("• [bold]Languages:[/bold] 50+ languages supported")
    console.print("• [bold]Cross-lingual:[/bold] Query in English, find docs in other languages")
    console.print("• [bold]Size:[/bold] 970MB download (much larger)")
    console.print("• [bold]Speed:[/bold] Slower than English-only models")

    console.print("\n[bold cyan]📊 What to expect:[/bold cyan]")
    console.print("• [bold]Download time:[/bold] ~5-10 minutes depending on connection")
    console.print("• [bold]Storage:[/bold] ~1GB disk space required")
    console.print("• [bold]Performance:[/bold] 2-3x slower than English models")
    console.print("• [bold]Use case:[/bold] Teams with documentation in multiple languages")

    console.print("\n[bold yellow]⚠️  Important:[/bold yellow]")
    console.print("• Only choose this if you have non-English documentation")
    console.print("• For English-only CLI docs, stick with faster models")
    console.print("• Requires more memory and processing power")

    if Confirm.ask("\nUse multilingual model?", default=False):
        console.print("[green]✅ Using multilingual model[/green]")
        console.print("[dim]Large model - will take several minutes to download[/dim]")
        return {"EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}

    return {}


def configure_custom_embedding() -> Dict[str, Any]:
    """Configure custom embedding model."""
    console.print("\n[bold cyan]🛠️ Custom Embedding Model[/bold cyan]")
    console.print("Enter any sentence-transformers model from HuggingFace.")

    model_name = Prompt.ask(
        "\nModel name (e.g., sentence-transformers/all-mpnet-base-v2)",
        default="sentence-transformers/all-MiniLM-L6-v2"
    )

    if model_name:
        console.print(f"[green]✅ Using custom model: {model_name}[/green]")
        return {"EMBEDDING_MODEL": model_name}

    return {}


def configure_llm_providers() -> Dict[str, Any]:
    """Configure LLM providers with enhanced developer experience."""
    console.clear()
    console.print(Panel(
        "[bold]🤖 LLM Provider Configuration[/bold]",
        style="green"
    ))

    # Check for existing environment keys first
    existing_keys = detect_existing_api_keys()
    config = {}

    if existing_keys:
        console.print("[bold green]🔍 Found existing API keys in environment![/bold green]")
        if import_existing_keys_interactive(existing_keys, config):
            console.print("[green]✅ Imported existing keys![/green]")
            console.input("\n[dim]Press Enter to continue...[/dim]")

    # Show comparison and smart recommendations
    show_llm_comparison()
    show_smart_recommendations(config)

    # Offer quick setup options
    if not config:
        quick_setup = offer_quick_setup()
        if quick_setup:
            config.update(quick_setup)

    # Interactive provider management loop
    while True:
        console.clear()
        console.print(Panel(
            "[bold]🤖 LLM Provider Configuration[/bold]",
            style="green"
        ))

        # Show current status with recommendations
        show_provider_status_enhanced(config)

        # Show menu options with smart defaults
        console.print("\n[bold yellow]📋 Configuration Options:[/bold yellow]")
        console.print("  [cyan]quick[/cyan]    - Quick setup (recommended providers)")
        console.print("  [cyan]add[/cyan]      - Add/configure a new provider")
        console.print("  [cyan]test[/cyan]     - Test existing providers")
        console.print("  [cyan]remove[/cyan]   - Remove a provider")
        console.print("  [cyan]status[/cyan]   - Show detailed provider status")
        console.print("  [cyan]done[/cyan]     - Finish configuration")
        console.print("  [cyan]q[/cyan]        - 🚪 Back to main configuration menu")

        if not has_valid_providers(config):
            console.print("\n[red]⚠️  You need at least one working provider![/red]")
            console.print("[yellow]💡 Try 'quick' for fast setup with recommended providers[/yellow]")

        valid_choices = ["quick", "add", "test", "remove", "status", "done", "q"]
        default_choice = "quick" if not has_valid_providers(config) else "done"

        try:
            choice = Prompt.ask(
                "\nChoose action",
                choices=valid_choices,
                default=default_choice
            )
        except KeyboardInterrupt:
            # Ctrl+C goes back to main configuration menu
            return config

        if choice == "quick":
            result = quick_setup_flow()
            if result:
                config.update(result)

        elif choice == "add":
            result = add_provider_with_guidance()
            if result:
                config.update(result)

        elif choice == "test":
            test_providers_with_progress(config)

        elif choice == "remove":
            remove_provider_interactive(config)

        elif choice == "status":
            show_detailed_provider_status(config)

        elif choice == "done":
            if not has_valid_providers(config):
                if not Confirm.ask("No working providers configured. Continue anyway?", default=False):
                    continue
            break

        elif choice == "q":
            # User chose to go back to main configuration menu
            return config

        if choice not in ["done", "q"]:
            console.input("\n[dim]Press Enter to continue...[/dim]")

    return config


def detect_existing_api_keys() -> Dict[str, str]:
    """Detect existing API keys in environment variables."""
    import os

    keys = {}
    env_vars = {
        "GEMINI_API_KEY": "gemini",
        "OPENAI_API_KEY": "openai",
        "ANTHROPIC_API_KEY": "anthropic"
    }

    for env_var, provider in env_vars.items():
        value = os.getenv(env_var)
        if value and not value.startswith("your_"):
            keys[provider] = value

    return keys


def import_existing_keys_interactive(existing_keys: Dict[str, str], config: Dict[str, Any]) -> bool:
    """Interactive import of existing API keys."""
    console.print("\nFound these API keys in your environment:")

    for provider, key in existing_keys.items():
        masked_key = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
        console.print(f"  🔮 {provider.title()}: {masked_key}")

    if Confirm.ask("\nImport and test these keys?", default=True):
        imported_count = 0

        for provider, key in existing_keys.items():
            console.print(f"\n[bold]Testing {provider.title()}...[/bold]")

            if test_provider_key(provider, key):
                config[f"{provider.upper()}_API_KEY"] = key
                console.print(f"[green]✅ {provider.title()} imported and working![/green]")
                imported_count += 1
            else:
                console.print(f"[red]❌ {provider.title()} key found but not working[/red]")
                if Confirm.ask(f"Import {provider.title()} anyway?", default=False):
                    config[f"{provider.upper()}_API_KEY"] = key
                    imported_count += 1

        return imported_count > 0

    return False


def show_smart_recommendations(config: Dict[str, Any]):
    """Show professional provider recommendations."""
    if has_valid_providers(config):
        return

    console.print("\n[bold blue]Provider Options:[/bold blue]")

    if not config:
        console.print("• [cyan]OpenAI GPT[/cyan] - Industry standard, reliable performance")
        console.print("• [cyan]Anthropic Claude[/cyan] - Advanced reasoning capabilities")
        console.print("• [cyan]Google Gemini[/cyan] - Fast processing, competitive pricing")
    else:
        working_providers = [k.replace("_API_KEY", "").lower() for k in config.keys() if k.endswith("_API_KEY")]

        if "openai" not in working_providers:
            console.print("• Consider [cyan]OpenAI[/cyan] for reliable performance")
        if "anthropic" not in working_providers:
            console.print("• Consider [cyan]Anthropic[/cyan] for advanced reasoning")
        if "gemini" not in working_providers:
            console.print("• Consider [cyan]Gemini[/cyan] for fast processing")
        if len(working_providers) == 1:
            console.print("• Multiple providers enable automatic fallback")


def offer_quick_setup() -> Dict[str, Any]:
    """Offer professional setup options."""
    console.print("\n[bold yellow]Setup Options:[/bold yellow]")
    console.print("1. [cyan]OpenAI Only[/cyan] - Industry standard LLM provider")
    console.print("2. [cyan]Claude Only[/cyan] - Anthropic's advanced reasoning model")
    console.print("3. [cyan]Gemini Only[/cyan] - Google's fast processing model")
    console.print("4. [cyan]OpenAI + Claude[/cyan] - Premium combination with fallback")
    console.print("5. [cyan]All Providers[/cyan] - Maximum compatibility and redundancy")
    console.print("6. [cyan]Custom Setup[/cyan] - Configure providers individually")

    console.print("  q. 🚪 Back - Return to main configuration menu")

    try:
        choice = Prompt.ask(
            "\nSetup option",
            choices=["1", "2", "3", "4", "5", "6", "q"],
            default="1"
        )
    except KeyboardInterrupt:
        # Ctrl+C goes back to main configuration menu
        return {}

    if choice == "1":
        return quick_setup_openai_only()
    elif choice == "2":
        return quick_setup_claude_only()
    elif choice == "3":
        return quick_setup_gemini_only()
    elif choice == "4":
        return quick_setup_openai_claude()
    elif choice == "5":
        return quick_setup_all_providers()
    elif choice == "6":
        return configure_custom_llm_setup()
    elif choice == "q":
        return {}
    else:
        return {}


def quick_setup_flow() -> Dict[str, Any]:
    """Enhanced quick setup flow."""
    console.print("\n[bold blue]🚀 Quick Provider Setup[/bold blue]")
    console.print("Let's get you set up with working providers quickly!")

    # Recommend based on what's missing
    recommendations = get_setup_recommendations()

    console.print(f"\n[bold]Recommended setup:[/bold] {recommendations['title']}")
    console.print(f"[dim]{recommendations['description']}[/dim]")

    try:
        if Confirm.ask(f"\nSet up {recommendations['title']}?", default=True):
            return execute_quick_setup(recommendations['providers'])
        else:
            return add_provider_with_guidance()
    except KeyboardInterrupt:
        # Ctrl+C goes back to main configuration menu
        return {}


def get_setup_recommendations() -> Dict[str, Any]:
    """Get professional setup recommendations."""
    return {
        "title": "OpenAI + Claude",
        "description": "Industry standard with advanced reasoning fallback",
        "providers": ["openai", "anthropic"]
    }


def execute_quick_setup(providers: List[str]) -> Dict[str, Any]:
    """Execute quick setup for recommended providers."""
    config = {}

    console.print(f"\n[bold]Setting up {len(providers)} providers...[/bold]")

    for i, provider in enumerate(providers, 1):
        console.print(f"\n[bold cyan]Step {i}/{len(providers)}: {provider.title()}[/bold cyan]")

        if provider == "gemini":
            result = configure_gemini_with_key()
        elif provider == "openai":
            result = configure_openai_with_key()
        elif provider == "anthropic":
            result = configure_anthropic_with_key()
        else:
            continue

        if result:
            # Test immediately with progress
            key = list(result.values())[0]
            console.print(f"🧪 Testing {provider.title()}...")

            if test_provider_key(provider, key):
                config.update(result)
                console.print(f"[green]✅ {provider.title()} working![/green]")
            else:
                console.print(f"[red]❌ {provider.title()} test failed[/red]")
                if Confirm.ask(f"Keep {provider.title()} anyway?", default=False):
                    config.update(result)
        else:
            console.print(f"[yellow]⏭️  Skipped {provider.title()}[/yellow]")

    if config:
        console.print(f"\n[green]✅ Quick setup complete! {len(config)} provider(s) configured[/green]")

    return config


def show_provider_status_enhanced(config: Dict[str, Any]):
    """Enhanced provider status with recommendations."""
    console.print("\n[bold]📊 Current Provider Status:[/bold]")

    providers = [
        ("GEMINI_API_KEY", "🔮 Google Gemini", "Free tier available"),
        ("OPENAI_API_KEY", "🧠 OpenAI GPT", "High quality responses"),
        ("ANTHROPIC_API_KEY", "🤖 Anthropic Claude", "Advanced reasoning")
    ]

    configured_count = 0
    working_count = 0

    for key, name, benefit in providers:
        if key in config:
            if test_provider_key(key.replace("_API_KEY", "").lower(), config[key]):
                status = "[green]✅ Working[/green]"
                working_count += 1
            else:
                status = "[yellow]⚠️ Configured but failing[/yellow]"
            configured_count += 1
        else:
            status = f"[red]❌ Not configured[/red] - [dim]{benefit}[/dim]"

        console.print(f"  {name}: {status}")

    # Show summary with recommendations
    if working_count == 0:
        console.print("\n[red]⚠️  No working providers! You need at least one.[/red]")
    elif working_count == 1:
        console.print(f"\n[yellow]📝 {working_count} working provider. Consider adding a backup![/yellow]")
    else:
        console.print(f"\n[green]✅ {working_count} working providers - excellent setup![/green]")


def add_provider_with_guidance() -> Dict[str, Any]:
    """Add provider with professional guidance."""
    console.print("\n[bold blue]➕ Add Provider[/bold blue]")

    providers = [
        ("openai", "🧠 OpenAI GPT", "Industry standard, reliable performance"),
        ("anthropic", "🤖 Anthropic Claude", "Advanced reasoning capabilities"),
        ("gemini", "🔮 Google Gemini", "Fast processing, competitive pricing")
    ]

    console.print("\nAvailable providers:")
    for i, (key, name, description) in enumerate(providers, 1):
        console.print(f"  {i}. {name}")
        console.print(f"     [dim]{description}[/dim]")

    choice = Prompt.ask(
        "Choose provider to add",
        choices=[str(i) for i in range(1, len(providers) + 1)],
        default="1"
    )

    provider = providers[int(choice) - 1][0]

    # Configure with immediate testing
    console.print(f"\n[bold]Configuring {provider.title()}...[/bold]")

    if provider == "gemini":
        result = configure_gemini_with_key()
    elif provider == "openai":
        result = configure_openai_with_key()
    elif provider == "anthropic":
        result = configure_anthropic_with_key()
    else:
        return {}

    if result:
        # Test with detailed feedback
        key = list(result.values())[0]
        console.print(f"\n[bold blue]Testing {provider.title()}...[/bold blue]")

        if test_provider_key(provider, key):
            console.print(f"[green]✅ {provider.title()} configured and tested successfully[/green]")
            return result
        else:
            console.print(f"[red]❌ {provider.title()} test failed[/red]")
            provide_troubleshooting_help(provider)

            if Confirm.ask(f"Add {provider.title()} anyway?", default=False):
                return result

    return {}


def test_providers_with_progress(config: Dict[str, Any]):
    """Test providers with progress indication."""
    console.print("\n[bold blue]🧪 Testing All Providers[/bold blue]")

    testable_providers = [(k.replace("_API_KEY", "").lower(), k, v)
                         for k, v in config.items()
                         if k.endswith("_API_KEY") and v and not v.startswith("your_")]

    if not testable_providers:
        console.print("[yellow]No configured providers to test![/yellow]")
        return

    console.print(f"Testing {len(testable_providers)} provider(s)...\n")

    results = []
    for i, (name, key, value) in enumerate(testable_providers, 1):
        console.print(f"[{i}/{len(testable_providers)}] Testing {name.title()}...", end=" ")

        success = test_provider_key(name, value)
        if success:
            console.print("[green]✅ Working[/green]")
            results.append((name, True, None))
        else:
            console.print("[red]❌ Failed[/red]")
            results.append((name, False, "Connection failed"))

    # Summary
    working = sum(1 for _, success, _ in results if success)
    total = len(results)

    console.print(f"\n[bold]Test Results:[/bold]")
    console.print(f"✅ Working: {working}/{total}")

    if working < total:
        console.print(f"❌ Failed: {total - working}/{total}")
        console.print("\n[yellow]💡 For failed providers, try:[/yellow]")
        console.print("• Check API key validity")
        console.print("• Verify network connection")
        console.print("• Check account billing/limits")


def provide_troubleshooting_help(provider: str):
    """Provide specific troubleshooting help for failed providers."""
    console.print(f"\n[yellow]🔧 Troubleshooting {provider.title()}:[/yellow]")

    if provider == "gemini":
        console.print("• Verify key at: https://makersuite.google.com/app/apikey")
        console.print("• Ensure key starts with 'AI'")
        console.print("• Check if Gemini API is enabled")

    elif provider == "openai":
        console.print("• Verify key at: https://platform.openai.com/api-keys")
        console.print("• Ensure key starts with 'sk-'")
        console.print("• Check billing and usage limits")
        console.print("• Verify account is in good standing")

    elif provider == "anthropic":
        console.print("• Verify key at: https://console.anthropic.com/")
        console.print("• Ensure key starts with 'sk-ant-'")
        console.print("• Check API access and billing")

    console.print("• Verify network connection")
    console.print("• Try again in a few minutes")


def quick_setup_openai_only() -> Dict[str, Any]:
    """Quick setup for OpenAI only."""
    console.print("\n[bold blue]🧠 Setting up OpenAI GPT[/bold blue]")
    return configure_openai_with_key()


def quick_setup_claude_only() -> Dict[str, Any]:
    """Quick setup for Claude only."""
    console.print("\n[bold purple]🤖 Setting up Anthropic Claude[/bold purple]")
    return configure_anthropic_with_key()


def quick_setup_gemini_only() -> Dict[str, Any]:
    """Quick setup for Gemini only."""
    console.print("\n[bold green]🔮 Setting up Google Gemini[/bold green]")
    return configure_gemini_with_key()


def quick_setup_openai_claude() -> Dict[str, Any]:
    """Quick setup for OpenAI + Claude."""
    console.print("\n[bold blue]🧠🤖 Setting up OpenAI + Claude[/bold blue]")
    config = {}

    # OpenAI first
    console.print("\n[bold]Step 1: OpenAI GPT[/bold]")
    openai_result = configure_openai_with_key()
    if openai_result:
        config.update(openai_result)

    # Claude second
    console.print("\n[bold]Step 2: Anthropic Claude[/bold]")
    claude_result = configure_anthropic_with_key()
    if claude_result:
        config.update(claude_result)

    return config


def quick_setup_all_providers() -> Dict[str, Any]:
    """Quick setup for all providers."""
    console.print("\n[bold purple]🧠🤖🔮 Setting up All Providers[/bold purple]")
    config = {}

    for i, (provider, name) in enumerate([
        ("openai", "OpenAI GPT"),
        ("anthropic", "Anthropic Claude"),
        ("gemini", "Google Gemini")
    ], 1):
        console.print(f"\n[bold]Step {i}: {name}[/bold]")

        if provider == "openai":
            result = configure_openai_with_key()
        elif provider == "anthropic":
            result = configure_anthropic_with_key()
        elif provider == "gemini":
            result = configure_gemini_with_key()

        if result:
            config.update(result)

    return config


def show_provider_status(config: Dict[str, Any]):
    """Show current provider configuration status."""
    console.print("\n[bold]📊 Current Provider Status:[/bold]")

    providers = [
        ("GEMINI_API_KEY", "🔮 Google Gemini"),
        ("OPENAI_API_KEY", "🧠 OpenAI GPT"),
        ("ANTHROPIC_API_KEY", "🤖 Anthropic Claude")
    ]

    configured_count = 0
    for key, name in providers:
        if key in config:
            status = "[green]✅ Configured[/green]"
            configured_count += 1
        elif config.get(f"_{key.replace('_API_KEY', '')}_SELECTED"):
            status = "[yellow]📝 Placeholder[/yellow]"
        else:
            status = "[red]❌ Not configured[/red]"

        console.print(f"  {name}: {status}")

    if configured_count == 0:
        console.print("\n[red]⚠️  No providers configured yet![/red]")
    else:
        console.print(f"\n[green]✅ {configured_count} provider(s) configured[/green]")


def has_valid_providers(config: Dict[str, Any]) -> bool:
    """Check if there's at least one valid provider."""
    provider_keys = ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    return any(config.get(key) and not config[key].startswith("your_") for key in provider_keys)


def add_provider_interactive() -> Dict[str, Any]:
    """Interactive provider addition."""
    console.print("\n[bold blue]➕ Add Provider[/bold blue]")

    providers = [
        ("gemini", "🔮 Google Gemini", "Free tier, fast, good for CLI commands"),
        ("openai", "🧠 OpenAI GPT", "High quality, reliable, well-tested"),
        ("anthropic", "🤖 Anthropic Claude", "Advanced reasoning, safety-focused")
    ]

    console.print("Available providers:")
    for i, (key, name, desc) in enumerate(providers, 1):
        console.print(f"  {i}. {name} - [dim]{desc}[/dim]")

    choice = Prompt.ask(
        "Choose provider to add",
        choices=[str(i) for i in range(1, len(providers) + 1)],
        default="1"
    )

    provider = providers[int(choice) - 1][0]

    if provider == "gemini":
        return configure_gemini_with_key()
    elif provider == "openai":
        return configure_openai_with_key()
    elif provider == "anthropic":
        return configure_anthropic_with_key()

    return {}


def test_providers_interactive(config: Dict[str, Any]):
    """Interactive provider testing."""
    console.print("\n[bold blue]🧪 Test Providers[/bold blue]")

    # Find configured providers
    testable_providers = []
    for key, value in config.items():
        if key.endswith("_API_KEY") and value and not value.startswith("your_"):
            provider_name = key.replace("_API_KEY", "").lower()
            testable_providers.append((provider_name, key, value))

    if not testable_providers:
        console.print("[yellow]No configured providers to test![/yellow]")
        return

    console.print("Available providers to test:")
    for i, (name, key, value) in enumerate(testable_providers, 1):
        console.print(f"  {i}. {name.title()}")

    console.print(f"  {len(testable_providers) + 1}. Test all")

    choice = Prompt.ask(
        "Choose provider to test",
        choices=[str(i) for i in range(1, len(testable_providers) + 2)],
        default="1"
    )

    choice_idx = int(choice) - 1

    if choice_idx < len(testable_providers):
        # Test single provider
        name, key, value = testable_providers[choice_idx]
        console.print(f"\n[bold]Testing {name.title()}...[/bold]")
        success = test_provider_key(name, value)
        if success:
            console.print(f"[green]✅ {name.title()} is working correctly![/green]")
        else:
            console.print(f"[red]❌ {name.title()} test failed![/red]")
    else:
        # Test all providers
        console.print("\n[bold]Testing all providers...[/bold]")
        for name, key, value in testable_providers:
            console.print(f"\nTesting {name.title()}...")
            success = test_provider_key(name, value)
            if success:
                console.print(f"  [green]✅ {name.title()} working[/green]")
            else:
                console.print(f"  [red]❌ {name.title()} failed[/red]")


def remove_provider_interactive(config: Dict[str, Any]):
    """Interactive provider removal."""
    console.print("\n[bold red]🗑️  Remove Provider[/bold red]")

    # Find configured providers
    removable_providers = []
    for key in list(config.keys()):
        if key.endswith("_API_KEY"):
            provider_name = key.replace("_API_KEY", "").lower()
            removable_providers.append((provider_name, key))

    if not removable_providers:
        console.print("[yellow]No providers to remove![/yellow]")
        return

    console.print("Configured providers:")
    for i, (name, key) in enumerate(removable_providers, 1):
        console.print(f"  {i}. {name.title()}")

    choice = Prompt.ask(
        "Choose provider to remove",
        choices=[str(i) for i in range(1, len(removable_providers) + 1)],
        default="1"
    )

    choice_idx = int(choice) - 1
    name, key = removable_providers[choice_idx]

    if Confirm.ask(f"Remove {name.title()} provider?", default=False):
        del config[key]
        console.print(f"[green]✅ {name.title()} removed[/green]")


def show_detailed_provider_status(config: Dict[str, Any]):
    """Show detailed provider status with testing."""
    console.clear()
    console.print(Panel(
        "[bold]📊 Detailed Provider Status[/bold]",
        style="blue"
    ))

    providers = [
        ("gemini", "GEMINI_API_KEY", "🔮 Google Gemini"),
        ("openai", "OPENAI_API_KEY", "🧠 OpenAI GPT"),
        ("anthropic", "ANTHROPIC_API_KEY", "🤖 Anthropic Claude")
    ]

    for provider_name, key, display_name in providers:
        console.print(f"\n[bold]{display_name}[/bold]")

        if key in config and config[key]:
            if config[key].startswith("your_"):
                console.print("  Status: [yellow]📝 Placeholder configured[/yellow]")
                console.print("  Action: Replace placeholder with real API key")
            else:
                console.print("  Status: [green]✅ API key configured[/green]")
                console.print("  Testing connection...")

                success = test_provider_key(provider_name, config[key])
                if success:
                    console.print("  Test: [green]✅ Working correctly[/green]")
                else:
                    console.print("  Test: [red]❌ Connection failed[/red]")
                    console.print("  Action: Check API key validity")
        else:
            console.print("  Status: [red]❌ Not configured[/red]")
            console.print("  Action: Add API key to use this provider")

    console.input("\n[dim]Press Enter to continue...[/dim]")


def test_provider_key(provider: str, api_key: str) -> bool:
    """Test if an API key works by making a simple request."""
    import logging

    # Suppress HTTP request logs during testing
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("google.generativeai").setLevel(logging.WARNING)

    try:
        if provider == "gemini":
            return test_gemini_key(api_key)
        elif provider == "openai":
            return test_openai_key(api_key)
        elif provider == "anthropic":
            return test_anthropic_key(api_key)
        else:
            return False
    except Exception as e:
        console.print(f"[red]Test error: {str(e)}[/red]")
        return False


def test_gemini_key(api_key: str) -> bool:
    """Test Gemini API key."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        # Try to list models - lightweight test
        models = genai.list_models()
        model_list = list(models)
        return len(model_list) > 0
    except ImportError:
        console.print("[yellow]⚠️  google-generativeai not installed, skipping test[/yellow]")
        return True  # Assume valid if can't test
    except Exception as e:
        console.print(f"[red]Gemini test failed: {str(e)}[/red]")
        return False


def test_openai_key(api_key: str) -> bool:
    """Test OpenAI API key."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        # Try to list models - lightweight test
        models = client.models.list()
        return len(models.data) > 0
    except ImportError:
        console.print("[yellow]⚠️  openai not installed, skipping test[/yellow]")
        return True  # Assume valid if can't test
    except Exception as e:
        console.print(f"[red]OpenAI test failed: {str(e)}[/red]")
        return False


def test_anthropic_key(api_key: str) -> bool:
    """Test Anthropic API key."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Try a minimal request
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1,
            messages=[{"role": "user", "content": "Hi"}]
        )
        return True
    except ImportError:
        console.print("[yellow]⚠️  anthropic not installed, skipping test[/yellow]")
        return True  # Assume valid if can't test
    except Exception as e:
        console.print(f"[red]Anthropic test failed: {str(e)}[/red]")
        return False


def show_llm_comparison():
    """Show LLM provider comparison."""
    table = Table(title="🤖 LLM Provider Comparison")
    table.add_column("Provider", style="cyan")
    table.add_column("Strengths", style="green")
    table.add_column("Cost", style="yellow")
    table.add_column("API", style="blue")

    table.add_row(
        "Google Gemini",
        "Fast, efficient, good for CLI commands",
        "Free tier available",
        "Simple setup"
    )

    table.add_row(
        "OpenAI GPT",
        "High quality, well-tested for code",
        "Pay per use",
        "Mature API"
    )

    table.add_row(
        "Anthropic Claude",
        "Advanced reasoning, very safe",
        "Pay per use",
        "Advanced features"
    )

    console.print(table)


def configure_gemini() -> Dict[str, Any]:
    """Configure Google Gemini (legacy function)."""
    return configure_gemini_with_key()


def configure_gemini_with_key() -> Dict[str, Any]:
    """Configure Google Gemini with security-conscious UX."""
    console.print("\n[bold blue]🔮 Google Gemini Setup[/bold blue]")
    console.print("Get your API key from: https://makersuite.google.com/app/apikey")

    # Security information
    console.print("\n[bold blue]🔒 Security Information:[/bold blue]")
    console.print("• Your API key will be stored in the local .env file")
    console.print("• The key is NOT transmitted over the network during setup")
    console.print("• Only you have access to this file on your local machine")
    console.print("• The key will be used for API calls to Google services")

    setup_choice = Prompt.ask(
        "\nHow would you like to configure Gemini?",
        choices=["enter", "later", "skip"],
        default="later"
    )

    if setup_choice == "skip":
        console.print("[yellow]Skipping Gemini configuration[/yellow]")
        return {}

    elif setup_choice == "later":
        console.print("[yellow]Gemini configuration will be added as placeholder in .env[/yellow]")
        console.print("You can edit .env later to add your actual API key")
        return {"_GEMINI_SELECTED": True}

    else:  # enter
        console.print("\n[bold]Enter your Gemini API key:[/bold]")
        console.print("[dim]The key will be masked as you type and stored in .env[/dim]")

        api_key = Prompt.ask("Gemini API key", password=True)
        if not api_key.strip():
            console.print("[yellow]No API key provided, will add placeholder instead[/yellow]")
            return {"_GEMINI_SELECTED": True}

        # Validate key format (basic check)
        if not api_key.startswith("AI") or len(api_key) < 20:
            console.print("[yellow]⚠️  This doesn't look like a valid Gemini API key (should start with 'AI')[/yellow]")
            if not Confirm.ask("Use this key anyway?", default=False):
                return {"_GEMINI_SELECTED": True}

        console.print("[green]✅ Gemini API key validated and will be stored in .env[/green]")
        return {"GEMINI_API_KEY": api_key}


def configure_openai() -> Dict[str, Any]:
    """Configure OpenAI (legacy function)."""
    return configure_openai_with_key()


def configure_openai_with_key() -> Dict[str, Any]:
    """Configure OpenAI with security-conscious UX."""
    console.print("\n[bold green]🧠 OpenAI Setup[/bold green]")
    console.print("Get your API key from: https://platform.openai.com/api-keys")

    # Security information
    console.print("\n[bold blue]🔒 Security Information:[/bold blue]")
    console.print("• Your API key will be stored in the local .env file")
    console.print("• The key is NOT transmitted over the network during setup")
    console.print("• Only you have access to this file on your local machine")
    console.print("• The key will be used for API calls to OpenAI services")

    setup_choice = Prompt.ask(
        "\nHow would you like to configure OpenAI?",
        choices=["enter", "later", "skip"],
        default="later"
    )

    if setup_choice == "skip":
        console.print("[yellow]Skipping OpenAI configuration[/yellow]")
        return {}

    elif setup_choice == "later":
        console.print("[yellow]OpenAI configuration will be added as placeholder in .env[/yellow]")
        console.print("You can edit .env later to add your actual API key")
        return {"_OPENAI_SELECTED": True}

    else:  # enter
        console.print("\n[bold]Enter your OpenAI API key:[/bold]")
        console.print("[dim]The key will be masked as you type and stored in .env[/dim]")

        api_key = Prompt.ask("OpenAI API key", password=True)
        if not api_key.strip():
            console.print("[yellow]No API key provided, will add placeholder instead[/yellow]")
            return {"_OPENAI_SELECTED": True}

        # Validate key format (basic check)
        if not api_key.startswith("sk-") or len(api_key) < 40:
            console.print("[yellow]⚠️  This doesn't look like a valid OpenAI API key (should start with 'sk-')[/yellow]")
            if not Confirm.ask("Use this key anyway?", default=False):
                return {"_OPENAI_SELECTED": True}

        console.print("[green]✅ OpenAI API key validated and will be stored in .env[/green]")
        return {"OPENAI_API_KEY": api_key}


def configure_anthropic() -> Dict[str, Any]:
    """Configure Anthropic (legacy function)."""
    return configure_anthropic_with_key()


def configure_anthropic_with_key() -> Dict[str, Any]:
    """Configure Anthropic with security-conscious UX."""
    console.print("\n[bold purple]🤖 Anthropic Claude Setup[/bold purple]")
    console.print("Get your API key from: https://console.anthropic.com/")

    # Security information
    console.print("\n[bold blue]🔒 Security Information:[/bold blue]")
    console.print("• Your API key will be stored in the local .env file")
    console.print("• The key is NOT transmitted over the network during setup")
    console.print("• Only you have access to this file on your local machine")
    console.print("• The key will be used for API calls to Anthropic services")

    setup_choice = Prompt.ask(
        "\nHow would you like to configure Anthropic?",
        choices=["enter", "later", "skip"],
        default="later"
    )

    if setup_choice == "skip":
        console.print("[yellow]Skipping Anthropic configuration[/yellow]")
        return {}

    elif setup_choice == "later":
        console.print("[yellow]Anthropic configuration will be added as placeholder in .env[/yellow]")
        console.print("You can edit .env later to add your actual API key")
        return {"_ANTHROPIC_SELECTED": True}

    else:  # enter
        console.print("\n[bold]Enter your Anthropic API key:[/bold]")
        console.print("[dim]The key will be masked as you type and stored in .env[/dim]")

        api_key = Prompt.ask("Anthropic API key", password=True)
        if not api_key.strip():
            console.print("[yellow]No API key provided, will add placeholder instead[/yellow]")
            return {"_ANTHROPIC_SELECTED": True}

        # Validate key format (basic check)
        if not api_key.startswith("sk-ant-") or len(api_key) < 40:
            console.print("[yellow]⚠️  This doesn't look like a valid Anthropic API key (should start with 'sk-ant-')[/yellow]")
            if not Confirm.ask("Use this key anyway?", default=False):
                return {"_ANTHROPIC_SELECTED": True}

        console.print("[green]✅ Anthropic API key validated and will be stored in .env[/green]")
        return {"ANTHROPIC_API_KEY": api_key}


def configure_project_setup() -> Dict[str, Any]:
    """Configure general project setup for RAG system."""
    console.clear()
    console.print(Panel(
        "[bold]📋 Project Setup[/bold]",
        style="yellow"
    ))

    console.print("[bold blue]What is this?[/bold blue]")
    console.print("• This helps the RAG system understand your project context")
    console.print("• Project name helps with relevant responses and context")
    console.print("• Domain describes what type of documentation/knowledge you're working with")
    console.print("• The system can help with ANY type of questions about your documentation")

    console.print("\n[bold green]📋 Examples of Project Types:[/bold green]")
    examples = [
        ("Kubernetes Deployment", "DevOps", "Help with K8s configs, troubleshooting, best practices"),
        ("API Documentation", "Software Development", "Help with API usage, endpoints, examples"),
        ("Product Manual", "Documentation", "Help users understand features and workflows"),
        ("Internal Wiki", "Knowledge Base", "Answer questions about processes and procedures"),
        ("Research Papers", "Academic", "Help understand concepts and methodologies"),
        ("Technical Guides", "Engineering", "Help with implementation and troubleshooting")
    ]

    from rich.table import Table
    examples_table = Table(title="🎯 Project Examples")
    examples_table.add_column("Project Name", style="cyan")
    examples_table.add_column("Domain", style="green")
    examples_table.add_column("Use Case", style="blue")

    for name, domain, use_case in examples:
        examples_table.add_row(name, domain, use_case)

    console.print(examples_table)

    config = {}

    console.print("\n[bold]Enter your project details:[/bold]")
    console.print("[dim]These help provide better context to the LLM[/dim]")

    try:
        project_name = console.input("[bold cyan]Project name[/bold cyan] (e.g., 'Kubernetes Docs', 'API Guide'): ").strip()

        if not project_name:
            console.print("[yellow]No project name provided, skipping project setup[/yellow]")
            return {}

        config["PROJECT_NAME"] = project_name
        console.print(f"[green]✅ Project name set to: {project_name}[/green]")

        # Optional domain
        console.print(f"\n[bold]Optional: Project domain/category[/bold]")
        console.print("[dim]Helps the system understand the context (e.g., 'DevOps', 'API Documentation', 'User Manual')[/dim]")

        domain = console.input("[cyan]Domain/Category[/cyan] (or press Enter to skip): ").strip()
        if domain:
            config["PROJECT_DOMAIN"] = domain
            console.print(f"[green]✅ Domain set to: {domain}[/green]")
        else:
            console.print("[dim]Domain not set - system will work without it[/dim]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Project setup cancelled[/yellow]")
        return {}

    return config


def show_cli_examples():
    """Show CLI tool examples."""
    examples = [
        ("kubectl", "Kubernetes cluster management", "kubectl get pods"),
        ("docker", "Container management", "docker ps"),
        ("git", "Version control", "git status"),
        ("aws", "AWS CLI", "aws s3 ls"),
        ("terraform", "Infrastructure as code", "terraform plan"),
        ("helm", "Kubernetes package manager", "helm list"),
        ("gcloud", "Google Cloud CLI", "gcloud compute instances list")
    ]

    table = Table(title="🛠️ Popular CLI Tools")
    table.add_column("Tool", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Example Command", style="green")

    for tool, desc, example in examples:
        table.add_row(tool, desc, example)

    console.print(table)


def detect_available_tools() -> List[str]:
    """Detect available CLI tools on the system."""
    import shutil

    common_tools = [
        "kubectl", "docker", "git", "aws", "terraform",
        "helm", "gcloud", "npm", "yarn", "pip", "cargo"
    ]

    detected = []
    for tool in common_tools:
        if shutil.which(tool):
            detected.append(tool)

    return detected


def configure_documentation_files() -> Dict[str, Any]:
    """Configure documentation file sources for RAG system."""
    console.clear()
    console.print(Panel(
        "[bold]📚 Documentation Files Configuration[/bold]",
        style="blue"
    ))

    console.print("[bold blue]What is this?[/bold blue]")
    console.print("• Configure where your documentation files are located")
    console.print("• The RAG system will index these files for question answering")
    console.print("• Works with any text-based documentation")

    console.print("\n[bold green]📄 Supported File Types:[/bold green]")
    file_types_table = Table(title="📁 File Type Support")
    file_types_table.add_column("Type", style="cyan")
    file_types_table.add_column("Extensions", style="green")
    file_types_table.add_column("Use Cases", style="blue")

    file_types_table.add_row(
        "Markdown",
        ".md, .markdown",
        "Documentation, READMEs, guides"
    )
    file_types_table.add_row(
        "Text Files",
        ".txt",
        "Plain text docs, notes"
    )
    file_types_table.add_row(
        "PDF Documents",
        ".pdf",
        "Manuals, papers, reports"
    )

    console.print(file_types_table)

    console.print("\n[bold yellow]📂 Setup Options:[/bold yellow]")
    console.print("  [cyan]single[/cyan]   - Index all supported files in one folder")
    console.print("  [cyan]multiple[/cyan] - Index files from multiple folders")
    console.print("  [cyan]current[/cyan]  - Use docs in this project (./docs, ./README.md)")
    console.print("  [cyan]q[/cyan]        - 🚪 Back to main configuration menu")

    config = {}

    # Check for existing documentation in current directory
    docs_found = find_documentation_sources()
    if docs_found:
        console.print(f"\n[bold green]📖 Found in current directory:[/bold green]")
        for doc_type, path in docs_found:
            exists_indicator = "✅" if Path(path).exists() else "❌"
            console.print(f"  {exists_indicator} {doc_type}: [cyan]{path}[/cyan]")

    try:
        setup_mode = Prompt.ask(
            "\nChoose setup mode",
            choices=["single", "multiple", "current", "q"],
            default="current" if docs_found else "single"
        )
    except KeyboardInterrupt:
        # Ctrl+C goes back to main configuration menu
        return {}

    if setup_mode == "current":
        return configure_current_project_docs(docs_found)
    elif setup_mode == "single":
        return configure_single_directory_docs()
    elif setup_mode == "multiple":
        return configure_multiple_directories_docs()
    elif setup_mode == "q":
        return {}
    else:
        return {}


def configure_current_project_docs(docs_found: List[tuple]) -> Dict[str, Any]:
    """Configure using current project documentation."""
    console.print("\n[bold green]📁 Using Current Project Documentation[/bold green]")

    config = {
        "DOCS_PATH": "./docs",
        "SUPPORTED_FILE_TYPES": "md,txt,pdf"
    }

    if docs_found:
        console.print("Will index the following:")
        for doc_type, path in docs_found:
            console.print(f"  ✅ {doc_type}: {path}")

    console.print(f"\n[green]✅ Configuration set:[/green]")
    console.print(f"  • Documentation folder: ./docs")
    console.print(f"  • Supported file types: Markdown (.md), Text (.txt), PDF (.pdf)")
    console.print(f"  • Will index all supported files in the docs folder")

    return config


def configure_single_directory_docs() -> Dict[str, Any]:
    """Configure documentation from a single directory."""
    console.print("\n[bold blue]📁 Single Directory Setup[/bold blue]")

    try:
        docs_path = console.input("[cyan]Documentation directory path[/cyan] (e.g., ./docs, /path/to/docs): ").strip()

        if not docs_path:
            console.print("[yellow]No path provided, using default ./docs[/yellow]")
            docs_path = "./docs"

        # Check if directory exists
        path_obj = Path(docs_path)
        if path_obj.exists():
            # Count files by type (only in immediate directory, not recursive)
            md_files = list(path_obj.glob("*.md"))
            txt_files = list(path_obj.glob("*.txt"))
            pdf_files = list(path_obj.glob("*.pdf"))

            total_files = len(md_files) + len(txt_files) + len(pdf_files)
            console.print(f"[green]📊 Found {total_files} files:[/green]")
            console.print(f"  • Markdown: {len(md_files)} files")
            console.print(f"  • Text: {len(txt_files)} files")
            console.print(f"  • PDF: {len(pdf_files)} files")
        else:
            console.print(f"[yellow]⚠️  Directory doesn't exist yet: {docs_path}[/yellow]")
            console.print("[dim]It will be created when you start indexing[/dim]")

        config = {
            "DOCS_PATH": docs_path,
            "SUPPORTED_FILE_TYPES": "md,txt,pdf"
        }

        console.print(f"\n[green]✅ Single directory configured: {docs_path}[/green]")
        return config

    except KeyboardInterrupt:
        console.print("\n[yellow]Single directory setup cancelled[/yellow]")
        return {}


def configure_multiple_directories_docs() -> Dict[str, Any]:
    """Configure documentation from multiple directories."""
    console.print("\n[bold purple]📁 Multiple Directories Setup[/bold purple]")
    console.print("Add multiple directories containing documentation files.")

    directories = []

    try:
        while True:
            if len(directories) == 0:
                prompt_text = "[cyan]First documentation directory[/cyan]: "
            else:
                prompt_text = f"[cyan]Additional directory[/cyan] (or 'done' to finish): "

            dir_path = console.input(prompt_text).strip()

            if dir_path.lower() == 'done':
                if len(directories) == 0:
                    console.print("[yellow]At least one directory is required[/yellow]")
                    continue
                break

            if not dir_path:
                if len(directories) > 0:
                    break
                console.print("[yellow]Please enter a directory path[/yellow]")
                continue

            path_obj = Path(dir_path)
            if path_obj.exists():
                md_files = len(list(path_obj.glob("*.md")))
                txt_files = len(list(path_obj.glob("*.txt")))
                pdf_files = len(list(path_obj.glob("*.pdf")))
                total = md_files + txt_files + pdf_files
                console.print(f"  ✅ Added: {dir_path} ({total} files)")
            else:
                console.print(f"  ⚠️  Added: {dir_path} (directory will be created)")

            directories.append(dir_path)

        if directories:
            # Join multiple directories with semicolon
            docs_path = ";".join(directories)
            config = {
                "DOCS_PATH": docs_path,
                "SUPPORTED_FILE_TYPES": "md,txt,pdf"
            }

            console.print(f"\n[green]✅ Multiple directories configured:[/green]")
            for i, dir_path in enumerate(directories, 1):
                console.print(f"  {i}. {dir_path}")

            return config
        else:
            return {}

    except KeyboardInterrupt:
        console.print("\n[yellow]Multiple directories setup cancelled[/yellow]")
        return {}


def configure_current_directory_docs(docs_found: List[tuple]) -> Dict[str, Any]:
    """Configure using current directory documentation."""
    console.print("\n[bold green]📁 Using Current Directory Documentation[/bold green]")

    config = {}

    # Smart defaults based on what was found
    readme_path = "./README.md"
    docs_path = "./docs"

    for doc_type, path in docs_found:
        if doc_type == "README":
            readme_path = path
        elif doc_type == "Documentation":
            docs_path = path

    # Confirm defaults or allow customization
    config["README_PATH"] = Prompt.ask("README file path", default=readme_path)
    config["DOCS_PATH"] = Prompt.ask("Documentation directory", default=docs_path)

    # Show what will be indexed
    console.print(f"\n[green]✅ Will index:[/green]")
    console.print(f"  • README: {config['README_PATH']}")
    console.print(f"  • Docs directory: {config['DOCS_PATH']}")

    return config


def configure_copy_from_repo() -> Dict[str, Any]:
    """Configure by copying docs from another local repository."""
    console.print("\n[bold blue]📂 Copy Documentation from Another Repository[/bold blue]")
    console.print("This will copy documentation files from another local project.")

    while True:
        source_repo = Prompt.ask("Path to source repository (e.g., ../my-project)")
        source_path = Path(source_repo).expanduser().resolve()

        if not source_path.exists():
            console.print(f"[red]❌ Path not found: {source_path}[/red]")
            if not Confirm.ask("Try another path?", default=True):
                return {}
            continue

        if not source_path.is_dir():
            console.print(f"[red]❌ Not a directory: {source_path}[/red]")
            if not Confirm.ask("Try another path?", default=True):
                return {}
            continue

        break

    # Find docs in source repo
    source_docs = find_documentation_sources_in_path(source_path)

    if not source_docs:
        console.print(f"[yellow]⚠️  No documentation found in {source_path}[/yellow]")
        return {}

    console.print(f"\n[bold green]📖 Found in {source_path}:[/bold green]")
    for doc_type, rel_path in source_docs:
        full_path = source_path / rel_path
        console.print(f"  • {doc_type}: [cyan]{rel_path}[/cyan]")

    if Confirm.ask(f"Copy documentation to ./docs/?", default=True):
        import shutil

        # Create local docs directory
        local_docs = Path("./docs")
        local_docs.mkdir(exist_ok=True)

        copied_files = []
        for doc_type, rel_path in source_docs:
            source_file = source_path / rel_path
            if source_file.is_file():
                dest_file = local_docs / source_file.name
                shutil.copy2(source_file, dest_file)
                copied_files.append(str(dest_file))
                console.print(f"  ✅ Copied: {source_file.name}")
            elif source_file.is_dir():
                dest_dir = local_docs / source_file.name
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(source_file, dest_dir)
                copied_files.append(str(dest_dir))
                console.print(f"  ✅ Copied directory: {source_file.name}")

        console.print(f"\n[green]✅ Copied {len(copied_files)} items to ./docs/[/green]")

        return {
            "README_PATH": "./README.md",
            "DOCS_PATH": "./docs",
            "_copied_from": str(source_path)
        }

    return {}


def configure_custom_docs() -> Dict[str, Any]:
    """Configure custom documentation paths."""
    console.print("\n[bold purple]🛠️ Custom Documentation Paths[/bold purple]")

    config = {}

    # README path
    readme_default = "./README.md" if Path("./README.md").exists() else ""
    readme_path = Prompt.ask("README file path (or leave empty)", default=readme_default)
    if readme_path:
        config["README_PATH"] = readme_path

    # Docs directory
    docs_default = "./docs" if Path("./docs").exists() else ""
    docs_path = Prompt.ask("Documentation directory (or leave empty)", default=docs_default)
    if docs_path:
        config["DOCS_PATH"] = docs_path

    # Additional files
    console.print("\n[bold]Additional documentation files:[/bold]")
    console.print("💡 [dim]Enter individual files you want to index (e.g., API.md, GUIDE.md)[/dim]")

    additional = []
    while True:
        doc_path = Prompt.ask("Additional file path (or 'done')", default="done")
        if doc_path.lower() == "done":
            break

        if Path(doc_path).exists():
            additional.append(doc_path)
            console.print(f"  ✅ Added: {doc_path}")
        else:
            console.print(f"  [yellow]⚠️  File not found: {doc_path}[/yellow]")
            if Confirm.ask("Add anyway?", default=False):
                additional.append(doc_path)

    if additional:
        config["ADDITIONAL_DOCS"] = ",".join(additional)

    return config


def configure_docs_interactive(docs_found: List[tuple]) -> Dict[str, Any]:
    """Interactive documentation configuration."""
    console.print("\n[bold cyan]🎮 Interactive Documentation Builder[/bold cyan]")
    console.print("Let's build your documentation configuration step by step!")

    config = {}

    # README configuration
    console.print("\n[bold]📄 README Configuration[/bold]")
    readme_options = ["./README.md", "./readme.md", "./README.txt"]

    # Add found README to options
    for doc_type, path in docs_found:
        if doc_type == "README" and path not in readme_options:
            readme_options.insert(0, path)

    console.print("Available README files:")
    for i, option in enumerate(readme_options, 1):
        exists = "✅" if Path(option).exists() else "❌"
        console.print(f"  {i}. {exists} [cyan]{option}[/cyan]")

    console.print(f"  {len(readme_options) + 1}. Custom path")
    console.print(f"  {len(readme_options) + 2}. Skip README")

    readme_choice = Prompt.ask(
        "Choose README option",
        choices=[str(i) for i in range(1, len(readme_options) + 3)],
        default="1"
    )

    choice_idx = int(readme_choice) - 1
    if choice_idx < len(readme_options):
        config["README_PATH"] = readme_options[choice_idx]
    elif choice_idx == len(readme_options):
        custom_readme = Prompt.ask("Custom README path")
        if custom_readme:
            config["README_PATH"] = custom_readme
    # else: skip README

    # Documentation directory
    console.print("\n[bold]📁 Documentation Directory[/bold]")
    docs_options = ["./docs", "./documentation", "./doc"]

    # Add found docs to options
    for doc_type, path in docs_found:
        if doc_type == "Documentation" and path not in docs_options:
            docs_options.insert(0, path)

    console.print("Available documentation directories:")
    for i, option in enumerate(docs_options, 1):
        exists = "✅" if Path(option).exists() else "❌"
        console.print(f"  {i}. {exists} [cyan]{option}[/cyan]")

    console.print(f"  {len(docs_options) + 1}. Custom path")
    console.print(f"  {len(docs_options) + 2}. Skip docs directory")

    docs_choice = Prompt.ask(
        "Choose docs directory",
        choices=[str(i) for i in range(1, len(docs_options) + 3)],
        default="1"
    )

    choice_idx = int(docs_choice) - 1
    if choice_idx < len(docs_options):
        config["DOCS_PATH"] = docs_options[choice_idx]
    elif choice_idx == len(docs_options):
        custom_docs = Prompt.ask("Custom docs directory path")
        if custom_docs:
            config["DOCS_PATH"] = custom_docs

    # Show final configuration
    if config:
        console.print(f"\n[green]✅ Documentation configuration:[/green]")
        for key, value in config.items():
            console.print(f"  • {key}: [cyan]{value}[/cyan]")

    return config


def find_documentation_sources_in_path(path: Path) -> List[tuple]:
    """Find documentation sources in a specific path."""
    sources = []

    # Check for README
    for readme in ["README.md", "readme.md", "README.txt", "README"]:
        readme_path = path / readme
        if readme_path.exists():
            sources.append(("README", readme))
            break

    # Check for docs directory
    for docs_dir in ["docs", "documentation", "doc"]:
        docs_path = path / docs_dir
        if docs_path.is_dir():
            sources.append(("Documentation", docs_dir))
            break

    # Check for common files
    common_files = ["CHANGELOG.md", "API.md", "USAGE.md", "GUIDE.md", "TUTORIAL.md"]
    for file in common_files:
        file_path = path / file
        if file_path.exists():
            sources.append(("Guide", file))

    return sources


def find_documentation_sources() -> List[tuple]:
    """Find common documentation sources."""
    sources = []

    # Check for README
    for readme in ["README.md", "readme.md", "README.txt", "README"]:
        if Path(readme).exists():
            sources.append(("README", readme))
            break

    # Check for docs directory
    for docs_dir in ["docs", "documentation", "doc"]:
        if Path(docs_dir).is_dir():
            sources.append(("Documentation", docs_dir))
            break

    # Check for other common files
    common_files = ["CHANGELOG.md", "API.md", "USAGE.md", "GUIDE.md"]
    for file in common_files:
        if Path(file).exists():
            sources.append(("Guide", file))

    return sources


def configure_advanced_settings() -> Dict[str, Any]:
    """Configure advanced settings with explanations."""
    console.clear()
    console.print(Panel(
        "[bold]🎛️ Advanced Settings[/bold]",
        style="magenta"
    ))

    show_advanced_help()

    try:
        if not Confirm.ask("Configure advanced settings?", default=False):
            return {}

        config = {}

        # Performance settings
        console.print("\n[bold blue]⚡ Performance Settings[/bold blue]")

        config["MAX_CONTEXT_LENGTH"] = Prompt.ask(
            "Max context length (tokens sent to LLM)",
            default="4000"
        )

        config["TOP_K_RESULTS"] = Prompt.ask(
            "Top K results (number of similar docs to retrieve)",
            default="5"
        )

        config["SIMILARITY_THRESHOLD"] = Prompt.ask(
            "Similarity threshold (0.0-1.0, higher = more strict)",
            default="0.7"
        )

        # Behavior settings
        console.print("\n[bold yellow]🎭 Behavior Settings[/bold yellow]")

        config["DRY_RUN"] = "true" if Confirm.ask(
            "Enable dry run by default? (show commands without executing)",
            default=False
        ) else "false"

        config["CONFIRM_BEFORE_EXECUTION"] = "true" if Confirm.ask(
            "Always confirm before executing commands?",
            default=True
        ) else "false"

        config["LOG_LEVEL"] = Prompt.ask(
            "Log level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default="INFO"
        )

        return config

    except KeyboardInterrupt:
        # Ctrl+C goes back to main configuration menu
        console.print("\n[yellow]Advanced settings configuration cancelled[/yellow]")
        return {}


def show_advanced_help():
    """Show help for advanced settings."""
    help_text = """
## ⚙️ Advanced Settings Explained

**Performance Settings:**
- **Max Context Length**: How much text to send to the LLM (more = better context, higher cost)
- **Top K Results**: How many similar docs to retrieve (more = better coverage, slower)
- **Similarity Threshold**: How similar docs must be to include (higher = more relevant, fewer results)

**Behavior Settings:**
- **Dry Run**: Show generated commands without executing them
- **Confirm Before Execution**: Ask permission before running commands
- **Log Level**: How much logging detail to show
    """

    console.print(Panel(
        Markdown(help_text),
        title="📖 Settings Guide",
        border_style="blue"
    ))


def show_configuration_overview(config: Dict[str, Any]):
    """Show complete configuration overview."""
    console.clear()
    console.print(Panel(
        "[bold]👀 Configuration Overview[/bold]",
        style="cyan"
    ))

    if not config:
        console.print("[yellow]No configuration set yet![/yellow]")
        return

    # Create a nice tree view
    tree = Tree("🔧 EasyPrompt Configuration")

    # Group by category
    categories = {
        "🗄️ Vector Database": ["VECTOR_DB_TYPE", "VECTOR_DB_URL", "PINECONE_API_KEY", "PINECONE_ENVIRONMENT", "PINECONE_INDEX_NAME", "WEAVIATE_URL", "WEAVIATE_API_KEY"],
        "🔤 Embedding Model": ["EMBEDDING_MODEL"],
        "🤖 LLM Providers": ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        "📋 Project": ["PROJECT_NAME", "PROJECT_DOMAIN"],
        "📚 Documentation": ["DOCS_PATH", "SUPPORTED_FILE_TYPES"],
        "🎛️ Advanced": ["MAX_CONTEXT_LENGTH", "TOP_K_RESULTS", "SIMILARITY_THRESHOLD", "DRY_RUN", "CONFIRM_BEFORE_EXECUTION", "LOG_LEVEL"]
    }

    for category, keys in categories.items():
        category_items = [(k, v) for k, v in config.items() if k in keys]
        if category_items:
            branch = tree.add(category)
            for key, value in category_items:
                display_value = "***" if "API_KEY" in key else str(value)
                branch.add(f"{key}: [green]{display_value}[/green]")

    console.print(tree)

    # Validation status
    console.print("\n[bold]✅ Configuration Validation:[/bold]")
    issues = validate_configuration(config)
    if not issues:
        console.print("[green]✅ Configuration looks good![/green]")
    else:
        for issue in issues:
            console.print(f"[red]❌ {issue}[/red]")


def show_help_and_tips():
    """Show help and tips."""
    console.clear()
    console.print(Panel(
        "[bold]❓ Help & Tips[/bold]",
        style="yellow"
    ))

    help_content = """
# 🚀 EasyPrompt Configuration Help

## 🎯 Quick Start Tips
- **Start with ChromaDB** for local development (no API keys needed)
- **Use Gemini** as your first LLM provider (generous free tier)
- **Point to your project's README** as documentation source

## 🔧 Configuration Tips
- You can always edit the `.env` file manually later
- Multiple LLM providers = automatic fallback if one fails
- Higher similarity threshold = more precise but fewer results

## 🛠️ Common Setups

### Development Setup
- Vector DB: ChromaDB (local)
- LLM: Gemini (free tier)
- CLI Tool: git or docker
- Docs: ./README.md

### Production Setup
- Vector DB: Pinecone (scalable)
- LLM: OpenAI GPT (reliable)
- CLI Tool: kubectl
- Docs: ./docs directory

## 🔗 Getting API Keys
- **Gemini**: https://makersuite.google.com/app/apikey
- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/
- **Pinecone**: https://app.pinecone.io/

## 🚨 Troubleshooting
- If CLI tool not found: specify full path
- If API errors: check your API keys
- If no results: lower similarity threshold
- If slow: reduce context length or top K
    """

    console.print(Markdown(help_content))


def validate_configuration(config: Dict[str, Any]) -> List[str]:
    """Validate configuration and return issues."""
    issues = []

    # Check required LLM provider
    has_llm = any("API_KEY" in k for k in config.keys())
    if not has_llm:
        issues.append("No LLM provider configured")

    # Check vector DB configuration
    if "VECTOR_DB_TYPE" not in config:
        issues.append("No vector database type specified")
    elif config["VECTOR_DB_TYPE"] == "pinecone":
        if "PINECONE_API_KEY" not in config:
            issues.append("Pinecone selected but no API key provided")

    # Check documentation
    if "DOCS_PATH" not in config:
        issues.append("No documentation sources configured")

    return issues


def auto_save_config(config: Dict[str, Any], env_file: Path) -> None:
    """Auto-save configuration after each section without validation prompts."""
    try:
        write_env_file(env_file, config)
        console.print(f"[dim]✅ Configuration saved automatically[/dim]")
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not auto-save: {e}[/yellow]")


def validate_and_save_config(config: Dict[str, Any], env_file: Path) -> bool:
    """Validate and save configuration."""
    console.clear()
    console.print(Panel(
        "[bold]💾 Save Configuration[/bold]",
        style="green"
    ))

    # Show final overview
    show_configuration_overview(config)

    # Validate
    issues = validate_configuration(config)
    if issues:
        console.print("\n[red]❌ Configuration has issues:[/red]")
        for issue in issues:
            console.print(f"  • {issue}")

        if not Confirm.ask("\nSave anyway?", default=False):
            return False

    # Save
    try:
        write_env_file(env_file, config)
        console.print(f"\n[green]✅ Configuration saved to {env_file}[/green]")

        # Create directories
        create_directories(config)

        # Show next steps
        show_next_steps(config)

        return True

    except Exception as e:
        console.print(f"[red]❌ Failed to save configuration: {e}[/red]")
        return False


def write_env_file(env_file: Path, config: dict):
    """Write configuration to .env file with proper placeholder handling."""
    with open(env_file, "w") as f:
        f.write("# EasyPrompt Configuration\n")
        f.write("# Generated by easyprompt init\n")
        f.write("# Edit this file to modify settings\n\n")

        # Handle LLM provider placeholders
        llm_providers = {}
        if config.get("_GEMINI_SELECTED"):
            llm_providers["GEMINI_API_KEY"] = ("your_gemini_api_key_here", "Google Gemini API key - Get from: https://makersuite.google.com/app/apikey")
        elif config.get("GEMINI_API_KEY"):
            llm_providers["GEMINI_API_KEY"] = (config["GEMINI_API_KEY"], "Google Gemini API key")

        if config.get("_OPENAI_SELECTED"):
            llm_providers["OPENAI_API_KEY"] = ("your_openai_api_key_here", "OpenAI API key - Get from: https://platform.openai.com/api-keys")
        elif config.get("OPENAI_API_KEY"):
            llm_providers["OPENAI_API_KEY"] = (config["OPENAI_API_KEY"], "OpenAI API key")

        if config.get("_ANTHROPIC_SELECTED"):
            llm_providers["ANTHROPIC_API_KEY"] = ("your_anthropic_api_key_here", "Anthropic Claude API key - Get from: https://console.anthropic.com/")
        elif config.get("ANTHROPIC_API_KEY"):
            llm_providers["ANTHROPIC_API_KEY"] = (config["ANTHROPIC_API_KEY"], "Anthropic Claude API key")

        # Define all possible settings with defaults and descriptions
        all_settings = {
            "Vector Database": {
                "VECTOR_DB_TYPE": (config.get("VECTOR_DB_TYPE", "chromadb"), "Vector database type: chromadb, pinecone, weaviate"),
                "VECTOR_DB_URL": (config.get("VECTOR_DB_URL", "./data/chroma.db"), "Database URL/path (for chromadb/weaviate)")
            },
            "Embedding Model": {
                "EMBEDDING_MODEL": (config.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"), "Sentence transformer model for embeddings")
            },
            "Project Setup": {
                "PROJECT_NAME": (config.get("PROJECT_NAME", ""), "Project name for context (optional)"),
                "PROJECT_DOMAIN": (config.get("PROJECT_DOMAIN", ""), "Project domain/category (optional)")
            },
            "Documentation Files": {
                "DOCS_PATH": (config.get("DOCS_PATH", "./docs"), "Documentation directory path(s) - use semicolon for multiple"),
                "SUPPORTED_FILE_TYPES": (config.get("SUPPORTED_FILE_TYPES", "md,txt,pdf"), "Supported file extensions (comma-separated)")
            },
            "LLM Providers": llm_providers,
            "Pinecone Settings": {
                "PINECONE_API_KEY": (config.get("PINECONE_API_KEY", ""), "Pinecone API key (if using Pinecone)"),
                "PINECONE_ENVIRONMENT": (config.get("PINECONE_ENVIRONMENT", ""), "Pinecone environment"),
                "PINECONE_INDEX_NAME": (config.get("PINECONE_INDEX_NAME", "easyprompt-index"), "Pinecone index name")
            },
            "Weaviate Settings": {
                "WEAVIATE_URL": (config.get("WEAVIATE_URL", "http://localhost:8080"), "Weaviate URL (if using Weaviate)"),
                "WEAVIATE_API_KEY": (config.get("WEAVIATE_API_KEY", ""), "Weaviate API key (optional)")
            },
            "Performance": {
                "MAX_CONTEXT_LENGTH": (config.get("MAX_CONTEXT_LENGTH", "4000"), "Maximum context length for LLM"),
                "TOP_K_RESULTS": (config.get("TOP_K_RESULTS", "5"), "Number of similar docs to retrieve"),
                "SIMILARITY_THRESHOLD": (config.get("SIMILARITY_THRESHOLD", "0.7"), "Similarity threshold (0.0-1.0)")
            },
            "Execution": {
                "DRY_RUN": (config.get("DRY_RUN", "false"), "Show commands without executing (true/false)"),
                "CONFIRM_BEFORE_EXECUTION": (config.get("CONFIRM_BEFORE_EXECUTION", "true"), "Ask before executing commands (true/false)"),
                "LOG_LEVEL": (config.get("LOG_LEVEL", "INFO"), "Logging level: DEBUG, INFO, WARNING, ERROR")
            }
        }

        for group_name, settings in all_settings.items():
            if not settings:  # Skip empty groups
                continue

            f.write(f"# {group_name}\n")
            for key, (value, description) in settings.items():
                f.write(f"# {description}\n")
                if value and not value.startswith("your_"):
                    # Real value
                    f.write(f"{key}={value}\n")
                elif value and value.startswith("your_"):
                    # Placeholder value - write as comment with clear instruction
                    f.write(f"# TODO: Replace with your actual API key\n")
                    f.write(f"# {key}={value}\n")
                else:
                    # Empty value
                    f.write(f"# {key}=\n")
            f.write("\n")

        # Add helpful footer for placeholder users
        if any(config.get(k) for k in ["_GEMINI_SELECTED", "_OPENAI_SELECTED", "_ANTHROPIC_SELECTED"]):
            f.write("# ==========================================\n")
            f.write("# NEXT STEPS:\n")
            f.write("# 1. Uncomment the API key lines above (remove #)\n")
            f.write("# 2. Replace 'your_*_api_key_here' with your actual API keys\n")
            f.write("# 3. Save this file\n")
            f.write("# 4. Run: easyprompt index ./docs\n")
            f.write("# ==========================================\n")


def create_directories(config: dict):
    """Create necessary directories."""
    dirs_to_create = []

    if "DOCS_PATH" in config:
        dirs_to_create.append(config["DOCS_PATH"])

    if config.get("VECTOR_DB_TYPE") == "chromadb" and "VECTOR_DB_URL" in config:
        db_path = Path(config["VECTOR_DB_URL"])
        if not db_path.is_absolute():
            dirs_to_create.append(str(db_path.parent))

    for dir_path in dirs_to_create:
        path = Path(dir_path)
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                console.print(f"[dim]Created directory: {path}[/dim]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not create directory {path}: {e}[/yellow]")


def show_next_steps(config: dict):
    """Show next steps to the user."""
    console.print(Panel(
        "[bold green]🎉 Configuration Complete![/bold green]",
        style="green"
    ))

    # Show what was configured
    configured_items = []
    if config.get("VECTOR_DB_TYPE"):
        configured_items.append(f"✅ Vector DB: {config['VECTOR_DB_TYPE']}")

    llm_providers = [k.replace('_API_KEY', '') for k in config.keys() if 'API_KEY' in k]
    if llm_providers:
        configured_items.append(f"✅ LLM Providers: {', '.join(llm_providers)}")

    if config.get("CLI_TOOL_NAME"):
        configured_items.append(f"✅ CLI Tool: {config['CLI_TOOL_NAME']}")

    if configured_items:
        console.print("\n[bold]What you configured:[/bold]")
        for item in configured_items:
            console.print(f"  {item}")

    # Next steps
    next_steps = []

    if config.get("CLI_TOOL_NAME"):
        next_steps.append(f"1. Ensure '[cyan]{config['CLI_TOOL_NAME']}[/cyan]' is installed and accessible")

    # Check if we need full dependencies
    needs_full = config.get("VECTOR_DB_TYPE") in ["pinecone", "weaviate"] or config.get("EMBEDDING_MODEL")
    if needs_full:
        next_steps.append("2. Install full dependencies: '[cyan]pip install -e \".[full]\"[/cyan]'")

    next_steps.extend([
        f"{len(next_steps) + 1}. Index your documentation: '[cyan]easyprompt index[/cyan]'",
        f"{len(next_steps) + 2}. Try a query: '[cyan]easyprompt query \"your request\"[/cyan]'",
        f"{len(next_steps) + 3}. Start interactive mode: '[cyan]easyprompt chat[/cyan]'",
        f"{len(next_steps) + 4}. Check status anytime: '[cyan]easyprompt status[/cyan]'"
    ])

    console.print(Panel(
        "\n".join(next_steps),
        title="🚀 Next Steps",
        border_style="green"
    ))

    console.print(f"\n[dim]Configuration saved to .env[/dim]")
    console.print(f"[dim]You can edit .env manually anytime to change settings[/dim]")


def configure_index_settings() -> Dict[str, Any]:
    """Configure indexing settings with simple choices."""
    console.clear()
    console.print(Panel(
        "[bold]⚙️ Index Settings Configuration[/bold]",
        style="blue"
    ))

    console.print("[bold blue]Index settings control how documents are processed for search.[/bold blue]")
    console.print("Configure chunking strategy, storage paths, and processing options.")

    # Load current settings to show current choice
    current_config = load_existing_env_file()
    current_strategy = current_config.get("CHUNKING_STRATEGY", "recursive")
    current_chunk_size = current_config.get("CHUNK_SIZE", "1000")
    current_overlap = current_config.get("CHUNK_OVERLAP", "200")

    # Show current settings
    console.print(f"\n[bold green]📋 Current Settings:[/bold green]")
    console.print(f"• Chunking Strategy: [yellow]{current_strategy.title()}[/yellow]")
    console.print(f"• Chunk Size: [yellow]{current_chunk_size}[/yellow] characters")
    console.print(f"• Chunk Overlap: [yellow]{current_overlap}[/yellow] characters")

    console.print("\n[bold yellow]Chunking Strategy Options:[/bold yellow]")
    # Highlight current choice
    strategies = [
        ("1", "Recursive", "Smart hierarchical splitting (recommended)"),
        ("2", "Fixed Size", "Simple fixed-length chunks"),
        ("3", "Sentence", "Split by sentence boundaries"),
        ("4", "Paragraph", "Split by paragraph boundaries")
    ]

    for num, name, desc in strategies:
        if name.lower().replace(" ", "") == current_strategy.replace("_", ""):
            console.print(f"{num}. [bold green]{name}[/bold green] - {desc} [green]← Current[/green]")
        else:
            console.print(f"{num}. [cyan]{name}[/cyan] - {desc}")

    console.print("  q. 🚪 Back - Return to main configuration menu")

    try:
        choice = Prompt.ask(
            "\nChoose chunking strategy",
            choices=["1", "2", "3", "4", "q"],
            default="1"
        )
    except KeyboardInterrupt:
        return {}

    if choice == "q":
        return {}

    # Map choices to strategy names
    strategy_map = {
        "1": "recursive",
        "2": "fixed",
        "3": "sentence",
        "4": "paragraph"
    }

    strategy = strategy_map.get(choice, "recursive")
    console.print(f"[green]✅ Using {strategy} chunking strategy[/green]")

    return {
        "CHUNKING_STRATEGY": strategy,
        "CHUNK_SIZE": "1000",
        "CHUNK_OVERLAP": "200",
        "INDEX_STORAGE_PATH": "./data/index",
        "CHUNKED_DOCS_PATH": "./data/chunked_docs",
        "ENABLE_METADATA_EXTRACTION": "true",
        "MIN_CHUNK_SIZE": "100"
    }

