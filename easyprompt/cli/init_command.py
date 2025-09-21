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


def init_configuration(force: bool = False):
    """Initialize EasyPrompt configuration with ultra-interactive interface."""
    env_file = Path(".env")

    # Check if .env already exists
    if env_file.exists() and not force:
        if not Confirm.ask(".env file already exists. Overwrite?"):
            console.print("[yellow]Configuration initialization cancelled[/yellow]")
            return

    # Show welcome with exploration options
    show_welcome()

    # Main configuration loop
    config = {}

    while True:
        console.clear()
        show_header()
        show_current_config(config)

        choice = show_main_menu()

        if choice == "1":
            config.update(configure_vector_database())
        elif choice == "2":
            config.update(configure_llm_providers())
        elif choice == "3":
            config.update(configure_cli_tool())
        elif choice == "4":
            config.update(configure_documentation())
        elif choice == "5":
            config.update(configure_advanced_settings())
        elif choice == "6":
            show_configuration_overview(config)
        elif choice == "7":
            show_help_and_tips()
        elif choice == "8":
            if validate_and_save_config(config, env_file):
                break
        elif choice == "q":
            if Confirm.ask("Exit without saving?"):
                console.print("[yellow]Configuration cancelled[/yellow]")
                return

        if choice != "6" and choice != "7":  # Don't pause for info screens
            console.input("\n[dim]Press Enter to continue...[/dim]")


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

    console.input("\n[bold cyan]Press Enter to start exploring...[/bold cyan]")


def show_header():
    """Show the main header."""
    console.print(Panel(
        "[bold blue]🔧 EasyPrompt Configuration Builder[/bold blue]",
        style="blue",
        padding=(0, 2)
    ))


def show_current_config(config: Dict[str, Any]):
    """Show current configuration status."""
    if not config:
        console.print("[dim]No configuration set yet. Start by choosing a section below![/dim]\n")
        return

    table = Table(title="📋 Current Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    table.add_column("Status", justify="center")

    # Group settings by category
    categories = {
        "Vector DB": ["VECTOR_DB_TYPE", "VECTOR_DB_URL", "PINECONE_API_KEY", "WEAVIATE_URL"],
        "LLM Providers": ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        "CLI Tool": ["CLI_TOOL_NAME", "CLI_TOOL_PATH"],
        "Documentation": ["DOCS_PATH", "README_PATH"],
        "Advanced": ["MAX_CONTEXT_LENGTH", "TOP_K_RESULTS", "SIMILARITY_THRESHOLD"]
    }

    for category, keys in categories.items():
        category_items = [(k, v) for k, v in config.items() if k in keys]
        if category_items:
            table.add_row(f"[bold]{category}[/bold]", "", "")
            for key, value in category_items:
                display_value = "***" if "API_KEY" in key else str(value)
                table.add_row(f"  {key}", display_value, "✅")

    console.print(table)
    console.print()


def show_main_menu() -> str:
    """Show main menu and get user choice."""
    console.print("[bold yellow]📂 Configuration Sections[/bold yellow]")

    menu_options = [
        ("1", "🗄️  Vector Database", "Choose and configure your vector database"),
        ("2", "🤖 LLM Providers", "Configure AI language model providers"),
        ("3", "⚙️  CLI Tool", "Set up your target CLI tool"),
        ("4", "📚 Documentation", "Configure documentation sources"),
        ("5", "🎛️  Advanced Settings", "Performance and behavior tuning"),
        ("6", "👀 Preview Configuration", "See complete configuration overview"),
        ("7", "❓ Help & Tips", "Get help and see examples"),
        ("8", "💾 Save & Exit", "Validate and save configuration"),
        ("q", "🚪 Quit", "Exit without saving")
    ]

    for key, title, desc in menu_options:
        console.print(f"  {key}. [bold cyan]{title}[/bold cyan] - [dim]{desc}[/dim]")

    console.print()
    choice = Prompt.ask(
        "[bold]Choose section",
        choices=[opt[0] for opt in menu_options],
        default="1"
    )

    return choice


def configure_vector_database() -> Dict[str, Any]:
    """Configure vector database with full exploration."""
    console.clear()
    console.print(Panel(
        "[bold]🗄️ Vector Database Configuration[/bold]",
        style="blue"
    ))

    # Show detailed comparison
    show_vector_db_comparison()

    while True:
        choice = Prompt.ask(
            "\nChoose action",
            choices=["compare", "chromadb", "pinecone", "weaviate", "back"],
            default="chromadb"
        )

        if choice == "compare":
            show_vector_db_comparison()
        elif choice == "chromadb":
            return configure_chromadb()
        elif choice == "pinecone":
            return configure_pinecone()
        elif choice == "weaviate":
            return configure_weaviate()
        elif choice == "back":
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

    path_choice = Prompt.ask("Choose path option", choices=["1", "2", "3"], default="1")

    if path_choice == "1":
        config["VECTOR_DB_URL"] = "./data/chroma.db"
    elif path_choice == "2":
        config["VECTOR_DB_URL"] = "~/.easyprompt/chroma.db"
    else:
        custom_path = Prompt.ask("Enter custom path", default="./data/chroma.db")
        config["VECTOR_DB_URL"] = custom_path

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

    if not Confirm.ask("\nDo you have a Pinecone account and API key?"):
        console.print("[yellow]ℹ️  Visit https://app.pinecone.io/ to create an account first[/yellow]")
        return {}

    config["PINECONE_API_KEY"] = Prompt.ask("Pinecone API key", password=True)
    config["PINECONE_ENVIRONMENT"] = Prompt.ask("Pinecone environment (e.g., 'us-east1-aws')")
    config["PINECONE_INDEX_NAME"] = Prompt.ask("Index name", default="easyprompt-index")

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

    setup_choice = Prompt.ask("Choose setup", choices=["1", "2", "3"], default="1")

    if setup_choice == "1":
        config["WEAVIATE_URL"] = "http://localhost:8080"
        console.print("\n[yellow]ℹ️  Make sure to run Weaviate locally:[/yellow]")
        console.print("docker run -p 8080:8080 semitechnologies/weaviate:latest")
    elif setup_choice == "2":
        config["WEAVIATE_URL"] = Prompt.ask("WCS cluster URL")
        config["WEAVIATE_API_KEY"] = Prompt.ask("WCS API key", password=True)
    else:
        config["WEAVIATE_URL"] = Prompt.ask("Weaviate URL", default="http://localhost:8080")
        if Confirm.ask("Does this instance require an API key?"):
            config["WEAVIATE_API_KEY"] = Prompt.ask("API key", password=True)

    return config


def configure_llm_providers() -> Dict[str, Any]:
    """Configure LLM providers with detailed exploration."""
    console.clear()
    console.print(Panel(
        "[bold]🤖 LLM Provider Configuration[/bold]",
        style="green"
    ))

    show_llm_comparison()

    config = {}

    while True:
        console.print("\n[bold]Configure Providers (you need at least one):[/bold]")

        providers = [
            ("gemini", "🔮 Google Gemini", "Fast, efficient, good for development"),
            ("openai", "🧠 OpenAI GPT", "High quality, widely used"),
            ("anthropic", "🤖 Anthropic Claude", "Advanced reasoning, safety-focused"),
            ("done", "✅ Done configuring", "Continue to next section")
        ]

        for key, name, desc in providers:
            status = "✅" if any(k.startswith(key.upper()) for k in config.keys()) else "⚪"
            console.print(f"  {status} {name} - [dim]{desc}[/dim]")

        choice = Prompt.ask(
            "\nChoose provider to configure",
            choices=[p[0] for p in providers],
            default="gemini"
        )

        if choice == "gemini":
            config.update(configure_gemini())
        elif choice == "openai":
            config.update(configure_openai())
        elif choice == "anthropic":
            config.update(configure_anthropic())
        elif choice == "done":
            if not any("API_KEY" in k for k in config.keys()):
                console.print("[red]❌ You must configure at least one provider![/red]")
                continue
            break

    return config


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
    """Configure Google Gemini."""
    console.print("\n[bold blue]🔮 Google Gemini Setup[/bold blue]")
    console.print("Get your API key from: https://makersuite.google.com/app/apikey")

    if not Confirm.ask("Do you have a Gemini API key?"):
        console.print("[yellow]Visit the link above to get one first[/yellow]")
        return {}

    api_key = Prompt.ask("Gemini API key", password=True)
    return {"GEMINI_API_KEY": api_key}


def configure_openai() -> Dict[str, Any]:
    """Configure OpenAI."""
    console.print("\n[bold green]🧠 OpenAI Setup[/bold green]")
    console.print("Get your API key from: https://platform.openai.com/api-keys")

    if not Confirm.ask("Do you have an OpenAI API key?"):
        console.print("[yellow]Visit the link above to get one first[/yellow]")
        return {}

    api_key = Prompt.ask("OpenAI API key", password=True)
    return {"OPENAI_API_KEY": api_key}


def configure_anthropic() -> Dict[str, Any]:
    """Configure Anthropic."""
    console.print("\n[bold purple]🤖 Anthropic Claude Setup[/bold purple]")
    console.print("Get your API key from: https://console.anthropic.com/")

    if not Confirm.ask("Do you have an Anthropic API key?"):
        console.print("[yellow]Visit the link above to get one first[/yellow]")
        return {}

    api_key = Prompt.ask("Anthropic API key", password=True)
    return {"ANTHROPIC_API_KEY": api_key}


def configure_cli_tool() -> Dict[str, Any]:
    """Configure CLI tool with examples and suggestions."""
    console.clear()
    console.print(Panel(
        "[bold]⚙️ CLI Tool Configuration[/bold]",
        style="yellow"
    ))

    show_cli_examples()

    config = {}

    # Detect common tools
    detected_tools = detect_available_tools()
    if detected_tools:
        console.print(f"\n[bold green]🔍 Detected tools on your system:[/bold green]")
        for tool in detected_tools:
            console.print(f"  • {tool}")

        use_detected = Prompt.ask(
            "\nUse one of the detected tools?",
            choices=detected_tools + ["custom"],
            default=detected_tools[0] if detected_tools else "custom"
        )

        if use_detected != "custom":
            config["CLI_TOOL_NAME"] = use_detected
            # Try to find the path
            import shutil
            tool_path = shutil.which(use_detected)
            if tool_path:
                config["CLI_TOOL_PATH"] = tool_path
                console.print(f"[green]✅ Found {use_detected} at {tool_path}[/green]")
            return config

    # Custom configuration
    console.print("\n[bold]Custom CLI Tool Setup:[/bold]")

    tool_name = Prompt.ask("CLI tool name (e.g., kubectl, docker, git)")
    config["CLI_TOOL_NAME"] = tool_name

    # Try to auto-detect path
    import shutil
    auto_path = shutil.which(tool_name)
    if auto_path:
        use_auto = Confirm.ask(f"Found {tool_name} at {auto_path}. Use this path?", default=True)
        if use_auto:
            config["CLI_TOOL_PATH"] = auto_path
        else:
            custom_path = Prompt.ask("Enter custom path (or leave empty)")
            if custom_path:
                config["CLI_TOOL_PATH"] = custom_path
    else:
        console.print(f"[yellow]⚠️  Could not find {tool_name} in PATH[/yellow]")
        custom_path = Prompt.ask("Enter full path to tool (or leave empty)")
        if custom_path:
            config["CLI_TOOL_PATH"] = custom_path

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


def configure_documentation() -> Dict[str, Any]:
    """Configure documentation sources."""
    console.clear()
    console.print(Panel(
        "[bold]📚 Documentation Configuration[/bold]",
        style="blue"
    ))

    console.print("EasyPrompt can index various documentation sources:")
    console.print("• README files")
    console.print("• Documentation directories")
    console.print("• Individual markdown files")
    console.print("• Man pages (future)")

    config = {}

    # Check for common documentation
    docs_found = find_documentation_sources()
    if docs_found:
        console.print(f"\n[bold green]📖 Found documentation:[/bold green]")
        for doc_type, path in docs_found:
            console.print(f"  • {doc_type}: [cyan]{path}[/cyan]")

    # Configure main sources
    readme_path = Prompt.ask("\nREADME file path", default="./README.md")
    config["README_PATH"] = readme_path

    docs_path = Prompt.ask("Documentation directory", default="./docs")
    config["DOCS_PATH"] = docs_path

    # Additional sources
    if Confirm.ask("Add additional documentation files?", default=False):
        additional = []
        while True:
            doc_path = Prompt.ask("Enter path (or 'done' to finish)", default="done")
            if doc_path.lower() == "done":
                break
            additional.append(doc_path)

        if additional:
            config["ADDITIONAL_DOCS"] = ",".join(additional)

    return config


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
        "🤖 LLM Providers": ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        "⚙️ CLI Tool": ["CLI_TOOL_NAME", "CLI_TOOL_PATH"],
        "📚 Documentation": ["DOCS_PATH", "README_PATH", "ADDITIONAL_DOCS"],
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

    # Check CLI tool
    if "CLI_TOOL_NAME" not in config:
        issues.append("No CLI tool specified")

    # Check documentation
    if "README_PATH" not in config and "DOCS_PATH" not in config:
        issues.append("No documentation sources configured")

    return issues


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
    """Write configuration to .env file."""
    with open(env_file, "w") as f:
        f.write("# EasyPrompt Configuration\n")
        f.write("# Generated by easyprompt init\n\n")

        # Group related settings
        groups = {
            "Vector Database": ["VECTOR_DB_TYPE", "VECTOR_DB_URL", "PINECONE_API_KEY", "PINECONE_ENVIRONMENT", "PINECONE_INDEX_NAME", "WEAVIATE_URL", "WEAVIATE_API_KEY"],
            "Embedding Model": ["EMBEDDING_MODEL", "EMBEDDING_DIMENSION"],
            "CLI Tool": ["CLI_TOOL_NAME", "CLI_TOOL_PATH"],
            "Documentation": ["DOCS_PATH", "README_PATH", "ADDITIONAL_DOCS"],
            "LLM Providers": ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
            "Performance": ["MAX_CONTEXT_LENGTH", "TOP_K_RESULTS", "SIMILARITY_THRESHOLD"],
            "Execution": ["DRY_RUN", "CONFIRM_BEFORE_EXECUTION", "LOG_LEVEL"]
        }

        for group_name, keys in groups.items():
            group_items = [(k, v) for k, v in config.items() if k in keys]
            if group_items:
                f.write(f"# {group_name}\n")
                for key, value in group_items:
                    f.write(f"{key}={value}\n")
                f.write("\n")


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
    next_steps = [
        "1. Run '[cyan]easyprompt index[/cyan]' to index your documentation",
        "2. Try a query: '[cyan]easyprompt query \"your natural language request\"[/cyan]'",
        "3. Start interactive mode: '[cyan]easyprompt chat[/cyan]'",
        "4. Check status: '[cyan]easyprompt status[/cyan]'"
    ]

    if config.get("CLI_TOOL_NAME"):
        next_steps.insert(0, f"0. Ensure '{config['CLI_TOOL_NAME']}' is installed and accessible")

    console.print(Panel(
        "\n".join(next_steps),
        title="Next Steps",
        border_style="green"
    ))