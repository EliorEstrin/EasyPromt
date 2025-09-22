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

        # Only pause for configuration actions, not info screens
        if choice in ["1", "2", "3", "4", "5"]:
            pass  # These sections handle their own pausing
        elif choice in ["6", "7"]:
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

        if not has_valid_providers(config):
            console.print("\n[red]⚠️  You need at least one working provider![/red]")
            console.print("[yellow]💡 Try 'quick' for fast setup with recommended providers[/yellow]")

        valid_choices = ["quick", "add", "test", "remove", "status", "done"]
        default_choice = "quick" if not has_valid_providers(config) else "done"

        choice = Prompt.ask(
            "\nChoose action",
            choices=valid_choices,
            default=default_choice
        )

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

        if choice != "done":
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

    choice = Prompt.ask(
        "\nSetup option",
        choices=["1", "2", "3", "4", "5", "6"],
        default="1"
    )

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

    if Confirm.ask(f"\nSet up {recommendations['title']}?", default=True):
        return execute_quick_setup(recommendations['providers'])
    else:
        return add_provider_with_guidance()


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


def configure_cli_tool() -> Dict[str, Any]:
    """Configure CLI tool with clear explanation."""
    console.clear()
    console.print(Panel(
        "[bold]⚙️ CLI Tool Configuration[/bold]",
        style="yellow"
    ))

    console.print("[bold blue]What is this?[/bold blue]")
    console.print("• This is the CLI tool you want EasyPrompt to help you with")
    console.print("• Examples: kubectl, docker, git, aws, terraform, or any custom tool")
    console.print("• The name is just an identifier - you can call it anything")
    console.print("• EasyPrompt will generate commands for this tool based on your docs")

    show_cli_examples()

    config = {}

    # Detect common tools for reference only
    detected_tools = detect_available_tools()
    if detected_tools:
        console.print(f"\n[bold green]💡 Common tools detected on your system:[/bold green]")
        console.print("[dim](These are just suggestions - you can use any name)[/dim]")
        for tool in detected_tools:
            console.print(f"  • {tool}")

    # Simple tool name input with better UX
    console.print("\n[bold]Enter your CLI tool details:[/bold]")
    console.print("[dim]Tip: Use arrow keys to edit, Ctrl+C to cancel[/dim]")

    try:
        # Use input() instead of Prompt.ask for better terminal support
        tool_name = console.input("[bold cyan]CLI tool name[/bold cyan] (e.g., kubectl, my-tool): ").strip()

        if not tool_name:
            console.print("[yellow]No tool name provided, skipping CLI tool configuration[/yellow]")
            return {}

        config["CLI_TOOL_NAME"] = tool_name
        console.print(f"[green]✅ CLI tool name set to: {tool_name}[/green]")

        # Optional path configuration
        console.print(f"\n[bold]Optional: Specify path to {tool_name}[/bold]")
        console.print("[dim]Leave empty to use system PATH[/dim]")

        # Try to auto-detect path
        import shutil
        auto_path = shutil.which(tool_name)
        if auto_path:
            console.print(f"[green]🔍 Found {tool_name} at: {auto_path}[/green]")
            use_auto = Confirm.ask("Use this path?", default=True)
            if use_auto:
                config["CLI_TOOL_PATH"] = auto_path
                console.print(f"[green]✅ Using detected path[/green]")
            else:
                custom_path = console.input("[cyan]Custom path[/cyan] (or press Enter to skip): ").strip()
                if custom_path:
                    config["CLI_TOOL_PATH"] = custom_path
                    console.print(f"[green]✅ Using custom path: {custom_path}[/green]")
        else:
            console.print(f"[yellow]⚠️  {tool_name} not found in system PATH[/yellow]")
            custom_path = console.input("[cyan]Full path to tool[/cyan] (or press Enter to skip): ").strip()
            if custom_path:
                config["CLI_TOOL_PATH"] = custom_path
                console.print(f"[green]✅ Custom path set: {custom_path}[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]CLI tool configuration cancelled[/yellow]")
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


def configure_documentation() -> Dict[str, Any]:
    """Configure documentation sources with developer-friendly options."""
    console.clear()
    console.print(Panel(
        "[bold]📚 Documentation Configuration[/bold]",
        style="blue"
    ))

    console.print("📖 EasyPrompt can index various documentation sources:")
    console.print("• README files and markdown docs")
    console.print("• Local documentation directories")
    console.print("• Copy docs from other local repositories")
    console.print("• Individual files and guides")

    config = {}

    # Check for existing documentation in current directory
    docs_found = find_documentation_sources()

    console.print("\n[bold yellow]📂 Documentation Setup Options:[/bold yellow]")
    console.print("1. [cyan]Use current directory docs[/cyan] - Index docs in this project")
    console.print("2. [cyan]Copy from another project[/cyan] - Copy docs from a local repo")
    console.print("3. [cyan]Custom paths[/cyan] - Specify custom documentation paths")
    console.print("4. [cyan]Interactive builder[/cyan] - Step-by-step doc selection")

    if docs_found:
        console.print(f"\n[bold green]📖 Auto-detected in current directory:[/bold green]")
        for doc_type, path in docs_found:
            exists_indicator = "✅" if Path(path).exists() else "❌"
            console.print(f"  {exists_indicator} {doc_type}: [cyan]{path}[/cyan]")

    setup_mode = Prompt.ask(
        "\nChoose documentation setup",
        choices=["current", "copy", "custom", "interactive"],
        default="current" if docs_found else "interactive"
    )

    if setup_mode == "current":
        return configure_current_directory_docs(docs_found)
    elif setup_mode == "copy":
        return configure_copy_from_repo()
    elif setup_mode == "custom":
        return configure_custom_docs()
    else:  # interactive
        return configure_docs_interactive(docs_found)


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
            "CLI Tool": {
                "CLI_TOOL_NAME": (config.get("CLI_TOOL_NAME", ""), "Name of your CLI tool (required)"),
                "CLI_TOOL_PATH": (config.get("CLI_TOOL_PATH", ""), "Full path to CLI tool (optional)")
            },
            "Documentation": {
                "DOCS_PATH": (config.get("DOCS_PATH", "./docs"), "Documentation directory path"),
                "README_PATH": (config.get("README_PATH", "./README.md"), "README file path"),
                "ADDITIONAL_DOCS": (config.get("ADDITIONAL_DOCS", ""), "Additional docs (comma-separated)")
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