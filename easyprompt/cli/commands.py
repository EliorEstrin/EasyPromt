"""CLI command implementations."""

import asyncio
import subprocess
import sys
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

from ..config import Settings
from ..indexer import DocumentIndexer
from ..query import QueryProcessor


class BaseCommand:
    """Base class for CLI commands."""

    def __init__(self, settings: Settings, console: Console):
        self.settings = settings
        self.console = console

    def run_async(self, coro):
        """Run an async coroutine."""
        try:
            return asyncio.run(coro)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Operation cancelled by user[/yellow]")
            sys.exit(1)
        except Exception as e:
            self.console.print(f"\n[red]Error:[/red] {e}")
            sys.exit(1)


class IndexCommand(BaseCommand):
    """Index documentation files."""

    def run(self, paths: Optional[List[str]], rebuild: bool, verbose: bool):
        """Run the index command."""
        self.run_async(self._async_run(paths, rebuild, verbose))

    async def _async_run(self, paths: Optional[List[str]], rebuild: bool, verbose: bool):
        """Async implementation of index command."""
        indexer = DocumentIndexer(self.settings)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Initializing indexer...", total=None)

            await indexer.initialize()
            progress.update(task, description="Indexing documents...")

            stats = await indexer.index_documentation(
                force_rebuild=rebuild,
                paths=paths
            )

            progress.update(task, description="Indexing complete!", completed=True)

        # Display results
        table = Table(title="Indexing Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Documents Processed", str(stats["documents"]))
        table.add_row("Chunks Created", str(stats["chunks"]))

        self.console.print(table)

        if verbose:
            db_stats = await indexer.get_index_stats()
            self.console.print(f"\nTotal chunks in database: {db_stats.get('total_chunks', 0)}")

        await indexer.cleanup()


class QueryCommand(BaseCommand):
    """Generate CLI command from natural language query."""

    def run(
        self,
        query_text: str,
        execute: bool,
        dry_run: bool,
        alternatives: int,
        explain: bool,
        provider: Optional[str]
    ):
        """Run the query command."""
        self.run_async(self._async_run(query_text, execute, dry_run, alternatives, explain, provider))

    def _is_qa_request(self, query_text: str) -> bool:
        """Determine if the query is a Q&A request vs a command generation request."""
        # Question patterns
        question_starters = [
            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'can you tell me',
            'tell me about', 'explain', 'describe', 'show me about', 'help me understand'
        ]

        # Command patterns
        command_patterns = [
            'run', 'execute', 'start', 'stop', 'install', 'create', 'delete', 'remove',
            'copy', 'move', 'list files', 'find files', 'search for', 'download', 'upload'
        ]

        query_lower = query_text.lower().strip()

        # Check for question indicators
        if query_lower.endswith('?'):
            return True

        # Check for question starters
        for starter in question_starters:
            if query_lower.startswith(starter):
                return True

        # Check for command patterns (if found, likely not Q&A)
        for pattern in command_patterns:
            if pattern in query_lower:
                return False

        # Default: if unclear, treat as Q&A since that's what the UI promises
        return True

    async def _async_run(
        self,
        query_text: str,
        execute: bool,
        dry_run: bool,
        alternatives: int,
        explain: bool,
        provider: Optional[str]
    ):
        """Async implementation of query command."""
        processor = QueryProcessor(self.settings)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Processing query...", total=None)

            await processor.initialize()

            # Detect if this is a Q&A request or command generation request
            is_qa = self._is_qa_request(query_text)

            if is_qa:
                # Process as documentation Q&A
                result = await processor.process_qa_query(query_text)
                progress.update(task, description="Query processed!", completed=True)
                self._display_qa_result(result)
            elif alternatives > 0:
                # Process as command generation with alternatives
                result = await processor.process_query_with_alternatives(
                    query_text, num_alternatives=alternatives, include_explanation=explain
                )
                progress.update(task, description="Query processed!", completed=True)
                self._display_alternatives_result(result)
            else:
                # Process as single command generation
                result = await processor.process_query(
                    query_text, include_explanation=explain
                )
                progress.update(task, description="Query processed!", completed=True)
                self._display_query_result(result, execute, dry_run)

        await processor.close()

    def _display_query_result(self, result, execute: bool, dry_run: bool):
        """Display the result of a query."""
        if not result.success:
            self.console.print(f"[red]Query failed:[/red] {result.error}")
            return

        # Display command
        command_panel = Panel(
            f"[bold green]{result.command}[/bold green]",
            title="Generated Command",
            border_style="green"
        )
        self.console.print(command_panel)

        # Display explanation if available
        if result.explanation:
            explanation_panel = Panel(
                result.explanation,
                title="Explanation",
                border_style="blue"
            )
            self.console.print(explanation_panel)

        # Display metadata if verbose
        if result.context_summary:
            self.console.print(f"\n[dim]Used {result.context_summary.get('total_chunks', 0)} context chunks "
                             f"from {len(result.context_summary.get('files', []))} files[/dim]")

        # Command validation
        if result.metadata.get("is_safe") is False:
            self.console.print("\n[red]⚠️  Warning: This command may be dangerous![/red]")

        # Execute command if requested
        if result.command != "UNCLEAR_REQUEST":
            if execute and not dry_run:
                self._execute_command(result.command)
            elif dry_run:
                self.console.print("\n[yellow]Dry run mode - command not executed[/yellow]")

    def _display_alternatives_result(self, result: dict):
        """Display alternatives result."""
        if not result.get("success"):
            self.console.print(f"[red]Query failed:[/red] {result.get('error')}")
            return

        # Main command
        main_panel = Panel(
            f"[bold green]{result['command']}[/bold green]",
            title="Primary Command",
            border_style="green"
        )
        self.console.print(main_panel)

        if result.get("explanation"):
            self.console.print(f"\n{result['explanation']}")

        # Alternatives
        if result.get("alternatives"):
            self.console.print("\n[bold]Alternative Commands:[/bold]")
            for i, alt in enumerate(result["alternatives"], 1):
                alt_panel = Panel(
                    f"[cyan]{alt['command']}[/cyan]\n\n{alt['explanation']}",
                    title=f"Alternative {i}",
                    border_style="cyan"
                )
                self.console.print(alt_panel)

    def _execute_command(self, command: str):
        """Execute a command."""
        if self.settings.confirm_before_execution:
            if not Confirm.ask(f"Execute command: {command}?"):
                self.console.print("[yellow]Command execution cancelled[/yellow]")
                return

        try:
            self.console.print(f"\n[dim]Executing: {command}[/dim]")
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.stdout:
                self.console.print(result.stdout)
            if result.stderr:
                self.console.print(f"[red]{result.stderr}[/red]")

            if result.returncode != 0:
                self.console.print(f"[red]Command failed with exit code {result.returncode}[/red]")

        except subprocess.TimeoutExpired:
            self.console.print("[red]Command timed out after 30 seconds[/red]")
        except Exception as e:
            self.console.print(f"[red]Failed to execute command:[/red] {e}")

    def _display_qa_result(self, result):
        """Display the result of a Q&A query."""
        if not result["success"]:
            self.console.print(f"[red]Query failed:[/red] {result['answer']}")
            return

        # Display the answer in a nice panel
        self.console.print(Panel(
            result["answer"],
            title="📖 Answer",
            style="green"
        ))

        # Show context information
        context_info = f"Used {len(result['context_used'])} context chunks from {len(result['files_used'])} files"
        self.console.print(f"\n[dim]{context_info}[/dim]")

        # Show processing time
        if result.get("processing_time"):
            self.console.print(f"[dim]⏱️ Processed in {result['processing_time']:.2f}s[/dim]")


class ChatCommand(BaseCommand):
    """Interactive chat session."""

    def run(self, provider: Optional[str]):
        """Run the chat command."""
        self.run_async(self._async_run(provider))

    async def _async_run(self, provider: Optional[str]):
        """Async implementation of chat command."""
        processor = QueryProcessor(self.settings)
        await processor.initialize()

        self.console.print("[bold green]EasyPrompt Interactive Chat[/bold green]")
        self.console.print("Type your queries in natural language. Type 'exit' to quit.\n")

        try:
            while True:
                query = Prompt.ask("[cyan]Query[/cyan]")

                if query.lower() in ["exit", "quit", "q"]:
                    break

                if not query.strip():
                    continue

                # Process query
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console,
                ) as progress:
                    task = progress.add_task("Processing...", total=None)
                    result = await processor.process_query(query, include_explanation=True)
                    progress.update(task, description="Done!", completed=True)

                # Display result
                if result.success and result.command != "UNCLEAR_REQUEST":
                    self.console.print(f"\n[bold green]Command:[/bold green] {result.command}")
                    if result.explanation:
                        self.console.print(f"[bold blue]Explanation:[/bold blue] {result.explanation}")

                    # Ask if user wants to execute
                    if Confirm.ask("Execute this command?", default=False):
                        self._execute_command(result.command)
                else:
                    self.console.print(f"\n[red]Could not generate command:[/red] {result.error or 'Unclear request'}")

                self.console.print()

        except KeyboardInterrupt:
            pass
        finally:
            self.console.print("\n[yellow]Goodbye![/yellow]")
            await processor.close()

    def _execute_command(self, command: str):
        """Execute a command in chat mode."""
        try:
            result = subprocess.run(command, shell=True, text=True)
            if result.returncode != 0:
                self.console.print(f"[red]Command failed with exit code {result.returncode}[/red]")
        except Exception as e:
            self.console.print(f"[red]Execution failed:[/red] {e}")


class StatusCommand(BaseCommand):
    """Show system status and configuration."""

    def run(self, verbose: bool):
        """Run the status command."""
        self.run_async(self._async_run(verbose))

    async def _async_run(self, verbose: bool):
        """Async implementation of status command."""
        # Temporarily suppress logs for cleaner UX
        import logging
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)

        processor = QueryProcessor(self.settings)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Checking system status...", total=None)

            try:
                await processor.initialize()
                status = await processor.get_system_status()
                progress.update(task, description="Status check complete!", completed=True)
            except Exception as e:
                progress.update(task, description="Status check failed!", completed=True)
                status = {"error": str(e)}

        # Display status
        self._display_status(status, verbose)

        if processor._initialized:
            await processor.close()

        # Restore original logging level after cleanup
        logging.getLogger().setLevel(original_level)

    def _display_status(self, status: dict, verbose: bool):
        """Display system status."""
        table = Table(title="EasyPrompt System Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="dim")

        if "error" in status:
            table.add_row("System", "[red]Error[/red]", status["error"])
            self.console.print(table)
            return

        # Basic status
        table.add_row("Initialization", "✅ Ready" if status.get("initialized") else "❌ Not Ready", "")
        table.add_row("Embedding Model", status.get("embedding_model", "Not configured"), "")
        table.add_row("Vector Database", status.get("vector_db_type", "Not configured"), "")

        # Vector DB status
        if status.get("vector_db_status"):
            db_status = "✅ Healthy" if status["vector_db_status"] == "healthy" else f"❌ {status['vector_db_status']}"
            db_details = f"{status.get('total_documents', 0)} documents" if status.get('total_documents') is not None else ""
            table.add_row("Vector DB Health", db_status, db_details)

        # LLM provider status
        if status.get("llm_provider"):
            llm_info = status["llm_provider"]
            if isinstance(llm_info, dict):
                provider_name = llm_info.get("provider", "Unknown")
                provider_status = "✅ Ready" if llm_info.get("status") == "ready" else f"❌ {llm_info.get('status')}"
                provider_details = llm_info.get("model", "") if verbose else ""
                table.add_row("LLM Provider", f"{provider_name} - {provider_status}", provider_details)

        self.console.print(table)

        # Additional details in verbose mode
        if verbose:
            self._display_verbose_status()

    def _display_verbose_status(self):
        """Display verbose status information."""
        # Configuration details
        config_table = Table(title="Configuration Details")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")

        config_table.add_row("Docs Path", self.settings.docs_path)
        config_table.add_row("README Path", self.settings.readme_path)
        config_table.add_row("Vector DB URL", self.settings.vector_db_url)
        config_table.add_row("Top K Results", str(self.settings.top_k_results))
        config_table.add_row("Similarity Threshold", str(self.settings.similarity_threshold))
        config_table.add_row("Max Context Length", str(self.settings.max_context_length))
        config_table.add_row("Dry Run", str(self.settings.dry_run))
        config_table.add_row("Confirm Before Execution", str(self.settings.confirm_before_execution))

        self.console.print("\n")
        self.console.print(config_table)