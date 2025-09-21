"""Find command examples related to a query."""

import asyncio
from rich.console import Console
from rich.panel import Panel

from ..config import Settings
from ..query import QueryProcessor


async def find_examples(settings: Settings, console: Console, query: str, limit: int):
    """Find command examples related to a query."""
    processor = QueryProcessor(settings)

    try:
        await processor.initialize()

        console.print(f"[bold]Finding examples for:[/bold] {query}")

        examples = await processor.find_command_examples(query)

        if not examples:
            console.print("[yellow]No command examples found[/yellow]")
            return

        # Limit results
        examples = examples[:limit]

        console.print(f"\n[green]Found {len(examples)} command examples:[/green]\n")

        for i, example in enumerate(examples, 1):
            metadata = example.get("metadata", {})
            content = example.get("content", "")
            similarity = example.get("similarity", 0)
            file_path = metadata.get("file_path", "Unknown")
            section = metadata.get("section", "main")

            title = f"Example {i} - {file_path}:{section} (similarity: {similarity:.3f})"

            panel = Panel(
                content,
                title=title,
                border_style="green"
            )
            console.print(panel)

    except Exception as e:
        console.print(f"[red]Failed to find examples:[/red] {e}")
    finally:
        await processor.close()