"""Search documentation command."""

import asyncio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ..config import Settings
from ..query import QueryProcessor


async def search_documentation(settings: Settings, console: Console, query: str, limit: int):
    """Search documentation without generating commands."""
    processor = QueryProcessor(settings)

    try:
        await processor.initialize()

        console.print(f"[bold]Searching documentation for:[/bold] {query}")

        results = await processor.search_documentation(query, top_k=limit)

        if not results:
            console.print("[yellow]No results found[/yellow]")
            return

        console.print(f"\n[green]Found {len(results)} results:[/green]\n")

        for i, result in enumerate(results, 1):
            metadata = result.get("metadata", {})
            content = result.get("content", "")
            similarity = result.get("similarity", 0)

            # Truncate content for display
            if len(content) > 200:
                content = content[:200] + "..."

            title = f"Result {i} - {metadata.get('file_path', 'Unknown')} ({similarity:.3f})"

            panel = Panel(
                content,
                title=title,
                border_style="blue"
            )
            console.print(panel)

    except Exception as e:
        console.print(f"[red]Search failed:[/red] {e}")
    finally:
        await processor.close()