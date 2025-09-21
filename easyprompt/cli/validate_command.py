"""Command validation utility."""

import asyncio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ..config import Settings
from ..query import QueryProcessor


async def validate_command(settings: Settings, console: Console, command: str):
    """Validate a command without executing it."""
    processor = QueryProcessor(settings)

    try:
        await processor.initialize()

        validation = await processor.validate_command(command)

        if "error" in validation:
            console.print(f"[red]Validation failed:[/red] {validation['error']}")
            return

        # Display command
        command_panel = Panel(
            command,
            title="Command to Validate",
            border_style="blue"
        )
        console.print(command_panel)

        # Validation results
        table = Table(title="Validation Results")
        table.add_column("Check", style="cyan")
        table.add_column("Result", style="bold")
        table.add_column("Details", style="dim")

        # Basic validation
        is_valid = validation.get("is_valid", False)
        table.add_row(
            "Syntax Valid",
            "✅ Pass" if is_valid else "❌ Fail",
            "Command appears to be syntactically correct" if is_valid else "Command has syntax issues"
        )

        # Safety check
        is_safe = validation.get("is_safe", False)
        table.add_row(
            "Safety Check",
            "✅ Safe" if is_safe else "⚠️  Potentially Dangerous",
            "No dangerous patterns detected" if is_safe else "Command contains potentially dangerous patterns"
        )

        # Command type
        cmd_type = validation.get("command_type", "unknown")
        table.add_row("Command Type", cmd_type.title(), f"Classified as {cmd_type} operation")

        console.print(table)

        # Recommendations
        recommendations = validation.get("recommendations", [])
        if recommendations:
            console.print("\n[bold]Recommendations:[/bold]")
            for rec in recommendations:
                console.print(f"  {rec}")

        # Overall assessment
        overall_status = "✅ Ready to execute" if is_valid and is_safe else "⚠️  Review before executing"
        console.print(f"\n[bold]Overall Assessment:[/bold] {overall_status}")

    except Exception as e:
        console.print(f"[red]Validation failed:[/red] {e}")
    finally:
        await processor.close()