"""CLI interface for EasyPrompt."""

from .main import app
from .commands import IndexCommand, QueryCommand, ChatCommand, StatusCommand

__all__ = ["app", "IndexCommand", "QueryCommand", "ChatCommand", "StatusCommand"]