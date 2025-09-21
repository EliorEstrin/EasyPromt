"""Tests for CLI commands."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typer.testing import CliRunner

from easyprompt.cli.main import app


class TestCLICommands:
    """Test CLI command implementations."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_settings(self, sample_settings):
        """Mock settings for CLI tests."""
        return sample_settings

    def test_main_help(self, runner):
        """Test main help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Natural Language to CLI Command Interface" in result.stdout

    @patch('easyprompt.cli.main.validate_config')
    def test_index_command(self, mock_validate, runner, mock_settings):
        """Test index command."""
        mock_validate.return_value = mock_settings

        with patch('easyprompt.cli.commands.DocumentIndexer') as mock_indexer_class:
            mock_indexer = AsyncMock()
            mock_indexer.initialize = AsyncMock()
            mock_indexer.index_documentation = AsyncMock(return_value={"documents": 5, "chunks": 20})
            mock_indexer.cleanup = AsyncMock()
            mock_indexer_class.return_value = mock_indexer

            result = runner.invoke(app, ["index"])

            assert result.exit_code == 0
            mock_indexer.initialize.assert_called_once()
            mock_indexer.index_documentation.assert_called_once()

    @patch('easyprompt.cli.main.validate_config')
    def test_index_command_with_rebuild(self, mock_validate, runner, mock_settings):
        """Test index command with rebuild flag."""
        mock_validate.return_value = mock_settings

        with patch('easyprompt.cli.commands.DocumentIndexer') as mock_indexer_class:
            mock_indexer = AsyncMock()
            mock_indexer.initialize = AsyncMock()
            mock_indexer.index_documentation = AsyncMock(return_value={"documents": 5, "chunks": 20})
            mock_indexer.cleanup = AsyncMock()
            mock_indexer_class.return_value = mock_indexer

            result = runner.invoke(app, ["index", "--rebuild"])

            assert result.exit_code == 0
            mock_indexer.index_documentation.assert_called_once_with(
                force_rebuild=True, paths=None
            )

    @patch('easyprompt.cli.main.validate_config')
    def test_query_command(self, mock_validate, runner, mock_settings):
        """Test query command."""
        mock_validate.return_value = mock_settings

        with patch('easyprompt.cli.commands.QueryProcessor') as mock_processor_class:
            mock_processor = AsyncMock()
            mock_processor.initialize = AsyncMock()
            mock_processor.process_query = AsyncMock()
            mock_processor.close = AsyncMock()

            # Mock query result
            from easyprompt.query.query_processor import QueryResult
            mock_result = QueryResult(
                query="list files",
                command="ls -la",
                explanation="Lists files with details",
                success=True
            )
            mock_processor.process_query.return_value = mock_result
            mock_processor_class.return_value = mock_processor

            result = runner.invoke(app, ["query", "list files"])

            assert result.exit_code == 0
            mock_processor.initialize.assert_called_once()
            mock_processor.process_query.assert_called_once()

    @patch('easyprompt.cli.main.validate_config')
    def test_query_command_with_alternatives(self, mock_validate, runner, mock_settings):
        """Test query command with alternatives."""
        mock_validate.return_value = mock_settings

        with patch('easyprompt.cli.commands.QueryProcessor') as mock_processor_class:
            mock_processor = AsyncMock()
            mock_processor.initialize = AsyncMock()
            mock_processor.process_query_with_alternatives = AsyncMock(return_value={
                "success": True,
                "command": "ls -la",
                "explanation": "Lists files",
                "alternatives": [
                    {"command": "ls -l", "explanation": "Long format"},
                    {"command": "find .", "explanation": "Find files"}
                ]
            })
            mock_processor.close = AsyncMock()
            mock_processor_class.return_value = mock_processor

            result = runner.invoke(app, ["query", "list files", "--alternatives", "2"])

            assert result.exit_code == 0
            mock_processor.process_query_with_alternatives.assert_called_once()

    @patch('easyprompt.cli.main.validate_config')
    def test_status_command(self, mock_validate, runner, mock_settings):
        """Test status command."""
        mock_validate.return_value = mock_settings

        with patch('easyprompt.cli.commands.QueryProcessor') as mock_processor_class:
            mock_processor = AsyncMock()
            mock_processor.initialize = AsyncMock()
            mock_processor.get_system_status = AsyncMock(return_value={
                "initialized": True,
                "cli_tool": "test-cli",
                "vector_db_status": "healthy",
                "total_documents": 42
            })
            mock_processor.close = AsyncMock()
            mock_processor_class.return_value = mock_processor

            result = runner.invoke(app, ["status"])

            assert result.exit_code == 0
            mock_processor.get_system_status.assert_called_once()

    @patch('easyprompt.cli.main.validate_config')
    def test_search_command(self, mock_validate, runner, mock_settings):
        """Test search command."""
        mock_validate.return_value = mock_settings

        with patch('easyprompt.cli.search_command.QueryProcessor') as mock_processor_class:
            mock_processor = AsyncMock()
            mock_processor.initialize = AsyncMock()
            mock_processor.search_documentation = AsyncMock(return_value=[
                {
                    "content": "Test documentation content",
                    "metadata": {"file_path": "test.md"},
                    "similarity": 0.8
                }
            ])
            mock_processor.close = AsyncMock()
            mock_processor_class.return_value = mock_processor

            result = runner.invoke(app, ["search", "test query"])

            assert result.exit_code == 0

    @patch('easyprompt.cli.main.validate_config')
    def test_validate_command(self, mock_validate, runner, mock_settings):
        """Test validate command."""
        mock_validate.return_value = mock_settings

        with patch('easyprompt.cli.validate_command.QueryProcessor') as mock_processor_class:
            mock_processor = AsyncMock()
            mock_processor.initialize = AsyncMock()
            mock_processor.validate_command = AsyncMock(return_value={
                "command": "ls -la",
                "is_valid": True,
                "is_safe": True,
                "command_type": "read",
                "recommendations": ["Command is safe to execute"]
            })
            mock_processor.close = AsyncMock()
            mock_processor_class.return_value = mock_processor

            result = runner.invoke(app, ["validate", "ls -la"])

            assert result.exit_code == 0

    @patch('easyprompt.cli.main.validate_config')
    def test_examples_command(self, mock_validate, runner, mock_settings):
        """Test examples command."""
        mock_validate.return_value = mock_settings

        with patch('easyprompt.cli.examples_command.QueryProcessor') as mock_processor_class:
            mock_processor = AsyncMock()
            mock_processor.initialize = AsyncMock()
            mock_processor.find_command_examples = AsyncMock(return_value=[
                {
                    "content": "Use `ls -la` to list files",
                    "metadata": {"file_path": "examples.md"},
                    "similarity": 0.9
                }
            ])
            mock_processor.close = AsyncMock()
            mock_processor_class.return_value = mock_processor

            result = runner.invoke(app, ["examples", "list files"])

            assert result.exit_code == 0

    def test_init_command(self, runner):
        """Test init command."""
        with patch('easyprompt.cli.init_command.init_configuration') as mock_init:
            result = runner.invoke(app, ["init"])

            assert result.exit_code == 0
            mock_init.assert_called_once()

    def test_init_command_force(self, runner):
        """Test init command with force flag."""
        with patch('easyprompt.cli.init_command.init_configuration') as mock_init:
            result = runner.invoke(app, ["init", "--force"])

            assert result.exit_code == 0
            mock_init.assert_called_once_with(True)

    @patch('easyprompt.cli.main.ConfigValidator.validate_environment')
    def test_config_validation_failure(self, mock_validate_env, runner):
        """Test CLI with configuration validation failure."""
        mock_validate_env.return_value = "No .env file found"

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 1
        assert "Configuration Error" in result.stdout

    @patch('easyprompt.cli.main.validate_config')
    def test_query_command_execution_error(self, mock_validate, runner, mock_settings):
        """Test query command with execution error."""
        mock_validate.return_value = mock_settings

        with patch('easyprompt.cli.commands.QueryProcessor') as mock_processor_class:
            mock_processor = AsyncMock()
            mock_processor.initialize = AsyncMock()
            mock_processor.process_query = AsyncMock()
            mock_processor.close = AsyncMock()

            # Mock failed query result
            from easyprompt.query.query_processor import QueryResult
            mock_result = QueryResult(
                query="invalid query",
                command="",
                explanation="",
                success=False,
                error="Processing failed"
            )
            mock_processor.process_query.return_value = mock_result
            mock_processor_class.return_value = mock_processor

            result = runner.invoke(app, ["query", "invalid query"])

            assert result.exit_code == 0  # CLI doesn't exit with error, just shows error
            assert "Query failed" in result.stdout or "Processing failed" in result.stdout

    @patch('easyprompt.cli.main.validate_config')
    def test_chat_command_initialization(self, mock_validate, runner, mock_settings):
        """Test chat command initialization."""
        mock_validate.return_value = mock_settings

        with patch('easyprompt.cli.commands.QueryProcessor') as mock_processor_class, \
             patch('easyprompt.cli.commands.Prompt') as mock_prompt:

            # Mock user input to exit immediately
            mock_prompt.ask.return_value = "exit"

            mock_processor = AsyncMock()
            mock_processor.initialize = AsyncMock()
            mock_processor.close = AsyncMock()
            mock_processor_class.return_value = mock_processor

            result = runner.invoke(app, ["chat"])

            assert result.exit_code == 0
            mock_processor.initialize.assert_called_once()

    def test_log_level_option(self, runner):
        """Test log level option."""
        with patch('easyprompt.cli.main.setup_logging') as mock_setup_logging, \
             patch('easyprompt.cli.main.validate_config'), \
             patch('easyprompt.cli.init_command.init_configuration'):

            result = runner.invoke(app, ["--log-level", "DEBUG", "init"])

            mock_setup_logging.assert_called_with("DEBUG")

    def test_config_file_option(self, runner):
        """Test config file option."""
        with patch('easyprompt.cli.init_command.init_configuration') as mock_init:
            result = runner.invoke(app, ["--config", "custom.env", "init"])

            # The config option is captured but not used in init command
            mock_init.assert_called_once()


class TestCLICommandClasses:
    """Test individual CLI command classes."""

    @pytest.fixture
    def mock_console(self):
        """Mock rich console."""
        return Mock()

    def test_base_command_run_async(self, sample_settings, mock_console):
        """Test BaseCommand run_async method."""
        from easyprompt.cli.commands import BaseCommand

        command = BaseCommand(sample_settings, mock_console)

        async def test_coro():
            return "test result"

        result = command.run_async(test_coro())
        assert result == "test result"

    def test_base_command_keyboard_interrupt(self, sample_settings, mock_console):
        """Test BaseCommand handling keyboard interrupt."""
        from easyprompt.cli.commands import BaseCommand

        command = BaseCommand(sample_settings, mock_console)

        async def test_coro():
            raise KeyboardInterrupt()

        with pytest.raises(SystemExit):
            command.run_async(test_coro())

        mock_console.print.assert_called_with("\n[yellow]Operation cancelled by user[/yellow]")

    def test_query_command_execute_command(self, sample_settings, mock_console, mock_subprocess_run):
        """Test QueryCommand execute_command method."""
        from easyprompt.cli.commands import QueryCommand

        command = QueryCommand(sample_settings, mock_console)
        command._execute_command("echo test")

        mock_subprocess_run.assert_called_once()

    def test_query_command_execute_with_confirmation(self, sample_settings, mock_console, mock_subprocess_run):
        """Test QueryCommand execute_command with confirmation."""
        from easyprompt.cli.commands import QueryCommand

        # Enable confirmation in settings
        sample_settings.confirm_before_execution = True
        command = QueryCommand(sample_settings, mock_console)

        with patch('easyprompt.cli.commands.Confirm.ask', return_value=False):
            command._execute_command("rm important_file")

            # Should not execute if user declines
            mock_subprocess_run.assert_not_called()
            mock_console.print.assert_called_with("[yellow]Command execution cancelled[/yellow]")

    def test_status_command_display_verbose(self, sample_settings, mock_console):
        """Test StatusCommand verbose display."""
        from easyprompt.cli.commands import StatusCommand

        command = StatusCommand(sample_settings, mock_console)

        status = {
            "initialized": True,
            "cli_tool": "test-cli",
            "vector_db_status": "healthy",
            "total_documents": 42
        }

        command._display_status(status, verbose=True)

        # Should display configuration table in verbose mode
        assert mock_console.print.call_count >= 2  # Status table + config table

    def test_status_command_display_error(self, sample_settings, mock_console):
        """Test StatusCommand error display."""
        from easyprompt.cli.commands import StatusCommand

        command = StatusCommand(sample_settings, mock_console)

        status = {"error": "System initialization failed"}

        command._display_status(status, verbose=False)

        # Should display error status
        mock_console.print.assert_called_once()