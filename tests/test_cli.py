"""Tests for CLI generation."""

from typing import Protocol

from click.testing import CliRunner
from pydantic import BaseModel

from tidyllm.adapters.cli import generate_cli


class SimpleArgs(BaseModel):
    """Simple test arguments."""

    name: str
    count: int = 5
    flag: bool = False


class ComplexArgs(BaseModel):
    """Complex test arguments."""

    items: list[str]
    config: dict[str, str] = {}
    optional_value: int | None = None


class CLITestContext(Protocol):
    """Test context for CLI tests."""

    project_root: str


def simple_tool(args: SimpleArgs) -> dict:
    """A simple test tool."""
    return {"message": f"Hello {args.name}", "count": args.count, "flag": args.flag}


def context_tool(args: SimpleArgs) -> dict:
    """Tool that requires context."""
    from tidyllm.context import get_tool_context

    ctx = get_tool_context()
    return {"message": f"Hello {args.name} from {ctx.config.notes_dir}"}


def complex_tool(args: ComplexArgs) -> dict:
    """Tool with complex arguments."""
    return {"items": args.items, "config": args.config}


class TestCLIGeneration:
    """Test CLI generation functionality."""

    def test_generate_cli_basic(self):
        """Test generating CLI for simple tool."""
        cli_command = generate_cli(simple_tool)

        assert cli_command.name == "simple_tool"
        assert cli_command.help is not None
        # Check that CLI was created successfully
        assert callable(cli_command)

    def test_cli_execution_with_individual_args(self):
        """Test executing CLI with individual arguments."""
        cli_command = generate_cli(simple_tool)
        runner = CliRunner()

        result = runner.invoke(
            cli_command, ["--name", "test", "--count", "10", "--flag"]
        )

        if result.exit_code != 0:
            print(f"CLI Error: {result.output}")
            print(f"Exception: {result.exception}")
        assert result.exit_code == 0
        # Check text output contains expected values
        assert "Hello test" in result.output
        assert "10" in result.output
        assert "true" in result.output

    def test_cli_execution_with_defaults(self):
        """Test CLI execution using default values."""
        cli_command = generate_cli(simple_tool)
        runner = CliRunner()

        result = runner.invoke(cli_command, ["--name", "test"])

        if result.exit_code != 0:
            print(f"CLI Error: {result.output}")
            print(f"Exception: {result.exception}")

        assert result.exit_code == 0
        # Check text output contains expected values
        assert "Hello test" in result.output
        assert "5" in result.output  # default count value
        assert "false" in result.output  # default flag value

    def test_cli_execution_with_json_input(self):
        """Test CLI execution with JSON input."""
        cli_command = generate_cli(simple_tool)
        runner = CliRunner()

        json_input = '{"name": "json_test", "count": 15, "flag": true}'

        result = runner.invoke(cli_command, ["--json", json_input])

        assert result.exit_code == 0
        # Check text output contains expected values
        assert "Hello json_test" in result.output
        assert "15" in result.output
        assert "true" in result.output

    def test_cli_json_overrides_individual_args(self):
        """Test that JSON input takes precedence over individual args."""
        cli_command = generate_cli(simple_tool)
        runner = CliRunner()

        json_input = '{"name": "json_name", "count": 20}'

        result = runner.invoke(
            cli_command,
            ["--json", json_input, "--name", "ignored_name", "--count", "999"],
        )

        assert result.exit_code == 0
        # Check text output contains JSON values, not the individual args
        assert "Hello json_name" in result.output
        assert "20" in result.output
        # Should not contain the ignored values
        assert "ignored_name" not in result.output
        assert "999" not in result.output

    def test_cli_invalid_json_input(self):
        """Test CLI with invalid JSON input."""
        cli_command = generate_cli(simple_tool)
        runner = CliRunner()

        result = runner.invoke(cli_command, ["--json", "invalid json"])

        assert result.exit_code == 0  # CLI might handle invalid JSON gracefully
        # Should contain error message
        assert "error" in result.output.lower() or "invalid" in result.output.lower()

    def test_cli_missing_required_arguments(self):
        """Test CLI with missing required arguments."""
        cli_command = generate_cli(simple_tool)
        runner = CliRunner()

        result = runner.invoke(
            cli_command, ["--count", "10"]
        )  # Missing required 'name'

        assert result.exit_code == 1  # Should exit with error for missing required args
        # The error should be shown to the user via exception or output
        error_text = str(result.exception) if result.exception else result.output
        assert "Field required" in error_text or "validation error" in error_text

    def test_cli_with_context_tool(self):
        """Test CLI generation for tool that uses context variables."""
        # Tools using contextvars should work with CLI when context_cls is provided
        from tidyllm.tools.context import ToolContext

        cli_command = generate_cli(context_tool, context_cls=ToolContext)
        runner = CliRunner()

        result = runner.invoke(cli_command, ["--name", "context_test"])
        assert result.exit_code == 0

        # Check text output contains expected values
        assert "Hello context_test from" in result.output

    def test_cli_complex_arguments(self):
        """Test CLI with complex argument types."""
        cli_command = generate_cli(complex_tool)
        runner = CliRunner()

        json_input = '{"items": ["item1", "item2", "item3"], "config": {"key1": "value1", "key2": "value2"}, "optional_value": 42}'

        result = runner.invoke(cli_command, ["--json", json_input])

        assert result.exit_code == 0
        # Check text output contains expected values
        assert "item1" in result.output
        assert "item2" in result.output
        assert "item3" in result.output
        assert "key1" in result.output
        assert "value1" in result.output

    def test_cli_tool_execution_error(self):
        """Test CLI when tool execution fails."""

        def failing_tool(args: SimpleArgs) -> dict:
            """Tool that always fails."""
            raise RuntimeError("Tool execution failed")

        cli_command = generate_cli(failing_tool)
        runner = CliRunner()

        result = runner.invoke(cli_command, ["--name", "test"])

        assert (
            result.exit_code == 1
        )  # Should exit with error for tool execution failure
        # The error should be shown to the user via exception or output
        error_text = str(result.exception) if result.exception else result.output
        assert "Tool execution failed" in error_text

    def test_cli_no_pydantic_args_model(self):
        """Test CLI generation for tool with non-Pydantic multiple args."""

        def multi_arg_tool(name: str, count: int = 5) -> dict:
            """Tool with non-Pydantic args."""
            return {"name": name, "count": count}

        # Should succeed now since we support multiple parameters
        cli_command = generate_cli(multi_arg_tool)
        assert cli_command is not None
        assert cli_command.name == "multi_arg_tool"

    def test_cli_option_name_transformation(self):
        """Test that underscores in field names become dashes in CLI options."""

        class ArgsWithUnderscore(BaseModel):
            field_with_underscore: str
            another_field: int = 10

        def underscore_tool(args: ArgsWithUnderscore) -> dict:
            """Tool with underscore fields."""
            return {"field": args.field_with_underscore, "another": args.another_field}

        cli_command = generate_cli(underscore_tool)
        runner = CliRunner()

        result = runner.invoke(
            cli_command,
            ["--field-with-underscore", "test_value", "--another-field", "20"],
        )

        assert result.exit_code == 0
        # Check text output contains expected values
        assert "test_value" in result.output
        assert "20" in result.output

    def test_cli_help_includes_descriptions(self):
        """Test that CLI help includes field descriptions."""

        class DocumentedArgs(BaseModel):
            name: str  # Should get description from field info
            count: int = 5

        # Add field descriptions
        DocumentedArgs.model_fields["name"].description = "The name parameter"
        DocumentedArgs.model_fields["count"].description = "Number of items"

        def documented_tool(args: DocumentedArgs) -> dict:
            """Well documented tool."""
            return {"name": args.name, "count": args.count}

        cli_command = generate_cli(documented_tool)
        runner = CliRunner()

        result = runner.invoke(cli_command, ["--help"])

        assert result.exit_code == 0
        assert "The name parameter" in result.output
        assert "Number of items" in result.output
