"""Integration tests for end-to-end functionality."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel

from tidyllm.cli import generate_cli
from tidyllm.registry import Registry, ToolError
from tidyllm.prompt import read_prompt
from tidyllm.registry import REGISTRY, register


class FileArgs(BaseModel):
    """File operation arguments."""

    path: str
    content: str = ""


class FileResult(BaseModel):
    """File operation result."""

    success: bool
    path: str
    message: str


class ProjectContext:
    """Project context for file operations."""
    
    def __init__(self, project_root: Path, dry_run: bool = False):
        self.project_root = project_root
        self.dry_run = dry_run


class TestEndToEndIntegration:
    """Test complete end-to-end workflows."""

    def setup_method(self):
        """Save registry state before each test."""
        self._saved_tools = REGISTRY._tools.copy()

    def teardown_method(self):
        """Restore registry state after each test."""
        REGISTRY._tools = self._saved_tools

    def test_register_and_execute_tool(self):
        """Test complete workflow: register tool -> create library -> execute."""

        @register()
        def write_file(args: FileArgs) -> FileResult:
            """Write content to a file."""
            # For testing purposes, we'll write to a temp directory
            # In real usage, tools would get context via get_tool_context()
            file_path = Path(args.path)
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(args.content)
            return FileResult(
                success=True,
                path=args.path,
                message=f"Successfully wrote to {args.path}",
            )

        # Create library and execute tool call
        with TemporaryDirectory() as tmpdir:
            library = Registry()
            for func in [write_file]:
                library.register(func)
            
            # Change to tmpdir so relative paths work
            import os
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                # Execute tool call
                result = library.call("write_file", {"path": "test.txt", "content": "Hello, world!"})

                assert isinstance(result, FileResult)
                assert result.success is True
                assert result.path == "test.txt"

                # Verify file was actually written
                written_file = Path(tmpdir) / "test.txt"
                assert written_file.exists()
                assert written_file.read_text() == "Hello, world!"
            finally:
                os.chdir(original_cwd)

    def test_multiple_tools_workflow(self):
        """Test workflow with multiple interacting tools."""

        @register()
        def create_dir(args: FileArgs) -> FileResult:
            """Create a directory."""
            dir_path = Path(args.path)
            
            dir_path.mkdir(parents=True, exist_ok=True)
            return FileResult(
                success=True,
                path=args.path,
                message=f"Created directory {args.path}",
            )

        @register()
        def write_file(args: FileArgs) -> FileResult:
            """Write content to a file."""
            file_path = Path(args.path)
            
            file_path.write_text(args.content)
            return FileResult(success=True, path=args.path, message=f"Wrote file {args.path}")

        with TemporaryDirectory() as tmpdir:
            library = Registry()
            for func in [create_dir, write_file]:
                library.register(func)
            
            # Change to tmpdir so relative paths work
            import os
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                # Step 1: Create directory
                dir_result = library.call("create_dir", {"path": "subdir"})
                assert isinstance(dir_result, FileResult)
                assert dir_result.success is True

                # Step 2: Write file in that directory
                file_result = library.call(
                    "write_file",
                    {"path": "subdir/config.json", "content": '{"setting": "value"}'},
                )

                assert isinstance(file_result, FileResult)
                assert file_result.success is True

                # Verify results
                config_file = Path(tmpdir) / "subdir" / "config.json"
                assert config_file.exists()
                assert json.loads(config_file.read_text()) == {"setting": "value"}
            finally:
                os.chdir(original_cwd)

    def test_tool_with_prompt_file(self):
        """Test tool registration with external prompt file."""

        with TemporaryDirectory() as tmpdir:
            # Create prompt file
            prompt_file = Path(tmpdir) / "PROMPT.md"
            prompt_file.write_text(
                """# File Writer Tool

A tool for writing content to files with validation.

## Parameters

- `path` (str): Path to the file to write
- `content` (str): Content to write to the file

## Returns

A result object indicating success or failure."""
            )

            @register(doc=read_prompt(str(prompt_file)))
            def documented_tool(args: FileArgs) -> FileResult:
                """Tool with external documentation."""
                return FileResult(success=True, path=args.path, message="Documented tool executed")

            # Check that schema uses prompt content
            func_desc = REGISTRY.get_description("documented_tool")
            schema = func_desc.function_schema
            assert "A tool for writing content to files" in schema["function"]["description"]
            assert "Parameters" in schema["function"]["description"]

    def test_cli_integration(self):
        """Test CLI generation and execution integration."""

        @register()
        def math_tool(args: FileArgs) -> dict:
            """Perform math operations."""
            # Use content as a number for this test
            try:
                value = int(args.content)
                return {"input": value, "doubled": value * 2, "squared": value**2}
            except ValueError:
                return {"error": "Content must be a number"}

        # Generate CLI
        cli_command = generate_cli(math_tool)

        # Test CLI execution - should fail due to Protocol context
        from click.testing import CliRunner

        runner = CliRunner()

        result = runner.invoke(cli_command, ["--path", "test", "--content", "5"])

        # CLI should work now with contextvar approach
        assert result.exit_code == 0
        # Result should contain the math operations
        assert "doubled" in result.output or "5" in result.output

    def test_error_propagation_integration(self):
        """Test that errors propagate correctly through the system."""

        @register()
        def failing_tool(args: FileArgs) -> FileResult:
            """Tool that demonstrates error handling."""
            if args.path == "fail":
                raise RuntimeError("Intentional failure")

            return FileResult(success=True, path=args.path, message="Success")

        library = Registry()
        library = Registry()
        for func in [failing_tool]:
            library.register(func)

        # Test successful call
        success_result = library.call("failing_tool", {"path": "success", "content": "test"})
        assert isinstance(success_result, FileResult)
        assert success_result.success is True

        # Test failing call
        fail_result = library.call("failing_tool", {"path": "fail", "content": "test"})
        assert isinstance(fail_result, ToolError)
        assert "Tool execution failed" in fail_result.error
        assert "Intentional failure" in fail_result.error

    def test_basic_functionality_integration(self):
        """Test basic functionality without context dependencies."""

        @register()
        def simple_tool(args: FileArgs) -> FileResult:
            """Simple tool for testing basic functionality."""
            return FileResult(
                success=True,
                path=args.path,
                message=f"Processed file: {args.path}",
            )

        library = Registry()
        library = Registry()
        for func in [simple_tool]:
            library.register(func)

        result = library.call("simple_tool", {"path": "test.txt", "content": "test"})
        assert isinstance(result, FileResult)
        assert result.success is True
        assert "Processed file: test.txt" in result.message

    def test_schema_generation_integration(self):
        """Test that schema generation works end-to-end."""

        @register()
        def complex_tool(args: FileArgs) -> FileResult:
            """Complex tool with detailed schema."""
            return FileResult(success=True, path=args.path, message="OK")

        library = Registry()
        library = Registry()
        for func in [complex_tool]:
            library.register(func)
        schemas = library.get_schemas()

        assert len(schemas) == 1
        schema = schemas[0]

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "complex_tool"
        assert "Complex tool" in schema["function"]["description"]

        # Check parameters schema
        params = schema["function"]["parameters"]
        assert "properties" in params
        assert "path" in params["properties"]
        assert "content" in params["properties"]

        # path should be required, content should have default
        required = params.get("required", [])
        assert "path" in required
        assert "content" not in required

    def test_prompt_includes_integration(self):
        """Test prompt system with includes in full integration."""

        with TemporaryDirectory() as tmpdir:
            # Create included files
            overview_file = Path(tmpdir) / "overview.md"
            overview_file.write_text("This tool processes files safely.")

            examples_file = Path(tmpdir) / "examples.md"
            examples_file.write_text(
                """
## Example Usage

```json
{"path": "test.txt", "content": "Hello World"}
```
"""
            )

            # Create main prompt with includes
            main_prompt = Path(tmpdir) / "PROMPT.md"
            main_prompt.write_text(
                """# File Tool

{{include: ./overview.md}}

## Parameters

- path: File path
- content: File content

{{include: ./examples.md}}
"""
            )

            @register(doc=read_prompt(str(main_prompt)))
            def include_tool(args: FileArgs) -> FileResult:
                """Tool with complex prompt."""
                return FileResult(success=True, path=args.path, message="Included")

            # Verify prompt was processed correctly
            func_desc = REGISTRY.get_description("include_tool")
            schema = func_desc.function_schema
            description = schema["function"]["description"]

            assert "This tool processes files safely" in description
            assert "Example Usage" in description
            assert "File path" in description
            assert "{{include:" not in description  # Should be processed
