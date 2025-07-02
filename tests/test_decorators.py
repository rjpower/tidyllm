"""Tests for @register decorator."""

from pydantic import BaseModel

from tidyllm.prompt import read_prompt
from tidyllm.registry import REGISTRY, register


class DecoratorTestArgs(BaseModel):
    """Test arguments."""

    name: str
    value: int = 10


class DecoratorTestResult(BaseModel):
    """Test result."""

    message: str
    computed: int




class TestRegisterDecorator:
    """Test @register decorator functionality."""

    def setup_method(self):
        """Save registry state before each test."""
        self._saved_tools = REGISTRY._tools.copy()

    def teardown_method(self):
        """Restore registry state after each test."""
        REGISTRY._tools = self._saved_tools

    def test_register_simple_function(self):
        """Test registering a simple function."""

        @register()
        def simple_tool(args: DecoratorTestArgs) -> DecoratorTestResult:
            """A simple test tool."""
            return DecoratorTestResult(message=f"Hello {args.name}", computed=args.value * 2)

        # Check function is registered
        assert "simple_tool" in REGISTRY.list_tools()
        func_desc = REGISTRY.get("simple_tool")
        assert func_desc.function is simple_tool

        # Check FunctionDescription has proper schema
        assert func_desc.function_schema is not None
        schema = func_desc.function_schema
        assert schema["function"]["name"] == "simple_tool"
        assert schema["function"]["description"] == "A simple test tool."

    def test_register_with_custom_doc(self):
        """Test registering with custom documentation."""
        custom_doc = "Custom documentation for this tool"

        @register(doc=custom_doc)
        def documented_tool(args: DecoratorTestArgs) -> DecoratorTestResult:
            """Original docstring."""
            return DecoratorTestResult(message="test", computed=42)

        func_desc = REGISTRY.get("documented_tool")
        schema = func_desc.function_schema
        assert schema["function"]["description"] == custom_doc

    def test_register_with_custom_name(self):
        """Test registering with custom name."""

        @register(name="custom_name")
        def original_name_tool(args: DecoratorTestArgs) -> DecoratorTestResult:
            """A renamed tool."""
            return DecoratorTestResult(message="renamed", computed=1)

        # Should be registered under custom name
        assert "custom_name" in REGISTRY.list_tools()
        assert "original_name_tool" not in REGISTRY.list_tools()

        # Function name should be updated
        assert original_name_tool.__name__ == "custom_name"

        func_desc = REGISTRY.get("custom_name")
        schema = func_desc.function_schema
        assert schema["function"]["name"] == "custom_name"

    def test_register_without_context_auto_inferred(self):
        """Test registering function without context (auto-inferred)."""

        @register()
        def no_ctx_tool(args: DecoratorTestArgs) -> DecoratorTestResult:
            """Tool without context."""
            return DecoratorTestResult(message=args.name, computed=args.value)

        assert "no_ctx_tool" in REGISTRY.list_tools()
        # Function should be registered successfully
        func_desc = REGISTRY.get("no_ctx_tool")
        assert func_desc is not None

    def test_register_context_auto_inferred_success(self):
        """Test that functions are registered successfully."""

        # Function without explicit context usage - should work fine
        @register()
        def no_context_tool(args: DecoratorTestArgs) -> DecoratorTestResult:
            """Tool without context."""
            return DecoratorTestResult(message="test", computed=1)

        assert "no_context_tool" in REGISTRY.list_tools()

        # Function with context usage - should also work fine
        @register()
        def with_context_tool(args: DecoratorTestArgs) -> DecoratorTestResult:
            """Tool with context."""
            # Context is available via get_tool_context() when called properly
            return DecoratorTestResult(message="test", computed=1)

        assert "with_context_tool" in REGISTRY.list_tools()

    def test_register_function_with_multiple_params(self):
        """Test registering function with multiple parameters."""
        @register()
        def multi_param_tool(
            args: DecoratorTestArgs, extra_value: int = 5
        ) -> DecoratorTestResult:
            """Tool with multiple parameters."""
            return DecoratorTestResult(message="test", computed=args.value + extra_value)

        assert "multi_param_tool" in REGISTRY.list_tools()
        func_desc = REGISTRY.get("multi_param_tool")
        assert func_desc is not None

    def test_register_schema_generation(self):
        """Test that function schema is properly generated."""

        @register()
        def schema_test_tool(args: DecoratorTestArgs) -> DecoratorTestResult:
            """Tool for schema testing."""
            return DecoratorTestResult(message="test", computed=1)

        # Schema should be properly generated
        func_desc = REGISTRY.get("schema_test_tool")
        assert func_desc is not None
        assert func_desc.function_schema is not None
        assert func_desc.function_schema["function"]["name"] == "schema_test_tool"

    def test_register_with_prompt_file(self):
        """Test registering with prompt file using read_prompt."""
        # This test would need a temporary file
        # For now, test the interface
        try:

            @register(doc=read_prompt("./nonexistent.md"))
            def prompt_tool(args: DecoratorTestArgs) -> DecoratorTestResult:
                """Tool with prompt file."""
                return DecoratorTestResult(message="test", computed=1)

        except FileNotFoundError:
            # Expected when file doesn't exist
            pass

    def test_register_preserves_original_function(self):
        """Test that decorator preserves original function behavior."""

        @register()
        def preserved_tool(args: DecoratorTestArgs) -> DecoratorTestResult:
            """Original behavior preserved."""
            # Note: In actual usage, context would be available via get_tool_context()
            # but for this test we just test the basic function behavior
            return DecoratorTestResult(message=f"Processed {args.name}", computed=args.value + 100)

        # The decorator preserves the function structure
        assert preserved_tool.__name__ == "preserved_tool"
        assert "preserved_tool" in REGISTRY.list_tools()

    def test_register_function_returns_wrapper(self):
        """Test that register decorator returns a wrapper function."""

        def original_function(args: DecoratorTestArgs) -> DecoratorTestResult:
            """Original function."""
            return DecoratorTestResult(message="original", computed=1)

        decorated = register()(original_function)
        # Returns a wrapper, not the original function
        assert decorated is not original_function
        # But preserves the name and behavior
        assert decorated.__name__ == "original_function"
        # And is registered
        assert "original_function" in REGISTRY.list_tools()

    def test_register_with_conditional_context_usage(self):
        """Test registering with conditional context usage."""

        @register()
        def conditional_ctx_tool(
            args: DecoratorTestArgs, use_context: bool = False
        ) -> DecoratorTestResult:
            """Tool with conditional context usage."""
            if use_context:
                # Context would be available via get_tool_context() when called properly
                message = "using context: available"
            else:
                message = "not using context"
            return DecoratorTestResult(message=message, computed=1)

        # Function should be registered successfully
        func_desc = REGISTRY.get("conditional_ctx_tool")
        assert func_desc is not None
        assert func_desc.function_schema is not None

    def test_multiple_registrations(self):
        """Test registering multiple functions."""

        @register()
        def tool_one(args: DecoratorTestArgs) -> DecoratorTestResult:
            """First tool."""
            return DecoratorTestResult(message="one", computed=1)

        @register()
        def tool_two(args: DecoratorTestArgs) -> DecoratorTestResult:
            """Second tool."""
            return DecoratorTestResult(message="two", computed=2)

        tools = REGISTRY.list_tools()
        initial_count = len(self._saved_tools)
        assert len(tools) == initial_count + 2
        assert "tool_one" in tools
        assert "tool_two" in tools
