"""Test FastMCP integration with registry."""


import pytest

from tidyllm.context import set_tool_context
from tidyllm.registry import REGISTRY
from tidyllm.tools.config import Config
from tidyllm.tools.context import ToolContext


def test_context_variable_integration():
    """Test that context variables work correctly with tool execution."""
    from tidyllm.context import get_tool_context

    # Create a test tool that uses context
    def test_context_tool() -> str:
        """Test tool that uses context."""
        ctx = get_tool_context()
        return f"Config notes_dir: {ctx.config.notes_dir}"

    # Register the tool
    from tidyllm.registry import register
    test_context_tool = register()(test_context_tool)

    try:
        # Create a test context
        config = Config(notes_dir="/test/notes")
        context = ToolContext(config=config)

        # Test that context is available within the context manager
        with set_tool_context(context):
            result = test_context_tool()
            assert "Config notes_dir: /test/notes" in result

        # Test that context is not available outside the context manager
        with pytest.raises(RuntimeError, match="No tool context available"):
            test_context_tool()

    finally:
        # Clean up
        if 'test_context_tool' in REGISTRY._tools:
            del REGISTRY._tools['test_context_tool']


def test_nested_context_support():
    """Test that contextvars support proper nesting."""
    from tidyllm.context import get_tool_context

    def test_nested_tool() -> str:
        """Test tool for nested context."""
        ctx = get_tool_context()
        return str(ctx.config.notes_dir)

    # Register the tool
    from tidyllm.registry import register
    test_nested_tool = register()(test_nested_tool)

    try:
        # Create two different contexts
        config1 = Config(notes_dir="/outer/notes")
        context1 = ToolContext(config=config1)

        config2 = Config(notes_dir="/inner/notes") 
        context2 = ToolContext(config=config2)

        # Test nested contexts
        with set_tool_context(context1):
            result1 = test_nested_tool()
            assert "/outer/notes" in result1

            with set_tool_context(context2):
                result2 = test_nested_tool()
                assert "/inner/notes" in result2

            # Should be back to outer context
            result3 = test_nested_tool()
            assert "/outer/notes" in result3

    finally:
        # Clean up
        if 'test_nested_tool' in REGISTRY._tools:
            del REGISTRY._tools['test_nested_tool']
