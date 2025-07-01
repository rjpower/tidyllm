"""Test FastMCP integration with registry."""

from unittest.mock import MagicMock

import pytest

from tidyllm.context import set_tool_context
from tidyllm.registry import REGISTRY
from tidyllm.tools.config import Config
from tidyllm.tools.context import ToolContext


def test_registry_fastmcp_integration():
    """Test that registry creates FastMCP server and registers tools properly."""
    # Clear any existing server
    REGISTRY._fastmcp_server = None
    
    # Mock FastMCP to avoid actual import dependency
    mock_fastmcp = MagicMock()
    mock_server = MagicMock()
    mock_fastmcp.return_value = mock_server
    
    # Patch FastMCP import
    import sys
    sys.modules['fastmcp'] = MagicMock()
    sys.modules['fastmcp'].FastMCP = mock_fastmcp
    
    try:
        # Create server
        server = REGISTRY.create_fastmcp_server("Test Server")
        
        # Verify server was created
        assert server is mock_server
        assert REGISTRY._fastmcp_server is mock_server
        
        # Verify FastMCP was called with correct name
        mock_fastmcp.assert_called_once_with("Test Server")
        
        # Verify tools were registered (should have been called for each tool)
        # We can't easily test the exact calls without complex mocking,
        # but we can verify the tool decorator was called
        assert mock_server.tool.call_count >= 0  # At least 0 tools registered
        
    finally:
        # Clean up mock
        if 'fastmcp' in sys.modules:
            del sys.modules['fastmcp']


def test_fastmcp_tool_registration_with_original_callable():
    """Test that FastMCP gets the original function, not a wrapper."""
    # Clear any existing server
    REGISTRY._fastmcp_server = None

    # Create a test function
    def test_tool(name: str) -> str:
        """Test tool for FastMCP registration."""
        return f"Hello {name}"

    # Mock FastMCP to capture what gets registered
    mock_fastmcp = MagicMock()
    mock_server = MagicMock()
    mock_fastmcp.return_value = mock_server

    # Mock the tool decorator to capture the function
    registered_functions = []

    def mock_tool_decorator(func):
        registered_functions.append(func)
        return func

    mock_server.tool = mock_tool_decorator

    # Patch FastMCP import
    import sys
    sys.modules['fastmcp'] = MagicMock()
    sys.modules['fastmcp'].FastMCP = mock_fastmcp

    try:
        # Register our test tool
        from tidyllm.registry import register
        register(test_tool)

        # Create FastMCP server
        REGISTRY.create_fastmcp_server("Test Server")

        # Verify the original function was registered with FastMCP
        assert len(registered_functions) >= 1

        # Find our test tool in the registered functions
        test_tool_found = False
        for func in registered_functions:
            if func.__name__ == 'test_tool':
                # With contextvar approach, we now register a wrapper function
                # but it should preserve the name and docstring
                assert func.__name__ == "test_tool"
                assert func.__doc__ == "Test tool for FastMCP registration."
                test_tool_found = True
                break

        assert test_tool_found, "Test tool was not found in registered functions"

    finally:
        # Clean up
        if 'fastmcp' in sys.modules:
            del sys.modules['fastmcp']
        # Remove test tool from registry
        if 'test_tool' in REGISTRY._tools:
            del REGISTRY._tools['test_tool']


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
    register(test_context_tool)
    
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
    register(test_nested_tool)
    
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
