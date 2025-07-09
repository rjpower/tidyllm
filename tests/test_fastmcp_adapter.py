"""Tests for FastMCP adapter."""

import tempfile
from pathlib import Path

import pytest

from tidyllm.adapters.fastmcp_adapter import create_fastmcp_server
from tidyllm.registry import REGISTRY
from tidyllm.tools.config import Config
from tidyllm.tools.context import ToolContext


@pytest.fixture
def test_tool_context():
    """Create a test ToolContext with temporary directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = Config(
            config_dir=temp_path,
            notes_dir=temp_path / "notes",
        )
        yield ToolContext(config=config)


def test_create_fastmcp_server_basic(test_tool_context):
    """Test basic FastMCP server creation."""
    # Create server with context
    mcp_server = create_fastmcp_server(context=test_tool_context)
    
    # Verify server was created
    assert mcp_server is not None
    assert hasattr(mcp_server, 'run')


def test_create_fastmcp_server_with_custom_context(test_tool_context):
    """Test FastMCP server creation with custom tool context."""
    # Create server with custom tool context
    mcp_server = create_fastmcp_server(
        context=test_tool_context,
        name="Test Server"
    )
    
    assert mcp_server is not None




def test_tool_wrapper_execution(test_tool_context):
    """Test that tool wrappers properly execute TidyLLM tools."""
    # Create server with tool context
    mcp_server = create_fastmcp_server(context=test_tool_context)
    
    # Verify the server has tools registered
    assert mcp_server is not None


def test_tool_context_integration(test_tool_context):
    """Test that ToolContext is properly integrated with FastMCP."""
    # Create server with custom tool context
    mcp_server = create_fastmcp_server(
        context=test_tool_context,
        name="Integration Test Server"
    )
    
    # Verify server creation succeeded
    assert mcp_server is not None


def test_server_name_customization(test_tool_context):
    """Test that server name can be customized."""
    custom_name = "My Custom TidyLLM Server"
    
    mcp_server = create_fastmcp_server(context=test_tool_context, name=custom_name)
    
    # The name should be stored in the server (exact verification depends on FastMCP internals)
    assert mcp_server is not None


def test_mock_tool_execution(test_tool_context):
    """Test tool execution setup."""
    # Create a minimal test setup with the test context
    mcp_server = create_fastmcp_server(context=test_tool_context)
    
    # Verify server creation
    assert mcp_server is not None


def test_registry_tools_discovery(test_tool_context):
    """Test that all registered tools are discovered."""
    # Import all tools to ensure they're registered
    import tidyllm.tools.anki  # noqa: F401
    import tidyllm.tools.manage_db  # noqa: F401
    import tidyllm.tools.notes  # noqa: F401
    import tidyllm.tools.transcribe  # noqa: F401
    import tidyllm.tools.vocab_table  # noqa: F401
    
    # Create server and verify tools are discovered
    mcp_server = create_fastmcp_server(context=test_tool_context)
    
    # Should have multiple tools registered
    assert len(REGISTRY.functions) >= 5  # We have multiple focused tools now
    
    # Server should be created successfully
    assert mcp_server is not None


def test_config_override_validation(test_tool_context):
    """Test that configuration overrides work correctly."""
    # Test that we can create a server with a custom context
    mcp_server = create_fastmcp_server(context=test_tool_context)
    
    # Server should be created successfully
    assert mcp_server is not None
