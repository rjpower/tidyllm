"""FastMCP adapter for TidyLLM registry functions."""

from typing import Any

from tidyllm.context import set_tool_context
from tidyllm.registry import REGISTRY
from tidyllm.tools.context import ToolContext


def create_fastmcp_server(
    name: str = "TidyLLM Tools",
    context: ToolContext | None = None,
    tool_context: ToolContext | None = None,
):
    """Create a FastMCP server that exposes TidyLLM tools.

    Args:
        name: Server name
        context: Optional ToolContext to use for tool execution
        function_library: Optional FunctionLibrary (for backwards compatibility)
        tool_context: Optional ToolContext (alternative to context parameter)

    Returns:
        FastMCP server with all registered tools

    Example:
        from tidyllm.adapters.fastmcp_adapter import create_fastmcp_server
        from tidyllm.tools.context import ToolContext

        # Create with default context
        server = create_fastmcp_server()

        # Create with custom context
        from tidyllm.tools.config import Config
        context = ToolContext(config=Config())
        server = create_fastmcp_server(context=context)
    """
    # Handle backward compatibility - prefer tool_context, then context
    if tool_context is not None:
        context = tool_context
    elif context is None:
        from tidyllm.tools.config import Config
        context = ToolContext(config=Config())

    server = REGISTRY.create_fastmcp_server(context=context, name=name)
    return server


def create_tidyllm_mcp_server(config_overrides: dict[str, Any] | None = None):
    """Create a FastMCP server specifically for TidyLLM tools with auto-discovery.

    Args:
        config_overrides: Dictionary of configuration overrides for ToolContext.config

    Returns:
        Configured FastMCP server with all discovered TidyLLM tools

    Example:
        from tidyllm.adapters.fastmcp_adapter import create_tidyllm_mcp_server

        # Create server with auto-discovered tools
        mcp = create_tidyllm_mcp_server()

        # Create with custom configuration
        mcp = create_tidyllm_mcp_server(config_overrides={"notes_dir": "/custom/notes"})
    """
    # Import all tools to ensure they're registered
    import tidyllm.tools.anki  # noqa: F401
    import tidyllm.tools.manage_db  # noqa: F401
    import tidyllm.tools.notes  # noqa: F401
    import tidyllm.tools.transcribe  # noqa: F401
    import tidyllm.tools.vocab_table  # noqa: F401

    # Create tool context
    from tidyllm.tools.config import Config
    config = Config()
    
    if config_overrides:
        # Allow configuration overrides
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    context = ToolContext(config=config)

    return create_fastmcp_server(
        name="TidyLLM Tools Server",
        context=context
    )


def run_tidyllm_mcp_server(config_overrides: dict[str, Any] | None = None):
    """Run the TidyLLM MCP server.

    Args:
        config_overrides: Optional configuration overrides for ToolContext.config

    Example:
        from tidyllm.adapters.fastmcp_adapter import run_tidyllm_mcp_server

        # Run with default configuration
        run_tidyllm_mcp_server()

        # Run with custom configuration
        run_tidyllm_mcp_server(config_overrides={"notes_dir": "/custom/notes"})
    """
    mcp = create_tidyllm_mcp_server(config_overrides)
    mcp.run()


async def run_tidyllm_mcp_server_async(config_overrides: dict[str, Any] | None = None):
    """Run the TidyLLM MCP server asynchronously.

    Args:
        config_overrides: Optional configuration overrides for ToolContext.config

    Example:
        import asyncio
        from tidyllm.adapters.fastmcp_adapter import run_tidyllm_mcp_server_async

        async def main():
            await run_tidyllm_mcp_server_async()

        asyncio.run(main())
    """
    mcp = create_tidyllm_mcp_server(config_overrides)
    await mcp.run_async()


# Create the default server instance that fastmcp can find
mcp = create_tidyllm_mcp_server()
