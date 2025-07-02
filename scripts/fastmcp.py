"""FastMCP server script with auto-discovery of TidyLLM tools."""

import argparse
from pathlib import Path
from typing import Any

from tidyllm.adapters.fastmcp_adapter import create_fastmcp_server
from tidyllm.discover import discover_tools_in_package
from tidyllm.tools.config import Config
from tidyllm.tools.context import ToolContext


def create_fastmcp_server_with_discovery(
    context: ToolContext | None = None,
    name: str = "TidyLLM Tools",
    tools_package: str = "tidyllm.tools",
):
    """
    Create a FastMCP server with auto-discovered TidyLLM tools.

    Args:
        context: Optional ToolContext for tools execution
        name: Server name
        tools_package: Package to discover tools from

    Returns:
        FastMCP server with all discovered tools

    Example:
        from scripts.fastmcp import create_fastmcp_server_with_discovery
        from tidyllm.tools.context import ToolContext

        # Create with custom context
        context = ToolContext()
        server = create_fastmcp_server_with_discovery(context=context)
    """
    # Auto-discover tools from the specified package
    discover_tools_in_package(tools_package)
    
    # Use provided context or create default
    if context is None:
        config = Config()
        context = ToolContext(config=config)

    return create_fastmcp_server(context=context, name=name)


def create_tidyllm_mcp_server(config_overrides: dict[str, Any] | None = None):
    """Create a FastMCP server specifically for TidyLLM tools with auto-discovery.

    Args:
        config_overrides: Dictionary of configuration overrides for ToolContext.config

    Returns:
        Configured FastMCP server with all discovered TidyLLM tools

    Example:
        from scripts.fastmcp import create_tidyllm_mcp_server

        # Create server with auto-discovered tools
        mcp = create_tidyllm_mcp_server()

        # Create with custom configuration
        mcp = create_tidyllm_mcp_server(config_overrides={"notes_dir": "/custom/notes"})
    """
    # Create tool context
    config = Config()

    if config_overrides:
        # Allow configuration overrides
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

    print("Starting server, config=", config)
    context = ToolContext(config=config)

    return create_fastmcp_server_with_discovery(
        context=context,
        name="TidyLLM Tools Server"
    )


def run_tidyllm_mcp_server(config_overrides: dict[str, Any] | None = None):
    """Run the TidyLLM MCP server.

    Args:
        config_overrides: Optional configuration overrides for ToolContext.config

    Example:
        from scripts.fastmcp import run_tidyllm_mcp_server

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
        from scripts.fastmcp import run_tidyllm_mcp_server_async

        async def main():
            await run_tidyllm_mcp_server_async()

        asyncio.run(main())
    """
    mcp = create_tidyllm_mcp_server(config_overrides)
    await mcp.run_stdio_async()


# Create the default server instance that fastmcp can find
mcp = create_tidyllm_mcp_server()


def main():
    """CLI entry point for running the FastMCP server."""
    parser = argparse.ArgumentParser(description="Run TidyLLM FastMCP server")
    parser.add_argument("--name", default="TidyLLM Tools Server", help="Server name")
    parser.add_argument("--notes-dir", help="Override notes directory")
    parser.add_argument("--user-db", help="Override user database path")
    parser.add_argument("--async", action="store_true", help="Run in async mode")
    
    args = parser.parse_args()
    
    # Create config with overrides
    config_overrides: dict[str, Any] = {}
    if args.notes_dir:
        config_overrides["notes_dir"] = Path(args.notes_dir)
    if args.user_db:
        config_overrides["user_db"] = Path(args.user_db)
    
    if getattr(args, 'async'):
        import asyncio
        asyncio.run(run_tidyllm_mcp_server_async(config_overrides))
    else:
        run_tidyllm_mcp_server(config_overrides)


if __name__ == "__main__":
    main()