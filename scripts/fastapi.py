"""FastAPI server script with auto-discovery of TidyLLM tools."""

import argparse
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from tidyllm.adapters.fastapi_adapter import create_fastapi_app
from tidyllm.discover import discover_tools_in_package
from tidyllm.tools.config import Config
from tidyllm.tools.context import ToolContext


def create_fastapi_app_with_discovery(
    context: ToolContext | None = None,
    title: str = "TidyLLM Tools API",
    description: str = "API for TidyLLM registered tools",
    version: str = "1.0.0",
    tools_package: str = "tidyllm.tools",
) -> FastAPI:
    """
    Create a FastAPI application with auto-discovered TidyLLM tools.

    Args:
        context: Optional ToolContext for tools execution
        title: API title
        description: API description
        version: API version
        tools_package: Package to discover tools from

    Returns:
        Configured FastAPI application with auto-discovered tools

    Example:
        from scripts.fastapi import create_fastapi_app_with_discovery
        from tidyllm.tools.context import ToolContext

        # Create with context
        context = ToolContext()
        app = create_fastapi_app_with_discovery(context)

        # Run with: uvicorn scripts.fastapi:app --reload
    """
    # Auto-discover tools from the specified package
    discover_tools_in_package(tools_package)
    
    # Use provided context or create default
    if context is None:
        config = Config()
        context = ToolContext(config=config)

    return create_fastapi_app(
        context=context,
        title=title,
        description=description,
        version=version
    )


def create_portkit_api(
    context: ToolContext | None = None,
    title: str = "PortKit Tools API", 
    description: str = "API for PortKit TinyAgent tools",
) -> FastAPI:
    """
    Create a FastAPI app specifically for PortKit tools.

    Args:
        context: ToolContext for tools execution
        title: API title
        description: API description

    Returns:
        Configured FastAPI application with PortKit tools

    Example:
        from scripts.fastapi import create_portkit_api
        from tidyllm.tools.context import ToolContext

        context = ToolContext()
        app = create_portkit_api(context=context)
    """
    return create_fastapi_app_with_discovery(
        context=context,
        title=title,
        description=description,
        tools_package="tidyllm.tools"
    )


# Default app instance for easy deployment
app = create_fastapi_app_with_discovery()


def main():
    """CLI entry point for running the FastAPI server."""
    parser = argparse.ArgumentParser(description="Run TidyLLM FastAPI server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--title", default="TidyLLM Tools API", help="API title")
    parser.add_argument("--notes-dir", help="Override notes directory")
    parser.add_argument("--user-db", help="Override user database path")
    
    args = parser.parse_args()
    
    # Create config with overrides
    config_kwargs: dict[str, Any] = {}
    if args.notes_dir:
        config_kwargs["notes_dir"] = Path(args.notes_dir)
    if args.user_db:
        config_kwargs["user_db"] = Path(args.user_db)
        
    config = Config(**config_kwargs)
    context = ToolContext(config=config)
    
    # Create app with custom context
    app = create_fastapi_app_with_discovery(
        context=context,
        title=args.title
    )
    
    # Run the server
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()