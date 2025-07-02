"""FastAPI adapter for TidyLLM registry functions."""


from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from fastapi import FastAPI

from tidyllm.context import ToolContext, set_tool_context
from tidyllm.library import FunctionLibrary


def create_fastapi_app(
    library: FunctionLibrary | None = None,
    context: ToolContext | None = None,
    title: str = "TidyLLM Tools API",
    description: str = "API for TidyLLM registered tools",
    version: str = "1.0.0",
) -> FastAPI:
    """
    Create a FastAPI application that exposes TidyLLM tools as individual endpoints.

    Each tool gets its own POST endpoint that accepts the tool's specific argument types
    directly, with FastAPI automatically generating the OpenAPI schema.

    Args:
        context: Optional ToolContext for tools execution
        title: API title
        description: API description
        version: API version

    Returns:
        Configured FastAPI application with individual tool endpoints

    Example:
        from tidyllm.adapters.fastapi_adapter import create_fastapi_app
        from tidyllm.tools.context import ToolContext

        # Create with context
        context = ToolContext()
        app = create_fastapi_app(context)

        # Run with: uvicorn module:app --reload
        # Tools will be available at /tools/{tool_name}
    """
    app = FastAPI(title=title, description=description, version=version)

    # Use provided context or create default
    if context is None:
        from tidyllm.tools.config import Config
        context = ToolContext(config=Config())

    if library is not None:
        function_descriptions = library.function_descriptions
    else:
        from tidyllm.registry import REGISTRY

        function_descriptions = REGISTRY.functions

    @app.get("/", summary="API Information")
    async def root():
        """Get basic API information."""
        tool_names = [desc.name for desc in function_descriptions]
        return {
            "title": title,
            "description": description,
            "version": version,
            "tools": tool_names,
            "endpoints": {f"/tools/{name}" for name in tool_names}
        }

    @app.get("/health", summary="Health Check")
    async def health_check():
        """Simple health check endpoint."""
        return {
            "status": "healthy",
            "available_tools": len(function_descriptions)
        }

    # Create individual endpoints for each tool
    for tool_desc in function_descriptions:
        _create_tool_endpoint(app, tool_desc, context)

    return app


P = ParamSpec("P")
R = TypeVar("R")


def context_fn(func: Callable[P, R], _desc, _context) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with set_tool_context(_context):
            return func(*args, **kwargs)

    return wrapper


def _create_tool_endpoint(app: FastAPI, tool_desc, context: Any):
    """Create a FastAPI endpoint for a specific tool."""
    tool_name = tool_desc.name
    app.post(
        f"/tools/{tool_name}",
        summary=f"Execute {tool_name}",
        description=tool_desc.function.__doc__ or f"Execute the {tool_name} tool",
        tags=["tools"],
    )(context_fn(tool_desc.function, tool_desc, context))


# Convenience function for common use case
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
        from tidyllm.adapters.fastapi_adapter import create_portkit_api
        from tidyllm.tools.context import ToolContext

        context = ToolContext()
        app = create_portkit_api(context=context)
    """
    # Import all tools to ensure they're registered
    import tidyllm.tools.anki  # noqa: F401
    import tidyllm.tools.manage_db  # noqa: F401
    import tidyllm.tools.notes  # noqa: F401
    import tidyllm.tools.transcribe  # noqa: F401
    import tidyllm.tools.vocab_table  # noqa: F401

    return create_fastapi_app(context=context, title=title, description=description)
