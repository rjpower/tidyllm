"""FastMCP adapter for TidyLLM registry functions."""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import pydantic

from tidyllm.context import set_tool_context
from tidyllm.registry import REGISTRY

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)

def context_fn(func: Callable[P, R], _desc, _context) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with set_tool_context(_context):
            return func(*args, **kwargs)

    return wrapper


def create_fastmcp_server(
    context: Any,
    name: str = "TidyLLM Tools",
):
    """Create a FastMCP server that exposes TidyLLM tools.

    Args:
        context: ToolContext instance for tool execution (required)
        name: Server name

    Returns:
        FastMCP server with all registered tools

    Example:
        from tidyllm.adapters.fastmcp_adapter import create_fastmcp_server
        from tidyllm.tools.context import ToolContext

        # Create with context
        context = ToolContext()
        server = create_fastmcp_server(context=context)
    """
    from mcp.server.fastmcp import FastMCP

    @asynccontextmanager
    async def app_lifespan(server: FastMCP) -> AsyncIterator[Any]:
        try:
            yield context
        finally:
            pass

    server = FastMCP(lifespan=app_lifespan)
    for tool_desc in REGISTRY.functions:
        fn = context_fn(tool_desc.function, tool_desc, context)
        try:
            server.tool()(fn)
        except pydantic.errors.PydanticSchemaGenerationError:
            logger.warning(f"Failed to register {fn.__name__}", stacklevel=1)

    return server
