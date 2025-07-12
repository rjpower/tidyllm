"""FastMCP adapter for TidyLLM registry functions."""

import base64
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from fastmcp.server import FastMCP
from fastmcp.utilities.types import Audio, Image

from tidyllm.context import set_tool_context
from tidyllm.registry import REGISTRY
from tidyllm.tools.context import ToolContext
from tidyllm.types.part import Part, is_audio_part, is_image_part, is_text_part

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


def _process_parts_in_result(result: Any) -> Any:
    """Recursively process tool result to pre-serialize all Parts while context is available."""
    # If it's a Part, serialize it now while we have context
    if isinstance(result, Part):
        return result.model_dump()
    
    # If it's a Pydantic model, serialize it (this will handle nested Parts)
    elif hasattr(result, 'model_dump'):
        return result.model_dump()
    
    # If it's a dict, process values recursively
    elif isinstance(result, dict):
        return {k: _process_parts_in_result(v) for k, v in result.items()}
    
    # If it's a list/tuple, process items recursively
    elif isinstance(result, list | tuple):
        processed = [_process_parts_in_result(item) for item in result]
        return type(result)(processed) if isinstance(result, tuple) else processed
    
    # For other types, return as-is
    else:
        return result


def context_fn(
    func: Callable[P, R], ctx: ToolContext
) -> Callable[P, R]:

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            with set_tool_context(ctx):
                result = func(*args, **kwargs)  # type: ignore
                # Process the result to pre-serialize any Parts while context is available
                processed_result = _process_parts_in_result(result)
                return processed_result  # type: ignore
        except Exception as e:
            logger.error(e, stack_info=True)
            raise

    return wrapper


def create_fastmcp_server(
    context: ToolContext,
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
    @asynccontextmanager
    async def app_lifespan(server: FastMCP) -> AsyncIterator[Any]:
        try:
            yield context
        finally:
            pass

    server = FastMCP(lifespan=app_lifespan)

    @server.tool
    def fetch_part_content(
        url: str, start: int = 0, limit: int = -1
    ) -> str | bytes | Image | Audio:
        """Fetch the data from a remote Part.

        If the result is an image or audio, it is returned as the appropriate object type.

        Args:
          url: The ref:// URL of the Part to fetch
          start: The start of the resulting bytes to return
          limit: The end of the resulting bytes to return
        """
        if not url.startswith("ref://"):
            raise ValueError("Can only fetch content for ref:// parts.")

        part = context.get_ref(url)
        if is_image_part(part):
            return Image(data=part.to_bytes("PNG"), format="png")
        elif is_audio_part(part):
            return Audio(data=part.to_wav_bytes(), format="wav")
        elif is_text_part(part):
            # For text parts (BasicPart), data is base64-encoded
            raw_data = base64.b64decode(part.data)
            return raw_data[start:limit].decode()

        # Fallback for other Part types
        if hasattr(part, 'data'):
            return part.data[start:limit]
        else:
            raise ValueError(f"Cannot fetch content for Part type: {type(part)}")

    for tool_desc in REGISTRY.functions:
        fn = context_fn(tool_desc.function, context)

        try:
            server.tool()(fn)
        except Exception:
            logger.exception(f"Failed to register {fn.__name__}")

    return server
