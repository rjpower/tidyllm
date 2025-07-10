"""FastMCP adapter for TidyLLM registry functions."""

import base64
import json
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, Iterable, ParamSpec, TypeVar

from fastmcp.server import FastMCP
from fastmcp.tools.tool import ToolResult
from fastmcp.utilities.types import Audio, Image
from mcp import Resource

from tidyllm.context import get_tool_context, set_tool_context
from tidyllm.registry import REGISTRY
from tidyllm.tools.context import ToolContext
from tidyllm.types.part import Part, is_audio_part, is_image_part, is_text_content_part

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


class ResourceStore:
    """Manages references to Part objects for MCP adapter."""

    def __init__(self):
        self._refs: dict[str, Part] = {}

    def store_part(self, part: Part) -> str:
        """Store a Part and return its reference ID."""
        ref_id = str(uuid.uuid4())
        self._refs[ref_id] = part
        return ref_id

    def get_part(self, ref_id: str) -> Part | None:
        """Retrieve a Part by reference ID."""
        return self._refs.get(ref_id)


def _handle_part_result(part: Part, ref_store: ResourceStore) -> Image | Audio | dict:
    """Handle Part objects by returning appropriate FastMCP format."""

    if is_image_part(part):
        return Image(data=part.data, format=part.mime_type.split("/")[1])
    elif is_audio_part(part):
        return Audio(data=part.data, format=part.mime_type.split("/")[1])
    else:
        logger.info("Text Part???")
        ref_id = ref_store.store_part(part)
        data_preview = part.data[:512]
        if is_text_content_part(part):
            data_preview = data_preview.decode()
        else:
            data_preview = base64.b64encode(data_preview)

        return {
            "_note": """This is a reference to a resource on the server. Pass { 'ref_id': ... } as an argument to use it in other tools.""",
            "ref_id": ref_id,
            "mime_type": part.mime_type,
            "data_preview": data_preview,
        }


def _process_result(result: Any, ref_store: ResourceStore) -> Any:
    """Process tool result, handling Part objects appropriately."""
    if isinstance(result, Part):
        return _handle_part_result(result, ref_store)
    elif isinstance(result, dict):
        return {k: _process_result(v, ref_store) for (k, v) in result.items()}
    elif isinstance(result, Iterable):
        return [_process_result(v, ref_store) for v in result]
    else:
        return result


def _rehydrate_args(
    args: tuple[Any, ...], kwargs: dict[str, Any], ref_store: ResourceStore
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Rehydrate any Part references in function arguments."""

    def rehydrate_value(value: Any) -> Any:
        if isinstance(value, dict) and "ref_id" in value:
            ref_id = value["ref_id"]
            part = ref_store.get_part(ref_id)
            if part is not None:
                return part
        elif isinstance(value, (list | tuple)):
            return type(value)(rehydrate_value(item) for item in value)
        elif isinstance(value, dict):
            return {k: rehydrate_value(v) for k, v in value.items()}
        return value

    new_args = tuple(rehydrate_value(arg) for arg in args)
    new_kwargs = {k: rehydrate_value(v) for k, v in kwargs.items()}
    return new_args, new_kwargs


def context_fn(func: Callable[P, R], _context) -> Callable[P, Image | Audio | R]:
    resource_store = _context.get_ref("ref_store", ResourceStore)

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Image | Audio | R:
        try:
            with set_tool_context(_context):
                new_args, new_kwargs = _rehydrate_args(args, kwargs, resource_store)
                logger.info("Calling")
                result = func(*new_args, **new_kwargs)  # type: ignore
                logger.info("Result: %s", result)
                processed = _process_result(result, resource_store)
                return processed
        except Exception as e:
            logger.error(e, stack_info=True)
            raise

    wrapper.__annotations__["return"] = Image | Audio | R
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
    @asynccontextmanager
    async def app_lifespan(server: FastMCP) -> AsyncIterator[Any]:
        try:
            yield context
        finally:
            pass

    server = FastMCP(lifespan=app_lifespan)

    def image_tool() -> Image:
        return Image(data=b"", format="image/png")

    server.tool()(image_tool)

    for tool_desc in REGISTRY.functions:
        fn = context_fn(tool_desc.function, context)

        try:
            server.tool()(fn)
        except Exception:
            logger.exception(f"Failed to register {fn.__name__}")

    return server
