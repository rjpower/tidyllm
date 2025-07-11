"""Global registry for tools."""

import logging
from collections import OrderedDict
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from tidyllm.function_schema import FunctionDescription, JSONSchema

logger = logging.getLogger(__name__)


class Registry:
    """Global registry for tools with execution capabilities."""

    def __init__(self):
        self._tools: dict[str, FunctionDescription] = OrderedDict()

    def register(
        self,
        func: Callable,
        name: str | None = None,
        doc_override: str | None = None,
        description: str = "",
        tags: list[str] | None = None,
    ) -> None:
        """Register a tool function and generate its schema automatically.

        Args:
            func: Function to register
            doc_override: Optional documentation override
            description: Tool description
            tags: List of tags for categorization
        """
        name = name or func.__name__

        if name in self._tools:
            logger.debug(f"Tool {name} already registered, skipping")
            return

        # Create FunctionDescription once at registration time
        func_desc = FunctionDescription(
            func,
            name=name,
            doc_override=doc_override,
            description=description,
            tags=tags,
        )

        self._tools[name] = func_desc
        logger.debug(f"Registered tool: {name}")

    @property
    def functions(self) -> list[FunctionDescription]:
        """Get all registered tool descriptions."""
        return list(self._tools.values())

    def get_description(self, name: str) -> FunctionDescription | None:
        """Get a tool description by name."""
        return self._tools.get(name)

    def get_function(self, name: str) -> Callable[..., Any]:
        """Get the raw function by name."""
        func_desc = self._tools.get(name)
        if func_desc is None:
            raise KeyError(f"Tool '{name}' not found")
        return func_desc.function

    def get_schemas(self) -> list[JSONSchema]:
        """Get OpenAI-format schemas for all tools."""
        return [func_desc.function_schema for func_desc in self._tools.values()]



# Global registry instance
REGISTRY = Registry()

P = ParamSpec("P")
T = TypeVar("T")


def register(
    *,
    doc: str | None = None,
    name: str | None = None,
    description: str = "",
    tags: list[str] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Register a function as a tool.

    Usage:
        @register()
        def my_tool(...): ...

        @register(doc="custom doc")
        def my_tool(...): ...

        @register(name="custom_name", description="Tool description", tags=["audio", "stream"])
        def my_tool(...): ...

    Args:
        doc: Override docstring (supports read_prompt())
        name: Override tool name
        description: Tool description
        tags: List of tags for categorization
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return func(*args, **kwargs)

        if name:
            wrapper.__name__ = name
        REGISTRY.register(wrapper, doc, description, tags)
        return wrapper

    return decorator
