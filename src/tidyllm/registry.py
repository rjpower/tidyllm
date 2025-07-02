"""Global registry for tools."""

import logging
from collections import OrderedDict
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, Protocol, TypeVar, cast

from fastapi.middleware import Middleware

from tidyllm.context import set_tool_context
from tidyllm.schema import (
    FunctionDescription,
)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T", covariant=True)


class CallableWithSchema(Protocol[P, T]):
    """Generic callable with schema that preserves function signature."""

    __tool_schema__: dict
    __name__: str

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T: ...


class InjectContextMiddleware(Middleware):
    def __init__(self, context: Any):
        self.context = context

    async def on_call_tool(
        self,
        fastmcp_context: Any,
        call_next: Callable,
    ):
        with set_tool_context(self.context):
            return await call_next(fastmcp_context)


class Registry:
    """Global registry for tools."""

    def __init__(self):
        self._tools: dict[str, FunctionDescription] = OrderedDict()

    def register(
        self,
        func: Callable,
        doc_override: str | None = None,
    ) -> None:
        """Register a tool function and generate its schema automatically.

        Args:
            func: Function to register
            doc_override: Optional documentation override
        """
        name = func.__name__

        if name in self._tools:
            # warnings.warn(
            #     f"Tool '{name}' already registered, previous definition: {self._tools[name].function.__code__.co_filename}. "
            #     f"Skipping duplicate registration from: {func.__code__.co_filename}",
            #     UserWarning,
            #     stacklevel=2,
            # )
            return

        # Create FunctionDescription once at registration time
        func_desc = FunctionDescription(func, doc_override)

        self._tools[name] = func_desc
        logger.info(f"Registered tool: {name}")

    @property
    def functions(self) -> list[FunctionDescription]:
        """Get all registered tool descriptions."""
        return list(self._tools.values())

    def get(self, name: str) -> FunctionDescription | None:
        """Get a tool description by name."""
        return self._tools.get(name)

    def get_function(self, name: str) -> CallableWithSchema[..., Any]:
        """Get the raw function by name with preserved typing."""
        func_desc = self._tools.get(name)
        if func_desc is None:
            raise KeyError(f"Tool '{name}' not found")
        return cast(CallableWithSchema[..., Any], func_desc.function)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

# Global registry instance
REGISTRY = Registry()


def register(
    *,
    doc: str | None = None,
    name: str | None = None,
) -> Callable[[Callable[P, T]], CallableWithSchema[P, T]]:
    """
    Register a function as a tool.

    Usage:
        @register()
        def my_tool(...): ...

        @register(doc="custom doc")
        def my_tool(...): ...

        @register(name="custom_name")
        def my_tool(...): ...

    Args:
        doc: Override docstring (supports read_prompt())
        name: Override tool name
    """
    def decorator(func: Callable[P, T]) -> CallableWithSchema[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return func(*args, **kwargs)

        # Override function name if provided
        if name:
            wrapper.__name__ = name

        # Register the function - registry will generate schema automatically
        REGISTRY.register(wrapper, doc)

        # Return the wrapper cast as CallableWithSchema to indicate it has __tool_schema__
        return cast(CallableWithSchema[P, T], wrapper)

    return decorator
