"""Global registry for tools."""

import logging
from collections import OrderedDict
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, Protocol, TypeVar, cast, overload

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


@overload
def register(
    func_or_doc: Callable[P, T], *, doc: str | None = None, name: str | None = None
) -> CallableWithSchema[P, T]: ...


@overload
def register(
    func_or_doc: str | None = None, *, doc: str | None = None, name: str | None = None
) -> Callable[[Callable[P, T]], CallableWithSchema[P, T]]: ...


def register(
    func_or_doc: Callable[P, T] | str | None = None,
    *,
    doc: str | None = None,
    name: str | None = None,
) -> CallableWithSchema[P, T] | Callable[[Callable[P, T]], CallableWithSchema[P, T]]:
    """
    Register a function as a tool.

    Can be used with or without parentheses:
        @register
        def my_tool(...): ...

        @register()
        def my_tool(...): ...

        @register(doc="custom doc")
        def my_tool(...): ...

    Args:
        func_or_doc: Function (when used without parentheses) or doc override
        doc: Override docstring (supports read_prompt())
        name: Override tool name
    """

    @wraps(func_or_doc)
    def _register_func(
        func: Callable[P, T], doc_override: str | None = None
    ) -> CallableWithSchema[P, T]:
        # Override function name if provided
        if name:
            func.__name__ = name

        # No context parameter validation needed with new contextvar approach

        # Register the function - registry will generate schema automatically
        REGISTRY.register(func, doc_override)

        # Return the function cast as CallableWithSchema to indicate it has __tool_schema__
        return cast(CallableWithSchema[P, T], func)

    # If first argument is a callable, this is direct usage (@register)
    if callable(func_or_doc):
        return _register_func(func_or_doc, doc)

    @wraps(func_or_doc)
    def decorator(func: Callable[P, T]) -> CallableWithSchema[P, T]:
        # Use func_or_doc as doc if it's a string, otherwise use doc parameter
        doc_override = func_or_doc if isinstance(func_or_doc, str) else doc
        return _register_func(func, doc_override)

    return decorator
