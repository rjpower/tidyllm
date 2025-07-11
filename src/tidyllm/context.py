"""Dependency injection helpers for passing additional context information to tools.

Typical usage is to use `with set_tool_context(MyToolContextClass())` at the entry point
to your program, e.g. at CLI startup or on each request to an API/MCP adapter.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Generic, TypeVar

T = TypeVar('T')

class ContextManager(Generic[T]):
    """Generic context manager using contextvars."""

    def __init__(self, name: str, default: T | None = None):
        self._var: ContextVar[T | None] = ContextVar(name, default=default)

    def get(self) -> T:
        """Get current context value."""
        value = self._var.get()
        if value is None:
            raise LookupError("No context value set")
        return value

    def set(self, value: T) -> Any:
        """Set context value, returns token for reset."""
        return self._var.set(value)

    def reset(self, token: Any) -> None:
        """Reset context to previous value using token."""
        self._var.reset(token)

    def __call__(self) -> T:
        """Shorthand for get()."""
        return self.get()


_tool_context: ContextManager[Any] = ContextManager("tool_context")


def get_tool_context() -> Any:
    """Get current tool context from contextvar.
    
    Returns:
        Current ToolContext instance
        
    Raises:
        RuntimeError: If no context is set (tools called outside proper adapter)
    """
    try:
        return _tool_context.get()
    except LookupError as e:
        raise RuntimeError(
            "No tool context available. Ensure tools are called within "
            "a properly configured adapter (FastAPI, FastMCP, CLI, etc.)"
        ) from e


@contextmanager
def set_tool_context(context: Any):
    """Context manager to set tool context for a block of code.
    
    Args:
        context: ToolContext to set
        
    Usage:
        with set_tool_context(my_context):
            result = some_tool()
    """
    token = _tool_context.set(context)
    try:
        yield
    finally:
        _tool_context.reset(token)
