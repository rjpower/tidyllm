"""Context management using contextvars for automatic context propagation."""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from tidyllm.tools.config import Config

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


class ToolContext(BaseModel):
    """Shared context for all tools."""

    config: Config = Field(default_factory=Config)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)
        self._db = None

    @property
    def db(self):
        """Get database instance, creating it if needed."""
        if self._db is None:
            from tidyllm.database import Database

            self._db = Database(str(self.config.user_db))
        return self._db

    def get_db_connection(self):
        """Get database instance (deprecated, use .db property)."""
        return self.db

    def ensure_notes_dir(self) -> Path:
        """Ensure notes directory exists and return it."""
        return self.config.ensure_notes_dir()

    def find_anki_db(self) -> Path | None:
        """Find Anki database path."""
        return self.config.find_anki_db()


# Global context manager for tool context
_tool_context: ContextManager[ToolContext] = ContextManager('tool_context')

def get_tool_context() -> ToolContext:
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
def set_tool_context(context: ToolContext):
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
