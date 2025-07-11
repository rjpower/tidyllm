import sqlite3
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from tidyllm.cache import CacheDbProtocol, SqlAdapter
from tidyllm.database import Database
from tidyllm.tools.config import Config

R = TypeVar("R")

class ToolContext(BaseModel):
    """Shared context for all tools."""

    model_config = {"arbitrary_types_allowed": True}
    config: Config = Field(default_factory=Config)
    db: Database = Field(default=None)  # type: ignore
    cache_db: CacheDbProtocol = Field(default=None)  # type: ignore
    refs: dict[str, Any] = Field(default_factory=dict, exclude=True)

    def model_post_init(self, _ctx: Any):
        self.db = Database(str(self.config.user_db))
        if not self.cache_db:
            cache_conn = sqlite3.connect(
                str(self.config.cache_db), check_same_thread=False
            )
            self.cache_db = SqlAdapter(cache_conn)

    def set_ref(self, key: str, value: Any):
        self.refs[key] = value

    def get_ref(self, key: str, load_fn: Callable[[], R] | None = None) -> R:
        """Get or create a cached reference using the provided loader function.

        Args:
            key: Unique key for the cached item
            load_fn: Function to call if item not in cache

        Returns:
            The cached or newly loaded item
        """
        if key not in self.refs:
            if not load_fn:
                raise KeyError(f"No ref for {key} found.")
            self.refs[key] = load_fn()
        return self.refs[key]
