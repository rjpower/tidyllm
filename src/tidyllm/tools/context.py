from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from tidyllm.database import Database
from tidyllm.tools.config import Config


class ToolContext(BaseModel):
    """Shared context for all tools."""

    model_config = {"arbitrary_types_allowed": True}
    config: Config = Field(default_factory=Config)
    db: Database = Field(default=None)  # type: ignore
    refs: dict[str, Any] = Field(default_factory=dict, exclude=True)

    def model_post_init(self, _ctx: Any):
        self.db = Database(str(self.config.user_db))

    def get_ref(self, key: str, load_fn: Callable[[], Any]) -> Any:
        """Get or create a cached reference using the provided loader function.

        Args:
            key: Unique key for the cached item
            load_fn: Function to call if item not in cache

        Returns:
            The cached or newly loaded item
        """
        if key not in self.refs:
            self.refs[key] = load_fn()
        return self.refs[key]
