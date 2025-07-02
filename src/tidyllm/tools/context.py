from pathlib import Path

from pydantic import BaseModel, Field

from tidyllm.tools.config import Config


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