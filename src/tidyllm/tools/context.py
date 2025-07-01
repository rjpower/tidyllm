"""Shared context for tidyllm tools."""

import sqlite3
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from tidyllm.tools.config import Config


class DBContext(Protocol):
    """Protocol for database context."""
    config: Config


class ToolContext(BaseModel):
    """Shared context for all tools."""
    
    config: Config = Field(default_factory=Config)
    
    model_config = {"arbitrary_types_allowed": True}
        
    def get_db_connection(self) -> sqlite3.Connection:
        """Get a new database connection."""
        return self.config.get_db_connection()
        
    def ensure_notes_dir(self) -> Path:
        """Ensure notes directory exists and return it."""
        return self.config.ensure_notes_dir()
        
    def find_anki_db(self) -> Path | None:
        """Find Anki database path."""
        return self.config.find_anki_db()