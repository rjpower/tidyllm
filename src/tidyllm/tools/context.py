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
        # Ensure parent directory exists
        self.config.user_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.config.user_db))
        conn.row_factory = sqlite3.Row
        return conn
        
    def ensure_notes_dir(self) -> Path:
        """Ensure notes directory exists and return it."""
        self.config.notes_dir.mkdir(parents=True, exist_ok=True)
        return self.config.notes_dir
        
    def find_anki_db(self) -> Path | None:
        """Find Anki database path."""
        if self.config.anki_path:
            return self.config.anki_path
            
        # Autodiscover on macOS
        anki_base = Path.home() / "Library" / "Application Support" / "Anki2"
        if anki_base.exists():
            # Look for collection.anki2 in profile directories
            for profile_dir in anki_base.iterdir():
                if profile_dir.is_dir():
                    collection = profile_dir / "collection.anki2"
                    if collection.exists():
                        return collection
        
        return None