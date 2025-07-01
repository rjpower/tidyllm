"""Configuration management for tidyllm tools."""

import sqlite3
from pathlib import Path

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Configuration for tidyllm tools."""

    notes_dir: Path = Path.home() / "Documents" / "Notes"
    user_db: Path = Path.home() / ".config" / "tidyllm" / "user.db"
    anki_path: Path | None = None  # Will autodiscover if unset
    fast_model: str = "gemini/gemini-2.5-flash"
    slow_model: str = "gemini/gemini-2.0-pro"

    model_config = {
        "env_prefix": "TIDYLLM_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    def model_post_init(self, __context) -> None:
        """Post-initialization setup."""
        # Auto-discover Anki path if not set
        if self.anki_path is None:
            discovered_path = self._autodiscover_anki_db()
            if discovered_path:
                self.anki_path = discovered_path

    def get_db_connection(self) -> sqlite3.Connection:
        """Get a new database connection."""
        # Ensure parent directory exists
        self.user_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.user_db))
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_notes_dir(self) -> Path:
        """Ensure notes directory exists and return it."""
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        return self.notes_dir

    def find_anki_db(self) -> Path | None:
        """Find Anki database path."""
        if self.anki_path:
            return self.anki_path
        return self._autodiscover_anki_db()

    def _autodiscover_anki_db(self) -> Path | None:
        """Autodiscover Anki database on macOS."""
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
