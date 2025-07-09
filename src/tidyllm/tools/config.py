"""Configuration management for tidyllm tools."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Configuration for tidyllm tools."""

    notes_dir: Path = Path.home() / "Documents" / "Notes"
    config_dir: Path = Path.home() / ".config" / "tidyllm"
    anki_path: Path | None = Field(default=None)
    fast_model: str = "gemini/gemini-2.5-flash"
    slow_model: str = "gemini/gemini-2.5-pro"

    model_config = {
        "env_prefix": "TIDYLLM_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "forbid",
    }

    @property
    def user_db(self):
        self.config_dir.mkdir(exist_ok=True)
        return self.config_dir / "user.db"

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
