"""Configuration management for tidyllm tools."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Configuration for tidyllm tools."""
    
    notes_dir: Path = Path.home() / "Documents" / "Notes"
    user_db: Path = Path.home() / ".config" / "tidyllm" / "user.db"
    anki_path: Path | None = None  # Will autodiscover if unset
    fast_model: str = "gemini/gemini-2.0-flash-exp"
    slow_model: str = "gemini/gemini-2.0-flash-thinking-exp-1219"
    
    model_config = {
        "env_prefix": "TIDYLLM_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }