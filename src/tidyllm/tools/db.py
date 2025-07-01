"""Database utilities for tidyllm tools."""

import json
import sqlite3
from typing import Any

from tidyllm.tools.context import DBContext


def init_database(ctx: DBContext) -> None:
    """Initialize database with required tables."""
    db_path = ctx.config.user_db
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Vocab table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vocab (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL UNIQUE,
            translation TEXT NOT NULL,
            examples TEXT,  -- JSON array
            tags TEXT,      -- JSON array
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create trigger to update updated_at
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS update_vocab_timestamp 
        AFTER UPDATE ON vocab
        BEGIN
            UPDATE vocab SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END
    ''')

    conn.commit()
    conn.close()


def json_encode(value: list[Any] | None) -> str | None:
    """Encode a list as JSON for storage."""
    if value is None:
        return None
    return json.dumps(value)


def json_decode(value: str | None) -> list[Any]:
    """Decode JSON string to list."""
    if not value:
        return []
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a sqlite3.Row to a dictionary."""
    return dict(zip(row.keys(), row, strict=False))
