"""Database class for SQLite operations with initialization utilities."""

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .database_models import ColumnSchema, Cursor, Row, Schema, TableSchema
from .tools.config import Config


class Database:
    """Database class managing SQLite connection and operations."""

    def __init__(
        self,
        path: str | Path,
        **connect_kwargs: Any
    ) -> None:
        """
        Initialize a Database instance.

        Args:
            path: Path to the SQLite database file. Use ':memory:' for in-memory.
            connect_kwargs: Optional keyword args forwarded to sqlite3.connect().
        """
        self.path = str(path)
        self.connect_kwargs = connect_kwargs
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Open the SQLite connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.path, **self.connect_kwargs)
            self._conn.row_factory = sqlite3.Row
            self.init_schema()

    def close(self) -> None:
        """Close the SQLite connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def query(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
    ) -> Cursor:
        """
        Execute a SELECT statement and return a Cursor.

        Returns:
            Cursor: An iterable of Rows.
        """
        self.connect()
        cur = self._conn.execute(sql, params or [])
        columns = [description[0] for description in cur.description or []]
        rows = [Row(**dict(zip(columns, r, strict=False))) for r in cur.fetchall()]
        return Cursor(columns=columns, rows=rows)

    def mutate(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
    ) -> int:
        """
        Execute an INSERT/UPDATE/DELETE statement.

        Returns:
            int: Number of affected rows.
        """
        self.connect()
        cur = self._conn.execute(sql, params or [])
        self._conn.commit()
        return cur.rowcount

    def schema(self) -> Schema:
        """
        Inspect the database schema.

        Returns:
            Schema: A Pydantic model of tables and columns.
        """
        self.connect()
        table_names = [
            r[0] for r in self._conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%';
                """
            )
        ]
        tables: list[TableSchema] = []
        for name in table_names:
            cols = []
            for col in self._conn.execute(f"PRAGMA table_info('{name}')"):
                cols.append(
                    ColumnSchema(
                        name=col['name'],
                        type=col['type'],
                        not_null=bool(col['notnull']),
                        default=col['dflt_value'],
                        primary_key=bool(col['pk']),
                    )
                )
            tables.append(TableSchema(name=name, columns=cols))
        return Schema(tables=tables)

    def __enter__(self) -> 'Database':
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback  # Unused parameters
        self.close()

    def init_schema(self) -> None:
        """Initialize database with required tables."""
        # Vocab table
        self.mutate('''
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
        self.mutate('''
            CREATE TRIGGER IF NOT EXISTS update_vocab_timestamp 
            AFTER UPDATE ON vocab
            BEGIN
                UPDATE vocab SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
        ''')


def init_database(config: Config) -> None:
    """Initialize database with required tables using legacy interface."""
    db_path = config.user_db
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    db = Database(str(db_path))
    db.init_schema()
    db.close()


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