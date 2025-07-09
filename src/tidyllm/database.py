"""Database class for SQLite operations with initialization utilities."""

import datetime
import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def adapt_date_iso(val):
    """Adapt datetime.date to ISO 8601 date."""
    return val.isoformat()


def adapt_datetime_iso(val):
    """Adapt datetime.datetime to timezone-naive ISO 8601 date."""
    return val.isoformat()


def adapt_datetime_epoch(val):
    """Adapt datetime.datetime to Unix timestamp."""
    return int(val.timestamp())


sqlite3.register_adapter(datetime.date, adapt_date_iso)
sqlite3.register_adapter(datetime.datetime, adapt_datetime_iso)
sqlite3.register_adapter(datetime.datetime, adapt_datetime_epoch)


def convert_date(val):
    """Convert ISO 8601 date to datetime.date object."""
    return datetime.date.fromisoformat(val.decode())


def convert_datetime(val):
    """Convert ISO 8601 datetime to datetime.datetime object."""
    return datetime.datetime.fromisoformat(val.decode())


def convert_timestamp(val):
    """Convert Unix epoch timestamp to datetime.datetime object."""
    return datetime.datetime.fromtimestamp(int(val))


sqlite3.register_converter("date", convert_date)
sqlite3.register_converter("datetime", convert_datetime)
sqlite3.register_converter("timestamp", convert_timestamp)


class ColumnSchema(BaseModel):
    """Represents a single column in a table."""

    name: str = Field(description="Column name")
    type: str = Field(description="Data type (e.g., TEXT, INTEGER)")
    not_null: bool = Field(description="Whether NOT NULL constraint exists")
    default: str | None = Field(default=None, description="Default value or None")
    primary_key: bool = Field(description="Whether column is part of the primary key")


class TableSchema(BaseModel):
    """Represents a database table."""

    name: str = Field(description="Table name")
    columns: list[ColumnSchema] = Field(description="List of ColumnSchema objects")


class Schema(BaseModel):
    """Represents the full database schema."""

    tables: list[TableSchema] = Field(description="List of TableSchema objects")

    def get_table(self, name: str) -> TableSchema | None:
        """Get table by name."""
        return next((t for t in self.tables if t.name == name), None)


class Row(BaseModel):
    """Wraps a single result row."""

    model_config = {"extra": "allow"}

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._data = data

    def __getitem__(self, key: str) -> Any:
        """Get item by key."""
        return getattr(self, key, self._data.get(key))

    def get(self, key: str, default: Any = None) -> Any:
        """Get item by key with default."""
        return getattr(self, key, self._data.get(key, default))

    def keys(self):
        """Get keys."""
        return self._data.keys()

    def values(self):
        """Get values."""
        return self._data.values()

    def items(self):
        """Get items."""
        return self._data.items()


class Cursor:
    """Iterator over Row objects with metadata."""

    def __init__(self, columns: list[str], rows: list[Row]):
        self.columns = columns
        self._rows = rows

    @property
    def rows(self) -> list[Row]:
        """Get all rows."""
        return self._rows

    def __iter__(self) -> Iterator[Row]:
        """Iterate over rows."""
        return iter(self._rows)

    def all(self) -> list[Row]:
        """Get all rows."""
        return self._rows

    def first(self) -> Row | None:
        """Get first row or None."""
        return self._rows[0] if self._rows else None

    def __len__(self) -> int:
        """Get number of rows."""
        return len(self._rows)

    def __bool__(self) -> bool:
        """Check if cursor has rows."""
        return bool(self._rows)


class Database:
    """Database class managing SQLite connection and operations."""

    _conn: sqlite3.Connection | None
    path: Path

    def __init__(self, path: Path | str, **connect_kwargs: Any) -> None:
        """
        Initialize a Database instance.

        Args:
            path: Path to the SQLite database file. Use ':memory:' for in-memory.
            connect_kwargs: Optional keyword args forwarded to sqlite3.connect().
        """
        self.path = Path(path)
        self.connect_kwargs = connect_kwargs
        self._conn = None
        self._schema_initialized = False

    def connection(self) -> sqlite3.Connection:
        """Get a connection as a context manager with proper transaction handling."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.path, **self.connect_kwargs)
            self._conn.row_factory = sqlite3.Row
            if not self._schema_initialized:
                self._init_schema_with_connection(self._conn)
                self._conn.commit()
                self._schema_initialized = True
        return self._conn

    def connect(self) -> sqlite3.Connection:
        """Open the SQLite connection (legacy method - use connection() instead)."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.path, **self.connect_kwargs)
            self._conn.row_factory = sqlite3.Row
            if not self._schema_initialized:
                self._init_schema_with_connection(self._conn)
                self._schema_initialized = True

        return self._conn

    def close(self) -> None:
        """Close the SQLite connection."""
        if self._conn:
            self._conn.close()

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
        with self.connection() as conn:
            cur = conn.execute(sql, params or [])
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
        with self.connection() as conn:
            cur = conn.execute(sql, params or [])
            conn.commit()
            return cur.rowcount

    def schema(self) -> Schema:
        """
        Inspect the database schema.

        Returns:
            Schema: A Pydantic model of tables and columns.
        """
        with self.connection() as conn:
            table_names = [
                r[0] for r in conn.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name NOT LIKE 'sqlite_%';
                    """
                )
            ]
            tables: list[TableSchema] = []
            for name in table_names:
                cols = []
                for col in conn.execute(f"PRAGMA table_info('{name}')"):
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
        with self.connection() as conn:
            self._init_schema_with_connection(conn)
            conn.commit()
        self._schema_initialized = True

    def _init_schema_with_connection(self, conn: sqlite3.Connection) -> None:
        """Initialize database schema using provided connection."""
        # Vocab table
        conn.execute('''
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
        conn.execute('''
            CREATE TRIGGER IF NOT EXISTS update_vocab_timestamp 
            AFTER UPDATE ON vocab
            BEGIN
                UPDATE vocab SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
        ''')


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a sqlite3.Row to a dictionary."""
    return dict(zip(row.keys(), row, strict=False))
