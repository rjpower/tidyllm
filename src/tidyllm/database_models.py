"""Pydantic models for database interface."""

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field


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