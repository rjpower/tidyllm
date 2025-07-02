# Database Interface Specification

## Overview

This document specifies the design and usage of a `Database` class to replace the individual helper functions in `src/tidyllm/db.py`. The goal is to encapsulate SQLite interactions behind a structured, object-oriented API, leveraging Pydantic models for typed access and introspection.

Key features:
- Single `Database` class managing connection lifecycle
- High-level methods: `query`, `mutate`, `schema`
- Pydantic models for rows, cursors, and schema objects
- Clean separation of concerns and easier testing

## Pydantic Models

### ColumnSchema

Represents a single column in a table.

```python
from pydantic import BaseModel
from typing import Optional

class ColumnSchema(BaseModel):
    name: str = Field(...)
    type: str = Field(...)
    not_null: bool
    default: Optional[str]
    primary_key: bool
```

Fields:
- `name`: Column name
- `type`: Data type (e.g., `TEXT`, `INTEGER`)
- `not_null`: Whether `NOT NULL` constraint exists
- `default`: Default value or `None`
- `primary_key`: Whether column is part of the primary key

### TableSchema

Represents a database table.

```python
from pydantic import BaseModel
from typing import List

class TableSchema(BaseModel):
    name: str = Field(...)
    columns: List[ColumnSchema]
```

- `name`: Table name
- `columns`: List of `ColumnSchema` objects

### Schema

Represents the full database schema.

```python
from pydantic import BaseModel
from typing import List, Optional

class Schema(BaseModel):
    tables: List[TableSchema]

    def get_table(self, name: str) -> Optional[TableSchema]:
        return next((t for t in self.tables if t.name == name), None)
```

### Row

Wraps a single result row.

```python
from pydantic import BaseModel
from typing import Any, Dict

class Row(BaseModel):
    __root__: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.__root__[key]
```

### Cursor

Iterator over `Row` objects with metadata.

```python
from pydantic import BaseModel
from typing import Iterator, List

class Cursor(BaseModel):
    columns: List[str]
    _rows: List[Row]

    def __iter__(self) -> Iterator[Row]:
        yield from self._rows

    def all(self) -> List[Row]:
        return self._rows
```

## Database Class API

### Class Definition

```python
import sqlite3
from pathlib import Path
from typing import Any, Optional, Sequence, Union, List
from .database_models import Row, Cursor, Schema, TableSchema, ColumnSchema

class Database:
    def __init__(
        self,
        path: Union[str, Path],
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
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Open the SQLite connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.path, **self.connect_kwargs)
            self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        """Close the SQLite connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def query(
        self,
        sql: str,
        params: Optional[Sequence[Any]] = None,
    ) -> Cursor:
        """
        Execute a SELECT statement and return a Cursor.

        Returns:
            Cursor: An iterable of Rows.
        """
        self.connect()
        cur = self._conn.execute(sql, params or [])
        columns = [description[0] for description in cur.description]
        rows = [Row(__root__=dict(zip(columns, r))) for r in cur.fetchall()]
        return Cursor(columns=columns, _rows=rows)

    def mutate(
        self,
        sql: str,
        params: Optional[Sequence[Any]] = None,
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
        tables: List[TableSchema] = []
        for name in table_names:
            cols = []
            for col in self._conn.execute(f"PRAGMA table_info('{name}')"):  # returns rows with keys
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

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback
    ) -> None:
        self.close()
```

### Method Summary

| Method    | Signature                                    | Returns    | Description                                    |
|-----------|----------------------------------------------|------------|------------------------------------------------|
| `connect` | `() -> None`                                 | `None`     | Open connection                                 |
| `close`   | `() -> None`                                 | `None`     | Close connection                                |
| `query`   | `(sql: str, params: Sequence[Any]) -> Cursor`| `Cursor`   | Execute SELECT queries                          |
| `mutate`  | `(sql: str, params: Sequence[Any]) -> int`   | `int`      | Execute INSERT/UPDATE/DELETE; return rowcount   |
| `schema`  | `() -> Schema`                               | `Schema`   | Introspect tables and columns                   |

### Context Manager Support

Use `with` syntax to auto-connect and close:

```python
from src.tidyllm.database import Database

with Database('db.sqlite3') as db:
    cursor = db.query("SELECT * FROM users WHERE active=?", (1,))
    for row in cursor:
        print(row['id'], row['username'])
```

## Usage Examples

### 1. Simple Query

```python
db = Database('data.db')
users: Cursor = db.query("SELECT id, name FROM user")
for user in users:
    print(user['id'], user['name'])
```

### 2. Mutations

```python
db = Database(':memory:')
# Create table
db.mutate("CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT)")
# Insert rows
count = db.mutate("INSERT INTO books(title) VALUES (?)", ("1984",))
print(f"Inserted {count} row(s)")
```

### 3. Schema Inspection

```python
schema = db.schema()
for table in schema.tables:
    print(f"Table: {table.name}")
    for col in table.columns:
        print(f"  - {col.name}: {col.type}")
```

## Implementation Notes

- The `Database` class uses `sqlite3.Row` factory for column access.
- Pydantic models ensure correct typing and validation.
- For advanced usage, extend `Database` with transaction support or custom row factories.

## Thread Safety

By default, SQLite connections are not thread-safe. For multi-threaded applications, open separate instances per thread with:

```python
db = Database('data.db', check_same_thread=False)
```

---

*End of specification.*
