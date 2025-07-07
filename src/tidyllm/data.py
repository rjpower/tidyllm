"""
Data model for tidyllm.

Defines core data structures and interfaces for tidyllm functions.
"""

import base64
import inspect
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Generic,
    Protocol,
    TypeAlias,
    TypeVar,
    Union,
    get_args,
    get_origin,
)
from uuid import UUID

from pydantic import BaseModel

Serializable: TypeAlias = Union[
    int,
    float,
    str,
    bytes,
    Enum,
    date,
    datetime,
    UUID,
    Path,
    Decimal,
    BaseModel,
    "dict[str, Serializable]",
    "list[Serializable]",
    "Table",
]


def parse_from_json(value: Serializable, value_type: type) -> Any:
    """Parse a value from JSON to the specified type using Pydantic conventions."""

    if value is None:
        return None

    # Handle Pydantic models
    if inspect.isclass(value_type) and issubclass(value_type, BaseModel):
        return value_type.model_validate(value)

    # Get origin and args for generic types
    origin = get_origin(value_type)
    args = get_args(value_type)

    # Handle list types
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"Expected list, got {type(value)}")
        item_type = args[0] if args else Any
        return [parse_from_json(item, item_type) for item in value]

    # Handle dict types
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"Expected dict, got {type(value)}")
        if len(args) > 0:
            return {
                key: parse_from_json(value, args[1]) for key, value in value.items()
            }
        else:
            # if we don't know the value type, just leave the dictionary as is
            return value

    if origin is set:
        if not isinstance(value, list):
            raise ValueError(f"Expected list for set, got {type(value)}")
        item_type = args[0] if args else Any
        return {parse_from_json(item, item_type) for item in value}

    # Handle primitive types
    if value_type is int:
        return int(value)
    elif value_type is float:
        return float(value)
    elif value_type is str:
        return str(value)
    elif value_type is bool:
        return bool(value)
    elif value_type is datetime:
        if isinstance(value, int):
            return datetime.fromtimestamp(value)
        return datetime.fromisoformat(value)
    elif value_type is date:
        return date.fromisoformat(value)
    elif value_type is time:
        return time.fromisoformat(value)
    elif value_type is bytes:
        return base64.b64decode(value)
    elif value_type is Decimal:
        return Decimal(str(value))
    elif value_type is UUID:
        return UUID(value)
    elif value_type is Path:
        return Path(value)
    else:
        warnings.warn(
            f"Unsupported result type: {value_type}. Returning raw result: {value}",
            UserWarning,
            stacklevel=2,
        )
        return value


def to_json_value(value: Serializable) -> Any:
    """Convert a Serializable value to a JSON-compatible format."""

    if value is None:
        return None

    # Handle Pydantic models
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")

    # Handle primitive types that are already JSON-compatible
    if isinstance(value, int | float | str | bool):
        return value

    # Handle lists
    if isinstance(value, list):
        return [to_json_value(item) for item in value]

    # Handle dicts
    if isinstance(value, dict):
        return {key: to_json_value(val) for key, val in value.items()}

    # Handle sets (convert to list)
    if isinstance(value, set):
        return [to_json_value(item) for item in value]

    # Handle Enum
    if isinstance(value, Enum):
        return value.value

    # Handle datetime, date, time
    if isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, date):
        return value.isoformat()
    elif isinstance(value, time):
        return value.isoformat()

    # Handle bytes (base64 encode)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("utf-8")

    # Handle Decimal
    if isinstance(value, Decimal):
        return str(value)

    # Handle UUID
    if isinstance(value, UUID):
        return str(value)

    # Handle Path
    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "to_json"):
        return value.to_json()  # type: ignore

    raise ValueError(f"Can't serialize {value} to JSON.")


ColumnSchema: TypeAlias = dict[str, type]


class Table(Protocol):
    @property
    def columns(self) -> ColumnSchema: ...

    def __iter__(self) -> Iterator: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...


T = TypeVar("T", bound=Serializable)


@dataclass
class ConcreteTable(Generic[T]):
    columns: dict[str, type]
    rows: list[Any]

    def __iter__(self) -> Iterator:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Any:
        return self.rows[index]

    @property
    def count(self):
        return len(self.rows)

    @staticmethod
    def from_dict(columns: dict[str, type[Serializable]], rows: list[Serializable]):
        return ConcreteTable(columns=columns, rows=rows)

    @staticmethod
    def from_pydantic(rows: list[BaseModel]):
        if len(rows) == 0:
            return ConcreteTable.empty()

        fields = {k: v.annotation for (k, v) in type(rows[0]).model_fields.items()}
        return ConcreteTable.from_dict(fields, rows)

    @staticmethod
    def empty():
        return ConcreteTable(columns={}, rows=[])

    def to_json(self):
        result = []
        for row in self.rows:
            result.append(to_json_value(row))
        return result

    def map(self, func: Callable) -> "ConcreteTable":
        """Apply a function to each row and return a new table.

        Args:
            func: Function that takes a row and returns a transformed row

        Returns:
            New ConcreteTable with transformed rows
        """
        if len(self.rows) == 0:
            return ConcreteTable.empty()

        new_rows = [func(row) for row in self.rows]

        # If all rows are the same type, infer columns from the first row
        if new_rows and all(type(row) == type(new_rows[0]) for row in new_rows):
            if hasattr(new_rows[0], "model_fields"):
                # Pydantic model
                fields = {
                    k: v.annotation for (k, v) in type(new_rows[0]).model_fields.items()
                }
                return ConcreteTable.from_dict(fields, new_rows)

        # For other cases, try to preserve existing column structure or create generic
        return ConcreteTable(columns=self.columns, rows=new_rows)

    def filter(self, predicate: Callable) -> "ConcreteTable":
        """Filter rows based on a predicate function.

        Args:
            predicate: Function that takes a row and returns True/False

        Returns:
            New ConcreteTable with filtered rows
        """
        filtered_rows = [row for row in self.rows if predicate(row)]
        return ConcreteTable(columns=self.columns, rows=filtered_rows)


class Sequence(Protocol[T]):
    def __iter__(self) -> Iterator[T]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> T: ...
    def __contains__(self, item: T) -> bool: ...
