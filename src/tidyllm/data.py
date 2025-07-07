"""
Data model for tidyllm.

Provides backwards compatibility imports for legacy code.
All functionality has been moved to specialized modules.
"""

# Backwards compatibility imports

# Legacy imports for protocols
from collections.abc import Iterator
from typing import Protocol, TypeVar

from tidyllm.linq import (
    ColumnSchema,
    Enumerable,
    Table,
    from_iterable,
)
from tidyllm.serialization import (
    Serializable,
    from_json_dict,
    to_json_dict,
)

T = TypeVar("T", bound=Serializable)

class Sequence(Protocol[T]):
    def __iter__(self) -> Iterator[T]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> T: ...
    def __contains__(self, item: T) -> bool: ...

__all__ = [
    "Serializable",
    "from_json_dict",
    "to_json_dict",
    "Table",
    "ColumnSchema",
    "Enumerable",
    "from_iterable",
    "Sequence",
]
