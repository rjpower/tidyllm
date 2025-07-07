"""
Pydantic-based serialization utilities for tidyllm.

Provides simple helpers for JSON serialization using Pydantic's robust system.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Any, TypeVar
from uuid import UUID

from pydantic import BaseModel, TypeAdapter

T = TypeVar('T')

Serializable = (
    BaseModel
    | str
    | int
    | float
    | bool
    | None
    | list
    | dict
    | UUID
    | date
    | datetime
    | Decimal
)


def to_json_dict(obj: Serializable) -> dict:
    """Convert any object to JSON dict using Pydantic."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode='json')

    # For non-Pydantic objects, create a TypeAdapter
    adapter = TypeAdapter(type(obj))
    return adapter.dump_python(obj, mode='json')


def to_json_string(obj: Serializable) -> str:
    """Convert any object to JSON string using Pydantic."""
    if isinstance(obj, BaseModel):
        return obj.model_dump_json()

    adapter = TypeAdapter(type(obj))
    result = adapter.dump_json(obj)
    if isinstance(result, bytes | bytearray):
        return result.decode('utf-8')
    return str(result)


def from_json_dict(data: Any, target_type: type[T]) -> T:
    """Parse JSON data to target type using Pydantic."""
    if data is None:
        return None  # type: ignore
        
    if isinstance(target_type, type) and issubclass(target_type, BaseModel):
        return target_type.model_validate(data)
    
    adapter = TypeAdapter(target_type)
    return adapter.validate_python(data)

def from_json_string(json_str: str, target_type: type[T]) -> T:
    """Parse JSON string to target type using Pydantic."""
    if issubclass(target_type, BaseModel):
        return target_type.model_validate_json(json_str)

    adapter = TypeAdapter(target_type)
    return adapter.validate_json(json_str)
