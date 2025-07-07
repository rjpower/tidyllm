"""
Pydantic-based serialization utilities for tidyllm.

Provides simple helpers for JSON serialization using Pydantic's robust system.
"""

from typing import Any, TypeVar, Union

from pydantic import BaseModel, TypeAdapter

T = TypeVar('T')

# Simple type alias - no custom registry needed
Serializable = Union[
    BaseModel,           # All our data structures
    str, int, float, bool, None,  # Primitives
    list, dict,          # Collections (handled by Pydantic)
]

def to_json_dict(obj: Any) -> dict:
    """Convert any object to JSON dict using Pydantic."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode='json')
    
    # For non-Pydantic objects, create a TypeAdapter
    adapter = TypeAdapter(type(obj))
    return adapter.dump_python(obj, mode='json')

def to_json_string(obj: Any) -> str:
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

# Legacy compatibility - these just delegate to Pydantic
parse_from_json = from_json_dict  # Backwards compatibility
to_json_value = to_json_dict      # Backwards compatibility