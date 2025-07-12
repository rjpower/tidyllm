"""
Pydantic-based serialization utilities for tidyllm.

Provides simple helpers for JSON serialization using Pydantic's robust system,
plus dynamic model creation utilities.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Any, TypeVar, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, TypeAdapter, create_model

T = TypeVar("T")

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
        return obj.model_dump(mode="json")

    # For non-Pydantic objects, create a TypeAdapter
    adapter = TypeAdapter(type(obj))
    return adapter.dump_python(obj, mode="json")


def to_json_str(obj: Serializable) -> str:
    """Convert any object to JSON string using Pydantic."""
    if isinstance(obj, BaseModel):
        return obj.model_dump_json()

    adapter = TypeAdapter(type(obj))
    return adapter.dump_json(obj).decode("utf-8")


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


# Dynamic model creation utilities
def create_model_from_field_definitions(
    model_name: str,
    field_definitions: dict[str, tuple[type, Any]],
    config: ConfigDict | None = None,
) -> type[BaseModel]:
    """Create a Pydantic model from field definitions.

    Args:
        model_name: Name for the generated model class
        field_definitions: Dict mapping field names to (type, default_value) tuples
        config: Optional Pydantic config, defaults to arbitrary_types_allowed=True

    Returns:
        Dynamically created Pydantic model class
    """
    if config is None:
        config = ConfigDict(
            arbitrary_types_allowed=True,
            json_schema_extra={"additionalProperties": False},
        )

    return create_model(model_name, __config__=config, **field_definitions)  # type: ignore


def infer_type_from_values(values: list[Any]) -> type | Any:
    """Infer a type from a list of sample values.

    Args:
        values: List of values to analyze

    Returns:
        Inferred type, falls back to Any for complex cases
    """
    if not values:
        return Any

    # Filter out None values for type inference
    non_none_values = [v for v in values if v is not None]

    if not non_none_values:
        return type(None)

    # Get unique types
    types = {type(v) for v in non_none_values}

    if len(types) == 1:
        # All values have the same type
        return next(iter(types))
    elif len(types) <= 3:
        # A few different types - create union
        return Union[tuple(types)]  # noqa: UP007
    else:
        # Too many types - fallback to Any
        return Any


def infer_field_types_from_data(data_samples: list[Any]) -> dict[str, tuple[type, Any]]:
    """Infer field types from data samples.

    Args:
        data_samples: List of data items to analyze

    Returns:
        Dict mapping field names to (type, default_value) tuples
    """
    if not data_samples:
        return {}

    # Handle different data types
    first_item = data_samples[0]

    if isinstance(first_item, BaseModel):
        # Extract from Pydantic model fields
        model_type = type(first_item)
        field_definitions = {}
        for field_name, field_info in model_type.model_fields.items():
            field_type = field_info.annotation
            # Determine if field is optional
            default_value = field_info.default if field_info.default is not ... else ...
            field_definitions[field_name] = (field_type, default_value)

        return field_definitions

    elif isinstance(first_item, dict):
        # Infer from dictionary structure
        all_keys = set()
        for item in data_samples:
            if isinstance(item, dict):
                all_keys.update(item.keys())

        field_definitions = {}
        for key in all_keys:
            # Collect all values for this key across samples
            values = []
            for item in data_samples:
                if isinstance(item, dict) and key in item:
                    values.append(item[key])

            # Infer type from values
            if values:
                inferred_type = infer_type_from_values(values)
                # Make field optional since not all samples may have all keys
                field_definitions[key] = (inferred_type | None, None)
            else:
                field_definitions[key] = (Any | None, None)

        return field_definitions

    else:
        # Primitive or other type - create single value field
        sample_type = infer_type_from_values(data_samples)
        return {"value": (sample_type, ...)}


def create_model_from_data_sample(
    data_samples: list[Any],
    model_name: str = "InferredSchema",
) -> type[BaseModel]:
    """Create a Pydantic model by inferring schema from data samples.

    Args:
        data_samples: List of data items to analyze
        model_name: Name for the generated model class

    Returns:
        Dynamically created Pydantic model class
    """
    if not data_samples:
        return create_model_from_field_definitions(model_name, {})

    # Special case: if all samples are the same Pydantic model type, return that type
    first_item = data_samples[0]
    if isinstance(first_item, BaseModel):
        first_type = type(first_item)
        if all(isinstance(item, first_type) for item in data_samples):
            return first_type

    # Infer field types from data
    field_definitions = infer_field_types_from_data(data_samples)

    return create_model_from_field_definitions(model_name, field_definitions)
