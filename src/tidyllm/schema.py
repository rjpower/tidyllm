"""Function schema extraction and JSON schema generation.

This module now provides backwards compatibility imports.
All functionality has been moved to tidyllm.function_schema.
"""

# Backwards compatibility imports
from tidyllm.function_schema import (
    FunctionDescription,
    FunctionSchema,
    JSONSchema,
    _process_union_type,
    function_schema_from_args,
)

__all__ = [
    "FunctionDescription",
    "FunctionSchema", 
    "JSONSchema",
    "function_schema_from_args",
    "_process_union_type",
]