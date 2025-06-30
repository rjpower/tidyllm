"""Utilities for extracting field information from Protocol classes and regular classes."""

import inspect
from pathlib import Path
from typing import get_type_hints


def get_protocol_fields(class_type: type) -> dict[str, type]:
    """Extract field names and types from a Protocol class or regular class.

    Returns:
        Dictionary mapping field names to their types
    """
    fields = {}
    
    # For regular classes, try to get constructor parameters
    if hasattr(class_type, "__init__"):
        try:
            sig = inspect.signature(class_type.__init__)
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                if param_name == "kwargs" and param.kind == param.VAR_KEYWORD:
                    continue
                    
                # Get the type annotation
                param_type = param.annotation
                if param_type == inspect.Parameter.empty:
                    param_type = str  # Default to string
                    
                fields[param_name] = param_type
        except (ValueError, TypeError):
            pass
    
    # Also try annotations (for Protocol classes or dataclasses)
    if hasattr(class_type, "__annotations__"):
        try:
            type_hints = get_type_hints(class_type)
        except (NameError, AttributeError):
            # Fallback to __annotations__ if get_type_hints fails
            type_hints = getattr(class_type, "__annotations__", {})

        # Add annotations, but constructor params take precedence
        for name, type_hint in type_hints.items():
            if not name.startswith("_") and name not in fields:  # Skip private attributes
                fields[name] = type_hint

    return fields


def get_cli_type_for_annotation(annotation: type) -> tuple[str, bool]:
    """Get the appropriate CLI type and whether it's a flag for a type annotation.

    Returns:
        Tuple of (click_type, is_flag)
    """
    # Handle common types
    if annotation is bool:
        return ("bool", True)
    elif annotation is int:
        return ("int", False)
    elif annotation is float:
        return ("float", False)
    elif annotation is str:
        return ("str", False)
    elif annotation is Path or annotation == Path:
        return ("path", False)
    elif hasattr(annotation, "__origin__"):
        # Handle generic types like list[str], set[str], etc.
        origin = getattr(annotation, "__origin__", None)
        if origin is list or origin is set:
            return ("str", False)  # Will be split by comma

    # Default to string for unknown types
    return ("str", False)

