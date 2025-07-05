"""Object manipulation tools for data extraction and processing."""

import json
import pickle
import re
import sys
from typing import Any

from tidyllm.adapters.cli import cli_main
from tidyllm.registry import register
from tidyllm.tools.context import ToolContext


def _parse_path(path: str) -> list[str | int]:
    """Parse dot/bracket notation path into segments.

    Examples:
        '.field' -> ['field']
        '.field.subfield' -> ['field', 'subfield']
        '.items[0]' -> ['items', 0]
        '.items[0].name' -> ['items', 0, 'name']
        '[0]' -> [0]
        '["key"]' -> ['key']
    """
    if not path.startswith(".") and not path.startswith("["):
        raise ValueError(f"Path must start with '.' or '[', got: {path}")

    segments = []

    # Remove leading dot
    if path.startswith("."):
        path = path[1:]

    # Split by dots, but handle brackets
    parts = []
    current = ""
    bracket_depth = 0

    for char in path:
        if char == "[":
            bracket_depth += 1
            current += char
        elif char == "]":
            bracket_depth -= 1
            current += char
        elif char == "." and bracket_depth == 0:
            if current:
                parts.append(current)
                current = ""
        else:
            current += char

    if current:
        parts.append(current)

    # Parse each part
    for part in parts:
        if not part:
            continue

        # Check for bracket notation
        bracket_match = re.match(r"^([^[]*)\[([^\]]+)\]$", part)
        if bracket_match:
            field_name, index_str = bracket_match.groups()
            if field_name:
                segments.append(field_name)

            # Parse index/key
            index_str = index_str.strip()
            if index_str.startswith('"') and index_str.endswith('"'):
                # String key
                segments.append(index_str[1:-1])
            elif index_str.startswith("'") and index_str.endswith("'"):
                # String key
                segments.append(index_str[1:-1])
            else:
                # Numeric index
                try:
                    segments.append(int(index_str))
                except ValueError:
                    segments.append(index_str)
        else:
            # Simple field name
            segments.append(part)

    return segments


def _read_stdin() -> Any:
    """Read and parse data from stdin."""
    if sys.stdin.isatty():
        raise ValueError("No input provided. Use stdin or provide data as argument.")

    data = sys.stdin.buffer.read()

    # Try to detect format
    try:
        # Try pickle first
        if data.startswith(b"\x80"):
            return pickle.loads(data)
    except Exception:
        pass

    try:
        return json.loads(data.decode("utf-8"))
    except Exception:
        pass

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data


def _traverse_path(data: Any, segments: list[str | int]) -> Any:
    """Traverse object path and return value."""
    current = data

    for segment in segments:
        if isinstance(current, list):
            current = current[int(segment)]
        elif isinstance(current, dict):
            current = current[segment]
        else:
            current = getattr(current, segment)

    return current


@register()
def get_attr(
    path: str, default: Any = None, strict: bool = True, input_data: Any = None
) -> Any:
    """Extract field from object using dot/bracket notation.

    Args:
        path: Path to extract (e.g., '.field.subfield', '.items[0].name')
        default: Default value if path not found (only used if strict=False)
        strict: If True, raise error on missing path; if False, return default
        input_data: Input data (if not provided, read from stdin)

    Returns:
        Extracted value

    Examples:
        echo '{"name": "John"}' | object.getattr '.name'
        echo '{"items": [1,2,3]}' | object.getattr '.items[0]'
        echo '{"user": {"profile": {"name": "John"}}}' | object.getattr '.user.profile.name'
    """
    if input_data is None:
        input_data = _read_stdin()

    segments = _parse_path(path)

    try:
        return _traverse_path(input_data, segments)
    except (KeyError, IndexError, AttributeError) as e:
        if strict:
            raise KeyError(f"Path not found: {path}") from e
        return default


@register()
def get_item(key: str | int, default: Any = None, input_data: Any = None) -> Any:
    """Extract item by key/index from object.

    Args:
        key: Key or index to extract
        default: Default value if key not found (only used if strict=False)
        input_data: Input data (if not provided, read from stdin)

    Returns:
        Extracted value

    Examples:
        echo '{"name": "John"}' | object.get_item 'name'
        echo '[1,2,3]' | object.getitem 0
        echo '{"items": [1,2,3]}' | object.get_item 'items'
    """
    if input_data is None:
        input_data = _read_stdin()

    if isinstance(input_data, dict):
        return input_data[key]
    elif isinstance(input_data, list | tuple):
        return input_data[int(key)]
    else:
        return getattr(input_data, str(key))


if __name__ == "__main__":
    cli_main([get_attr, get_item], context_cls=ToolContext)
