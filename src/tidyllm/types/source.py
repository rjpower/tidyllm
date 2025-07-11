"""Source types for handling file and byte data in tidyllm.

Provides a unified interface for reading data from files or bytes objects.
Tools can accept Source parameters which automatically convert from:
- File paths (str/Path) -> open() file handle
- Raw bytes -> BytesIO

Example usage:
    @register()
    def my_tool(data: Source):
        content = data.read()
"""

import base64
import binascii
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Protocol, runtime_checkable

from pydantic import BeforeValidator, PlainSerializer, WithJsonSchema


@runtime_checkable
class BytesReader(Protocol):
    def read(self, sz: int = -1) -> bytes:
        ...


def _validate_source(value: Any) -> Any:
    """Validate and convert input to a BytesIO object."""
    if hasattr(value, 'read'):
        # Already a file-like object, wrap in BytesIO if needed
        if isinstance(value, BytesIO):
            return value
        # Read all data and wrap in BytesIO
        data = value.read()
        if isinstance(data, str):
            data = data.encode('utf-8')
        return BytesIO(data)
    
    if isinstance(value, str | Path):
        # Check if it's a data URI first
        if isinstance(value, str) and value.startswith('data:'):
            try:
                _, b64_part = value.split(',', 1)
                decoded = base64.b64decode(b64_part)
                return BytesIO(decoded)
            except (ValueError, binascii.Error):
                # Fall back to treating as file path
                pass
        # Open file and read into BytesIO
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return BytesIO(path.read_bytes())
    
    if isinstance(value, bytes):
        return BytesIO(value)
    
    raise ValueError(f"Cannot convert {type(value)} to Source. Expected str, Path, bytes, or file-like object.")


def _serialize_source(source: Any) -> str:
    """Serialize a source for JSON output."""
    # Reset position to read all data
    pos = source.tell()
    source.seek(0)
    data = source.read()
    source.seek(pos)  # Restore position
    
    # Try to detect if it looks like a file path
    try:
        text = data.decode('utf-8')
        if len(text) < 256 and '\n' not in text and Path(text.strip()).exists():
            return text.strip()
    except UnicodeDecodeError:
        pass
    
    # Encode as data URI
    b64_data = base64.b64encode(data).decode('ascii')
    return f"data:application/octet-stream;base64,{b64_data}"


Source = Annotated[
    BytesReader,
    BeforeValidator(_validate_source),
    PlainSerializer(_serialize_source, return_type=str),
    WithJsonSchema({
        'type': 'string',
        'description': 'File path or base64-encoded data URI (data:mime/type;base64,...)',
        'examples': [
            '/path/to/file.txt',
            'data:text/plain;base64,SGVsbG8gd29ybGQ='
        ]
    }, mode='validation')
]


def as_source(value: Any) -> Any:
    """Convert any input to a Source object."""
    return _validate_source(value)


def read_bytes(source: Source, size: int = -1) -> bytes:
    """Read bytes from a source.
    
    Args:
        source: The source to read from
        size: Number of bytes to read (-1 for all)
        
    Returns:
        The bytes read from the source
    """
    data = source.read(size)
    return data


def read_text(source: Source, encoding: str = 'utf-8', size: int = -1) -> str:
    """Read text from a source.
    
    Args:
        source: The source to read from
        encoding: Text encoding to use
        size: Number of bytes to read (-1 for all)
        
    Returns:
        The text read from the source
    """
    data = read_bytes(source, size)
    return data.decode(encoding)