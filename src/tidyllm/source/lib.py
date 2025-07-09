"""Core source library functions."""

from pathlib import Path
from typing import Any, cast

from tidyllm.source.gdrive import GDriveSource, parse_gdrive_url
from tidyllm.source.model import ByteSource, FileSource, Source, SourceLike


def is_source_like(value: Any) -> bool:
    """Check if a value is SourceLike without expensive conversion.
    
    Args:
        value: Value to check
        
    Returns:
        True if value can be used as a SourceLike parameter
    """
    return (
        isinstance(value, (Path | str | bytes)) or 
        (hasattr(value, 'read') and callable(value.read))
    )


def as_source(data: SourceLike) -> Source:
    """Convert various input types to a Source."""
    if hasattr(data, 'read') and callable(data.read): # type: ignore
        return cast(Source, data)
    elif isinstance(data, Path):
        return FileSource(path=data)
    elif isinstance(data, str):
        # Handle URL-like strings
        if data.startswith('gdrive://'):
            path = parse_gdrive_url(data)
            return GDriveSource(path=path)
        elif data.startswith('/') or Path(data).exists():
            return FileSource(path=Path(data))
        else:
            raise ValueError(f"Unsupported URL scheme: {data}")
    elif isinstance(data, bytes):
        return ByteSource(data=data)
    else:
        raise TypeError(f"Cannot convert {type(data)} to Source")


def read_bytes(source: SourceLike) -> bytes:
    """Convenience function to read all bytes from a source-like object."""
    src = as_source(source)
    data = src.read()
    if isinstance(data, bytes):
        return data
    elif isinstance(data, str):
        return data.encode('utf-8')
    else:
        raise TypeError(f"Cannot convert {type(data)} to bytes")


def read_text(source: SourceLike, encoding: str = 'utf-8') -> str:
    """Convenience function to read all text from a source-like object."""
    src = as_source(source)
    data = src.read()
    if isinstance(data, str):
        return data
    elif isinstance(data, bytes):
        return data.decode(encoding)
    else:
        raise TypeError(f"Cannot convert {type(data)} to text")


class SourceManager:
    """Manages lifecycle of Sources in a context."""
    
    def __init__(self):
        self._active_sources: list[Source] = []
    
    def register(self, source: Source) -> Source:
        """Register source for cleanup."""
        self._active_sources.append(source)
        return source
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for source in self._active_sources:
            if hasattr(source, 'close'):
                source.close()  # type: ignore