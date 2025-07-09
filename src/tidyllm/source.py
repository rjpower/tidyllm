"""Source library for unified data source abstraction.

Provides a unified interface for reading from various data sources including
files, bytes, stdin, and remote resources using fsspec.
"""

from pathlib import Path
from typing import Any, Protocol, TypeVar, Union, cast, overload, runtime_checkable

import fsspec
from pydantic import BaseModel, Field

ValueType = TypeVar("ValueType", covariant=True)
SliceType = TypeVar("SliceType", covariant=True)


@runtime_checkable
class Sliceable(Protocol[ValueType, SliceType]):
    """Protocol for objects that support slicing operations."""
    
    @overload
    def __getitem__(self, key: int) -> ValueType: ...
    
    @overload
    def __getitem__(self, key: slice) -> SliceType: ...

    def __getitem__(self, key: int | slice) -> ValueType | SliceType: ...


@runtime_checkable
class Source(Protocol):
    """Base protocol for all data sources."""
    
    def read(self, size: int = -1) -> bytes | str:
        """Read data from the source."""
        ...


class ByteSource(BaseModel):
    """Source backed by byte data."""
    
    data: bytes = Field(description="The byte data")
    pos: int = Field(default=0, description="Current position in the data")

    def read(self, size: int = -1) -> bytes:
        """Read bytes from the source."""
        if size == -1:
            result = self.data[self.pos:]
            self.pos = len(self.data)
        else:
            result = self.data[self.pos:self.pos + size]
            self.pos += len(result)
        return result



class SliceSource(BaseModel):
    """Source backed by sliceable data (str, bytes, list, etc)."""
    
    data: Any = Field(description="The sliceable data")
    pos: int = Field(default=0, description="Current position in the data")

    def read(self, size: int = -1) -> Any:
        """Read data from the source."""
        if size == -1:
            result = self.data[self.pos:]
            self.pos = len(self.data)
        else:
            result = self.data[self.pos:self.pos + size]
            self.pos += len(result)
        return result


class FileSource(BaseModel):
    """Source backed by a file path."""
    
    path: Path = Field(description="Path to the file")
    mode: str = Field(default="rb", description="File open mode")

    def __init__(self, **data):
        super().__init__(**data)
        self._file = None

    def _open(self):
        """Lazily open the file."""
        if self._file is None:
            self._file = open(self.path, self.mode)

    def read(self, size: int = -1) -> bytes:
        """Read data from the file."""
        self._open()
        return self._file.read(size)

    def close(self):
        """Close the file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        """Clean up file handle."""
        self.close()



class FSSpecSource(BaseModel):
    """Source backed by fsspec filesystem."""
    
    url: str = Field(description="URL or path to the resource")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Additional fsspec kwargs")

    def __init__(self, **data):
        super().__init__(**data)
        self._fs = None
        self._file = None

    def _open(self):
        """Lazily open the file."""
        if self._file is None:
            self._fs, path = fsspec.core.url_to_fs(self.url, **self.kwargs)
            self._file = self._fs.open(path, 'rb')

    def read(self, size: int = -1) -> bytes:
        """Read data from the source."""
        self._open()
        return self._file.read(size)

    def close(self):
        """Close the file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        """Clean up file handle."""
        self.close()



# Type aliases
SourceLike = Union[Path, str, bytes, Source]


def is_source_like(value: Any) -> bool:
    """Check if a value is SourceLike without expensive conversion.
    
    Args:
        value: Value to check
        
    Returns:
        True if value can be used as a SourceLike parameter
    """
    return (
        isinstance(value, (Path, str, bytes)) or 
        (hasattr(value, 'read') and callable(getattr(value, 'read')))
    )


def as_source(data: SourceLike) -> Source:
    """Convert various input types to a Source."""
    if hasattr(data, 'read') and callable(data.read):
        return cast(data, Source)
    elif isinstance(data, Path):
        return FileSource(path=data)
    elif isinstance(data, str):
        # Try to determine if it's a URL/path or literal string data
        if '://' in data or data.startswith('/') or Path(data).exists():
            return FSSpecSource(url=data)
        else:
            return SliceSource(data=data)
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


def open_url(url: str, **kwargs) -> FSSpecSource:
    """Open a URL as a Source using fsspec."""
    return FSSpecSource(url=url, kwargs=kwargs)