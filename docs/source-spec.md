# Source Library Specification

## Overview

The Source library provides a unified abstraction for reading from and writing
to various data sources in tidyllm. This eliminates code duplication and
provides a consistent interface for handling files, bytes, stdin, and remote
resources.

fsspec is used to abstract around files, and helper functions are provided to
read slices from "source like" objects, such as bytes or strings.

## Core Concepts

### Source Protocol

A Source represents any readable data source with lazy evaluation - data is only
read when explicitly requested. Sources are typed and return a sliceable:

```python
# source.py
SliceType = TypeVar("SliceType")
ValueType = TypeVar("ValueType")

class Sliceable(Protocol[ValueType, SliceType]):
    @overload
    def __getitem__(self, key: int) -> ValueType: ...
    
    @overload
    def __getitem__(self, key: slice) -> SliceType: ...

    def __getitem__(self, key: Union[int, slice]) -> Union[ValueType, SliceType]: ...


class Source(Protocol, Generic[ValueType, SliceType]):
    """Base protocol for all data sources."""
    
    def read(self, size: int = -1) -> Sliceable[ValueType, SliceType]:
        """Read bytes from the source."""
        ...

```

### Source Factory Functions

We need to create sources somehow, so we'll provide wrappers for common types.

```python
# source.py
from typing import Union
from pathlib import Path

SourceLike = Union[Path, str, bytes, Source]

class ByteSource(BaseModel):
    data: Base64Bytes
    pos: int = 0

    def read(self, size: int = -1):
        return ...

class SliceSource:
    data: T
    pos: int = 0
    ...

class FileSource:
    ...

```

Reading from cloud fileysystems:

We want to be able to connect gdrive etc to our "Source" system. We'll use
fsspec to open arbitrary files using the fsspec handler.

```python
# src/tidyllm/connectors/fsspec.py
from typing import Any
import fsspec
from tidyllm.source import Source

class FSSpecSource:
    """Source backed by fsspec filesystem."""
    
    def __init__(self, url: str, **kwargs):
        self.url = url
        self.kwargs = kwargs
        self._fs = None
        self._file = None
    
    def _open(self):
        """Lazily open the file."""
        if self._file is None:
            self._fs, path = fsspec.core.url_to_fs(self.url, **self.kwargs)
            self._file = self._fs.open(path, 'rb')
    
    def read(self, size: int = -1) -> bytes:
        self._open()
        return self._file.read(size)
    
    def __del__(self):
        """Clean up file handle."""
        if self._file is not None:
            self._file.close()

```

Provide helpers for creating a source from existing types:

```python
def as_source(data: SourceLike) -> Source:
    """Convert various input types to a Source."""
    if isinstance(data, Source):
        return data
    elif isinstance(data, Path):
        return FileSource(data)
    elif isinstance(data, str|bytes):
        return SliceSource(data)
    else:
        raise TypeError(f"Cannot convert {type(data)} to Source")


def read_bytes(source: SourceLike) -> bytes:
    """Convenience function to read all bytes from a source-like object."""
    return as_source(source).read()

def read_text(source: SourceLike, encoding: str = 'utf-8') -> str:
    """Convenience function to read all text from a source-like object."""
    return as_source(source).read().decode(encoding)
```

## Integration Examples

### Updating transcribe.py

```python
from tidyllm.source import SourceLike, as_source
from tidyllm.tools.transcribe import TranscriptionResult

@register()
@cached_function
def transcribe_audio(
    source: SourceLike,
    source_language: str | None = None,
    target_language: str = "en",
) -> TranscriptionResult:
    """Transcribe audio from any source."""
    audio_data = read_bytes(source)
    mimetype = filetype.guess(audio_data)
    ...

## No backwards compat, delete transcribe_data, transcribe_file
```


## Usage Examples

```python
from tidyllm.source import as_source, read_bytes, read_text
from tidyllm.connectors.fsspec import open_url
from pathlib import Path

# 1. Reading from file
source = as_source(Path("audio.mp3"))
data = source.read()

# 2. Reading from bytes
audio_bytes = b"..."
source = as_source(audio_bytes)
transcribe_source(source)

# 3. Reading from stdin
source = StdinSource()
transcribe_source(source)

# 4. Reading from URL
source = open_url("github://org:repo@main/audio/sample.wav")
transcribe_source(source)

# 5. Convenience functions
text = read_text("document.txt")
data = read_bytes("image.png")

# 6. In tool functions
def process_audio(source: SourceLike) -> Any:
    src = as_source(source)
    print(f"Processing {src.metadata.get('size', 'unknown')} bytes")
    data = src.read()
    # ... process data
```

## CLI Integration and Serialization

We'll add Pydantic serialization to Sources using __get_pydantic_core_schema__.

File sources will be serialized as { type: "FSSchemaSource", url: "..." }
Byte sources will be auto-serialized { type: "ByteSource", data: Base64 }

   @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Any
    ) -> core_schema.CoreSchema:
        """Serialization logic for Enumerable.

        Enumerables are serialized by first materializing them to `Table` form
        then inferring the underlying model type for the schema.
        """

        def serialize_enumerable(instance: "Enumerable[Any]") -> dict[str, Any]:
            """Serialize Enumerable by materializing to Table."""
            table = instance.to_table()
            schema = table.table_schema().model_json_schema()

            return {"rows": table.rows, "table_schema": schema, "_type": "Table"}

        def deserialize_enumerable(data: Any) -> Any:
            """Deserialize to Table or pass through if already an Enumerable."""
            if isinstance(data, Enumerable):
                return data
            if isinstance(data, dict) and "rows" in data:
                # For now, create Table without schema - it will infer from rows
                return Table(rows=data["rows"], table_schema=None)
            return data

        # Create an any schema that accepts Enumerable instances
        return core_schema.no_info_before_validator_function(
            deserialize_enumerable,
            core_schema.any_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize_enumerable,
                info_arg=False,
                return_schema=core_schema.dict_schema(),
            ),
        )

### CLI Integration

```python
def parse_source_parameter(value: str) -> Source:
    """Parse CLI string parameter to Source."""
    return FSSpecSource(value)

# Update CLI type converter
def get_click_type(annotation: type) -> Any:
    """Convert Python type to Click type."""
    if annotation == Source or get_origin(annotation) == Union and Source in get_args(annotation):
        return click.STRING  # Will be converted via parse_source_parameter
    # ... existing logic
```

### Context and Resource Management

Sources should NOT be context managers directly. Instead, use a resource manager pattern:

```python
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
                source.close()

# Usage in tools
def transcribe_source(source: SourceLike) -> TranscriptionResult:
    with SourceManager() as manager:
        src = manager.register(as_source(source))
        audio_data = src.read()
        mime_type = filetype.guess(audio_data)
        return transcribe_bytes(audio_data, mime_type)
```

## Authentication and OAuth Workflow

We won't handle it ourselves, we'll use for example
[Use gdrivefs for ](https://github.com/fsspec/gdrivefs) connecting to google drive and handling auth.

## Remaining Open Questions

1. **Error Handling**: What exceptions should we define?

Don't define any new exceptions, just let normal exceptions flow and throw
ValueError for the rare cases we need to.

2. **Connector Plugin System**: Should we use a plugin system for different source types?
   ```python
   class SourceConnector(Protocol):
       def can_handle(self, uri: str) -> bool: ...
       def create_source(self, uri: str) -> Source: ...
```

NO - we're using fsspec for pluggability.

## Requirements

1. **Backward Compatibility**: REPLACE ALL EXISTING USAGE. REMOVE ALL DEPRECATED FUNCTIONS.
2. **Type Safety**: Full type annotations with Protocol support
3. **Performance**: Lazy evaluation, minimal overhead
4. **Extensibility**: Easy to add new source types
5. **Integration**: Work with existing tidyllm patterns (caching, registration, context)

## Implementation Plan

1. **Phase 1**: Core Source protocol and basic implementations (FileSource, BytesSource)
2. **Phase 2**: Update existing tools to use Source abstraction
3. **Phase 3**: Add FSSpec connector for remote sources
4. **Phase 4**: Add streaming and advanced features
