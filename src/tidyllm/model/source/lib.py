"""Core source library functions."""

import base64
from pathlib import Path
from types import UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    Protocol,
    Union,
    cast,
    get_args,
    get_origin,
    runtime_checkable,
)

from pydantic import Base64Bytes, BaseModel, Field
from pydantic_core import core_schema

from tidyllm.model.source.gdrive import GDriveSource


@runtime_checkable
class Source(Protocol):
    """Base protocol for all data sources."""

    def read(self, size: int = -1) -> bytes:
        """Read data from the source."""
        ...


class ByteSource(BaseModel):
    """Source backed by byte data."""

    type: Literal["ByteSource"] = "ByteSource"
    data: Base64Bytes = Field(description="The byte data as a base64 encoded string")

    def model_post_init(self, _ctx: Any):
        self._pos = 0

    def read(self, size: int = -1) -> bytes:
        """Read bytes from the source."""
        if size == -1:
            result = self.data[self._pos :]
            self._pos = len(self.data)
        else:
            result = self.data[self._pos : self._pos + size]
            self._pos += len(result)
        return result


class FileSource(BaseModel):
    """Source backed by a file path."""

    type: Literal["FileSource"] = "FileSource"
    path: Path = Field(description="Path to the file")

    def model_post_init(self, _ctx: Any):
        self._file = None

    def _open(self):
        """Lazily open the file."""
        if self._file is None:
            self._file = open(self.path, "rb")

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


def url_mapping(url: str):
    if url.startswith("gdrive://"):
        return GDriveSource(url=url)
    if url.startswith("file://"):
        return FileSource(path=Path(url.split("file://")[1]))
    raise ValueError(f"Unknonw source type {url}")


def as_source(data: Any) -> Source:
    """Convert various input types to a Source."""
    if hasattr(data, 'read') and callable(data.read): # type: ignore
        return cast(Source, data)
    elif isinstance(data, Path):
        if "://" in str(data):
            return url_mapping(str(data))
        return FileSource(path=data)
    elif isinstance(data, str):
        if "://" in data:
            return url_mapping(data)
        return FileSource(path=Path(data))
    elif isinstance(data, bytes):
        return ByteSource(data=base64.b64encode(data))
    else:
        raise TypeError(f"Cannot convert {str(data)[:100]} to Source")


class SourceLikeAdapter:
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Any
    ) -> core_schema.CoreSchema:

        def validate_source_like(value: Any) -> Any:
            if isinstance(value, dict):
                if value.get("type") == "FileSource":
                    return FileSource(path=Path(value["path"]))
                elif value.get("type") == "ByteSource":
                    return ByteSource(data=value["data"])
                else:
                    raise ValueError(f"Cannot convert dict {value} to SourceLike")
            else:
                return as_source(value)

        def serialize_source_like(
            value: Any, info: core_schema.SerializationInfo
        ) -> dict:
            if isinstance(value, FileSource | ByteSource):
                return value.model_dump(mode=info.mode)
            elif isinstance(value, Path):
                return FileSource(path=value).model_dump(mode=info.mode)
            raise ValueError(f"Unexpected type for SourceLike {type(value)}")

        return core_schema.no_info_after_validator_function(
            validate_source_like,
            core_schema.union_schema(
                [
                    core_schema.is_instance_schema(Path),
                    core_schema.str_schema(),
                    core_schema.bytes_schema(),
                    FileSource.__pydantic_core_schema__,
                    ByteSource.__pydantic_core_schema__,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize_source_like,
                info_arg=True,
                return_schema=core_schema.dict_schema(),
            ),
        )


SourceLike = Annotated[Path | Source, SourceLikeAdapter]


def is_source_like_type(tp: Any) -> bool:
    if get_origin(tp) is Annotated:
        tp = get_args(tp)[0]
        return is_source_like_type(tp)

    if get_origin(tp) is Union:
        return any(is_source_like_type(arg) for arg in get_args(tp))

    if isinstance(tp, UnionType):
        return any(is_source_like_type(arg) for arg in tp.__args__)

    if tp == Path:
        return True

    return hasattr(tp, "read") and callable(tp.read)


def read_bytes(source: SourceLike) -> bytes:
    """Convenience function to read all bytes from a source-like object."""
    if isinstance(source, bytes):
        return source

    return as_source(source).read()


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
