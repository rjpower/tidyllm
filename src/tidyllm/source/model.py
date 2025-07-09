"""Source data models."""

import base64
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, TypeVar, overload, runtime_checkable

from pydantic import Base64Bytes, BaseModel, Field
from pydantic_core import core_schema

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
    type: Literal["ByteSource"] = "ByteSource"
    data: Base64Bytes = Field(description="The byte data as a base64 encoded string")

    def __init__(self):
        self._pos = 0

    def read(self, size: int = -1) -> bytes:
        """Read bytes from the source."""
        if size == -1:
            result = self.data[self._pos :]
            self.pos = len(self.data)
        else:
            result = self.data[self._pos : self._pos + size]
            self._pos += len(result)
        return result


class FileSource(BaseModel):
    """Source backed by a file path."""
    type: Literal["FileSource"] = "FileSource"
    path: Path = Field(description="Path to the file")

    def __init__(self, **data):
        super().__init__(**data)
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


class SourceLikeAdapter:
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Any
    ) -> core_schema.CoreSchema:
        def validate_source_like(value: Any) -> Any:
            if isinstance(value, dict):
                # Handle deserialization from JSON
                if value.get("type") == "FileSource":
                    return FileSource(path=Path(value["path"]))
                elif value.get("type") == "ByteSource":
                    return ByteSource(data=base64.b64decode(value["data"]))
            elif isinstance(value, Path):
                return FileSource(path=value)
            elif isinstance(value, bytes):
                return ByteSource(data=value)
            elif hasattr(value, "read"):
                return value
            raise ValueError(f"Cannot convert {type(value)} to SourceLike")

        def serialize_source_like(value: Any) -> dict:
            if isinstance(value, FileSource | ByteSource):
                return value.model_dump(mode="json")
            raise ValueError(f"Unexpected type for SourceLike {type(value)}")

        return core_schema.no_info_after_validator_function(
            validate_source_like,
            core_schema.union_schema([
                FileSource.__pydantic_core_schema__,
                ByteSource.__pydantic_core_schema__,
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize_source_like,
                return_schema=core_schema.dict_schema(),
            ),
        )


# Apply the adapter to SourceLike
SourceLike = Annotated[Path | str | bytes | Source, SourceLikeAdapter]
