"""Source data models."""

import base64
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import Base64Bytes, BaseModel, Field
from pydantic_core import core_schema


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
            elif isinstance(value, Path):
                return FileSource(path=value)
            elif hasattr(value, "read"):
                return value
            raise ValueError(f"Cannot convert {type(value)} to SourceLike")

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
