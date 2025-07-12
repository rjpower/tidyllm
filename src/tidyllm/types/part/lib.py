"""Core Part type for tidyllm.

Provides registry to resolve a `Part` from a URL during validation and
type guards for common part types. Typical usage is to define your tool
as accepting some kind of Part:

@register
def my_image_analyzer(image: ImagePart):
  img = image_part_to_pil(image)

When using the CLI/API/MCP adapters, your tool will automatically accept
Parts serialized as JSON, but also data: and any custom URLs which have
been loaded. So for example:

tidyllm my_image_analyzer --image=file://foo.png

Will automatically load the image file from disk.
"""

import base64
from pathlib import Path
from typing import Any, Protocol

import filetype
from pydantic import (
    Base64Bytes,
    BaseModel,
    model_validator,
)
from pydantic_core import Url

from tidyllm.types.linq import Enumerable, Table


class Part(BaseModel):
    mime_type: str

    @classmethod
    def from_url(cls, url: str | Url) -> "Enumerable[Part]":
        """Create Part stream from URL using registry."""
        if isinstance(url, str):
            url = Url(url)

        if url.scheme == "data":
            # handle the common data:image/png;base64,... format
            mime_type, b64 = url.path.split(";", maxsplit=1)
            b64_prefix, payload = b64.split(",", maxsplit=1)
            assert b64_prefix == "base64"
            part = PART_SOURCE_REGISTRY.from_dict({"mime_type": mime_type, "data": payload})
            return Table.from_rows([part])

        return PART_SOURCE_REGISTRY.from_url(url)

    @classmethod
    def from_value(cls, value: Any) -> Any:
        """Validate Part input: str | Url | {"mimetype": ..., "data": ...}"""
        if isinstance(value, dict):
            return PART_SOURCE_REGISTRY.from_dict(value)

        if isinstance(value, str):
            value = Url(value)

        assert isinstance(value, Url), f"Unknown value type for Part {value}."
        return cls.from_url(value)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """
        Customize schema to show Part accepts: Url | {"mimetype": ..., "data": ...}
        """
        original_schema = handler(core_schema)
        return {
            "title": "Part",
            "anyOf": [
                {
                    "type": "string",
                    "format": "uri",
                    "description": "Resource URL to load into a part",
                },
                original_schema,
            ],
        }


class BasicPart(Part):
    """A basic Part implementation that stores raw data."""

    data: Base64Bytes

    @classmethod
    def from_base64(cls, data: bytes, mime_type: str) -> "BasicPart":
        """Create BasicPart from raw bytes."""
        return cls.model_construct(mime_type=mime_type, data=data)

    @property
    def text(self):
        """Get text content by decoding data."""
        return self.data.decode()

    @property
    def base64_bytes(self):
        """Get base64-encoded bytes."""
        return base64.b64encode(self.data)


TextPart = BasicPart
HtmlPart = BasicPart
PngPart = BasicPart
JpegPart = BasicPart


class BasicPartSource:
    """PartSource for basic data types."""

    def from_url(self, url: Url) -> "Enumerable[Part]":
        """BasicPartSource doesn't support URL loading - only dictionary creation."""
        raise NotImplementedError("BasicPartSource only supports from_dict, not from_url")

    def from_dict(self, d: dict) -> Part:
        """Create BasicPart from dictionary."""
        return BasicPart(**d)


class PartSource(Protocol):
    """Protocol for sources that can stream Parts."""

    def from_url(self, url: Url) -> "Enumerable[Part]": ...

    def from_dict(self, dict: dict) -> "Part": ...


class PartSourceRegistry(dict[str, PartSource]):
    """Registry mapping URL schemes to PartSource implementations."""

    def register_scheme(self, scheme: str, source: PartSource):
        """Register a PartSource for a URL scheme."""
        self[scheme] = source

    def register_mimetype(self, mimetype: str, source: PartSource):
        self[mimetype] = source

    def from_url(self, url: Url) -> "Enumerable[Part]":
        """Create a Part stream from a URL."""
        if url.scheme not in self:
            raise KeyError(f"Unregistered URL scheme: {url.scheme}")
        return self[url.scheme].from_url(url)

    def from_dict(self, d: dict) -> "Part":
        mime_type = d["mime_type"]

        # Try exact match first
        if mime_type in self:
            return self[mime_type].from_dict(d)

        # If no exact match, try base mime type (strip parameters)
        base_mime_type = mime_type.split(";")[0]
        if base_mime_type in self:
            return self[base_mime_type].from_dict(d)

        # No handler found - fall back to BasicPart
        data_str = d.get("data", "")
        if isinstance(data_str, str):
            data_bytes = data_str.encode() if data_str else b""
        else:
            data_bytes = data_str

        return BasicPart.model_construct(mime_type=mime_type, data=data_bytes)


PART_SOURCE_REGISTRY = PartSourceRegistry()


class LocalFilePartSource:
    def __init__(self, allowed_dirs: list[Path]):
        self._allowed_dirs = [dir.resolve() for dir in allowed_dirs]

    def from_url(self, url: Url) -> "Enumerable[Part]":
        path = Path(url.path).resolve()
        for dir in self._allowed_dirs:
            if path.is_relative_to(dir):
                data = path.read_bytes()
                mime_type = filetype.guess_mime(data) or "application/octet-stream"

                # Check if there's a specific handler for this mime type
                base_mime = mime_type.split(";")[0]
                if base_mime in PART_SOURCE_REGISTRY:
                    # Let the specific handler create the part
                    part_dict = {
                        "mime_type": mime_type,
                        "data": base64.b64encode(data).decode(),
                    }
                    part = PART_SOURCE_REGISTRY.from_dict(part_dict)
                else:
                    # Fall back to BasicPart for unregistered types
                    part = BasicPart(mime_type=mime_type, data=base64.b64encode(data))

                return Table.from_rows([part])

        raise ValueError(
            f"URL path {path} not found in allowed set of directories {self._allowed_dirs}"
        )

    def from_dict(self, d: dict) -> "Part":
        """LocalFilePartSource doesn't support from_dict - only URL loading.""" 
        raise NotImplementedError("LocalFilePartSource only supports from_url, not from_dict")


PART_SOURCE_REGISTRY.register_scheme("file", LocalFilePartSource([Path(".")]))

# Register BasicPartSource for common mime types
basic_part_source = BasicPartSource()
PART_SOURCE_REGISTRY.register_mimetype("text/plain", basic_part_source)
PART_SOURCE_REGISTRY.register_mimetype("text/html", basic_part_source)
PART_SOURCE_REGISTRY.register_mimetype("text/css", basic_part_source)
PART_SOURCE_REGISTRY.register_mimetype("text/javascript", basic_part_source)
PART_SOURCE_REGISTRY.register_mimetype("application/json", basic_part_source)
PART_SOURCE_REGISTRY.register_mimetype("application/pdf", basic_part_source)
