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
            data_bytes = base64.b64decode(payload)
            part = PART_SOURCE_REGISTRY.from_bytes(mime_type, data_bytes)
            return Table.from_rows([part])

        return PART_SOURCE_REGISTRY.from_url(url)

    @classmethod
    def from_bytes(cls, mimetype: str, data: bytes) -> "Part":
        """Create Part from mimetype and raw bytes."""
        return PART_SOURCE_REGISTRY.from_bytes(mimetype, data)

    @classmethod
    def from_json(cls, data: dict) -> "Part":
        """Create Part from JSON dictionary (serialized Part)."""
        return PART_SOURCE_REGISTRY.from_json(data)

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


class UrlLoader(Protocol):
    """Protocol for sources that can stream Parts."""

    def __call__(self, url: Url) -> Enumerable[Part]: ...


class MimeLoader(Protocol):
    def from_json(self, d: dict[str, Any]) -> Part:
        """Load a Part from a serialized dictionary (e.g. Pydantic representation)."""
        raise NotImplementedError

    def from_bytes(self, mime_type: str, data: bytes) -> Part:
        """If supported, load `Part` from the specified data."""
        raise NotImplementedError


class PartSourceRegistry:
    """Registry mapping URL schemes to PartSource implementations."""
    _url_loaders: dict[str, UrlLoader] = {}
    _mime_loaders: dict[str, MimeLoader] = {}

    def register_scheme(self, scheme: str, loader: UrlLoader):
        self._url_loaders[scheme] = loader

    def register_mimetype(self, mimetype: str, loader: MimeLoader):
        self._mime_loaders[mimetype] = loader

    def from_url(self, url: Url) -> "Enumerable[Part]":
        """Create a Part stream from a URL."""
        if url.scheme not in self._url_loaders:
            raise KeyError(f"Unregistered URL scheme: {url.scheme}")
        return self._url_loaders[url.scheme](url)

    def _find_loader(self, mime_type: str) -> MimeLoader | None:
        if mime_type in self._mime_loaders:
            return self._mime_loaders[mime_type]
        base_mime_type = mime_type.split(";")[0]
        if base_mime_type in self._mime_loaders:
            return self._mime_loaders[base_mime_type]

        return None

    def from_bytes(self, mime_type: str, data: bytes) -> "Part":
        """Create a Part from mimetype and raw bytes."""
        loader = self._find_loader(mime_type)
        if loader:
            return loader.from_bytes(mime_type, data)

        return BasicPart.from_base64(base64.b64encode(data), mime_type)

    def from_json(self, d: dict) -> "Part":
        mime_type = d["mime_type"]
        loader = self._find_loader(mime_type)
        if loader:
            return loader.from_json(d)
        return BasicPart.model_validate(d)


PART_SOURCE_REGISTRY = PartSourceRegistry()


class BasicPartLoader:
    """MimeLoader for basic data types."""

    def from_json(self, d: dict[str, Any]) -> Part:
        """Create BasicPart from JSON dictionary."""
        return BasicPart.model_validate(d)

    def from_bytes(self, mime_type: str, data: bytes) -> Part:
        """Create BasicPart from raw bytes."""
        return BasicPart.from_base64(data, mime_type)


class LocalFilePartSource:
    def __init__(self, allowed_dirs: list[Path]):
        self._allowed_dirs = [dir.resolve() for dir in allowed_dirs]

    def __call__(self, url: Url) -> "Enumerable[Part]":
        path = Path(url.path).resolve()
        for dir in self._allowed_dirs:
            if path.is_relative_to(dir):
                data = path.read_bytes()
                mime_type = filetype.guess_mime(data) or "application/octet-stream"

                # Use registry to create appropriate Part type
                part = PART_SOURCE_REGISTRY.from_bytes(mime_type, data)

                return Table.from_rows([part])

        raise ValueError(
            f"URL path {path} not found in allowed set of directories {self._allowed_dirs}"
        )


PART_SOURCE_REGISTRY.register_scheme("file", LocalFilePartSource([Path(".")]))

# Register BasicPartLoader for common mime types
basic_part_loader = BasicPartLoader()
PART_SOURCE_REGISTRY.register_mimetype("text/plain", basic_part_loader)
PART_SOURCE_REGISTRY.register_mimetype("text/html", basic_part_loader)
PART_SOURCE_REGISTRY.register_mimetype("text/css", basic_part_loader)
PART_SOURCE_REGISTRY.register_mimetype("text/javascript", basic_part_loader)
PART_SOURCE_REGISTRY.register_mimetype("application/json", basic_part_loader)
