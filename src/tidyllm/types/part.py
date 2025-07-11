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
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias, TypeGuard

import filetype
from pydantic import (
    Base64Bytes,
    BaseModel,
    model_validator,
)
from pydantic_core import Url


class PartRegistry(dict):
    def register_part_creator(self, scheme: str, creator: Callable[[Url], "Part"]):
        self[scheme] = creator

    def create(self, url: Url) -> "Part":
        return self[url.scheme](url)


PART_REGISTRY = PartRegistry()


class Part(BaseModel):
    mime_type: str
    data: Base64Bytes = b""

    @property
    def text(self):
        return self.data.decode()

    @property
    def base64_bytes(self):
        return base64.b64encode(self.data)

    @model_validator(mode="before")
    @classmethod
    def validate_part_input(cls, value: Any) -> Any:
        """Validate Part input: str | Url | {"mimetype": ..., "data": ...}"""
        if isinstance(value, dict):
            return value

        if isinstance(value, str):
            value = Url(value)

        assert isinstance(value, Url), f"Unknown value type for Part {value}."

        if value.scheme == "data":
            # handle the common data:image/png;base64,... format
            mime_type, b64 = value.path.split(";", maxsplit=1)
            b64_prefix, payload = b64.split(",", maxsplit=1)
            assert b64_prefix == "base64"
            return {"mime_type": mime_type, "data": payload}

        if value.scheme not in PART_REGISTRY:
            raise KeyError(f"Unregistered URL type {value.scheme} from {value}.")

        return PART_REGISTRY[value](value)

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


class LocalFileHandler:
    def __init__(self, allowed_dirs: list[Path]):
        self._allowed_dirs = [dir.resolve() for dir in allowed_dirs]

    def __call__(self, url: Url) -> Part:
        path = Path(url.path).resolve()
        for dir in self._allowed_dirs:
            if path.is_relative_to(dir):
                data = path.read_bytes()
                mime_type = filetype.guess_mime(data)
                return Part(mime_type=mime_type, data=base64.b64encode(data))

        raise ValueError(
            f"URL path {path} not found in allowed set of directories {self._allowed_dirs}"
        )


PART_REGISTRY.register_part_creator("file", LocalFileHandler([Path(".")]))

ImagePart: TypeAlias = Part
AudioPart: TypeAlias = Part
TextPart: TypeAlias = Part
PngPart: TypeAlias = Part
HtmlPart: TypeAlias = Part


def is_image_part(part: Part) -> TypeGuard[ImagePart]:
    return part.mime_type in ("image/png", "image/jpeg")


def is_audio_part(part: Part) -> TypeGuard[AudioPart]:
    return part.mime_type in ("audio/wav", "audio/mp3", "audio/mov")


def is_text_content_part(part: Part) -> TypeGuard[TextPart]:
    return part.mime_type in ("text/html", "text/plain")
