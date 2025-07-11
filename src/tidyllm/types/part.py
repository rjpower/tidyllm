"""Core Part type for tidyllm.
Tracks a mime_type and a data payload, with subtypes to simplify type-based
pattern matching and annotation.
Adapted from the google.genai package.
"""

import base64
import uuid
from typing import Any, TypeGuard

from pydantic import (
    Base64Bytes,
    BaseModel,
    model_serializer,
    model_validator,
)
from pydantic_core import Url

from tidyllm.context import get_tool_context


class Part(BaseModel):
    mime_type: str
    data: Base64Bytes = b""
    url: str = ""

    @property
    def text(self):
        return self.data.decode()

    @property
    def base64_bytes(self):
        return base64.b64encode(self.data)

    @model_validator(mode="before")
    @classmethod
    def validate_part_input(cls, value: Any) -> Any:
        """Validate Part input: Url | {"mimetype": ..., "data": ...}"""
        # Handle URL format (ref:// URLs)
        if isinstance(value, Url):
            value = str(value)

        if isinstance(value, str) and value.startswith("ref://"):
            ctx = get_tool_context()
            part = ctx.get_ref(value)
            # Return the dict representation to avoid nested validation issues
            return {
                "mime_type": part.mime_type,
                "data": base64.b64encode(part.data).decode(),
                "url": part.url,
            }
        # Handle dict format with mimetype/data
        if isinstance(value, dict):
            # If it's a ref:// dict from serialization, deserialize from context
            if "url" in value and value["url"].startswith("ref://"):
                ref_id = value["url"]
                ctx = get_tool_context()
                part = ctx.get_ref(ref_id)
                return {
                    "mime_type": part.mime_type,
                    "data": base64.b64encode(part.data).decode(),
                    "url": part.url,
                }

            # Handle direct {"mimetype": ..., "data": ...} format
            if "mimetype" in value and "data" in value:
                data = value["data"]
                # If data is a string, assume it's base64 encoded
                if isinstance(data, str):
                    data_bytes = base64.b64decode(data)
                elif isinstance(data, bytes):
                    data_bytes = data
                else:
                    raise ValueError(f"Data must be string or bytes, got {type(data)}")

                return {
                    "mime_type": value["mimetype"],
                    "data": base64.b64encode(data_bytes).decode(),
                    "url": value.get("url", ""),
                }

            # Handle standard Part dict format
            if "mime_type" in value:
                return value

        # If it's a regular string (not ref://), treat as text content
        if isinstance(value, str):
            return {
                "mime_type": "text/plain",
                "data": base64.b64encode(value.encode()).decode(),
                "url": "",
            }

        # Otherwise, let Pydantic handle normal validation
        return value

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """
        Customize schema to show Part accepts: Url | {"mimetype": ..., "data": ...}
        """
        # Get the original schema
        original_schema = handler(core_schema)

        # Create the union schema
        return {
            "title": "Part",
            "anyOf": [
                {
                    "type": "string",
                    "format": "uri",
                    "description": "Reference URL to a stored Part (ref://...)",
                },
                {
                    "type": "object",
                    "properties": {
                        "mimetype": {
                            "type": "string",
                            "description": "MIME type of the content",
                        },
                        "data": {
                            "anyOf": [
                                {
                                    "type": "string",
                                    "description": "Base64 encoded content",
                                },
                                {
                                    "type": "string",
                                    "format": "binary",
                                    "description": "Raw bytes content",
                                },
                            ]
                        },
                    },
                    "required": ["mimetype", "data"],
                    "additionalProperties": False,
                },
                original_schema,
            ],
        }

    @model_serializer
    def serialize_part(self) -> dict[str, Any]:
        """Serialize Part to RemotePart with ref:// URL."""
        # Generate a unique reference ID
        ref_id = f"ref://{uuid.uuid4()}"

        # Store the Part in the tool context
        ctx = get_tool_context()
        ctx.set_ref(ref_id, self)

        # Create preview data
        if is_text_content_part(self):
            data_preview = self.data[:128].decode()
        else:
            data_preview = base64.b85encode(self.data[:128]).decode()

        if len(self.data) > 512:
            data_preview += "[truncated]..."

        # Return RemotePart as dict
        return {
            "url": ref_id,
            "mime_type": self.mime_type,
            "data": data_preview,
            "note": """
This is a reference to a resource on the server.
You can pass it to any function which expects a `Part` by passing the URL directly e.g. tool("ref://...")
You can fetch the raw content using `fetch_part_content`.
""",
        }


class PngPart(Part):
    mime_type: str = "image/png"

    @staticmethod
    def from_bytes(data: bytes):
        return PngPart(data=base64.b64encode(data))


class JpegPart(Part):
    mime_type: str = "image/jpeg"


class WavPart(Part):
    mime_type: str = "audio/wav"


class Mp3Part(Part):
    mime_type: str = "audio/mp3"


class MovPart(Part):
    mime_type: str = "audio/mov"


class HtmlPart(Part):
    mime_type: str = "text/html"


class TextPart(Part):
    mime_type: str = "text/plain"

ImagePart = PngPart | JpegPart
TextContentPart = TextPart | HtmlPart


def is_image_part(part: Part) -> TypeGuard[PngPart | JpegPart]:
    return part.mime_type in ("image/png", "image/jpeg")


def is_audio_part(part: Part) -> TypeGuard[WavPart | Mp3Part | MovPart]:
    return part.mime_type in ("audio/wav", "audio/mp3", "audio/mov")


def is_text_content_part(part: Part) -> TypeGuard[HtmlPart | TextPart]:
    return part.mime_type in ("text/html", "text/plain")
