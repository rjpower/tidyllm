"""Core Part type for tidyllm.
Tracks a mime_type and a data payload, with subtypes to simplify type-based
pattern matching and annotation.
Adapted from the google.genai package.
"""

import base64
from typing import TypeGuard

from pydantic import Base64Bytes, BaseModel


class Part(BaseModel):
    mime_type: str
    data: Base64Bytes

    @property
    def text(self):
        return self.data.decode()
    
    @property
    def base64_bytes(self):
        return base64.b64encode(self.data)

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
