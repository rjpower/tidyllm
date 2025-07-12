"""Part types and utilities for tidyllm.

This module provides a unified interface for handling different types of media
and data parts (text, audio, images, etc.) with automatic serialization and
type-specific processing capabilities.
"""

from typing import TypeGuard

# Import specialized Part types - these will register themselves with the registry
from tidyllm.types.part.audio import AudioPart
from tidyllm.types.part.gdrive import GDriveSource
from tidyllm.types.part.image import ImagePart
from tidyllm.types.part.pdf import PdfPart

# Import core Part infrastructure
from tidyllm.types.part.lib import (
    PART_SOURCE_REGISTRY,
    BasicPart,
    BasicPartSource,
    HtmlPart,
    JpegPart,
    LocalFilePartSource,
    Part,
    PartSource,
    PartSourceRegistry,
    PngPart,
    TextPart,
)


# Type guard functions with proper TypeGuard annotations
def is_image_part(part: Part) -> TypeGuard[ImagePart]:
    """Check if a Part is an ImagePart."""
    return part.mime_type.startswith("image/")


def is_audio_part(part: Part) -> TypeGuard[AudioPart]:
    """Check if a Part is an AudioPart."""
    return part.mime_type.startswith("audio/")


def is_text_part(part: Part) -> TypeGuard[TextPart]:
    """Check if a Part is a TextPart."""
    return part.mime_type.startswith("text/")


def is_video_part(part: Part) -> TypeGuard[Part]:
    """Check if a Part is a video part."""
    return part.mime_type.startswith("video/")


def is_pdf_part(part: Part) -> TypeGuard[PdfPart]:
    """Check if a Part is a PdfPart."""
    return part.mime_type.startswith("application/pdf")

# Export the main types for easy access
__all__ = [
    # Core infrastructure
    "Part",
    "BasicPart", 
    "PartSource",
    "PartSourceRegistry",
    "PART_SOURCE_REGISTRY",
    "LocalFilePartSource",
    "BasicPartSource",
    
    # Type aliases for basic parts
    "TextPart",
    "HtmlPart", 
    "PngPart",
    "JpegPart",
    
    # Specialized part types
    "AudioPart",
    "ImagePart", 
    "PdfPart",
    "GDriveSource",
    
    # Type guards
    "is_audio_part",
    "is_image_part", 
    "is_text_part",
    "is_video_part",
    "is_pdf_part",
]