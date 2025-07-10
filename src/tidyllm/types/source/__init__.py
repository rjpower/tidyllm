"""Source library for unified data source abstraction.

Provides a unified interface for reading from various data sources including
files, bytes, stdin, and remote resources.
"""

# Import all core components for backwards compatibility
from tidyllm.types.source.gdrive import GDriveSource
from tidyllm.types.source.lib import (
    ByteSource,
    FileSource,
    Source,
    SourceLike,
    SourceManager,
    as_source,
    is_source_like_type,
    read_bytes,
    read_text,
)

__all__ = [
    "GDriveSource",
    "ByteSource",
    "FileSource",
    "Source",
    "SourceLike",
    "SourceManager",
    "as_source",
    "is_source_like_type",
    "read_bytes",
    "read_text",
]

