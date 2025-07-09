"""Source library for unified data source abstraction.

Provides a unified interface for reading from various data sources including
files, bytes, stdin, and remote resources.
"""

# Import all core components for backwards compatibility
from tidyllm.source.gdrive import GDriveSource
from tidyllm.source.lib import (
    SourceManager,
    as_source,
    is_source_like,
    read_bytes,
    read_text,
)
from tidyllm.source.model import (
    ByteSource,
    FileSource,
    Source,
    SourceLike,
)

__all__ = [
    "Source",
    "SourceLike",
    "ByteSource",
    "FileSource",
    "GDriveSource",
    "as_source",
    "read_bytes",
    "read_text",
    "is_source_like",
    "SourceManager",
]
