"""Source library for unified data source abstraction.

Provides a unified interface for reading from various data sources including
files, bytes, stdin, and remote resources.
"""

# Import all core components for backwards compatibility
from tidyllm.source.gdrive import GDriveSource
from tidyllm.source.lib import (
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
