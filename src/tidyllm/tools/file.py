"""File manipulation tools for reading and writing files."""

import sys
from pathlib import Path

from tidyllm.adapters.cli import cli_main
from tidyllm.registry import register
from tidyllm.tools.context import ToolContext
from tidyllm.types.source import SourceLike, read_bytes, read_text


@register()
def source_read_bytes(source: SourceLike) -> bytes:
    """Read file contents as binary data from any source.

    Args:
        source: Source to read from (file path, URL, bytes, etc.)

    Returns:
        Binary file contents

    Examples:
        file.read 'config.json'
        file.read '/path/to/audio.mp3'
        file.read 'https://example.com/data.json'
    """
    return read_bytes(source)


@register()
def source_read_text(source: SourceLike, encoding: str = "utf-8") -> str:
    """Read file contents as text from any source.

    Args:
        source: Source to read from (file path, URL, bytes, etc.)
        encoding: Text encoding to use (default: utf-8)

    Returns:
        Text file contents

    Examples:
        file.read_text 'document.txt'
        file.read_text 'https://example.com/data.txt'
    """
    return read_text(source, encoding)


@register()
def file_write(path: str | Path, data: bytes | None) -> str:
    """Write binary data from stdin to file.

    Args:
        path: Path to file to write

    Returns:
        Success message with file path

    Examples:
        echo "Hello" | file.write 'output.txt'
        some_binary_command | file.write 'output.bin'
    """
    file_path = Path(path)

    if sys.stdin.isatty():
        raise ValueError("No input provided. Use stdin.")

    if data is None:
        data = sys.stdin.buffer.read()

    # Create parent directories if needed
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write binary data
    file_path.write_bytes(data)

    return f"Written {len(data)} bytes to {file_path}"


if __name__ == "__main__":
    cli_main([source_read_bytes, source_read_text, file_write], context_cls=ToolContext)
