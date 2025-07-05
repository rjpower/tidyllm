"""File manipulation tools for reading and writing files."""

import sys
from pathlib import Path

from tidyllm.adapters.cli import cli_main
from tidyllm.registry import register
from tidyllm.tools.context import ToolContext


@register()
def read_file(path: str | Path) -> bytes:
    """Read file contents as binary data.

    Args:
        path: Path to file to read

    Returns:
        Binary file contents

    Examples:
        file.read 'config.json'
        file.read '/path/to/audio.mp3'
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return file_path.read_bytes()


@register()
def write_file(path: str | Path, data: bytes | None) -> str:
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
    cli_main([read_file, write_file], context_cls=ToolContext)
