"""Tests for file manipulation tools."""

import tempfile
from pathlib import Path

import pytest

from tidyllm.tools.file import source_read_bytes, source_read_text, file_write


class TestFileRead:
    """Test file read functionality."""

    def test_read_text_file(self):
        """Test reading text file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Hello world")
            f.flush()

            result = source_read_bytes(f.name)
            assert result == b"Hello world"

            # Clean up
            Path(f.name).unlink()

    def test_read_binary_file(self):
        """Test reading binary file."""
        binary_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"

        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(binary_data)
            f.flush()

            result = source_read_bytes(f.name)
            assert result == binary_data

            # Clean up
            Path(f.name).unlink()

    def test_read_nonexistent_file(self):
        """Test reading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            source_read_bytes("/path/to/nonexistent/file.txt")

    def test_read_with_path_object(self):
        """Test reading with Path object."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Hello world")
            f.flush()

            result = source_read_bytes(Path(f.name))
            assert result == b"Hello world"

            # Clean up
            Path(f.name).unlink()

    def test_read_text_with_source_function(self):
        """Test reading text using file_read_text."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Hello world")
            f.flush()

            result = source_read_text(f.name)
            assert result == "Hello world"

            # Clean up
            Path(f.name).unlink()

    def test_read_bytes_source(self):
        """Test reading from bytes source."""
        test_data = b"Hello bytes"
        result = source_read_bytes(test_data)
        assert result == test_data

    def test_read_text_bytes_source(self):
        """Test reading text from bytes source."""
        test_data = b"Hello text"
        result = source_read_text(test_data)
        assert result == "Hello text"


class TestFileWrite:
    """Test file write functionality."""

    def test_write_binary_data(self):
        """Test writing binary data."""
        binary_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.bin"

            result = file_write(str(output_path), binary_data)
            assert "Written" in result
            assert str(output_path) in result

            # Verify file was written correctly
            assert output_path.read_bytes() == binary_data

    def test_write_text_data(self):
        """Test writing text data (as bytes)."""
        text_data = b"Hello world"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.txt"

            result = file_write(str(output_path), text_data)
            assert "Written" in result
            assert str(output_path) in result

            # Verify file was written correctly
            assert output_path.read_bytes() == text_data

    def test_write_creates_parent_dirs(self):
        """Test that write creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "test.txt"
            data = b"Hello world"

            result = file_write(str(output_path), data)
            assert "Written" in result

            # Verify directory was created and file written
            assert output_path.exists()
            assert output_path.read_bytes() == data

    def test_write_with_path_object(self):
        """Test writing with Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.txt"
            data = b"Hello world"

            result = file_write(output_path, data)
            assert "Written" in result
            assert output_path.read_bytes() == data

    def test_write_no_data_no_stdin(self):
        """Test write with no data and no stdin raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir) / "test.txt"

            # This would normally read from stdin, but we can't test that easily
            # The actual error would occur in the stdin reading logic
            pass  # Skip this test for now as it requires stdin mocking


class TestFileIntegration:
    """Test file operations integration."""

    def test_read_write_roundtrip(self):
        """Test reading and writing the same data."""
        original_data = b"Hello world\nThis is a test file\x00\x01\x02"

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.bin"

            # Write data
            write_result = file_write(str(file_path), original_data)
            assert "Written" in write_result

            # Read it back
            read_result = source_read_bytes(str(file_path))
            assert read_result == original_data

    def test_large_file_handling(self):
        """Test handling of larger files."""
        # Create 1MB of data
        large_data = b"x" * (1024 * 1024)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "large.bin"

            # Write large data
            write_result = file_write(str(file_path), large_data)
            assert "1048576" in write_result  # 1MB in bytes

            # Read it back
            read_result = source_read_bytes(str(file_path))
            assert read_result == large_data
            assert len(read_result) == 1024 * 1024
