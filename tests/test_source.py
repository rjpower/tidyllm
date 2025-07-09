"""Tests for source library functionality."""

import tempfile
from pathlib import Path

import pytest

from tidyllm.source import (
    ByteSource,
    FileSource,
    FSSpecSource,
    SliceSource,
    SourceManager,
    as_source,
    read_bytes,
    read_text,
)


class TestByteSource:
    """Test ByteSource functionality."""

    def test_byte_source_read(self):
        """Test reading from ByteSource."""
        data = b"Hello world"
        source = ByteSource(data=data)
        
        result = source.read()
        assert result == data
        
    def test_byte_source_partial_read(self):
        """Test partial reading from ByteSource."""
        data = b"Hello world"
        source = ByteSource(data=data)
        
        result = source.read(5)
        assert result == b"Hello"
        
        result = source.read(6)
        assert result == b" world"


class TestSliceSource:
    """Test SliceSource functionality."""

    def test_slice_source_string(self):
        """Test SliceSource with string data."""
        data = "Hello world"
        source = SliceSource(data=data)
        
        result = source.read()
        assert result == data
        
    def test_slice_source_partial_read(self):
        """Test partial reading from SliceSource."""
        data = "Hello world"
        source = SliceSource(data=data)
        
        result = source.read(5)
        assert result == "Hello"


class TestFileSource:
    """Test FileSource functionality."""

    def test_file_source_read(self):
        """Test reading from FileSource."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Hello world")
            f.flush()
            
            source = FileSource(path=Path(f.name))
            result = source.read()
            assert result == b"Hello world"
            
            source.close()
            Path(f.name).unlink()


class TestAsSource:
    """Test as_source conversion function."""

    def test_as_source_path(self):
        """Test converting Path to Source."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Hello world")
            f.flush()
            
            source = as_source(Path(f.name))
            assert isinstance(source, FileSource)
            
            result = source.read()
            assert result == b"Hello world"
            
            source.close()
            Path(f.name).unlink()

    def test_as_source_bytes(self):
        """Test converting bytes to Source."""
        data = b"Hello bytes"
        source = as_source(data)
        assert isinstance(source, ByteSource)
        
        result = source.read()
        assert result == data

    def test_as_source_string_literal(self):
        """Test converting string literal to Source."""
        data = "Hello string"
        source = as_source(data)
        assert isinstance(source, SliceSource)
        
        result = source.read()
        assert result == data

    def test_as_source_string_path(self):
        """Test converting string path to Source."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Hello world")
            f.flush()
            
            # Use absolute path to ensure it's detected as file
            source = as_source(f.name)
            assert isinstance(source, FSSpecSource)
            
            result = source.read()
            assert result == b"Hello world"
            
            source.close()
            Path(f.name).unlink()


class TestReadFunctions:
    """Test convenience read functions."""

    def test_read_bytes_from_path(self):
        """Test read_bytes with file path."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Hello world")
            f.flush()
            
            result = read_bytes(f.name)
            assert result == b"Hello world"
            
            Path(f.name).unlink()

    def test_read_bytes_from_bytes(self):
        """Test read_bytes with bytes data."""
        data = b"Hello bytes"
        result = read_bytes(data)
        assert result == data

    def test_read_text_from_path(self):
        """Test read_text with file path."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Hello world")
            f.flush()
            
            result = read_text(f.name)
            assert result == "Hello world"
            
            Path(f.name).unlink()

    def test_read_text_from_bytes(self):
        """Test read_text with bytes data."""
        data = b"Hello text"
        result = read_text(data)
        assert result == "Hello text"

    def test_read_text_from_string(self):
        """Test read_text with string data."""
        data = "Hello string"
        result = read_text(data)
        assert result == data


class TestSourceManager:
    """Test SourceManager functionality."""

    def test_source_manager_cleanup(self):
        """Test SourceManager properly closes sources."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Hello world")
            f.flush()
            
            with SourceManager() as manager:
                source = manager.register(FileSource(path=Path(f.name)))
                result = source.read()
                assert result == b"Hello world"
            
            # Source should be closed automatically
            Path(f.name).unlink()


class TestPydanticSerialization:
    """Test Pydantic serialization of Sources."""

    def test_byte_source_serialization(self):
        """Test ByteSource serialization."""
        data = b"Hello world"
        source = ByteSource(data=data)
        
        # Test that it can be created and used
        result = source.read()
        assert result == data

    def test_file_source_serialization(self):
        """Test FileSource serialization."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Hello world")
            f.flush()
            
            source = FileSource(path=Path(f.name))
            
            # Test that it can be created and used
            result = source.read()
            assert result == b"Hello world"
            
            source.close()
            Path(f.name).unlink()