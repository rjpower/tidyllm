"""Tests for source library functionality."""

import tempfile
from pathlib import Path

import pytest

from tidyllm.source import (
    ByteSource,
    FileSource,
    SourceManager,
    as_source,
    read_bytes,
    read_text,
)


class TestAsSource:
    """Test as_source conversion function."""

    def test_as_source_path(self):
        """Test converting Path to Source."""
        with tempfile.NamedTemporaryFile(mode="w") as f:
            f.write("Hello world")
            f.flush()

            source = as_source(Path(f.name))
            assert isinstance(source, FileSource)

            result = source.read()
            assert result == b"Hello world"

            source.close()

    def test_as_source_bytes(self):
        """Test converting bytes to Source."""
        data = b"Hello bytes"
        source = as_source(data)
        assert isinstance(source, ByteSource)

        result = source.read()
        assert result == data

    def test_as_source_string_path(self):
        """Test converting string path to Source."""
        with tempfile.NamedTemporaryFile(mode="w") as f:
            f.write("Hello world")
            f.flush()

            # Use absolute path to ensure it's detected as file
            source = as_source(f.name)
            assert isinstance(source, FileSource)

            result = source.read()
            assert result == b"Hello world"

            source.close()


class TestReadFunctions:
    """Test convenience read functions."""

    def test_read_bytes_from_path(self):
        """Test read_bytes with file path."""
        with tempfile.NamedTemporaryFile(mode="w") as f:
            f.write("Hello world")
            f.flush()

            result = read_bytes(f.name)
            assert result == b"Hello world"

    def test_read_bytes_from_bytes(self):
        """Test read_bytes with bytes data."""
        data = b"Hello bytes"
        result = read_bytes(data)
        assert result == data

    def test_read_text_from_path(self):
        """Test read_text with file path."""
        with tempfile.NamedTemporaryFile(mode="w") as f:
            f.write("Hello world")
            f.flush()

            result = read_text(f.name)
            assert result == "Hello world"

    def test_read_text_from_bytes(self):
        """Test read_text with bytes data."""
        data = b"Hello text"
        result = read_text(data)
        assert result == "Hello text"


class TestSourceManager:
    """Test SourceManager functionality."""

    def test_source_manager_cleanup(self):
        """Test SourceManager properly closes sources."""
        with tempfile.NamedTemporaryFile(mode="w") as f:
            f.write("Hello world")
            f.flush()

            with SourceManager() as manager:
                source = manager.register(FileSource(path=Path(f.name)))
                result = source.read()
                assert result == b"Hello world"


class TestPydanticSerialization:
    """Test Pydantic serialization of Sources."""

    def test_byte_source_serialization(self):
        """Test ByteSource serialization."""
        data = b"Hello world"
        source = as_source(data)

        # Test that it can be created and used
        result = source.read()
        assert result == data

    def test_file_source_serialization(self):
        """Test FileSource serialization."""
        with tempfile.NamedTemporaryFile(mode="w") as f:
            f.write("Hello world")
            f.flush()

            source = FileSource(path=Path(f.name))

            # Test that it can be created and used
            result = source.read()
            assert result == b"Hello world"

            source.close()

    def test_byte_source_dict_serialization(self):
        """Test ByteSource serialization to dict format."""
        from pydantic import BaseModel
        from tidyllm.source.model import SourceLike

        class TestModel(BaseModel):
            source: SourceLike

        data = b"Hello world"
        source = as_source(data)

        # Test serialization to dict
        serialized = source.model_dump(mode="json")
        assert serialized["type"] == "ByteSource"
        assert "data" in serialized

        # Test deserialization through SourceLike adapter
        model = TestModel(source=serialized)
        deserialized = model.source
        assert isinstance(deserialized, ByteSource)
        assert deserialized.read() == data

    def test_file_source_dict_serialization(self):
        """Test FileSource serialization to dict format."""
        from pydantic import BaseModel
        from tidyllm.source.model import SourceLike

        class TestModel(BaseModel):
            source: SourceLike

        with tempfile.NamedTemporaryFile(mode="w") as f:
            f.write("Hello world")
            f.flush()

            source = FileSource(path=Path(f.name))

            # Test serialization to dict
            serialized = source.model_dump(mode="json")
            assert serialized["type"] == "FileSource"
            assert serialized["path"] == str(Path(f.name))

            # Test deserialization through SourceLike adapter
            model = TestModel(source=serialized)
            deserialized = model.source
            assert isinstance(deserialized, FileSource)
            assert deserialized.read() == b"Hello world"

            source.close()
            deserialized.close()
