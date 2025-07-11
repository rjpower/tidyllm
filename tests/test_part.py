"""Tests for Part serialization and validation."""

import base64
import json
import pytest

from tidyllm.context import set_tool_context
from tidyllm.tools.context import ToolContext
from tidyllm.types.part import Part, TextPart, PngPart


@pytest.fixture
def tool_context():
    """Provide a fresh ToolContext for each test."""
    return ToolContext()


class TestPartValidation:
    """Test Part validation from different input formats."""

    def test_part_from_string(self, tool_context):
        """Test creating Part from plain string."""
        with set_tool_context(tool_context):
            part = Part.model_validate("Hello World")
            
            assert part.mime_type == "text/plain"
            assert part.text == "Hello World"
            assert part.url == ""

    def test_part_from_dict(self, tool_context):
        """Test creating Part from dictionary."""
        with set_tool_context(tool_context):
            part_data = {
                "mime_type": "text/html",
                "data": base64.b64encode(b"<h1>Hello</h1>").decode(),
                "url": ""
            }
            part = Part.model_validate(part_data)
            
            assert part.mime_type == "text/html"
            assert part.text == "<h1>Hello</h1>"

    def test_part_from_mimetype_data_dict(self, tool_context):
        """Test creating Part from {"mimetype": ..., "data": ...} format."""
        with set_tool_context(tool_context):
            # Test with base64 string data
            part_data = {
                "mimetype": "text/html",
                "data": base64.b64encode(b"<h1>Hello</h1>").decode()
            }
            part = Part.model_validate(part_data)
            
            assert part.mime_type == "text/html"
            assert part.text == "<h1>Hello</h1>"
            assert part.url == ""

    def test_part_from_mimetype_data_bytes(self, tool_context):
        """Test creating Part from {"mimetype": ..., "data": bytes} format."""
        with set_tool_context(tool_context):
            # Test with raw bytes data
            part_data = {
                "mimetype": "image/png",
                "data": b"fake png data"
            }
            part = Part.model_validate(part_data)
            
            assert part.mime_type == "image/png"
            assert part.data == b"fake png data"

    def test_part_from_ref_url(self, tool_context):
        """Test creating Part from ref:// URL."""
        with set_tool_context(tool_context):
            # First create and serialize a part
            original = Part.model_validate("Test content")
            serialized = original.model_dump()
            ref_url = serialized["url"]
            
            # Now validate from the ref URL
            restored = Part.model_validate(ref_url)
            
            assert restored.text == "Test content"
            assert restored.mime_type == "text/plain"

    def test_part_from_ref_dict(self, tool_context):
        """Test creating Part from dict with ref:// URL."""
        with set_tool_context(tool_context):
            # First create and serialize a part
            original = Part.model_validate("Test content")
            serialized = original.model_dump()
            
            # Now validate from the serialized dict
            restored = Part.model_validate(serialized)
            
            assert restored.text == "Test content"
            assert restored.mime_type == "text/plain"


class TestPartSerialization:
    """Test Part serialization functionality."""

    def test_part_serialization_creates_ref(self, tool_context):
        """Test that Part serialization creates ref:// URL and stores in context."""
        with set_tool_context(tool_context):
            part = TextPart(data=base64.b64encode(b"Hello World"))
            serialized = part.model_dump()
            
            # Check serialized structure
            assert "url" in serialized
            assert serialized["url"].startswith("ref://")
            assert serialized["mime_type"] == "text/plain"
            assert "data" in serialized  # Preview data
            assert "note" in serialized
            
            # Check that Part is stored in context
            ref_url = serialized["url"]
            stored_part = tool_context.get_ref(ref_url)
            assert stored_part.text == "Hello World"

    def test_serialization_preview_data(self, tool_context):
        """Test that serialization creates appropriate preview data."""
        with set_tool_context(tool_context):
            # Test text content
            text_part = Part.model_validate("Hello World")
            text_serialized = text_part.model_dump()
            assert text_serialized["data"] == "Hello World"
            
            # Test binary content (should be base85 encoded)
            binary_data = b"Binary content here"
            png_part = PngPart(data=base64.b64encode(binary_data))
            png_serialized = png_part.model_dump()
            expected_preview = base64.b85encode(binary_data[:128]).decode()
            assert png_serialized["data"] == expected_preview

    def test_serialization_truncation(self, tool_context):
        """Test that large content gets truncated in preview."""
        with set_tool_context(tool_context):
            # Create large content
            large_content = "x" * 1000
            part = Part.model_validate(large_content)
            serialized = part.model_dump()
            
            # Should be truncated with indicator
            assert "[truncated]..." in serialized["data"]
            assert len(serialized["data"]) < len(large_content)

    def test_round_trip_serialization(self, tool_context):
        """Test that serialize -> deserialize preserves data."""
        with set_tool_context(tool_context):
            original_text = "This is test content"
            
            # Create original part
            part1 = Part.model_validate(original_text)
            
            # Serialize
            serialized = part1.model_dump()
            
            # Deserialize
            part2 = Part.model_validate(serialized)
            
            # Should be identical
            assert part2.text == original_text
            assert part2.mime_type == part1.mime_type

    def test_multiple_parts_different_refs(self, tool_context):
        """Test that different Parts get different ref URLs."""
        with set_tool_context(tool_context):
            part1 = Part.model_validate("Content 1")
            part2 = Part.model_validate("Content 2")
            
            serialized1 = part1.model_dump()
            serialized2 = part2.model_dump()
            
            # Should have different ref URLs
            assert serialized1["url"] != serialized2["url"]
            assert serialized1["url"].startswith("ref://")
            assert serialized2["url"].startswith("ref://")


class TestPartSubtypes:
    """Test Part subtype functionality."""

    def test_text_part_creation(self, tool_context):
        """Test TextPart specific functionality."""
        with set_tool_context(tool_context):
            part = Part.model_validate("Hello")
            assert part.mime_type == "text/plain"

    def test_png_part_creation(self, tool_context):
        """Test PngPart functionality."""
        with set_tool_context(tool_context):
            binary_data = b"PNG data here"
            png_part = PngPart(data=base64.b64encode(binary_data))
            
            assert png_part.mime_type == "image/png"
            assert png_part.data == binary_data

    def test_png_part_from_bytes(self, tool_context):
        """Test PngPart.from_bytes method."""
        with set_tool_context(tool_context):
            binary_data = b"PNG data here"
            png_part = PngPart.from_bytes(binary_data)
            
            assert png_part.mime_type == "image/png"
            assert png_part.data == binary_data


class TestPartIntegration:
    """Test Part integration with existing systems."""

    def test_part_json_serialization(self, tool_context):
        """Test that serialized Parts can be converted to JSON."""
        with set_tool_context(tool_context):
            part = Part.model_validate("Test content")
            serialized = part.model_dump()
            
            # Should be JSON serializable
            json_str = json.dumps(serialized)
            assert isinstance(json_str, str)
            
            # Should be deserializable
            parsed = json.loads(json_str)
            assert parsed["url"].startswith("ref://")

    def test_part_in_complex_structure(self, tool_context):
        """Test Part serialization within complex data structures."""
        with set_tool_context(tool_context):
            part = Part.model_validate("Embedded content")
            
            # Embed in complex structure
            data = {
                "id": 123,
                "content": part,
                "metadata": {"type": "test"}
            }
            
            # Pydantic should handle Part serialization automatically
            from pydantic import BaseModel
            
            class TestModel(BaseModel):
                id: int
                content: Part
                metadata: dict
            
            model = TestModel(**data)
            serialized = model.model_dump()
            
            # Part should be serialized with ref URL
            assert serialized["content"]["url"].startswith("ref://")
            assert "data" in serialized["content"]  # Preview