"""Tests for Part validation and functionality."""

import base64
import pytest
from pathlib import Path

from tidyllm.context import set_tool_context
from tidyllm.tools.context import ToolContext
from tidyllm.types.part import Part, BasicPart, TextPart, HtmlPart, AudioPart, ImagePart
import numpy as np
from PIL import Image


@pytest.fixture
def tool_context():
    """Provide a fresh ToolContext for each test."""
    return ToolContext()


class TestPartConstruction:
    """Test the 3 ways to construct Parts: URL, bytes, serialized dict."""

    def test_part_from_bytes(self, tool_context):
        """Test creating Part from raw bytes."""
        with set_tool_context(tool_context):
            part = Part.from_bytes("text/plain", b"Hello World")
            
            assert isinstance(part, BasicPart)
            assert part.mime_type == "text/plain"
            assert part.text == "Hello World"

    def test_part_from_url_data_scheme(self, tool_context):
        """Test creating Part from data URL."""
        with set_tool_context(tool_context):
            data_url = "data:text/plain;base64," + base64.b64encode(b"Hello World").decode()
            parts = Part.from_url(data_url)
            part = parts.first()
            
            assert isinstance(part, BasicPart)
            assert part.mime_type == "text/plain"
            assert part.text == "Hello World"

    def test_part_from_json_serialized(self, tool_context):
        """Test creating Part from serialized dictionary."""
        with set_tool_context(tool_context):
            # Create original part
            original = Part.from_bytes("text/plain", b"Hello World")
            
            # Serialize it
            serialized = original.model_dump()
            
            # Load it back using from_json
            restored = Part.from_json(serialized)
            
            assert isinstance(restored, BasicPart)
            assert restored.mime_type == "text/plain"
            assert restored.text == "Hello World"

    def test_image_part_from_bytes(self, tool_context):
        """Test creating ImagePart from raw bytes."""
        with set_tool_context(tool_context):
            # Create a simple PNG image
            test_image = Image.new("RGB", (10, 10), color="red")
            import io
            buffer = io.BytesIO()
            test_image.save(buffer, format="PNG")
            png_bytes = buffer.getvalue()
            
            part = Part.from_bytes("image/png", png_bytes)
            
            assert isinstance(part, ImagePart)
            assert part.mime_type.startswith("image/png")
            assert part.width == 10
            assert part.height == 10

    def test_image_part_serialization_roundtrip(self, tool_context):
        """Test ImagePart serialization and deserialization."""
        with set_tool_context(tool_context):
            # Create test image
            test_image = Image.new("RGB", (30, 20), color="cyan")
            original = ImagePart.from_pil(test_image)
            
            # Serialize
            serialized = original.model_dump()
            
            # Deserialize using from_json
            restored = Part.from_json(serialized)
            
            assert isinstance(restored, ImagePart)
            assert restored.width == 30
            assert restored.height == 20

    def test_unregistered_mime_type_fallback(self, tool_context):
        """Test that unregistered mime types fall back to BasicPart."""
        with set_tool_context(tool_context):
            part = Part.from_bytes("unknown/type", b"test")
            assert isinstance(part, BasicPart)
            assert part.mime_type == "unknown/type"