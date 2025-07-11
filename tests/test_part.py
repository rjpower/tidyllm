"""Tests for Part validation and functionality."""

import base64
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

    def test_part_from_dict(self, tool_context):
        """Test creating Part from dictionary."""
        with set_tool_context(tool_context):
            part_data = {
                "mime_type": "text/html",
                "data": base64.b64encode(b"<h1>Hello</h1>").decode()
            }
            part = Part.model_validate(part_data)

            assert part.mime_type == "text/html"
            assert part.text == "<h1>Hello</h1>"

    def test_part_from_data_url(self, tool_context):
        """Test creating Part from data: URL."""
        with set_tool_context(tool_context):
            # Create a data URL
            data_content = b"Hello World"
            b64_content = base64.b64encode(data_content).decode()
            data_url = f"data:text/plain;base64,{b64_content}"
            
            part = Part.model_validate(data_url)
            
            assert part.mime_type == "text/plain"
            assert part.data == data_content


class TestPartSubtypes:
    """Test Part subtype functionality."""

    def test_part_type_aliases(self, tool_context):
        """Test that type aliases work correctly."""
        with set_tool_context(tool_context):
            # All type aliases should be the same as Part
            assert TextPart is Part
            assert PngPart is Part

    def test_part_properties(self, tool_context):
        """Test Part properties like text and base64_bytes."""
        with set_tool_context(tool_context):
            text_data = b"Hello World"
            part = Part(mime_type="text/plain", data=base64.b64encode(text_data))
            
            assert part.text == "Hello World"
            assert part.base64_bytes == base64.b64encode(text_data)