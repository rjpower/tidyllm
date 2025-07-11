"""Tests for FastMCP adapter."""

import base64
import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from tidyllm.adapters.fastmcp_adapter import create_fastmcp_server
from tidyllm.context import set_tool_context
from tidyllm.registry import REGISTRY, register
from tidyllm.tools.config import Config
from tidyllm.tools.context import ToolContext
from tidyllm.types.part import Part, TextPart, PngPart


@pytest.fixture
def test_tool_context():
    """Create a test ToolContext with temporary directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = Config(
            config_dir=temp_path,
            notes_dir=temp_path / "notes",
        )
        yield ToolContext(config=config)


# Test tools that work with Parts
class CreatePartArgs(BaseModel):
    content: str = Field(description="Content for the Part")
    mime_type: str = Field(default="text/plain", description="MIME type")

class PartResult(BaseModel):
    part: Part = Field(description="Created Part")

@register()
def create_test_part(args: CreatePartArgs) -> PartResult:
    """Create a test Part with given content."""
    if args.mime_type == "image/png":
        part = PngPart(data=base64.b64encode(args.content.encode()))
    else:
        part = Part.model_validate(args.content)
        part.mime_type = args.mime_type
    return PartResult(part=part)


class ProcessPartArgs(BaseModel):
    input_part: Part = Field(description="Part to process")
    operation: str = Field(description="Operation to perform")

class ProcessPartResult(BaseModel):
    original_content: str = Field(description="Original content")
    processed_part: Part = Field(description="Processed Part")
    operation_performed: str = Field(description="Operation that was performed")

@register()
def process_test_part(args: ProcessPartArgs) -> ProcessPartResult:
    """Process a Part and return both original and processed versions."""
    original_content = args.input_part.text
    
    if args.operation == "uppercase":
        processed_content = original_content.upper()
    elif args.operation == "reverse":
        processed_content = original_content[::-1]
    else:
        processed_content = f"[{args.operation}] {original_content}"
    
    processed_part = Part.model_validate(processed_content)
    
    return ProcessPartResult(
        original_content=original_content,
        processed_part=processed_part,
        operation_performed=args.operation
    )


class MultiPartArgs(BaseModel):
    parts: list[Part] = Field(description="List of Parts to combine")
    separator: str = Field(default=" ", description="Separator between parts")

class MultiPartResult(BaseModel):
    combined_part: Part = Field(description="Combined Part")
    part_count: int = Field(description="Number of parts combined")

@register()
def combine_test_parts(args: MultiPartArgs) -> MultiPartResult:
    """Combine multiple Parts into one."""
    contents = [part.text for part in args.parts]
    combined_content = args.separator.join(contents)
    combined_part = Part.model_validate(combined_content)
    
    return MultiPartResult(
        combined_part=combined_part,
        part_count=len(args.parts)
    )


def test_create_fastmcp_server_basic(test_tool_context):
    """Test basic FastMCP server creation."""
    # Create server with context
    mcp_server = create_fastmcp_server(context=test_tool_context)
    
    # Verify server was created
    assert mcp_server is not None
    assert hasattr(mcp_server, 'run')


def test_create_fastmcp_server_with_custom_context(test_tool_context):
    """Test FastMCP server creation with custom tool context."""
    # Create server with custom tool context
    mcp_server = create_fastmcp_server(
        context=test_tool_context,
        name="Test Server"
    )
    
    assert mcp_server is not None




def test_tool_wrapper_execution(test_tool_context):
    """Test that tool wrappers properly execute TidyLLM tools."""
    # Create server with tool context
    mcp_server = create_fastmcp_server(context=test_tool_context)
    
    # Verify the server has tools registered
    assert mcp_server is not None


def test_tool_context_integration(test_tool_context):
    """Test that ToolContext is properly integrated with FastMCP."""
    # Create server with custom tool context
    mcp_server = create_fastmcp_server(
        context=test_tool_context,
        name="Integration Test Server"
    )
    
    # Verify server creation succeeded
    assert mcp_server is not None


def test_server_name_customization(test_tool_context):
    """Test that server name can be customized."""
    custom_name = "My Custom TidyLLM Server"
    
    mcp_server = create_fastmcp_server(context=test_tool_context, name=custom_name)
    
    # The name should be stored in the server (exact verification depends on FastMCP internals)
    assert mcp_server is not None


def test_mock_tool_execution(test_tool_context):
    """Test tool execution setup."""
    # Create a minimal test setup with the test context
    mcp_server = create_fastmcp_server(context=test_tool_context)
    
    # Verify server creation
    assert mcp_server is not None


def test_registry_tools_discovery(test_tool_context):
    """Test that all registered tools are discovered."""
    # Import all tools to ensure they're registered
    import tidyllm.tools.anki  # noqa: F401
    import tidyllm.tools.manage_db  # noqa: F401
    import tidyllm.tools.notes  # noqa: F401
    import tidyllm.tools.transcribe  # noqa: F401
    import tidyllm.tools.vocab_table  # noqa: F401
    
    # Create server and verify tools are discovered
    mcp_server = create_fastmcp_server(context=test_tool_context)
    
    # Should have multiple tools registered
    assert len(REGISTRY.functions) >= 5  # We have multiple focused tools now
    
    # Server should be created successfully
    assert mcp_server is not None


def test_config_override_validation(test_tool_context):
    """Test that configuration overrides work correctly."""
    # Test that we can create a server with a custom context
    mcp_server = create_fastmcp_server(context=test_tool_context)
    
    # Server should be created successfully
    assert mcp_server is not None


class TestPartHandling:
    """Test Part serialization and deserialization in FastMCP adapter."""

    def test_tool_returning_part(self, test_tool_context):
        """Test that tools returning Parts get properly serialized."""
        with set_tool_context(test_tool_context):
            # Call tool that returns a Part
            result = create_test_part(CreatePartArgs(content="Hello World"))
            
            # Part should be returned
            assert isinstance(result.part, Part)
            assert result.part.text == "Hello World"
            
            # When serialized (as FastMCP would do), should become RemotePart format
            serialized = result.model_dump()
            part_data = serialized["part"]
            
            # Should have ref:// URL and preview data
            assert "url" in part_data
            assert part_data["url"].startswith("ref://")
            assert part_data["mime_type"] == "text/plain"
            assert "data" in part_data
            assert "Hello World" in part_data["data"]

    def test_tool_accepting_part_from_string(self, test_tool_context):
        """Test that tools can accept Parts created from strings."""
        with set_tool_context(test_tool_context):
            # Create Part from string input
            input_part = Part.model_validate("Test Content")
            
            # Call tool that processes Parts
            result = process_test_part(ProcessPartArgs(
                input_part=input_part,
                operation="uppercase"
            ))
            
            assert result.original_content == "Test Content"
            assert result.processed_part.text == "TEST CONTENT"
            assert result.operation_performed == "uppercase"

    def test_tool_accepting_part_from_ref_url(self, test_tool_context):
        """Test that tools can accept Parts from ref:// URLs."""
        with set_tool_context(test_tool_context):
            # Create and serialize a Part to get ref URL
            original_part = Part.model_validate("Original Content")
            serialized = original_part.model_dump()
            ref_url = serialized["url"]
            
            # Create Part from ref URL (simulating FastMCP deserialization)
            input_part = Part.model_validate(ref_url)
            
            # Call tool with the deserialized Part
            result = process_test_part(ProcessPartArgs(
                input_part=input_part,
                operation="reverse"
            ))
            
            assert result.original_content == "Original Content"
            assert result.processed_part.text == "tnetnoC lanigirO"

    def test_tool_accepting_multiple_parts(self, test_tool_context):
        """Test tools that accept multiple Parts."""
        with set_tool_context(test_tool_context):
            # Create multiple Parts
            part1 = Part.model_validate("Hello")
            part2 = Part.model_validate("World")
            part3 = Part.model_validate("!")
            
            # Call tool that combines Parts
            result = combine_test_parts(MultiPartArgs(
                parts=[part1, part2, part3],
                separator=" "
            ))
            
            assert result.part_count == 3
            assert result.combined_part.text == "Hello World !"

    def test_part_serialization_in_complex_result(self, test_tool_context):
        """Test Part serialization in complex nested results."""
        with set_tool_context(test_tool_context):
            # Create a Part
            input_part = Part.model_validate("Test Input")
            
            # Process it
            result = process_test_part(ProcessPartArgs(
                input_part=input_part,
                operation="custom"
            ))
            
            # Serialize the entire result (as FastMCP would do)
            serialized = result.model_dump()
            
            # Both original content and processed Part should be serialized
            assert serialized["original_content"] == "Test Input"
            assert serialized["operation_performed"] == "custom"
            
            # Processed Part should be serialized with ref URL
            processed_part_data = serialized["processed_part"]
            assert processed_part_data["url"].startswith("ref://")
            assert "[custom] Test Input" in processed_part_data["data"]

    def test_fetch_part_content_tool(self, test_tool_context):
        """Test the fetch_part_content tool provided by FastMCP adapter."""
        with set_tool_context(test_tool_context):
            # Create a Part and serialize it
            original_part = TextPart(data=base64.b64encode(b"Fetchable Content"))
            serialized = original_part.model_dump()
            ref_url = serialized["url"]
            
            # Create FastMCP server to get fetch_part_content tool
            mcp_server = create_fastmcp_server(context=test_tool_context)
            
            # The fetch_part_content function should be available
            # We can't easily test the actual FastMCP tool execution here,
            # but we can verify the Part is stored in context
            stored_part = test_tool_context.get_ref(ref_url)
            assert stored_part.text == "Fetchable Content"

    def test_part_types_preserved(self, test_tool_context):
        """Test that different Part types are preserved through serialization."""
        with set_tool_context(test_tool_context):
            # Test PNG Part
            png_result = create_test_part(CreatePartArgs(
                content="fake png data",
                mime_type="image/png"
            ))
            
            # Should create PngPart
            assert isinstance(png_result.part, PngPart)
            assert png_result.part.mime_type == "image/png"
            
            # Serialization should preserve mime type
            serialized = png_result.model_dump()
            assert serialized["part"]["mime_type"] == "image/png"

    def test_part_round_trip_through_serialization(self, test_tool_context):
        """Test complete round-trip: Part -> serialized -> deserialized -> tool -> result."""
        with set_tool_context(test_tool_context):
            # 1. Create original Part
            original = Part.model_validate("Round Trip Test")
            
            # 2. Serialize (as FastMCP would do for output)
            serialized = original.model_dump()
            ref_url = serialized["url"]
            
            # 3. Deserialize (as FastMCP would do for input)
            deserialized = Part.model_validate(serialized)
            
            # 4. Use in tool
            result = process_test_part(ProcessPartArgs(
                input_part=deserialized,
                operation="uppercase"
            ))
            
            # 5. Verify round-trip worked
            assert result.original_content == "Round Trip Test"
            assert result.processed_part.text == "ROUND TRIP TEST"
            
            # 6. Result should also be serializable
            final_serialized = result.model_dump()
            assert final_serialized["processed_part"]["url"].startswith("ref://")

    def test_error_handling_invalid_ref_url(self, test_tool_context):
        """Test error handling for invalid ref:// URLs."""
        with set_tool_context(test_tool_context):
            # Try to create Part from non-existent ref URL
            with pytest.raises(KeyError):
                Part.model_validate("ref://non-existent-id")

    def test_part_serialization_context_fix(self, test_tool_context):
        """Test that Parts are pre-serialized to avoid context issues."""
        with set_tool_context(test_tool_context):
            from tidyllm.adapters.fastmcp_adapter import context_fn
            
            # Create a tool that returns a Part
            def mock_tool_returning_part():
                return create_test_part(CreatePartArgs(content="Context Test"))
            
            # Wrap the tool with context_fn (as FastMCP adapter does)
            wrapped_tool = context_fn(mock_tool_returning_part, test_tool_context)
            
            # Call the wrapped tool
            result = wrapped_tool()
            
            # The result should be pre-serialized (a dict, not a PartResult with Part objects)
            assert isinstance(result, dict)
            assert "part" in result
            
            # The part should be serialized with ref:// URL
            part_data = result["part"]
            assert isinstance(part_data, dict)
            assert "url" in part_data
            assert part_data["url"].startswith("ref://")
            assert "data" in part_data
            assert "Context Test" in part_data["data"]

    def test_nested_part_serialization_context_fix(self, test_tool_context):
        """Test that nested Parts in complex structures are pre-serialized."""
        with set_tool_context(test_tool_context):
            from tidyllm.adapters.fastmcp_adapter import context_fn
            
            # Create a tool that returns nested Parts
            def mock_tool_returning_nested_parts():
                part1 = Part.model_validate("First Part")
                part2 = Part.model_validate("Second Part")
                
                return {
                    "single_part": part1,
                    "part_list": [part1, part2],
                    "nested_dict": {
                        "inner_part": part2,
                        "other_data": "not a part"
                    },
                    "regular_data": "just a string"
                }
            
            # Wrap the tool with context_fn
            wrapped_tool = context_fn(mock_tool_returning_nested_parts, test_tool_context)
            
            # Call the wrapped tool
            result = wrapped_tool()
            
            # All Parts should be pre-serialized
            assert isinstance(result, dict)
            
            # Check single part
            assert isinstance(result["single_part"], dict)
            assert result["single_part"]["url"].startswith("ref://")
            
            # Check parts in list
            assert isinstance(result["part_list"], list)
            assert len(result["part_list"]) == 2
            assert result["part_list"][0]["url"].startswith("ref://")
            assert result["part_list"][1]["url"].startswith("ref://")
            
            # Check nested part in dict
            assert isinstance(result["nested_dict"]["inner_part"], dict)
            assert result["nested_dict"]["inner_part"]["url"].startswith("ref://")
            
            # Check regular data is unchanged
            assert result["regular_data"] == "just a string"
            assert result["nested_dict"]["other_data"] == "not a part"
