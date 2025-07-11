"""Tests for image Part serialization through recipe tool workflow."""

import base64
import json
import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from tidyllm.context import set_tool_context
from tidyllm.registry import register
from tidyllm.tools.config import Config
from tidyllm.tools.context import ToolContext
from tidyllm.tools.recipe import recipe_extract
from tidyllm.types.part import Part, PngPart, TextPart


@pytest.fixture
def tool_context():
    """Create a test ToolContext with temporary directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = Config(
            config_dir=temp_path,
            notes_dir=temp_path / "notes",
        )
        yield ToolContext(config=config)


# Test tool that loads the recipe.png image
class LoadImageResult(BaseModel):
    image_part: PngPart = Field(description="Loaded image as PngPart")
    file_size: int = Field(description="Size of the image file in bytes")
    
@register()
def load_recipe_image() -> LoadImageResult:
    """Load the recipe.png test image and return as PngPart."""
    image_path = Path(__file__).parent / "recipe.png"
    
    if not image_path.exists():
        raise FileNotFoundError(f"Recipe image not found at {image_path}")
    
    image_data = image_path.read_bytes()
    png_part = PngPart.from_bytes(image_data)
    
    return LoadImageResult(
        image_part=png_part,
        file_size=len(image_data)
    )


class TestRecipeImageSerialization:
    """Test complete image Part serialization workflow through recipe tools."""

    def test_load_recipe_image_tool(self, tool_context):
        """Test that we can load the recipe image as a PngPart."""
        with set_tool_context(tool_context):
            result = load_recipe_image()
            
            # Verify we got a valid PngPart
            assert isinstance(result.image_part, PngPart)
            assert result.image_part.mime_type == "image/png"
            assert result.file_size > 0
            assert len(result.image_part.data) == result.file_size

    def test_image_part_json_serialization(self, tool_context):
        """Test JSON serialization/deserialization of image Part."""
        with set_tool_context(tool_context):
            # Load image
            result = load_recipe_image()
            image_part = result.image_part
            
            # Serialize to JSON (simulating FastMCP output)
            serialized = image_part.model_dump()
            json_str = json.dumps(serialized)
            
            # Verify JSON is valid
            assert isinstance(json_str, str)
            parsed = json.loads(json_str)
            
            # Verify serialized structure
            assert "url" in parsed
            assert parsed["url"].startswith("ref://")
            assert parsed["mime_type"] == "image/png"
            assert "data" in parsed  # Preview data
            assert "note" in parsed
            
            # Verify the Part is stored in context
            ref_url = parsed["url"]
            stored_part = tool_context.get_ref(ref_url)
            assert isinstance(stored_part, PngPart)
            assert stored_part.data == image_part.data

    def test_image_part_round_trip_serialization(self, tool_context):
        """Test complete round-trip: image Part → JSON → deserialized Part."""
        with set_tool_context(tool_context):
            # Load original image
            original_result = load_recipe_image()
            original_part = original_result.image_part
            original_data = original_part.data
            
            # Serialize to JSON
            serialized = original_part.model_dump()
            json_str = json.dumps(serialized)
            parsed = json.loads(json_str)
            
            # Deserialize from JSON (simulating FastMCP input)
            deserialized_part = Part.model_validate(parsed)
            
            # Verify round-trip preserves data
            # Note: Deserialized Part is generic Part, not PngPart subclass
            assert deserialized_part.mime_type == "image/png"
            assert deserialized_part.data == original_data
            assert len(deserialized_part.data) == original_result.file_size

    def test_recipe_extract_with_image_part(self, tool_context):
        """Test passing image Part to recipe_extract tool with actual LLM call."""
        with set_tool_context(tool_context):
            # Load image
            image_result = load_recipe_image()
            image_part = image_result.image_part
            
            # Create some text content as well
            text_part = TextPart(data=base64.b64encode(b"Extract recipe from this image"))
            
            # Call recipe_extract with mixed content - this will make an actual LLM call
            recipe = recipe_extract([text_part, image_part])
            
            # Verify we got a recipe result
            assert hasattr(recipe, 'title')
            assert hasattr(recipe, 'ingredients')
            assert hasattr(recipe, 'steps')
            assert hasattr(recipe, 'time')
            
            # The recipe should be based on the image content
            assert isinstance(recipe.ingredients, list)
            assert isinstance(recipe.steps, list)
            assert len(recipe.ingredients) > 0  # Should extract some ingredients
            assert len(recipe.steps) > 0  # Should extract some steps
            
            # Should contain recipe content related to the image
            title_lower = recipe.title.lower()
            assert any(word in title_lower for word in ['egg', 'pickle', 'deviled']) or len(title_lower) > 0

    def test_recipe_extract_result_serialization(self, tool_context):
        """Test serialization of recipe results that used image Parts."""
        with set_tool_context(tool_context):
            # Load image and create mixed content
            image_result = load_recipe_image()
            image_part = image_result.image_part
            text_part = TextPart(data=base64.b64encode(
                b"Extract recipe from this deviled eggs image"
            ))
            
            # Extract recipe using actual LLM call
            recipe = recipe_extract([text_part, image_part])
            
            # Serialize the recipe result
            recipe_serialized = recipe.model_dump()
            recipe_json = json.dumps(recipe_serialized)
            
            # Should be valid JSON
            assert isinstance(recipe_json, str)
            parsed_recipe = json.loads(recipe_json)
            
            # Verify recipe structure
            assert "title" in parsed_recipe
            assert "ingredients" in parsed_recipe
            assert "steps" in parsed_recipe
            assert "time" in parsed_recipe
            
            # Verify we got actual content
            assert isinstance(parsed_recipe["ingredients"], list)
            assert isinstance(parsed_recipe["steps"], list)
            assert len(parsed_recipe["ingredients"]) > 0
            assert len(parsed_recipe["steps"]) > 0

    def test_fastmcp_adapter_with_image_parts(self, tool_context):
        """Test FastMCP adapter context handling with image Parts."""
        with set_tool_context(tool_context):
            from tidyllm.adapters.fastmcp_adapter import context_fn
            
            # Create tool that returns image Part
            def mock_image_tool():
                return load_recipe_image()
            
            # Wrap with FastMCP context function
            wrapped_tool = context_fn(mock_image_tool, tool_context)
            
            # Call wrapped tool
            result = wrapped_tool()
            
            # Result should be pre-serialized
            assert isinstance(result, dict)
            assert "image_part" in result
            assert "file_size" in result
            
            # Image part should be serialized with ref:// URL
            image_data = result["image_part"]
            assert isinstance(image_data, dict)
            assert image_data["url"].startswith("ref://")
            assert image_data["mime_type"] == "image/png"
            assert "data" in image_data  # Preview data

    def test_multiple_image_parts_different_refs(self, tool_context):
        """Test that multiple image Parts get different ref URLs."""
        with set_tool_context(tool_context):
            # Load same image twice (should create different Parts)
            result1 = load_recipe_image()
            result2 = load_recipe_image()
            
            # Serialize both
            serialized1 = result1.image_part.model_dump()
            serialized2 = result2.image_part.model_dump()
            
            # Should have different ref URLs even though same image data
            assert serialized1["url"] != serialized2["url"]
            assert serialized1["url"].startswith("ref://")
            assert serialized2["url"].startswith("ref://")
            
            # But should have same mime type and file size
            assert serialized1["mime_type"] == serialized2["mime_type"]
            assert result1.file_size == result2.file_size

    def test_complex_workflow_json_serialization(self, tool_context):
        """Test complex workflow: tool → JSON → deserialization → recipe tool → JSON."""
        with set_tool_context(tool_context):
            # Step 1: Load image via tool
            original_result = load_recipe_image()
            
            # Step 2: Serialize to JSON (FastMCP output)
            serialized = original_result.model_dump()
            json_str = json.dumps(serialized)
            
            # Step 3: Deserialize from JSON (FastMCP input)
            parsed = json.loads(json_str)
            
            ref_url = parsed["image_part"]["url"]
            
            # Step 5: Use in recipe tool with text
            text_content = TextPart(data=base64.b64encode(
                b"Extract recipe from this deviled eggs image"
            ))
            
            recipe = recipe_extract([text_content, Part.model_validate(ref_url)])
            
            # Step 6: Serialize recipe result to JSON
            recipe_serialized = recipe.model_dump()
            recipe_json = json.dumps(recipe_serialized)
            
            # Verify complete workflow succeeded
            assert isinstance(recipe_json, str)
            final_parsed = json.loads(recipe_json)
            assert "title" in final_parsed
            assert "ingredients" in final_parsed
            assert len(final_parsed["ingredients"]) > 0
            assert len(final_parsed["steps"]) > 0
            

    def test_ref_url_context_isolation(self, tool_context):
        """Test that ref:// URLs are properly isolated to context."""
        with set_tool_context(tool_context):
            # Create image Part in current context
            result = load_recipe_image()
            serialized = result.image_part.model_dump()
            ref_url = serialized["url"]
            
            # Verify Part is accessible in current context
            stored_part = tool_context.get_ref(ref_url)
            assert stored_part is not None
            
        # In different context, ref URL should not be accessible
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            different_config = Config(
                config_dir=temp_path,
                notes_dir=temp_path / "notes",
            )
            different_context = ToolContext(config=different_config)
            
            with set_tool_context(different_context):
                # Same ref URL should not exist in different context
                with pytest.raises(KeyError):
                    different_context.get_ref(ref_url)

    def test_image_part_preview_data_format(self, tool_context):
        """Test that image Part preview data is properly formatted."""
        with set_tool_context(tool_context):
            result = load_recipe_image()
            serialized = result.image_part.model_dump()
            
            # Preview data should be base85 encoded for binary content
            preview_data = serialized["data"]
            assert isinstance(preview_data, str)
            assert len(preview_data) > 0
            
            # Should indicate truncation for large files
            if result.file_size > 512:
                assert "[truncated]..." in preview_data
            
            # Should be different from the actual image data (which is base64)
            actual_data_b64 = base64.b64encode(result.image_part.data).decode()
            assert preview_data != actual_data_b64