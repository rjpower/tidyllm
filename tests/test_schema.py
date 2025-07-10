"""Tests for function schema extraction."""

from typing import Any

from pydantic import BaseModel

from tidyllm.function_schema import (
    FunctionDescription,
)
from tidyllm.model.source import SourceLike


class ExampleModel(BaseModel):
    """Example model for testing."""
    field1: str
    field2: int = 42




def multi_param_function(
    name: str, count: int, tags: list[str], config: dict | None = None
) -> dict:
    """Function with multiple parameters.

    Args:
        name: The name parameter
        count: Number of items
        tags: List of tags
        config: Optional configuration dict
    """
    return {"name": name, "count": count, "tags": tags, "config": config}


def mixed_types_function(
    text: str, number: int, flag: bool, items: list[str], metadata: dict[str, Any]
) -> dict:
    """Function with mixed parameter types.

    Args:
        text: A text string
        number: An integer number
        flag: A boolean flag
        items: List of string items
        metadata: Dictionary of metadata
    """
    return {
        "text": text,
        "number": number,
        "flag": flag,
        "items": items,
        "metadata": metadata,
    }


def optional_params_function(
    required_param: str, optional_int: int = 42, optional_str: str = "default"
) -> dict:
    """Function with optional parameters.

    Args:
        required_param: This parameter is required
        optional_int: Optional integer with default
        optional_str: Optional string with default
    """
    return {
        "required": required_param,
        "opt_int": optional_int,
        "opt_str": optional_str,
    }


def single_primitive_function(message: str) -> dict:
    """Function with single primitive parameter.

    Args:
        message: The message to process
    """
    return {"processed": message}


def function_with_sourcelike(data: SourceLike, name: str) -> dict:
    """Function with SourceLike parameter.

    Args:
        data: Source-like data input
        name: A string name
    """
    return {"name": name, "data_type": type(data).__name__}


def function_with_basemodel(config: ExampleModel, enabled: bool = True) -> dict:
    """Function with BaseModel parameter.

    Args:
        config: Configuration model
        enabled: Whether feature is enabled
    """
    return {"config": config.model_dump(), "enabled": enabled}


class TestFunctionDescription:
    """Test FunctionDescription for argument parsing and validation."""

    def test_function_description_creation_multi_param(self):
        """Test creating FunctionDescription for multi-parameter function."""
        func_desc = FunctionDescription(multi_param_function)

        assert func_desc.name == "multi_param_function"
        assert func_desc.args_model is not None

    def test_function_description_creation_single_primitive(self):
        """Test creating FunctionDescription for single primitive function."""
        func_desc = FunctionDescription(single_primitive_function)

        assert func_desc.name == "single_primitive_function"
        assert func_desc.args_model is not None

    def test_validate_and_parse_args_multi_param(self):
        """Test validation and parsing for multi-parameter function."""
        func_desc = FunctionDescription(multi_param_function)
        json_args = {
            "name": "test",
            "count": 5,
            "tags": ["a", "b"],
            "config": {"key": "value"},
        }

        result = func_desc.validate_and_parse_args(json_args)
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["count"] == 5
        assert result["tags"] == ["a", "b"]
        assert result["config"] == {"key": "value"}

    def test_validate_and_parse_args_single_primitive(self):
        """Test validation and parsing for single primitive function."""
        func_desc = FunctionDescription(single_primitive_function)
        json_args = {"message": "hello"}

        result = func_desc.validate_and_parse_args(json_args)
        assert isinstance(result, dict)
        assert result["message"] == "hello"

    def test_call_multi_param(self):
        """Test calling function with parsed args for multi-parameter function."""
        func_desc = FunctionDescription(multi_param_function)
        json_args = {"name": "test", "count": 5, "tags": ["a", "b"]}

        parsed_args = func_desc.validate_and_parse_args(json_args)
        result = func_desc.call(**parsed_args)
        assert result["name"] == "test"
        assert result["count"] == 5
        assert result["tags"] == ["a", "b"]

    def test_call_single_primitive(self):
        """Test calling function with parsed args for single primitive parameter."""
        func_desc = FunctionDescription(single_primitive_function)
        json_args = {"message": "hello world"}

        parsed_args = func_desc.validate_and_parse_args(json_args)
        result = func_desc.call(**parsed_args)
        assert result["processed"] == "hello world"


def create_test_schema(func, doc_override=None):
    """Helper to create schema using FunctionDescription."""
    func_desc = FunctionDescription(func, doc_override=doc_override)
    return func_desc.function_schema


class TestToolSchemaGeneration:
    """Test tool schema generation."""

    def test_generate_schema_multiple_parameters(self):
        """Test schema generation for function with multiple parameters."""
        schema = create_test_schema(multi_param_function)

        assert schema["function"]["name"] == "multi_param_function"
        params = schema["function"]["parameters"]

        # Check all parameters are present
        assert "name" in params["properties"]
        assert "count" in params["properties"]
        assert "tags" in params["properties"]
        assert "config" in params["properties"]

        # Check types
        assert params["properties"]["name"]["type"] == "string"
        assert params["properties"]["count"]["type"] == "integer"
        assert params["properties"]["tags"]["type"] == "array"
        assert params["properties"]["tags"]["items"]["type"] == "string"
        # Check that config is properly typed as optional object
        config_prop = params["properties"]["config"]
        assert "anyOf" in config_prop
        assert any(item.get("type") == "object" for item in config_prop["anyOf"])
        assert any(item.get("type") == "null" for item in config_prop["anyOf"])

        # Check required parameters
        required = params.get("required", [])
        assert "name" in required
        assert "count" in required
        assert "tags" in required
        assert "config" not in required  # Has default value

    def test_generate_schema_mixed_types(self):
        """Test schema generation for function with mixed parameter types."""
        schema = create_test_schema(mixed_types_function)

        params = schema["function"]["parameters"]

        # Check all basic types
        assert params["properties"]["text"]["type"] == "string"
        assert params["properties"]["number"]["type"] == "integer"
        assert params["properties"]["flag"]["type"] == "boolean"
        assert params["properties"]["items"]["type"] == "array"
        assert params["properties"]["items"]["items"]["type"] == "string"
        assert params["properties"]["metadata"]["type"] == "object"

        # All parameters should be required
        required = params.get("required", [])
        assert len(required) == 5
        assert all(param in required for param in ["text", "number", "flag", "items", "metadata"])

    def test_generate_schema_single_primitive(self):
        """Test schema generation for function with single primitive parameter."""
        schema = create_test_schema(single_primitive_function)

        params = schema["function"]["parameters"]

        # Should have message parameter
        assert "message" in params["properties"]
        assert params["properties"]["message"]["type"] == "string"

        # Should be required
        required = params.get("required", [])
        assert "message" in required

    def test_multi_param_parameter_descriptions(self):
        """Test that parameter descriptions work with multiple parameters."""
        schema = create_test_schema(multi_param_function)

        params = schema["function"]["parameters"]

        # Check descriptions from docstring
        assert params["properties"]["name"]["description"] == "The name parameter"
        assert params["properties"]["count"]["description"] == "Number of items"
        assert params["properties"]["tags"]["description"] == "List of tags"
        assert params["properties"]["config"]["description"] == "Optional configuration dict"

    def test_generate_schema_with_sourcelike(self):
        """Test schema generation for function with SourceLike parameter."""
        schema = create_test_schema(function_with_sourcelike)

        params = schema["function"]["parameters"]

        # Check that SourceLike parameter is present
        assert "data" in params["properties"]
        assert "name" in params["properties"]

        # Both should be required
        required = params.get("required", [])
        assert "data" in required
        assert "name" in required

    def test_generate_schema_with_basemodel(self):
        """Test schema generation for function with BaseModel parameter."""
        schema = create_test_schema(function_with_basemodel)

        params = schema["function"]["parameters"]

        # Check that BaseModel parameter is present
        assert "config" in params["properties"]
        assert "enabled" in params["properties"]

        # Check required parameters
        required = params.get("required", [])
        assert "config" in required
        assert "enabled" not in required  # Has default value
