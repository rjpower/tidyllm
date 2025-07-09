"""Tests for tool registry system."""

import pytest
from pydantic import BaseModel

from tidyllm.registry import REGISTRY, Registry, ToolError




def multi_param_function(name: str, count: int, enabled: bool = True) -> dict:
    """Function with multiple parameters."""
    return {"name": name, "count": count, "enabled": enabled}


def primitive_param_function(message: str) -> dict:
    """Function with single primitive parameter."""
    return {"processed": message.upper()}


@pytest.fixture
def registry():
    """Create a registry for testing."""
    return Registry()

def test_register_function(registry):
    """Test registering a function."""
    registry.register(multi_param_function)

    assert "multi_param_function" in registry._tools
    assert registry.get_description("multi_param_function") is not None


def test_register_duplicate_function(registry):
    """Test that registering duplicate function name is silently ignored."""
    registry.register(multi_param_function)
    assert len(registry.functions) == 1

    # Registering again should be silently ignored
    registry.register(multi_param_function)
    assert len(registry.functions) == 1


def test_register_with_context_type(registry):
    """Test registering function works properly."""

    # Test that registration works for functions regardless of context needs
    registry.register(multi_param_function)

    func_desc = registry.get_description("multi_param_function")
    assert func_desc is not None
    # Verify basic function properties are preserved
    assert func_desc.name == "multi_param_function"
    assert func_desc.function is multi_param_function


def test_get_existing_tool(registry):
    """Test getting an existing tool."""
    registry.register(multi_param_function)

    func_desc = registry.get_description("multi_param_function")
    assert func_desc is not None
    assert func_desc.function is multi_param_function

    # Test get_function method
    tool_func = registry.get_function("multi_param_function")
    assert tool_func is multi_param_function


def test_get_nonexistent_tool(registry):
    """Test getting a non-existent tool."""
    func_desc = registry.get_description("nonexistent")
    assert func_desc is None


def test_list_tools(registry):
    """Test listing all registered tools."""
    registry.register(multi_param_function)
    registry.register(primitive_param_function)

    functions = registry.functions
    assert len(functions) == 2
    names = [f.name for f in functions]
    assert "multi_param_function" in names
    assert "primitive_param_function" in names


def test_registry_independence(registry):
    """Test that separate registry instances are independent."""
    other_registry = Registry()

    registry.register(multi_param_function)
    assert len(registry.functions) == 1
    assert len(other_registry.functions) == 0

    other_registry.register(primitive_param_function)
    assert len(registry.functions) == 1
    assert len(other_registry.functions) == 1


def test_function_metadata_attached(registry):
    """Test that metadata is attached to registered functions."""
    registry.register(multi_param_function)

    # Check function metadata via registry
    func_desc = registry.get_description("multi_param_function")
    assert func_desc is not None
    assert func_desc.function_schema["function"]["name"] == "multi_param_function"
    # Check FunctionDescription has schema
    assert func_desc.function_schema is not None


@pytest.fixture
def saved_global_registry():
    """Save and restore global registry state."""
    saved_tools = REGISTRY._tools.copy()
    yield saved_tools
    REGISTRY._tools = saved_tools

def test_global_registry_exists(saved_global_registry):
    """Test that global REGISTRY exists."""
    assert REGISTRY is not None
    assert isinstance(REGISTRY, Registry)


def test_global_registry_register(saved_global_registry):
    """Test registering with global registry."""
    REGISTRY.register(multi_param_function)

    names = [f.name for f in REGISTRY.functions]
    assert "multi_param_function" in names


def test_global_registry_isolation(saved_global_registry):
    """Test that tests don't interfere with each other."""
    # This test should start with the saved registry state
    initial_count = len(saved_global_registry)
    assert len(REGISTRY.functions) == initial_count

    REGISTRY.register(multi_param_function)

    assert len(REGISTRY.functions) == initial_count + 1


@pytest.fixture
def execution_registry():
    """Create a registry with execution functions for testing."""
    registry = Registry()
    registry.register(multi_param_function)
    registry.register(primitive_param_function)
    return registry

def test_call_nonexistent_function(execution_registry):
    """Test calling a non-existent function returns ToolError."""
    result = execution_registry.call("nonexistent", {})
    
    assert isinstance(result, ToolError)
    assert "not found" in result.error


def test_call_with_invalid_arguments(execution_registry):
    """Test calling function with invalid arguments returns ToolError."""
    # Missing required argument
    result = execution_registry.call("multi_param_function", {"count": 5})
    
    assert isinstance(result, ToolError)
    assert "Invalid arguments" in result.error


def test_call_multi_param_function(execution_registry):
    """Test calling function with multiple parameters."""
    result = execution_registry.call("multi_param_function", {
        "name": "test", 
        "count": 42, 
        "enabled": False
    })
    
    assert result == {"name": "test", "count": 42, "enabled": False}


def test_call_multi_param_function_with_defaults(execution_registry):
    """Test calling function with default parameter values."""
    result = execution_registry.call("multi_param_function", {
        "name": "test", 
        "count": 42
    })
    
    assert result == {"name": "test", "count": 42, "enabled": True}


def test_call_primitive_param_function(execution_registry):
    """Test calling function with single primitive parameter."""
    result = execution_registry.call("primitive_param_function", {"message": "hello"})
    
    assert result == {"processed": "HELLO"}


def test_get_schemas(execution_registry):
    """Test getting OpenAI-format schemas for all tools."""
    schemas = execution_registry.get_schemas()
    
    assert len(schemas) == 2
    schema_names = [s["function"]["name"] for s in schemas]
    assert "multi_param_function" in schema_names
    assert "primitive_param_function" in schema_names


def test_call_with_json_response(execution_registry):
    """Test call_with_json_response method."""
    result = execution_registry.call_with_json_response(
        "multi_param_function", 
        {"name": "test", "count": 5, "enabled": True},
        "test_id"
    )
    
    # Should return JSON string
    assert isinstance(result, str)
    import json
    parsed = json.loads(result)
    assert parsed["name"] == "test"
    assert parsed["count"] == 5
    assert parsed["enabled"] == True


def test_call_with_json_response_error(execution_registry):
    """Test call_with_json_response with error."""
    result = execution_registry.call_with_json_response(
        "nonexistent", 
        {},
        "test_id"
    )
    
    # Should return JSON error string
    assert isinstance(result, str)
    import json
    parsed = json.loads(result)
    assert "error" in parsed
