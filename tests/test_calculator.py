"""Tests for calculator tool."""

import pytest

from tidyllm.registry import ToolError
from tidyllm.tools.calculator import (
    CalculatorResult,
    calculator,
    perform_calculation,
)


@pytest.fixture
def registry():
    """Create a fresh registry with calculator registered."""
    from tidyllm.registry import Registry
    
    registry = Registry()
    registry.register(calculator)
    return registry


class TestCalculatorLib:
    """Test core calculator functionality."""

    def test_addition(self):
        """Test addition operation."""
        result = perform_calculation(operation="add", left=10, right=5)

        assert isinstance(result, CalculatorResult)
        assert result.result == 15
        assert result.operation == "add"
        assert result.expression == "10 + 5 = 15"

    def test_subtraction(self):
        """Test subtraction operation."""
        result = perform_calculation(operation="subtract", left=10, right=3)

        assert result.result == 7
        assert result.operation == "subtract"
        assert "10 - 3 = 7" in result.expression

    def test_multiplication(self):
        """Test multiplication operation."""
        result = perform_calculation(operation="multiply", left=4, right=6)

        assert result.result == 24
        assert result.operation == "multiply"

    def test_division(self):
        """Test division operation."""
        result = perform_calculation(operation="divide", left=15, right=3)

        assert result.result == 5
        assert result.operation == "divide"

    def test_division_by_zero(self):
        """Test division by zero error."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            perform_calculation(operation="divide", left=10, right=0)

    def test_float_operations(self):
        """Test operations with floating point numbers."""
        result = perform_calculation(operation="add", left=3.14, right=2.86)

        assert abs(result.result - 6.0) < 0.0001  # Float precision

    def test_negative_numbers(self):
        """Test operations with negative numbers."""
        result = perform_calculation(operation="multiply", left=-5, right=3)

        assert result.result == -15


class TestCalculatorTool:
    """Test calculator tool registration and execution."""

    def test_tool_registered(self, registry):
        """Test that calculator tool is properly registered."""
        tool_desc = registry.get_description("calculator")
        assert tool_desc is not None
        assert tool_desc.function.__name__ == "calculator"

        # Check tool has schema via FunctionDescription
        schema = tool_desc.function_schema
        assert schema["function"]["name"] == "calculator"
        assert "operation" in schema["function"]["parameters"]["properties"]
        assert "left" in schema["function"]["parameters"]["properties"]
        assert "right" in schema["function"]["parameters"]["properties"]

    def test_tool_execution_success(self):
        """Test successful tool execution."""
        result = calculator(operation="add", left=8, right=7)

        assert isinstance(result, CalculatorResult)
        assert result.result == 15
        assert result.operation == "add"

    def test_tool_execution_error(self):
        """Test tool execution with error."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            calculator(operation="divide", left=10, right=0)

    def test_schema_generation(self, registry):
        """Test that schema is generated correctly."""
        tool_desc = registry.get_description("calculator")
        schema = tool_desc.function_schema

        # Check basic structure
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "calculator"

        # Check parameters
        params = schema["function"]["parameters"]["properties"]

        # Check operation parameter
        assert "operation" in params
        operation_param = params["operation"]
        # Operation parameter should be a string
        assert operation_param["type"] == "string"

        # Check number parameters
        assert "left" in params
        assert "right" in params
        assert params["left"]["type"] == "number"
        assert params["right"]["type"] == "number"

        # Check required parameters
        required = schema["function"]["parameters"]["required"]
        assert "operation" in required
        assert "left" in required
        assert "right" in required


class TestCalculatorIntegration:
    """Test calculator integration with FunctionLibrary."""

    def test_library_execution(self, registry):
        """Test calculator execution through Registry."""
        # Execute through library
        result = registry.call(
            "calculator", {"operation": "multiply", "left": 6, "right": 7}
        )

        assert isinstance(result, CalculatorResult)
        assert result.result == 42
        assert result.operation == "multiply"

    def test_library_error_handling(self, registry):
        """Test error handling through Registry."""
        # Runtime errors are raised directly, not wrapped in ToolError
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            registry.call(
                "calculator", {"operation": "divide", "left": 5, "right": 0}
            )

    def test_invalid_arguments(self, registry):
        """Test handling of invalid arguments."""
        # Missing required argument - validation errors are raised directly
        with pytest.raises(Exception):  # ValidationError or similar
            registry.call(
                "calculator", {"operation": "add", "left": 5}
            )