"""Tests for calculator tool."""

import pytest

from tidyllm import REGISTRY
from tidyllm.registry import ToolError
from tidyllm.tools.calculator import calculator
from tidyllm.tools.calculator.lib import (
    CalculatorArgs,
    CalculatorResult,
    perform_calculation,
)


class TestCalculatorLib:
    """Test core calculator functionality."""

    def test_addition(self):
        """Test addition operation."""
        args = CalculatorArgs(operation="add", left=10, right=5)
        result = perform_calculation(args)

        assert isinstance(result, CalculatorResult)
        assert result.result == 15
        assert result.operation == "add"
        assert result.expression == "10.0 + 5.0 = 15.0"

    def test_subtraction(self):
        """Test subtraction operation."""
        args = CalculatorArgs(operation="subtract", left=10, right=3)
        result = perform_calculation(args)

        assert result.result == 7
        assert result.operation == "subtract"
        assert "10.0 - 3.0 = 7.0" in result.expression

    def test_multiplication(self):
        """Test multiplication operation."""
        args = CalculatorArgs(operation="multiply", left=4, right=6)
        result = perform_calculation(args)

        assert result.result == 24
        assert result.operation == "multiply"

    def test_division(self):
        """Test division operation."""
        args = CalculatorArgs(operation="divide", left=15, right=3)
        result = perform_calculation(args)

        assert result.result == 5
        assert result.operation == "divide"

    def test_division_by_zero(self):
        """Test division by zero error."""
        args = CalculatorArgs(operation="divide", left=10, right=0)

        with pytest.raises(ValueError, match="Cannot divide by zero"):
            perform_calculation(args)

    def test_float_operations(self):
        """Test operations with floating point numbers."""
        args = CalculatorArgs(operation="add", left=3.14, right=2.86)
        result = perform_calculation(args)

        assert abs(result.result - 6.0) < 0.0001  # Float precision

    def test_negative_numbers(self):
        """Test operations with negative numbers."""
        args = CalculatorArgs(operation="multiply", left=-5, right=3)
        result = perform_calculation(args)

        assert result.result == -15


class TestCalculatorTool:
    """Test calculator tool registration and execution."""

    def setup_method(self):
        """Ensure calculator is registered."""
        # If calculator isn't already registered, register it
        if REGISTRY.get_description("calculator") is None:
            # Import will cause the @register decorator to execute
            import importlib

            import tidyllm.tools.calculator

            importlib.reload(tidyllm.tools.calculator)

    def test_tool_registered(self):
        """Test that calculator tool is properly registered."""
        tool_desc = REGISTRY.get_description("calculator")
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
        args = CalculatorArgs(operation="add", left=8, right=7)
        result = calculator(args)

        assert isinstance(result, CalculatorResult)
        assert result.result == 15
        assert result.operation == "add"

    def test_tool_execution_error(self):
        """Test tool execution with error."""
        args = CalculatorArgs(operation="divide", left=10, right=0)

        with pytest.raises(ValueError, match="Cannot divide by zero"):
            calculator(args)

    def test_schema_generation(self):
        """Test that schema is generated correctly."""
        tool_desc = REGISTRY.get_description("calculator")
        schema = tool_desc.function_schema

        # Check basic structure
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "calculator"

        # Check parameters
        params = schema["function"]["parameters"]["properties"]

        # Check operation parameter
        assert "operation" in params
        operation_param = params["operation"]
        assert "enum" in operation_param  # Should have limited values
        assert set(operation_param["enum"]) == {"add", "subtract", "multiply", "divide"}

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

    def setup_method(self):
        """Ensure calculator is registered."""
        # If calculator isn't already registered, register it
        if REGISTRY.get_description("calculator") is None:
            # Import will cause the @register decorator to execute
            import importlib

            import tidyllm.tools.calculator

            importlib.reload(tidyllm.tools.calculator)

    def test_library_execution(self):
        """Test calculator execution through Registry."""
        from tidyllm.registry import Registry
        
        # Create a new registry and register the function
        library = Registry()
        library.register(calculator)

        # Execute through library
        result = library.call(
            "calculator", {"operation": "multiply", "left": 6, "right": 7}
        )

        assert isinstance(result, CalculatorResult)
        assert result.result == 42
        assert result.operation == "multiply"

    def test_library_error_handling(self):
        """Test error handling through Registry."""
        from tidyllm.registry import Registry
        
        library = Registry()
        library.register(calculator)

        result = library.call(
            "calculator", {"operation": "divide", "left": 5, "right": 0}
        )

        assert isinstance(result, ToolError)
        assert "divide by zero" in result.error.lower()

    def test_invalid_arguments(self):
        """Test handling of invalid arguments."""
        from tidyllm.registry import Registry
        
        library = Registry()
        library.register(calculator)

        # Missing required argument
        result = library.call(
            "calculator", {"operation": "add", "left": 5}
        )

        # Should return ToolError due to validation failure
        assert isinstance(result, ToolError)
