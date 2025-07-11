"""Calculator tool for basic mathematical operations."""

from typing import Literal

from pydantic import BaseModel, Field

from tidyllm.registry import register


class CalculatorResult(BaseModel):
    """Result of calculator operation."""

    result: float = Field(description="The calculated result")
    operation: str = Field(description="The operation that was performed")
    expression: str = Field(description="Human-readable expression (e.g., '10 + 5 = 15')")


@register()
def calculator(operation: str, left: float, right: float) -> CalculatorResult:
    """Perform the requested mathematical operation.

    Args:
        operation: Mathematical operation to perform
        left: Left operand (first number)
        right: Right operand (second number)

    Returns:
        CalculatorResult with the computed value

    Raises:
        ValueError: For invalid operations or division by zero
    """

    if operation == "add":
        result = left + right
        expression = f"{left} + {right} = {result}"
    elif operation == "subtract":
        result = left - right
        expression = f"{left} - {right} = {result}"
    elif operation == "multiply":
        result = left * right
        expression = f"{left} * {right} = {result}"
    elif operation == "divide":
        if right == 0:
            raise ValueError("Cannot divide by zero")
        result = left / right
        expression = f"{left} / {right} = {result}"
    else:
        raise ValueError(f"Unknown operation: {operation}")

    return CalculatorResult(result=result, operation=operation, expression=expression)
