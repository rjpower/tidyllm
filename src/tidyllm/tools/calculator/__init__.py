"""Calculator tool for basic mathematical operations."""

from tidyllm.prompt import module_dir, read_prompt
from tidyllm.registry import register
from tidyllm.tools.calculator.lib import (
    CalculatorResult,
    perform_calculation,
)


@register(doc=read_prompt(module_dir(__file__) / "prompt.md"))
def calculator(operation: str, left: float, right: float) -> CalculatorResult:
    """Perform basic mathematical operations.
    
    Args:
        operation: Mathematical operation to perform (add, subtract, multiply, divide)
        left: Left operand (first number)
        right: Right operand (second number)
    
    Example usage: calculator("add", 10, 5) or calculator("divide", 20, 4)
    """
    return perform_calculation(operation, left, right)
