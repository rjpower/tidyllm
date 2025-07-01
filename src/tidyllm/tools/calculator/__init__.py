"""Calculator tool for basic mathematical operations."""

from tidyllm.prompt import module_dir, read_prompt
from tidyllm.registry import register
from tidyllm.tools.calculator.lib import (
    CalculatorArgs,
    CalculatorResult,
    perform_calculation,
)


@register(doc=read_prompt(module_dir(__file__) / "prompt.md"))
def calculator(args: CalculatorArgs) -> CalculatorResult:
    """Perform basic mathematical operations.
    
    Example usage: calculator({"operation": "add", "left": 10, "right": 5}) or calculator({"operation": "divide", "left": 20, "right": 4})
    """
    return perform_calculation(args)
