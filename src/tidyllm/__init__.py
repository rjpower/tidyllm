"""TidyAgent - Clean tool management for LLMs."""

from tidyllm.cli import cli_main
from tidyllm.discover import (
    discover_tools_in_directory,
    discover_tools_in_package,
)
from tidyllm.prompt import read_prompt
from tidyllm.registry import REGISTRY, REGISTRY as FunctionLibrary, ToolError, ToolResult, register
from tidyllm.schema import FunctionDescription

__all__ = [
    "ToolError",
    "ToolResult",
    "register",
    "read_prompt",
    "REGISTRY",
    "FunctionLibrary",
    "FunctionDescription",
    "cli_main",
    "discover_tools_in_directory",
    "discover_tools_in_package",
]
