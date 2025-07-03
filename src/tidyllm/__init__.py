"""TidyAgent - Clean tool management for LLMs."""

from tidyllm.adapters.cli import cli_main
from tidyllm.discover import (
    discover_tools_in_directory,
    discover_tools_in_package,
)
from tidyllm.prompt import read_prompt
from tidyllm.registry import REGISTRY, ToolError, ToolResult, register
from tidyllm.registry import REGISTRY as FunctionLibrary
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
