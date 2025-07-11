"""TidyLLM - Python library for LLM tool management."""

# Core registry and tool management
# CLI support
from tidyllm.adapters.cli import cli_main

# Tool discovery
from tidyllm.discover import discover_tools_in_directory, discover_tools_in_package

# Schema generation and function descriptions
from tidyllm.function_schema import FunctionDescription
from tidyllm.registry import REGISTRY, register

# Data models and serialization
from tidyllm.types.linq import Enumerable, Table, from_iterable
from tidyllm.types.serialization import Serializable, from_json_dict, to_json_dict

# Context management - import if available
try:
    from tidyllm.context import get_tool_context, set_tool_context
except ImportError:
    get_tool_context = None
    set_tool_context = None

# Backwards compatibility aliases
FunctionLibrary = REGISTRY


__all__ = [
    # Core functionality
    "register",
    "REGISTRY",
    "FunctionLibrary",
    # Schema and function management
    "FunctionDescription",
    # Data models
    "Table",
    "Enumerable",
    "from_iterable",
    "Serializable",
    "from_json_dict",
    "to_json_dict",
    # Tool discovery
    "discover_tools_in_directory",
    "discover_tools_in_package",
    # CLI support
    "cli_main",
    # Context management
    "get_tool_context",
    "set_tool_context",
]
