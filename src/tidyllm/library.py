"""Function library for tools with shared context."""

import json
import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from tidyllm.models import ToolError
from tidyllm.registry import REGISTRY
from tidyllm.schema import FunctionDescription, JSONSchema

logger = logging.getLogger(__name__)


class FunctionLibrary:
    """Container for tools with shared context."""

    def __init__(
        self,
        functions: list[Callable | FunctionDescription] | None = None,
        context: Any | None = None,
        registry=None,
    ):
        """
        Initialize library.

        Args:
            functions: List of functions with __tool_schema__ attribute or FunctionDescription objects
            context: Shared context dict
            registry: Registry to use (defaults to global REGISTRY)
        """
        self.context = context or {}
        self.registry = registry or REGISTRY

        # Create own function dictionary for faster lookups
        self._function_descriptions: dict[str, FunctionDescription] = {}

        if functions is not None:
            # Handle mixed list of callables and FunctionDescription objects
            for item in functions:
                if isinstance(item, FunctionDescription):
                    # Use provided FunctionDescription object directly
                    self._function_descriptions[item.name] = item
                else:
                    func_desc = FunctionDescription(item)
                    self._function_descriptions[func_desc.name] = func_desc
        else:
            # Default: use all functions from the registry
            for func_desc in self.registry.functions:
                self._function_descriptions[func_desc.name] = func_desc

        # Only assert if no functions were provided and none found in registry
        if functions is None:
            assert self._function_descriptions, "No functions provided or found in registry"

    def add_tool(self, tool_name: str, tool_call: Callable) -> None:
        """
        Add a new tool to the library.

        Args:
            tool_name: Name of the tool
            tool_call: Callable function implementing the tool

        Raises:
            ValueError: If tool_name already exists
        """
        if tool_name in self._function_descriptions:
            raise ValueError(f"Tool '{tool_name}' already exists in the library")

        func_desc = FunctionDescription(tool_call)
        func_desc.name = tool_name
        self._function_descriptions[tool_name] = func_desc
        logger.info(f"Added tool: {tool_name}")

    def call(self, tool_name: str, arguments: dict) -> Any:
        """
        Execute a function call with JSON arguments.

        Args:
            tool_name: Name of the function to call
            arguments: JSON dictionary of arguments

        Returns:
            Result from the function call
        """
        logger.info(f"Calling tool: {tool_name} with arguments: {arguments}")

        # Get tool description from internal dictionary
        func_desc = self._function_descriptions.get(tool_name)
        if not func_desc:
            error = f"Tool '{tool_name}' not found"
            logger.error(error)
            return ToolError(error=error)

        # Use pre-created FunctionDescription for validation
        try:
            call_kwargs = func_desc.validate_and_parse_args(arguments)
        except ValidationError as e:
            error = f"Invalid arguments: {e}"
            logger.error(error)
            return ToolError(error=error, details={"validation_errors": e.errors()})
        except Exception as e:
            error = f"Invalid arguments: {str(e)}"
            logger.error(error)
            return ToolError(error=error)

        try:
            # All tools now use contextvars for context access
            result = func_desc.function(**call_kwargs)

            # Handle async functions - they need to be awaited by the caller
            if func_desc.is_async:
                # Return the coroutine - it will be awaited by the caller
                return result

            logger.info(f"Tool {tool_name} completed successfully")
            return result

        except Exception as e:
            error = f"Tool execution failed: {str(e)}"
            logger.exception(e, stack_info=True)
            return ToolError(error=error)

    @property
    def function_descriptions(self) -> list[FunctionDescription]:
        """Get all function descriptions."""
        return list(self._function_descriptions.values())

    def get_schemas(self) -> list[JSONSchema]:
        """Get OpenAI-format schemas for all tools."""
        return [
            func_desc.function_schema
            for func_desc in self._function_descriptions.values()
        ]

    def validate_context(self, tool_name: str) -> bool:
        """Check if context satisfies tool requirements.
        
        Note: With contextvar approach, context validation is no longer needed
        at the library level. Tools access context directly via get_tool_context().
        """
        # Still return False for nonexistent tools
        func_desc = self._function_descriptions.get(tool_name)
        if not func_desc:
            return False
        return True

    def call_with_json_response(self, name: str, args: dict, id: str) -> str:
        """Execute a tool call, returning a tool call message with the result or error."""
        try:
            result = self.call(name, args)

            # use the custom string serializer if specified
            if hasattr(result, "to_message"):
                result = result.to_message()  # type: ignore
            elif isinstance(result, BaseModel):
                result = result.model_dump_json()
            else:
                result = json.dumps(result)

            return result
        except Exception as e:
            logger.exception(e, stack_info=True)
            return json.dumps({"error": str(e), "type": type(e).__name__})
