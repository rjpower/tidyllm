"""Function schema extraction and JSON schema generation."""

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel, create_model

# make a typed dict for the json schema
from typing_extensions import TypedDict

from tidyllm.docstring import (
    enhance_schema_with_docs,
)


class FunctionSchema(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]


class JSONSchema(TypedDict):
    type: str
    function: FunctionSchema


def generate_tool_schema(args_json_schema: dict, name: str, doc: str) -> JSONSchema:
    """Generate OpenAI-compatible tool schema from function using Pydantic models."""
    parameters_schema = args_json_schema

    description = doc
    description = description.strip()

    schema: JSONSchema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters_schema,
        },
    }

    # Enhance schema with griffe-extracted documentation
    schema = enhance_schema_with_docs(schema, doc)  # type: ignore

    return schema


class FunctionDescription:
    """Description of a function for tool calling with proper Pydantic validation."""

    function: Callable
    function_schema: JSONSchema
    name: str
    args_model: type[BaseModel]
    args_json_schema: dict

    def __init__(self, func: Callable, doc_override: str | None = None):
        """Initialize function description with generated Pydantic model for validation.

        Args:
            func: The function to wrap
        """
        self.function = func
        self.name = func.__name__

        # Check if function takes context and extract context type
        sig = inspect.signature(func)
        self.takes_ctx = "ctx" in sig.parameters

        # Extract context type from type hints if available
        self.context_type: type | None = None
        if self.takes_ctx:
            try:
                hints = get_type_hints(func)
                self.context_type = hints.get("ctx")
            except (NameError, AttributeError):
                # Fallback if type hints can't be resolved
                self.context_type = None

        self.is_async = inspect.iscoroutinefunction(func)

        # Generate Pydantic model for argument validation
        self.args_model = self._create_args_model(func)
        self.args_json_schema = self.args_model.model_json_schema()

        # Use doc_override if provided, otherwise use function's docstring or generate default
        doc_text = doc_override or (func.__doc__ or f"Function {self.name}")
        self.function_schema = generate_tool_schema(self.args_json_schema, self.name, doc_text)

    def _create_args_model(self, func: Callable) -> type[BaseModel]:
        """Create a Pydantic model for function arguments.

        Args:
            func: Function to analyze

        Returns:
            Dynamically created Pydantic model class
        """
        sig = inspect.signature(func)
        hints = get_type_hints(func)

        # Get non-context parameters
        non_ctx_params = {name: param for name, param in sig.parameters.items() if name != "ctx"}

        if len(non_ctx_params) == 1:
            # Single parameter - check if it's already a Pydantic model
            param_name, param = next(iter(non_ctx_params.items()))
            param_type = hints.get(param_name, Any)

            if inspect.isclass(param_type) and issubclass(param_type, BaseModel):
                # Already a Pydantic model - use it directly
                return param_type

        # Create dynamic model for multiple parameters or single non-model parameter
        field_definitions = {}

        for param_name, param in non_ctx_params.items():
            param_type = hints.get(param_name, Any)

            if param.default is not inspect.Parameter.empty:
                # Parameter has default value
                field_definitions[param_name] = (param_type, param.default)
            else:
                # Required parameter
                field_definitions[param_name] = (param_type, ...)

        # Create the dynamic model
        model_name = f"{func.__name__.title()}Args"
        return create_model(model_name, **field_definitions)

    def validate_and_parse_args(self, json_args: dict) -> dict:
        """Validate JSON arguments and return parsed values.

        Args:
            json_args: Raw JSON arguments from tool call

        Returns:
            Validated and parsed arguments ready for function call
        """
        # Use the Pydantic model to validate and parse
        validated_model = self.args_model.model_validate(json_args)

        # Check if original function has single Pydantic model parameter
        sig = inspect.signature(self.function)
        hints = get_type_hints(self.function)
        non_ctx_params = {name: param for name, param in sig.parameters.items() if name != "ctx"}

        if len(non_ctx_params) == 1:
            param_name, param = next(iter(non_ctx_params.items()))
            param_type = hints.get(param_name, Any)

            if inspect.isclass(param_type) and issubclass(param_type, BaseModel):
                # Single Pydantic model - return the model instance
                return {param_name: validated_model}

        # Multiple parameters or single primitive - return field values
        return validated_model.model_dump()

    def call_with_json_args(self, json_args: dict, context: Any = None) -> Any:
        """Call the function with JSON arguments after validation.

        Args:
            json_args: Raw JSON arguments
            context: Optional context object for injection

        Returns:
            Function result
        """
        # Validate and parse arguments
        parsed_args = self.validate_and_parse_args(json_args)

        # Add context if function expects it
        if self.takes_ctx:
            parsed_args["ctx"] = context

        return self.call(**parsed_args)

    def call(self, *args, **kwargs) -> Any:
        """Call the function directly with args/kwargs, handling async properly.
        Always returns the actual result synchronously (caller doesn't need to await).
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result (never a coroutine)
        """
        result = self.function(*args, **kwargs)

        if self.is_async:
            # Always run async functions and return the actual result
            try:
                # Check if we're already in an async context
                asyncio.get_running_loop()
                # We're in an async context but want to block - use new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, result)
                    return future.result()
            except RuntimeError:
                # No event loop running, so we can use asyncio.run directly
                return asyncio.run(result)

        return result

    async def call_async(self, *args, **kwargs) -> Any:
        """Call the function asynchronously.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        if self.is_async:
            return await self.function(*args, **kwargs)
        else:
            return self.function(*args, **kwargs)
