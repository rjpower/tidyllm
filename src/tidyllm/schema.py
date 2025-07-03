"""Function schema extraction and JSON schema generation."""

import asyncio
import base64
import inspect
import warnings
from collections.abc import Callable, Iterable
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints
from uuid import UUID

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


def parse_from_json(result: Any, result_type: type) -> Any:
    """Parse a result from JSON to the specified type using Pydantic conventions."""

    if result is None:
        return None

    # Handle Pydantic models
    if inspect.isclass(result_type) and issubclass(result_type, BaseModel):
        return result_type.model_validate(result)

    # Get origin and args for generic types
    origin = get_origin(result_type)
    args = get_args(result_type)

    # Handle list types
    if origin is list:
        if not isinstance(result, list):
            raise ValueError(f"Expected list, got {type(result)}")
        item_type = args[0] if args else Any
        return [parse_from_json(item, item_type) for item in result]

    # Handle dict types
    if origin is dict:
        if not isinstance(result, dict):
            raise ValueError(f"Expected dict, got {type(result)}")
        value_type = args[1] if len(args) > 1 else Any
        return {
            key: parse_from_json(value, value_type) for key, value in result.items()
        }

    if origin is set:
        if not isinstance(result, list):
            raise ValueError(f"Expected list for set, got {type(result)}")
        item_type = args[0] if args else Any
        return {parse_from_json(item, item_type) for item in result}

    # Handle primitive types
    if result_type is int:
        return int(result)
    elif result_type is float:
        return float(result)
    elif result_type is str:
        return str(result)
    elif result_type is bool:
        return bool(result)
    elif result_type is datetime:
        return datetime.fromisoformat(result.replace("Z", "+00:00"))
    elif result_type is date:
        return date.fromisoformat(result)
    elif result_type is time:
        return time.fromisoformat(result)
    elif result_type is bytes:
        return base64.b64decode(result)
    elif result_type is Decimal:
        return Decimal(str(result))
    elif result_type is UUID:
        return UUID(result)
    elif result_type is Path:
        return Path(result)
    else:
        warnings.warn(
            f"Unsupported result type: {result_type}. Returning raw result: {result}",
            UserWarning,
            stacklevel=2,
        )
        return result


class FunctionDescription:
    """Description of a function for tool calling with proper Pydantic validation."""

    function: Callable
    function_schema: JSONSchema
    name: str
    description: str
    tags: list[str]

    result_type: type
    args_model: type[BaseModel]
    args_json_schema: dict

    def __init__(
        self, 
        func: Callable, 
        doc_override: str | None = None,
        description: str = "",
        tags: list[str] | None = None,
    ):
        """Initialize function description with generated Pydantic model for validation.

        Args:
            func: The function to wrap
            doc_override: Override for function documentation
            description: Tool description
            tags: List of tags for categorization
        """
        self.function = func
        self.name = func.__name__
        self.description = description
        self.tags = tags or []

        self.sig = inspect.signature(func)
        self.is_async = inspect.iscoroutinefunction(func)

        # Extract return type
        hints = get_type_hints(func)
        self.result_type = hints.get('return', Any)

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

        # Get all parameters (no context filtering needed with contextvar approach)
        all_params = sig.parameters

        if len(all_params) == 1:
            # Single parameter - check if it's already a Pydantic model
            param_name, param = next(iter(all_params.items()))
            param_type = hints.get(param_name, Any)

            if inspect.isclass(param_type) and issubclass(param_type, BaseModel):
                # Already a Pydantic model - use it directly
                return param_type

        # Create dynamic model for multiple parameters or single non-model parameter
        field_definitions = {}

        for param_name, param in all_params.items():
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

    def arg_model_from_args(
        self, *args: Iterable[Any], **kwargs: dict[str, Any]
    ) -> BaseModel:
        """Construct the argument model from args & kwargs"""
        bound_args = self.sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        args_instance = self.args_model(**bound_args.arguments)
        return args_instance

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
        all_params = sig.parameters

        if len(all_params) == 1:
            param_name, param = next(iter(all_params.items()))
            param_type = hints.get(param_name, Any)

            if inspect.isclass(param_type) and issubclass(param_type, BaseModel):
                # Single Pydantic model - return the model instance
                return {param_name: validated_model}

        # Multiple parameters or single primitive - return field values
        return validated_model.model_dump()

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
