"""Function schema extraction and JSON schema generation."""

import asyncio
import inspect
from collections.abc import Callable, Iterable
from typing import Any, get_type_hints

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from tidyllm.docstring import (
    DocstringInfo,
    extract_docs_from_string,
)
from tidyllm.types.serialization import (
    create_model_from_field_definitions,
)


class FunctionSchema(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]


class JSONSchema(TypedDict):
    type: str
    function: FunctionSchema


def function_schema_from_args(
    args_json_schema: dict, name: str, doc: str
) -> JSONSchema:
    """Generate OpenAI-compatible tool schema from function using Pydantic models."""
    parameters_schema = args_json_schema

    schema: JSONSchema = {
        "type": "function",
        "function": {
            "name": name,
            "description": doc.strip(),
            "parameters": parameters_schema,
        },
    }

    return schema


class FunctionDescription:
    """Description of a function for tool calling with proper Pydantic validation."""

    function: Callable
    function_schema: JSONSchema
    name: str
    description: str
    tags: list[str]
    docstring_info: DocstringInfo

    result_type: type
    args_model: type[BaseModel]
    args_json_schema: dict

    def __init__(
        self,
        func: Callable,
        name: str | None = None,
        doc_override: str | None = None,
        description: str = "",
        tags: list[str] | None = None,
    ):
        """Initialize function description with generated Pydantic model for validation.

        Args:
            func: The function to wrap
            name: Override for function name
            doc_override: Override for function documentation
            description: Tool description
            tags: List of tags for categorization
        """
        self.function = func
        self.name = name or self.function.__name__

        self.description = description or self.function.__doc__ or ""
        self.tags = tags or []

        self.sig = inspect.signature(func)
        self.is_async = inspect.iscoroutinefunction(func)

        # Extract return type
        hints = get_type_hints(func)
        self.result_type = hints.get('return', Any)

        # Parse docstring early and cache result
        doc_text = doc_override or (func.__doc__ or f"Function {self.name}")
        self.docstring_info = extract_docs_from_string(doc_text)

        # Generate Pydantic model for argument validation
        self.args_model = self._create_args_model(func)

        # Pydantic handles schema generation robustly
        self.args_json_schema = self.args_model.model_json_schema()

        # Use parsed docstring info for schema generation
        self.function_schema = function_schema_from_args(
            self.args_json_schema,
            self.name,
            self.docstring_info.description or doc_text,
        )

    def _create_args_model(self, func: Callable) -> type[BaseModel]:
        """Create a Pydantic model for function arguments.

        Args:
            func: Function to analyze

        Returns:
            Dynamically created Pydantic model class
        """
        sig = inspect.signature(func)
        get_type_hints(func)

        # Get all parameters (no context filtering needed with contextvar approach)
        all_params = sig.parameters

        # Always create a unified model - no special casing for single Pydantic models
        field_definitions = {}

        for param_name, param in all_params.items():
            # Use the raw annotation from the signature to preserve Annotated types
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any

            # Get parameter description from docstring info
            param_description = self.docstring_info.parameters.get(param_name, "")

            if param.default is not inspect.Parameter.empty:
                field_definitions[param_name] = (
                    param_type,
                    Field(default=param.default, description=param_description),
                )
            else:
                field_definitions[param_name] = (
                    param_type,
                    Field(description=param_description),
                )

        # Create the dynamic model using the refactored utility
        model_name = f"{self.name.title()}Args"
        return create_model_from_field_definitions(model_name, field_definitions)

    def arg_model_from_args(
        self, *args: Iterable[Any], **kwargs: dict[str, Any]
    ) -> BaseModel:
        """Construct the argument model from args & kwargs"""
        bound_args = self.sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Always use the unified args model
        args_instance = self.args_model(**bound_args.arguments)
        return args_instance

    def validate_and_parse_args(self, json_args: dict) -> dict:
        """Validate JSON arguments and return parsed values.

        Args:
            json_args: Raw JSON arguments from tool call

        Returns:
            Validated and parsed arguments ready for function call
        """
        parsed_args = self.args_model.model_validate(json_args)
        return {k: getattr(parsed_args, k) for k in self.args_model.model_fields.keys()}

    def call(self, *args, **kwargs) -> Any:
        """Call the function directly with args/kwargs, handling async properly.

        Async functions are dispatched in a threadpool.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result (never a coroutine)
        """
        result = self.function(*args, **kwargs)

        if self.is_async:
            try:
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
