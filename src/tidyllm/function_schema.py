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
from tidyllm.serialization import (
    create_model_from_field_definitions,
    transform_argument_type,
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
        hints = get_type_hints(func)

        # Get all parameters (no context filtering needed with contextvar approach)
        all_params = sig.parameters

        if len(all_params) == 1:
            # Single parameter - check if it's already a Pydantic model
            param_name, param = next(iter(all_params.items()))
            param_type = hints.get(param_name, Any)

            if inspect.isclass(param_type) and issubclass(param_type, BaseModel):
                # Check if we have docstring info that could enhance the model
                if any(field_name in self.docstring_info.parameters for field_name in param_type.model_fields):
                    # We have parameter descriptions that could enhance the existing model
                    # Create a new model with enhanced descriptions
                    field_definitions = {}
                    for field_name, field_info in param_type.model_fields.items():
                        field_type = field_info.annotation
                        field_type = transform_argument_type(field_type)
                        param_description = self.docstring_info.parameters.get(field_name, "")

                        if field_info.default is not ...:
                            field_definitions[field_name] = (
                                field_type,
                                Field(default=field_info.default, description=param_description),
                            )
                        else:
                            field_definitions[field_name] = (
                                field_type,
                                Field(description=param_description),
                            )

                    model_name = f"{self.name.title()}Args"
                    return create_model_from_field_definitions(model_name, field_definitions)
                else:
                    # No parameter descriptions - use the model directly
                    return param_type

        # Create dynamic model for multiple parameters or single non-model parameter
        field_definitions = {}

        for param_name, param in all_params.items():
            param_type = hints.get(param_name, Any)
            param_type = transform_argument_type(param_type)

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

        # Special case: function takes single Pydantic model, but we're called with it wrapped
        if (len(bound_args.arguments) == 1 and 
            len(self.sig.parameters) == 1):

            param_name = list(self.sig.parameters.keys())[0]
            param_value = bound_args.arguments[param_name]

            # If the parameter value is already a Pydantic model matching the expected type
            if isinstance(param_value, BaseModel):
                param_annotation = list(self.sig.parameters.values())[0].annotation
                if isinstance(param_value, param_annotation):
                    return param_value

        # Original behavior
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
        # In this case, we let the user pass the "flattened" arguments to the
        # model directly instead of { "args": { "x": 1}}, the user can write {"x": 1}
        sig = inspect.signature(self.function)
        hints = get_type_hints(self.function)
        all_params = sig.parameters

        if len(all_params) == 1:
            param_name, _ = next(iter(all_params.items()))
            param_type = hints.get(param_name, Any)

            if inspect.isclass(param_type) and issubclass(param_type, BaseModel):
                # Single Pydantic model - return the model instance
                return {param_name: validated_model}

        # Multiple parameters or single primitive - return field values
        # The annotated types handle conversion automatically
        return validated_model.model_dump()

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
