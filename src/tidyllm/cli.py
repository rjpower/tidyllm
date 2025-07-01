"""CLI generation from function signatures."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
from pydantic import BaseModel

from tidyllm.protocol_utils import (
    get_cli_type_for_annotation,
    get_protocol_fields,
)
from tidyllm.schema import FunctionDescription
from tidyllm.context import set_tool_context
from tidyllm.tools.context import ToolContext


def create_context_from_cli_args(protocol_type: type, cli_args: dict[str, Any]) -> Any:
    """Create a context which matches `protocol_type` from CLI arguments.

    Args:
        protocol_type: The Protocol class defining the context interface
        cli_args: Dictionary of CLI arguments with ctx_ prefix stripped

    Returns:
        Mock context object with the specified attributes
    """

    class CliProtocol:
        pass

    context = CliProtocol()

    # Get expected fields from the protocol
    fields = get_protocol_fields(protocol_type)

    # Set attributes based on CLI args, with type conversion
    for field_name, field_type in fields.items():
        cli_value = cli_args.get(field_name)
        if cli_value is not None:
            # Convert CLI string values to appropriate types
            if field_type is bool:
                # Boolean flags are already handled by Click
                setattr(context, field_name, cli_value)
            elif field_type is int:
                setattr(context, field_name, int(cli_value))
            elif field_type is float:
                setattr(context, field_name, float(cli_value))
            elif field_type is Path or field_type == Path:
                setattr(context, field_name, Path(cli_value))
            elif hasattr(field_type, "__origin__"):
                # Handle generic types like set[str]
                origin = getattr(field_type, "__origin__", None)
                if origin is set:
                    # Split comma-separated values into a set
                    setattr(context, field_name, set(cli_value.split(",")))
                elif origin is list:
                    # Split comma-separated values into a list
                    setattr(context, field_name, cli_value.split(","))
                else:
                    setattr(context, field_name, cli_value)
            else:
                setattr(context, field_name, cli_value)
        else:
            # Set default values for missing fields
            if field_type is bool:
                setattr(context, field_name, False)
            elif field_type is str:
                setattr(context, field_name, "")
            elif field_type is Path or field_type == Path:
                setattr(context, field_name, Path("."))
            elif hasattr(field_type, "__origin__"):
                origin = getattr(field_type, "__origin__", None)
                if origin is set:
                    setattr(context, field_name, set())
                elif origin is list:
                    setattr(context, field_name, [])
                else:
                    setattr(context, field_name, None)
            else:
                setattr(context, field_name, None)

    return context


@dataclass
class FieldInfo:
    """Represents information about a field in a Pydantic model."""

    name: str
    type: type
    is_required: bool
    help_text: str


def get_pydantic_field_info(model: type[BaseModel], field_name: str) -> FieldInfo:
    """Extract field info from Pydantic model.
    
    Returns:
        tuple: (field_type, is_required, help_text)
    """
    field_info = model.model_fields.get(field_name)
    if field_info is None:
        return FieldInfo(
            name=field_name,
            type=Any,
            is_required=True,
            help_text=f"Value for {field_name}",
        )

    field_type = field_info.annotation or Any
    is_required = field_info.is_required()
    help_text = field_info.description or f"Value for {field_name}"

    return FieldInfo(
        name=field_name,
        type=field_type,
        is_required=is_required,
        help_text=help_text,
    )


class CliOption:
    """Represents a CLI option configuration."""

    def __init__(
        self,
        name: str,
        param_name: str,
        type_annotation: type,
        help_text: str,
        is_flag: bool = False,
        multiple: bool = False,
        required: bool = True,
    ):
        self.name = name
        self.param_name = param_name
        self.type_annotation = type_annotation
        self.help_text = help_text
        self.is_flag = is_flag
        self.multiple = multiple
        self.required = required


def create_click_option(option: CliOption, is_context_option: bool = False):
    """Create a Click option decorator from a CliOption configuration."""
    if option.is_flag:
        return click.option(
            option.name,
            option.param_name,
            is_flag=True,
            help=option.help_text,
            required=False,  # Never required at Click level - we validate in the function
        )
    elif option.multiple:
        return click.option(
            option.name,
            option.param_name,
            multiple=True,
            help=f"{option.help_text} (can be specified multiple times)",
        )
    else:
        # Map type annotations to Click types
        click_type = str  # Default
        cli_type, _ = get_cli_type_for_annotation(option.type_annotation)
        if cli_type == "int":
            click_type = int
        elif cli_type == "float":
            click_type = float
        elif cli_type == "path":
            click_type = click.Path(exists=False)

        return click.option(
            option.name,
            option.param_name,
            type=click_type,
            help=option.help_text,
            required=False,  # Never required at Click level - we validate in the function
        )


def collect_function_options(func_desc: FunctionDescription) -> list[CliOption]:
    """Collect CLI options from function arguments."""
    options = []

    for field_name in func_desc.args_model.model_fields:
        field_info = get_pydantic_field_info(func_desc.args_model, field_name)

        option_name = f"--{field_name.replace('_', '-')}"
        param_name = field_name.replace("-", "_")

        # Determine option type
        is_flag = field_info.type is bool
        is_list = (
            hasattr(field_info.type, "__origin__")
            and field_info.type.__origin__ is list
        )

        options.append(
            CliOption(
                name=option_name,
                param_name=param_name,
                type_annotation=field_info.type,
                help_text=field_info.help_text,
                is_flag=is_flag,
                multiple=is_list,
                required=field_info.is_required,
            )
        )

    return options


def collect_context_options(func_desc: FunctionDescription, context_class: type | None = None) -> list[CliOption]:
    """Collect CLI options from Config fields automatically."""
    options = []

    # Automatically enumerate Config fields using Pydantic model fields
    from tidyllm.tools.config import Config
    
    for field_name, field_info in Config.model_fields.items():
        option_name = f"--ctx-{field_name.replace('_', '-')}"
        param_name = f"ctx_{field_name}"
        
        field_type = field_info.annotation
        help_text = field_info.description or f"Config field: {field_name}"
        
        _, is_flag = get_cli_type_for_annotation(field_type)
        
        options.append(
            CliOption(
                name=option_name,
                param_name=param_name,
                type_annotation=field_type,
                help_text=help_text,
                is_flag=is_flag,
                multiple=False,
                required=field_info.is_required(),
            )
        )

    return options


def parse_cli_arguments(
    kwargs: dict[str, Any], func_options: list[CliOption], ctx_options: list[CliOption]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse CLI arguments into function args and context args."""
    args_dict = {}
    ctx_args = {}

    # Parse function arguments
    for option in func_options:
        value = kwargs.get(option.param_name)
        if value is not None:
            if option.multiple and isinstance(value, tuple):
                # Convert tuple from click multiple option to list
                args_dict[option.param_name] = list(value)
            else:
                args_dict[option.param_name] = value

    # Parse context arguments
    for option in ctx_options:
        value = kwargs.get(option.param_name)
        if value is not None:
            # Remove 'ctx_' prefix for context field name
            ctx_field_name = option.param_name[4:]
            ctx_args[ctx_field_name] = value

    return args_dict, ctx_args


def generate_cli(func: Callable) -> click.Command:
    """Generate a Click CLI for a registered tool using FunctionDescription."""
    func_desc = FunctionDescription(func)
    return _generate_cli_from_description(func_desc)


def cli_main(func: Callable):
    try:
        generate_cli(func)(standalone_mode=False)
    except Exception as _:
        import traceback
        click.echo(traceback.format_exc())
        click.echo(json.dumps({"error": "An error occurred while generating CLI"}))


def _generate_cli_from_description(func_desc: FunctionDescription) -> click.Command:
    """Generate CLI from a FunctionDescription."""

    # Collect all CLI options
    func_options = collect_function_options(func_desc)
    ctx_options = collect_context_options(func_desc)

    @click.command(name=func_desc.name)
    @click.option("--json", "json_input", help="JSON input for all arguments")
    def cli(json_input: str | None, **kwargs):
        """Auto-generated CLI for tool."""
        if json_input:
            # Parse JSON input
            try:
                args_dict = json.loads(json_input)
                ctx_args = {}  # JSON mode doesn't support context args
            except json.JSONDecodeError as e:
                click.echo(json.dumps({"error": f"Invalid JSON: {str(e)}"}))
                return
        else:
            # Parse CLI arguments using helper function
            args_dict, ctx_args = parse_cli_arguments(kwargs, func_options, ctx_options)

        # Create context from CLI arguments 
        from tidyllm.tools.config import Config
        
        # Create config with CLI overrides
        config_kwargs = {}
        for key, value in ctx_args.items():
            if value is not None:  # Only set non-None values
                config_kwargs[key] = value
        
        config = Config(**config_kwargs)
        context = ToolContext(config=config)

        # Execute tool with context variable
        with set_tool_context(context):
            # Use the new direct approach - validate args and call function
            parsed_args = func_desc.validate_and_parse_args(args_dict)
            result = func_desc.call(**parsed_args)

        # Handle async functions by running them in asyncio
        if hasattr(result, "__await__"):
            import asyncio

            result = asyncio.run(result)

        # Output as JSON
        if isinstance(result, BaseModel):
            output = result.model_dump()
        else:
            output = result

        click.echo(json.dumps(output))

    # Add all CLI options using helper function
    for option in func_options:
        cli_option = create_click_option(option, is_context_option=False)
        cli = cli_option(cli)

    for option in ctx_options:
        cli_option = create_click_option(option, is_context_option=True)
        cli = cli_option(cli)

    return cli
