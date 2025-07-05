"""CLI generation from function signatures."""

import json
import pickle
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_origin

import click
from pydantic import BaseModel, ValidationError

from tidyllm.context import set_tool_context
from tidyllm.schema import FunctionDescription


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


def get_cli_type_for_annotation(type_annotation: type) -> tuple[str, bool]:
    """Convert Python type annotation to CLI type and flag status."""
    if type_annotation is bool:
        return "bool", True
    elif type_annotation is int:
        return "int", False
    elif type_annotation is float:
        return "float", False
    elif type_annotation is Path:
        return "path", False
    else:
        return "str", False


def create_click_option(option: CliOption):
    """Create a Click option decorator from a CliOption configuration."""
    # Click automatically converts hyphens to underscores in parameter names
    # So --audio-file-path becomes audio_file_path in the callback kwargs
    click_param_name = option.name.lstrip("-").replace("-", "_")

    if option.is_flag:
        return click.option(
            option.name,
            click_param_name,
            is_flag=True,
            help=option.help_text,
            required=option.required,
        )
    elif option.multiple:
        return click.option(
            option.name,
            click_param_name,
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
            click_param_name,
            type=click_type,
            help=option.help_text,
            required=option.required,
        )


def collect_function_options(func_desc: FunctionDescription) -> list[CliOption]:
    """Collect CLI options from function arguments."""
    options = []

    for field_name, field_info in func_desc.args_model.model_fields.items():
        option_name = f"--{field_name.replace('_', '-')}"
        param_name = field_name

        field_type = field_info.annotation or Any
        help_text = field_info.description or f"Value for {field_name}"
        # Make options not required for Click, we'll validate manually
        is_required = False

        # Determine option type
        is_flag = field_type is bool
        origin = get_origin(field_type)
        is_list = origin is list

        options.append(
            CliOption(
                name=option_name,
                param_name=param_name,
                type_annotation=field_type,
                help_text=help_text,
                is_flag=is_flag,
                multiple=is_list,
                required=is_required,
            )
        )

    return options


def parse_cli_arguments(
    kwargs: dict[str, Any], func_options: list[CliOption]
) -> dict[str, Any]:
    """Parse CLI arguments into function args."""
    args_dict = {}

    # Parse function arguments
    for option in func_options:
        value = kwargs.get(option.param_name)
        if value is not None:
            if option.multiple and isinstance(value, tuple):
                # Convert tuple from click multiple option to list
                args_dict[option.param_name] = list(value)
            else:
                args_dict[option.param_name] = value

    return args_dict


def _generate_cli_from_description(
    func_desc: FunctionDescription, context_cls: type[BaseModel] | None = None
) -> click.Command:
    """Generate CLI from a FunctionDescription."""

    # Collect CLI options from function arguments
    func_options = collect_function_options(func_desc)

    # Get the first line of the tool's docstring for the CLI help
    tool_doc = func_desc.description or "CLI for tool"
    first_line = tool_doc.split('\n')[0].strip()

    @click.command(name=func_desc.name, help=first_line)
    @click.option("--json", "json_input", help="JSON input for all arguments")
    @click.option(
        "--format",
        "output_format",
        type=click.Choice(["json", "pickle", "raw"]),
        default="json",
        help="Output format",
    )
    def cli(json_input: str | None, output_format: str, **kwargs):
        if json_input:
            # Parse JSON input
            try:
                args_dict = json.loads(json_input)
            except json.JSONDecodeError as e:
                click.echo(json.dumps({"error": f"Invalid JSON: {str(e)}"}))
                return
        else:
            # Parse CLI arguments
            args_dict = parse_cli_arguments(kwargs, func_options)

        # Validate and execute tool
        try:
            if context_cls:
                context = context_cls()
                with set_tool_context(context):
                    parsed_args = func_desc.validate_and_parse_args(args_dict)
                    result = func_desc.call(**parsed_args)
            else:
                parsed_args = func_desc.validate_and_parse_args(args_dict)
                result = func_desc.call(**parsed_args)
        except ValidationError as e:
            click.echo(json.dumps({"error": f"Validation error: {str(e)}"}))
            return

        # Output in specified format
        if output_format == "json":
            if isinstance(result, BaseModel):
                output = result.model_dump_json(indent=2)
            else:
                output = json.dumps(result, indent=2)
            click.echo(output)
        elif output_format == "pickle":
            pickled_data = pickle.dumps(result)
            sys.stdout.buffer.write(pickled_data)
        elif output_format == "raw":
            sys.stdout.buffer.write(result)

    # Add all CLI options
    for option in func_options:
        cli_option = create_click_option(option)
        cli = cli_option(cli)

    return cli


def generate_cli(
    func: Callable, context_cls: type[BaseModel] | None = None
) -> click.Command:
    """Generate a Click CLI for a registered tool using FunctionDescription."""
    func_desc = FunctionDescription(func)
    return _generate_cli_from_description(func_desc, context_cls)


def cli_main(
    functions: list[Callable] | Callable,
    context_cls: type[BaseModel] | None = None,
):
    """Create a click CLI that dispatches to multiple functions.

    Args:
        functions: List of functions to create CLI commands for
        default_function: Name of default function if no command specified
        context_cls: Context class to instantiate for each function call
    """

    if not isinstance(functions, list):
        functions = [functions]

    @click.group()
    @click.pass_context
    def cli(ctx):
        """Multi-function CLI."""
        pass

    for func in functions:
        func_desc = FunctionDescription(func)
        cmd = _generate_cli_from_description(func_desc, context_cls)
        cli.add_command(cmd)

    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname).1s %(asctime)s %(filename)s:%(lineno)d - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    cli(standalone_mode=True)
