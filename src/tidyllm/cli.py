"""CLI generation from function signatures."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_origin

import click
from pydantic import BaseModel

from tidyllm.schema import FunctionDescription
from tidyllm.context import set_tool_context


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
    if option.is_flag:
        return click.option(
            option.name,
            option.param_name,
            is_flag=True,
            help=option.help_text,
            required=False,
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
            required=False,
        )


def collect_function_options(func_desc: FunctionDescription) -> list[CliOption]:
    """Collect CLI options from function arguments."""
    options = []

    for field_name, field_info in func_desc.args_model.model_fields.items():
        option_name = f"--{field_name.replace('_', '-')}"
        param_name = field_name.replace("-", "_")

        field_type = field_info.annotation or Any
        help_text = field_info.description or f"Value for {field_name}"
        is_required = field_info.is_required()

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


def generate_cli(func: Callable, context_cls: type[BaseModel] = None) -> click.Command:
    """Generate a Click CLI for a registered tool using FunctionDescription."""
    func_desc = FunctionDescription(func)
    return _generate_cli_from_description(func_desc, context_cls)


def cli_main(func: Callable, context_cls: type[BaseModel] = None):
    try:
        generate_cli(func, context_cls)(standalone_mode=False)
    except Exception as _:
        import traceback
        click.echo(traceback.format_exc())
        click.echo(json.dumps({"error": "An error occurred while generating CLI"}))


def _generate_cli_from_description(func_desc: FunctionDescription, context_cls: type[BaseModel] = None) -> click.Command:
    """Generate CLI from a FunctionDescription."""

    # Collect CLI options from function arguments
    func_options = collect_function_options(func_desc)

    @click.command(name=func_desc.name)
    @click.option("--json", "json_input", help="JSON input for all arguments")
    def cli(json_input: str | None, **kwargs):
        """Auto-generated CLI for tool."""
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

        # Execute tool with context if provided
        if context_cls:
            context = context_cls()
            with set_tool_context(context):
                parsed_args = func_desc.validate_and_parse_args(args_dict)
                result = func_desc.call(**parsed_args)
        else:
            parsed_args = func_desc.validate_and_parse_args(args_dict)
            result = func_desc.call(**parsed_args)

        # Output as JSON
        if isinstance(result, BaseModel):
            output = result.model_dump()
        else:
            output = result

        click.echo(json.dumps(output))

    # Add all CLI options
    for option in func_options:
        cli_option = create_click_option(option)
        cli = cli_option(cli)

    return cli


def multi_cli_main(functions: list[Callable], default_function: str | None = None, context_cls: type[BaseModel] = None):
    """Create a click CLI that dispatches to multiple functions.

    Args:
        functions: List of functions to create CLI commands for
        default_function: Name of default function if no command specified
        context_cls: Context class to instantiate for each function call
    """

    @click.group(invoke_without_command=True)
    @click.pass_context
    def cli(ctx):
        """Multi-function CLI."""
        if ctx.invoked_subcommand is None:
            if default_function and default_function in functions:
                # Invoke default function
                ctx.invoke(cli.commands[default_function])
            else:
                # Show help
                click.echo(ctx.get_help())

    # Add subcommands for each function
    for func in functions:
        cmd_name = func.__name__
        func_desc = FunctionDescription(func)
        cmd = _generate_cli_from_description(func_desc, context_cls)

        # Wrap the command to handle exceptions properly
        def make_wrapper(original_cmd, _cmd_name=cmd_name):
            def wrapper(*args, **kwargs):
                try:
                    return original_cmd.callback(*args, **kwargs)
                except Exception as e:
                    import traceback

                    click.echo(traceback.format_exc())
                    click.echo(json.dumps({"error": f"Error in {_cmd_name}: {str(e)}"}))

            return wrapper

        # Replace the callback with our wrapper
        wrapped_cmd = click.command(name=cmd_name)(make_wrapper(cmd))

        # Copy over the parameters from the original command
        wrapped_cmd.params = cmd.params[:]  # Copy params list

        cli.add_command(wrapped_cmd)

    try:
        cli(standalone_mode=False)
    except Exception as e:
        import traceback

        click.echo(traceback.format_exc())
        click.echo(json.dumps({"error": f"CLI error: {str(e)}"}))
