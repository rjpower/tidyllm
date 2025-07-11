"""CLI generation from function signatures."""

import json
import pickle
import sys
from collections.abc import Callable
from inspect import isclass
from pathlib import Path
from typing import Any, get_args, get_origin

import click
from pydantic import BaseModel

from tidyllm.context import set_tool_context
from tidyllm.function_schema import FunctionDescription
from tidyllm.types.linq import Enumerable
from tidyllm.types.part import Part
from tidyllm.types.serialization import to_json_dict


def get_click_type(annotation: type) -> Any:
    """Convert Python type to Click type."""
    if annotation == Path:
        return click.Path(exists=False, path_type=Path)
    if annotation in (str, int, float, bool):
        return annotation
    return str


def parse_cli_kwargs(kwargs: dict[str, Any], func_desc: FunctionDescription) -> dict[str, Any]:
    """Parse CLI kwargs into function arguments."""
    args_dict = {}
    fields = func_desc.args_model.model_fields

    if not fields:
        return args_dict

    # Direct field mapping
    for field_name in fields:
        value = kwargs.get(field_name)
        if value is not None:
            args_dict[field_name] = value

    return args_dict


def add_cli_options(cli_func: click.Command, func_desc: FunctionDescription) -> click.Command:
    """Add CLI options for function parameters."""
    fields = func_desc.args_model.model_fields

    if not fields:
        return cli_func

    # Direct field mapping
    for field_name, field_info in fields.items():
        option_name = f"--{field_name.replace('_', '-')}"
        field_type = field_info.annotation or str
        help_text = field_info.description or f"Value for {field_name}"

        if field_type is bool:
            option = click.option(option_name, field_name, is_flag=True, help=help_text)
        elif (origin := get_origin(field_type)) is not None and isclass(origin) and issubclass(
            origin, list | tuple
        ):
            option = click.option(
                option_name,
                field_name,
                multiple=True,
                help=f"{help_text}",
                type=get_args(field_type)[0],
            )
        else:
            option = click.option(
                option_name, field_name, type=get_click_type(field_type), help=help_text
            )
        cli_func = option(cli_func)
    return cli_func


def write_raw_output(result: Any):
    from tidyllm.types.part import BasicPart, is_audio_part, is_image_part

    print(type(result))
    if isinstance(result, bytes):
        sys.stdout.buffer.write(result)
    elif isinstance(result, Part):
        if is_image_part(result):
            # ImagePart - write PNG bytes
            sys.stdout.buffer.write(result.to_bytes("PNG"))
        elif is_audio_part(result):
            # AudioPart - write WAV bytes
            sys.stdout.buffer.write(result.to_bytes())
        else:
            sys.stdout.write(str(result))
    elif isinstance(result, Enumerable):
        print("Enumable: ", type(result))
        for row in result:
            write_raw_output(row)
    else:
        click.echo(result)


def output_result(result: Any, format: str) -> None:
    """Output result in specified format."""
    print("Output", type(result))
    if result is None:
        return
    if format == "json":
        result = to_json_dict(result)
        click.echo(json.dumps(result, indent=2))
    elif format == "pickle":
        sys.stdout.buffer.write(pickle.dumps(result))
    elif format == "raw":
        write_raw_output(result)


def _generate_cli_from_description(
    func_desc: FunctionDescription, context_cls: type[BaseModel] | None = None
) -> click.Command:
    """Generate CLI from a FunctionDescription."""
    tool_doc = func_desc.description or "CLI for tool"
    first_line = tool_doc.split('\n')[0].strip()

    @click.command(name=func_desc.name, help=first_line)
    @click.option("--json", "json_input", help="JSON input for all arguments")
    @click.option(
        "--format",
        "output_format", 
        type=click.Choice(["json", "pickle", "raw"]),
        default="json",
        help="Output format"
    )
    def cli(json_input: str | None, output_format: str, **kwargs):
        if json_input:
            try:
                args_dict = json.loads(json_input)
            except json.JSONDecodeError as e:
                click.echo(json.dumps({"error": f"Invalid JSON: {str(e)}"}))
                return
        else:
            args_dict = parse_cli_kwargs(kwargs, func_desc)

        parsed_args = func_desc.validate_and_parse_args(args_dict)

        if context_cls:
            context = context_cls()
            with set_tool_context(context):
                result = func_desc.call(**parsed_args)
        else:
            result = func_desc.call(**parsed_args)

        output_result(result, output_format)

    return add_cli_options(cli, func_desc)


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
    """Create a click CLI that dispatches to multiple functions."""
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
