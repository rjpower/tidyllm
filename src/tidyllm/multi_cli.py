"""Multi-function CLI support with click dispatch."""

import json
from collections.abc import Callable
from typing import Any

import click
from pydantic import BaseModel

from tidyllm.cli import _generate_cli_from_description
from tidyllm.schema import FunctionDescription


def multi_cli_main(functions: dict[str, Callable], default_function: str | None = None):
    """Create a click CLI that dispatches to multiple functions.
    
    Args:
        functions: Dict mapping command names to functions
        default_function: Name of default function if no command specified
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
    for cmd_name, func in functions.items():
        func_desc = FunctionDescription(func)
        cmd = _generate_cli_from_description(func_desc)
        
        # Wrap the command to handle exceptions properly
        def make_wrapper(original_cmd):
            def wrapper(*args, **kwargs):
                try:
                    return original_cmd.callback(*args, **kwargs)
                except Exception as e:
                    import traceback
                    click.echo(traceback.format_exc())
                    click.echo(json.dumps({"error": f"Error in {cmd_name}: {str(e)}"}))
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


def simple_cli_main(functions: list[Callable], default_function: str | None = None):
    """Simple version that just takes a list of functions and uses their names."""
    func_dict = {func.__name__: func for func in functions}
    multi_cli_main(func_dict, default_function)