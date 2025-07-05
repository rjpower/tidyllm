"""TidyLLM CLI - Main entry point for all TidyLLM tools."""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname).1s %(asctime)s %(filename)s:%(lineno)d - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)

from tidyllm.adapters.cli import cli_main
from tidyllm.discover import discover_tools_in_package
from tidyllm.registry import REGISTRY
from tidyllm.tools.context import ToolContext


def main():
    """CLI entry point for TidyLLM tools."""
    discover_tools_in_package("tidyllm.tools")

    # Get all registered functions
    functions = [func_desc.function for func_desc in REGISTRY.functions]

    if not functions:
        print("No tools discovered. Make sure tools are properly registered.")
        return

    # Use cli_main to create the CLI
    cli_main(
        functions=functions,
        context_cls=ToolContext
    )


if __name__ == "__main__":
    main()
