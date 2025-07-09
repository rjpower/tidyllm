"""FastMCP server script with auto-discovery of TidyLLM tools."""

import argparse
from pathlib import Path
from typing import Any

from tidyllm.adapters.fastmcp_adapter import create_fastmcp_server
from tidyllm.discover import discover_tools_in_directory, discover_tools_in_package
from tidyllm.tools.config import Config
from tidyllm.tools.context import ToolContext


def create_fastmcp_server_with_discovery(
    context: ToolContext | None = None,
    name: str = "TidyLLM Tools",
    tools_package: str = "tidyllm.tools",
):
    discover_tools_in_package(tools_package)
    discover_tools_in_directory(Path("apps/"))

    config = Config()
    context = ToolContext(config=config)

    return create_fastmcp_server(context=context, name=name)


mcp = create_fastmcp_server_with_discovery()


def main():
    mcp.run()


if __name__ == "__main__":
    main()
