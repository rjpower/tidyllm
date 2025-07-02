#!/usr/bin/env python3
"""DXT entry point for TidyLLM MCP Server."""

import logging

logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("tidyllm").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)

import asyncio

from tidyllm.adapters.fastmcp_adapter import (
    run_tidyllm_mcp_server_async,
)

if __name__ == "__main__":
    asyncio.run(run_tidyllm_mcp_server_async())
