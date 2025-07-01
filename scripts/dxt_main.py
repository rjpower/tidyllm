#!/usr/bin/env python3
"""DXT entry point for TidyLLM MCP Server."""

from tidyllm.adapters.fastmcp_adapter import run_tidyllm_mcp_server

if __name__ == "__main__":
    # Config class automatically reads TIDYLLM_* environment variables
    # set by the DXT manifest user_config
    run_tidyllm_mcp_server()