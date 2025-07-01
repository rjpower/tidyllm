# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

Use `uv` for all package management and execution:

For individual test runs with output: `uv run pytest -s path/to/test.py`

## Architecture Overview

**tidyllm** is a Python library for LLM tool management with automatic schema generation and CLI creation.

### Core Components

- **Registry System** (`registry.py`): Global `REGISTRY` with `@register` decorator for tool registration
- **Function Library** (`library.py`): Container for tools with shared context, JSON validation, async support
- **Schema Generation** (`schema.py`): OpenAI-compatible schemas from function signatures and docstrings
- **CLI Generation** (`cli.py`): Automatic command-line interfaces from tool schemas
- **Models** (`models.py`): Core types (`ToolError`, `ToolResult`)
- **Discovery** (`discover.py`): Automatic tool discovery in packages/directories

### Import Patterns

The codebase uses `portkit.tidyllm.*` imports internally. Key exports from `tidyllm.__init__.py`:
- `register`, `REGISTRY` - tool registration
- `FunctionLibrary` - tool container
- `ToolError`, `ToolResult` - core types
- `cli_main` - CLI entry point

### Tool Structure

Tools follow this pattern (see `tools/calculator/`):
- Pydantic models for arguments and results
- Full type annotations
- Comprehensive docstrings
- Error handling with `ToolError`

### Framework Integration

- **FastAPI Adapter** (`adapters/fastapi_adapter.py`): Convert function libraries to FastAPI endpoints
- **Async Support**: Native async/await function support throughout

## Code Standards

- **Python Version**: 3.11-3.13 only, use modern syntax
- **Type Annotations**: Full typing required, use `str | None` not `Optional[str]`
- **Imports**: Absolute imports only, `from portkit.tidyllm.module import ...`
- **Line Length**: 100 characters (ruff configured)
- **Testing**: Inline tests preferred for simple cases, `tests/` for complex ones
- **Comments**: Explanatory only, avoid obvious descriptions

## Project Structure

- `src/tidyllm/` - Main source code
- `src/tidyllm/tools/` - Built-in tools
- `src/tidyllm/adapters/` - Framework integrations  
- `src/tidyllm/tests/` - Test suite
- `tests/` - Additional tests
- `.cursor/rules/` - Development guidelines (generates this file via `make agent-rules`)