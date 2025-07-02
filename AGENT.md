# AGENT.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

Use `uv` for all package management and execution:

For individual test runs with output: `uv run pytest -s path/to/test.py`
Run full test suite: `uv run pytest`

**IMPORTANT**: Always run `uv run pytest` (full test suite) after making changes to verify nothing is broken.

## Architecture Overview

**tidyllm** is a Python library for LLM tool management with automatic schema generation, CLI creation, and adapter integrations for FastAPI and FastMCP.

### Core Components

- **Registry System** (`registry.py`): Global `REGISTRY` with `@register` decorator for tool registration and FastMCP server integration
- **Context Management** (`context.py`): ContextVar-based context system using `get_tool_context()` and `set_tool_context()`
- **Function Library** (`library.py`): Container for tools with validation and execution, no longer handles context passing
- **Schema Generation** (`schema.py`): OpenAI-compatible schemas from function signatures and docstrings
- **CLI Generation** (`cli.py`): Automatic command-line interfaces with contextvar integration
- **Models** (`models.py`): Core types (`ToolError`, `ToolResult`)
- **Discovery** (`discover.py`): Automatic tool discovery in packages/directories

### Import Patterns

The codebase uses `tidyllm.*` imports internally. Key exports from `tidyllm.__init__.py`:
- `register`, `REGISTRY` - tool registration
- `FunctionLibrary` - tool container
- `ToolError`, `ToolResult` - core types
- `cli_main` - CLI entry point
- `get_tool_context`, `set_tool_context` - context management

### Tool Structure

Tools follow this updated pattern:
- **Function signature**: `def tool_name(args: ArgsModel) -> ResultModel`
- **Context access**: Use `ctx = get_tool_context()` inside function body
- **No explicit context parameters**: Context accessed via contextvars
- **Pydantic models**: For arguments and results with field descriptions
- **Full type annotations**: Required for schema generation
- **Comprehensive docstrings**: With parameter documentation
- **Error handling**: With `ToolError` or standard exceptions

### Framework Integration

- **FastAPI Adapter** (`adapters/fastapi_adapter.py`): Convert tools to FastAPI endpoints with contextvar support
- **FastMCP Adapter** (`adapters/fastmcp_adapter.py`): Integrate with FastMCP using registry's built-in server creation
- **CLI Integration**: Automatic CLI generation with context variable support
- **Async Support**: Native async/await function support throughout

## Code Standards

- **Python Version**: 3.11-3.13 only, use modern syntax
- **Type Annotations**: Full typing required, use `str | None` not `Optional[str]`
- **Imports**: Absolute imports only, `from tidyllm.module import ...`
- **Line Length**: 100 characters (ruff configured)
- **Testing**: Comprehensive test suite with 238 tests covering all components
- **Comments**: Explanatory only, avoid obvious descriptions
- **Context Access**: Always use `get_tool_context()`, never explicit context parameters
- **Tool Registration**: Use `@register()` decorator for all tools
- **Field Documentation**: All Pydantic model fields must have descriptions

### Context Management System

**New ContextVar Approach** (current):
- Tools use `ctx = get_tool_context()` to access context
- Context automatically propagated via Python's contextvars
- Adapters use `with set_tool_context(context):` to establish context
- No explicit `*, ctx: ToolContext` parameters in function signatures

**Key Context Functions**:
- `get_tool_context() -> ToolContext`: Get current context (raises RuntimeError if none)
- `set_tool_context(context) -> ContextManager`: Set context for a block

### FastMCP Integration

- **Registry Integration**: `REGISTRY.create_fastmcp_server()` creates server with all tools
- **Context Wrapping**: Tools automatically wrapped to provide context via contextvars
- **Adapter Functions**: `create_fastmcp_server()`, `create_tidyllm_mcp_server()` for different use cases
- **Tool Registration**: Original functions wrapped with async context handlers for FastMCP

## Project Structure

- `src/tidyllm/` - Main source code
  - `context.py` - ContextVar-based context management
  - `registry.py` - Tool registration with FastMCP integration
  - `library.py` - Tool container and execution (simplified)
  - `cli.py` - CLI generation with contextvar support
- `src/tidyllm/tools/` - Built-in tools (notes, anki, transcribe, vocab, manage_db)
- `src/tidyllm/adapters/` - Framework integrations (FastAPI, FastMCP)
- `tests/` - Test suite (238 tests, all passing)
- `.cursor/rules/` - Development guidelines (generates this file via `make agent-rules`)