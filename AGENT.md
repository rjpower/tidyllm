# AGENT.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Development Commands

Use `uv` for all package management and execution:

- Individual test runs: `uv run pytest -s path/to/test.py`
- Full test suite: `uv run pytest`

**IMPORTANT**: Always run `uv run pytest` after making changes to verify nothing is broken.

## Architecture Overview

**tidyllm** is a Python library for LLM tool management with automatic schema generation, CLI creation, and adapter integrations for FastAPI and FastMCP.

### Core Components

- **Registry System** (`registry.py`): Global `REGISTRY` with `@register` decorator for tool registration
- **Context Management** (`context.py`): ContextVar-based system using `get_tool_context()` and `set_tool_context()`
- **Schema Generation** (`schema.py`): OpenAI-compatible schemas from function signatures and docstrings
- **CLI Generation** (`adapters/cli.py`): Automatic command-line interfaces with contextvar integration
- **Database** (`database.py`): SQLite wrapper with Pydantic models and automatic schema initialization
- **Caching** (`cache.py`): Function result caching with `@cached_function` decorator
- **Streaming** (`stream.py`): `Stream[T]` abstraction for lazy data processing
- **Duration** (`duration.py`): Time duration utilities for audio processing
- **Discovery** (`discover.py`): Automatic tool discovery in packages/directories

### Key Exports

From `tidyllm.__init__.py`:
- `register`, `REGISTRY` - tool registration and access
- `FunctionLibrary` - alias for `REGISTRY`
- `ToolError`, `ToolResult` - core exception and result types
- `cli_main` - CLI entry point
- `FunctionDescription` - schema generation
- `discover_tools_in_directory`, `discover_tools_in_package` - tool discovery

### Tool Structure

Tools use this pattern:
```python
from pydantic import BaseModel, Field
from tidyllm.registry import register
from tidyllm.context import get_tool_context

class ToolArgs(BaseModel):
    param: str = Field(description="Parameter description")

class ToolResult(BaseModel):
    output: str = Field(description="Output description")

@register()
def tool_name(args: ToolArgs) -> ToolResult:
    """Tool description with example usage."""
    ctx = get_tool_context()
    # Implementation using ctx.db, ctx.config, etc.
    return ToolResult(output="result")
```

### Context Management

- Tools access context via `ctx = get_tool_context()`
- Adapters use `with set_tool_context(context):` to establish context
- Context provides: `ctx.db` (database), `ctx.config` (configuration), `ctx.get_ref()` (cached resources)

## Project Structure

```
src/tidyllm/          # Core library
├── tools/            # Built-in tools
├── adapters/         # Framework integrations (CLI, FastAPI, FastMCP)
apps/                 # Example applications
tests/                # Test suite (217 tests)
```

## Built-in Tools

### Audio Processing (`tools/audio.py`)
- **AudioChunk/AudioFormat**: Audio data models with timestamp/format info
- **mic()**, **file()**: Audio streaming from microphone/file
- **chunk_by_vad_stream()**: Voice activity detection segmentation
- **merge_chunks()**, **chunk_to_wav_bytes()**: Audio manipulation utilities

### Transcription (`tools/transcribe.py`)
- **transcribe_bytes()**: Cached audio transcription via Gemini Flash
- **transcribe()**: File-based transcription wrapper

### Vocabulary (`tools/vocab_table.py`)
- **vocab_add/search/update/delete()**: Database CRUD operations
- Full-text search with tags and examples

### Notes (`tools/notes.py`)
- **note_add/search/list/open/recent/tags()**: Markdown notes with YAML frontmatter
- Content search via ripgrep

### Anki (`tools/anki.py`)
- **anki_query/create/list()**: Flashcard management and generation

### Other Tools
- **Calculator** (`tools/calculator/`): Mathematical evaluation
- **Database Management** (`tools/manage_db.py`): Schema operations

## Framework Integrations

### CLI Adapter (`adapters/cli.py`)
```python
# Single function CLI
from tidyllm.adapters.cli import cli_main
cli_main(my_function, context_cls=ToolContext)

# Multi-function CLI
from tidyllm.adapters.cli import multi_cli_main
multi_cli_main([func1, func2], default_function="func1", context_cls=ToolContext)
```

### FastAPI Adapter (`adapters/fastapi_adapter.py`)
```python
from tidyllm.adapters.fastapi_adapter import create_fastapi_app
app = create_fastapi_app(context=ToolContext())
```

### FastMCP Adapter (`adapters/fastmcp_adapter.py`)
```python
from tidyllm.adapters.fastmcp_adapter import create_fastmcp_server
server = create_fastmcp_server(context=ToolContext())
```

## Adding New Tools

1. **Create Pydantic models** for arguments and results with field descriptions
2. **Use `@register()` decorator** on functions
3. **Access context** via `ctx = get_tool_context()`
4. **Database operations** via `ctx.db.query()` and `ctx.db.mutate()`
5. **Add to CLI** using `cli_main()` or `multi_cli_main()`

## Writing Tests

```python
import pytest
from tidyllm.context import set_tool_context
from tidyllm.tools.context import ToolContext

@pytest.fixture
def tool_context():
    return ToolContext()

def test_tool(tool_context):
    with set_tool_context(tool_context):
        result = my_tool(MyToolArgs(param="test"))
        assert result.success
```

**Key patterns**:
- Use `tool_context` fixture with `ToolContext()`
- Wrap tests in `with set_tool_context(tool_context):`
- Each test gets fresh database state
- Test count: 217 tests covering all components

## Code Standards

- **Python 3.11-3.13** with modern syntax (`str | None` not `Optional[str]`)
- **Absolute imports**: `from tidyllm.module import ...`
- **Type annotations** required for schema generation
- **Pydantic models** with field descriptions
- **Context access** via `get_tool_context()`, never explicit parameters
- **Line length**: 100 characters (ruff configured)

## Performance Notes

- **Caching**: Use `@cached_function` for expensive operations
- **Streaming**: Use `Stream[T]` for large datasets
- **Resource management**: Use `ctx.get_ref()` for expensive resources like models
- **Database**: Use parameterized queries, batch operations when possible