# tidyllm

Clean tool management for LLMs with automatic schema generation and multi-protocol support.

## Features

- **Zero-config tool registration** with `@register()` decorator
- **Multi-protocol support** - CLI, REST API, MCP, direct LLM integration
- **Type-safe** with full Pydantic validation
- **Context management** using modern Python contextvars
- **Built-in tools** - notes, flashcards, transcription, vocabulary
- **Rich UI** for real-time LLM interaction monitoring

## Quick Start

```python
from tidyllm import register, get_tool_context
from pydantic import BaseModel

class AddArgs(BaseModel):
    a: int
    b: int

@register()
def add(args: AddArgs) -> int:
    """Add two numbers."""
    return args.a + args.b
```

## Usage
```bash
uv add tidyllm
```