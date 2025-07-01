# Registry FastMCP Integration Specification

## Overview

This specification outlines the architectural changes to integrate FastMCP server directly into the TidyLLM registry system and replace explicit context passing with Python contextvars for automatic context propagation.

## Current State Problems

1. **Dual Context Systems**: FastMCP and TidyLLM maintain separate context systems
2. **Manual Context Management**: Requires explicit `{"ctx": tool_context}` passing to `FunctionLibrary`
3. **Fragmented Registration**: Registry and FastMCP server are created separately
5. **Complex Setup**: Each adapter must manually configure context and server lifecycle

## Proposed Solution

### Core Architecture Changes

#### 1. Context Management with contextvars

Replace explicit context passing with Python's `contextvars` module for automatic context propagation.

**New Context API:**
```python
# src/tidyllm/context.py
from contextvars import ContextVar
from typing import TypeVar, Generic

T = TypeVar('T')

class ContextManager(Generic[T]):
    def __init__(self, name: str, default: T | None = None):
        self._var: ContextVar[T] = ContextVar(name, default=default)
    
    def get(self) -> T:
        return self._var.get()
    
    def set(self, value: T) -> None:
        self._var.set(value)
    
    def __call__(self) -> T:
        return self.get()

# Global context managers
tool_context: ContextManager[ToolContext] = ContextManager('tool_context')

def get_tool_context()...
```

**Tool Usage Pattern:**
```python
# Before (explicit context)
@register
def note_add(title: str, content: str, *, ctx: ToolContext) -> NoteResult:
    db_conn = ctx.get_db_connection()
    # ...

# After (automatic context)
@register  
def note_add(title: str, content: str) -> NoteResult:
    ctx = get_tool_context()
    db_conn = ctx.get_db_connection()
    # ...
```

#### 2. Registry-Integrated FastMCP Server

Embed FastMCP server creation and lifecycle management directly into the registry system.

**Enhanced Registry:**
```python
# src/tidyllm/registry.py
from fastmcp import FastMCP
from tidyllm.context import ContextManager, ToolContext

class Registry:
    def __init__(self):
        self._tools: dict[str, FunctionDescription] = OrderedDict()
        self._fastmcp_server: FastMCP | None = None
        self._context_manager: ContextManager[ToolContext] | None = None
    
    def create_fastmcp_server(self, 
                             context: ToolContext | None = None,
                             server_info: ServerInfo | None = None) -> FastMCP:
        """Create and configure FastMCP server with all registered tools."""
        if self._fastmcp_server is None:
            self._fastmcp_server = FastMCP("TidyLLM Tools", server_info)
            self._context_manager = ContextManager('tool_context', context)
            
            # Register all existing tools
            for tool_desc in self._tools.values():
                self._register_tool_with_fastmcp(tool_desc)
        
        return self._fastmcp_server
    
    def register(self, func: Callable, doc_override: str | None = None) -> None:
        """Register tool and add to FastMCP server if it exists."""
        # Existing registration logic
        func_desc = FunctionDescription(func, doc_override)
        self._tools[func.__name__] = func_desc
        
        # Auto-register with FastMCP if server exists
        if self._fastmcp_server is not None:
            self._register_tool_with_fastmcp(func_desc)
    
    def _register_tool_with_fastmcp(self, func_desc: FunctionDescription) -> None:
        """Register individual tool with FastMCP server."""
        async def fastmcp_wrapper(arguments: dict) -> Any:
            # Set context for this execution
            if self._context_manager:
                token = self._context_manager._var.set(self._context_manager.get())
                try:
                    # Call tool with automatic context propagation
                    return await self._execute_tool_with_context(func_desc, arguments)
                finally:
                    self._context_manager._var.reset(token)
            else:
                return await self._execute_tool_with_context(func_desc, arguments)
        
        self._fastmcp_server.tool(
            name=func_desc.name,
            description=func_desc.description,
            parameters=func_desc.parameters_schema
        )(fastmcp_wrapper)
```

#### 3. Simplified Tool Context Access

**New Context Access Function:**
```python
# src/tidyllm/context.py
def get_tool_context() -> ToolContext:
    """Get current tool context from contextvar."""
    try:
        return tool_context.get()
    except LookupError:
        raise RuntimeError(
            "No tool context available. Ensure tools are called within "
            "a properly configured adapter (FastAPI, FastMCP, etc.)"
        )
```

**Updated Tool Pattern:**
```python
# All tools use this pattern now
@register
def note_search(query: str, limit: int = 10) -> NoteSearchResult:
    """Search notes by content and metadata.
    
    Args:
        query: Search query to match against note content and metadata
        limit: Maximum number of results to return (default: 10)
        
    Returns:
        NoteSearchResult with matching notes and search metadata
        
    Example:
        result = note_search("python functions", limit=5)
        for note in result.notes:
            print(f"{note.title}: {note.excerpt}")
    """
    ctx = get_tool_context()
    # Implementation...
```

#### 4. Adapter Simplification

**FastMCP Adapter:**
```python
# src/tidyllm/adapters/fastmcp_adapter.py
from tidyllm.registry import REGISTRY
from tidyllm.context import ToolContext

def create_fastmcp_server(
    context: ToolContext | None = None,
    server_info: ServerInfo | None = None
) -> FastMCP:
    """Create FastMCP server with automatic tool registration."""
    # Context defaults
    if context is None:
        context = ToolContext()
    
    return REGISTRY.create_fastmcp_server(context, server_info)

# Usage becomes:
def main():
    context = ToolContext(config=Config())
    server = create_fastmcp_server(context)
    # Server has all registered tools automatically
```

**FastAPI Adapter:**
```python
# src/tidyllm/adapters/fastapi_adapter.py  
def create_fastapi_app(context: ToolContext | None = None) -> FastAPI:
    """Create FastAPI app with automatic context management."""
    app = FastAPI()
    
    if context is None:
        context = ToolContext()
    
    for tool_desc in REGISTRY.functions:
        _create_tool_endpoint_with_context(app, tool_desc, context)
    
    return app

def _create_tool_endpoint_with_context(app, tool_desc, context):
    async def tool_endpoint(args):
        # Set context for this request
        token = tool_context._var.set(context)
        try:
            return await _execute_tool(tool_desc, args.model_dump())
        finally:
            tool_context._var.reset(token)
    
    app.post(f"/tools/{tool_desc.name}")(tool_endpoint)
```

## Implementation Plan

### Phase 1: Context Infrastructure
1. Create `src/tidyllm/context.py` with `ContextManager` and `get_tool_context()`
2. Update `ToolContext` class with improved type hints

### Phase 2: Registry Integration  
1. Add FastMCP server creation methods to `Registry` class
2. Implement automatic tool registration with FastMCP
3. Add context propagation to tool execution

### Phase 3: Tool Migration
1. Update all tools in `src/tidyllm/tools/` to use `get_tool_context()`
2. Remove `*, ctx` parameters from tool signatures
3. Update tool documentation with new patterns

### Phase 4: Adapter Updates
1. Simplify FastMCP adapter to use registry integration
2. Update FastAPI adapter to use contextvars
3. Update CLI adapter for context management

### Phase 5: Testing and Documentation
1. Add comprehensive tests for context propagation
2. Test FastMCP server integration
3. Update all tool documentation with examples
4. Performance testing for contextvar overhead

## Testing Strategy

### Unit Tests
```python
# tests/test_context_management.py
def test_context_propagation():
    context = ToolContext(config=Config())
    
    with context_manager.set_context(context):
        result = note_search("test")
        assert result is not None

def test_context_isolation():
    # Test that context doesn't leak between tool calls
    pass

def test_missing_context_error():
    with pytest.raises(RuntimeError, match="No tool context available"):
        get_tool_context()
```

### Integration Tests
```python
# tests/test_fastmcp_integration.py
def test_registry_fastmcp_integration():
    context = ToolContext()
    server = REGISTRY.create_fastmcp_server(context)
    
    # Verify all registered tools are available
    tools = server.list_tools()
    assert "note_search" in tools
    
    # Test tool execution with context
    result = server.call_tool("note_search", {"query": "test"})
    assert result is not None
```

## Migration Guide

### For Tool Developers
1. Remove `*, ctx: ToolContext` from function signatures
2. Add `ctx = get_tool_context()` at function start
3. Update docstrings with clear usage examples

### For Adapter Users
1. FastMCP: Use `create_fastmcp_server(context)` instead of manual setup
2. FastAPI: Pass context to `create_fastapi_app(context)`
3. CLI: Context automatically configured from config files

## Backward Compatibility

- Keep existing `FunctionLibrary` class for gradual migration
- Support both old and new context patterns during transition
- Deprecation warnings for old patterns

## Performance Considerations

- Contextvars have minimal overhead (~10-20ns per access)
- FastMCP server creation is one-time cost at startup

## Security Considerations  

- Context isolation between concurrent tool executions
- Sensitive data in context (API keys, tokens) properly scoped

## Questions and Considerations

1. **Context Scoping**: Should we support nested contexts for complex workflows?
2. **Error Handling**: How should context-related errors be surfaced to tool users?
3. **Context Validation**: Move validation to adapter setup time vs tool execution time?
4. **Async Support**: Ensure contextvar propagation works correctly with async tools
5. **Testing Isolation**: Best practices for context isolation in test suites?
6. **Context Inheritance**: Should child contexts inherit from parent contexts?
7. **Performance Monitoring**: Add metrics for context access patterns?

## Success Criteria

1. ✅ All tools work without explicit `*, ctx` parameters
2. ✅ FastMCP server automatically includes all registered tools  
3. ✅ Context setup reduced to single function call per adapter
4. ✅ Zero runtime context validation errors in well-configured environments
5. ✅ Complete test coverage for context propagation scenarios
6. ✅ Documentation includes clear usage examples for every tool