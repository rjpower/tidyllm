"""Tests for context injection and validation."""

from pathlib import Path

import pytest
from pydantic import BaseModel

from tidyllm.context import get_tool_context, set_tool_context
from tidyllm.registry import Registry
from tidyllm.tools.config import Config
from tidyllm.tools.context import ToolContext


class SimpleArgs(BaseModel):
    """Simple test arguments."""

    name: str


class SimpleResult(BaseModel):
    """Simple test result."""

    message: str
    context_info: dict


def basic_context_tool(args: SimpleArgs) -> SimpleResult:
    """Tool requiring basic context."""
    ctx = get_tool_context()
    return SimpleResult(
        message=f"Hello {args.name}",
        context_info={"project_root": str(ctx.config.notes_dir), "debug": True},
    )


def extended_context_tool(args: SimpleArgs) -> SimpleResult:
    """Tool requiring extended context."""
    ctx = get_tool_context()
    return SimpleResult(
        message=f"Hello {args.name}",
        context_info={
            "project_root": str(ctx.config.notes_dir),
            "debug": True,
            "api_key": "test_key...",  # Mock value
            "timeout": 30,
        },
    )


def no_context_tool(args: SimpleArgs) -> SimpleResult:
    """Tool that doesn't require context."""
    return SimpleResult(message=f"Hello {args.name}", context_info={})


@pytest.fixture
def context_registry():
    """Create a registry with context tools for testing."""
    registry = Registry()
    
    # Register tools with different context requirements
    registry.register(basic_context_tool)
    registry.register(extended_context_tool)
    registry.register(no_context_tool)
    
    return registry

def test_context_injection_basic(context_registry):
    """Test basic context injection."""
    config = Config(notes_dir=Path("/test/project"))
    context = ToolContext(config=config)

    with set_tool_context(context):
        result = basic_context_tool(SimpleArgs(name="test"))
        assert result.message == "Hello test"
        assert "/test/project" in result.context_info["project_root"]
        assert result.context_info["debug"] is True


def test_context_injection_extended(context_registry):
    """Test extended context injection with more fields."""
    config = Config(notes_dir=Path("/test/project"))
    context = ToolContext(config=config)

    with set_tool_context(context):
        result = extended_context_tool(SimpleArgs(name="extended"))
        assert result.message == "Hello extended"
        assert "/test/project" in result.context_info["project_root"]
        assert result.context_info["debug"] is True
        assert result.context_info["api_key"] == "test_key..."
        assert result.context_info["timeout"] == 30


def test_no_context_tool_execution(context_registry):
    """Test tool that doesn't require context."""
    # No context tool should work without setting context
    result = no_context_tool(SimpleArgs(name="no_ctx"))
    assert result.message == "Hello no_ctx"
    assert result.context_info == {}


def test_context_validation_extra_fields_allowed(context_registry):
    """Test that context works with all available fields."""
    config = Config(notes_dir=Path("/test"))
    context = ToolContext(config=config)

    with set_tool_context(context):
        result = basic_context_tool(SimpleArgs(name="test"))
        assert result.message == "Hello test"


def test_empty_context(context_registry):
    """Test with minimal context."""
    config = Config()  # Uses default values
    context = ToolContext(config=config)

    with set_tool_context(context):
        result = basic_context_tool(SimpleArgs(name="empty"))
        assert result.message == "Hello empty"


def test_context_with_none_values(context_registry):
    """Test context with None values in config."""
    config = Config(anki_path=None)  # Explicitly set None
    context = ToolContext(config=config)

    with set_tool_context(context):
        result = basic_context_tool(SimpleArgs(name="none_vals"))
        assert result.message == "Hello none_vals"
