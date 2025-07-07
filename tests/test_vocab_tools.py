"""Tests for vocabulary tools."""

import tempfile
from pathlib import Path

import pytest

from tidyllm.context import set_tool_context
from tidyllm.tools.config import Config
from tidyllm.tools.context import ToolContext
from tidyllm.tools.vocab_table import (
    vocab_add,
    vocab_delete,
    vocab_search,
    vocab_update,
)


@pytest.fixture
def test_context():
    """Create a test context with temporary database."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = Config(
            user_db=temp_path / "test.db",
            notes_dir=temp_path / "notes",
        )
        yield ToolContext(config=config)


def test_vocab_add_basic(test_context):
    """Test adding a basic vocabulary word."""
    with set_tool_context(test_context):
        vocab_add(
            word="hello",
            translation="hola",
            examples=["Hello world", "Hello there"],
            tags=["greeting", "basic"]
        )
    
    # Function completed successfully if no exception raised


def test_vocab_add_duplicate(test_context):
    """Test adding a duplicate word fails."""
    with set_tool_context(test_context):
        # Add first time
        vocab_add(word="hello", translation="hola")
        # Function completed successfully if no exception raised
        
        # Add second time should fail with constraint error
        with pytest.raises(Exception):  # IntegrityError or similar
            vocab_add(word="hello", translation="hola")


def test_vocab_search_empty(test_context):
    """Test searching in empty database."""
    with set_tool_context(test_context):
        result = vocab_search()
    
    # Function completed successfully if no exception raised
    assert len(result) == 0


def test_vocab_search_with_results(test_context):
    """Test searching with results."""
    # Add some words first
    with set_tool_context(test_context):
        vocab_add(word="hello", translation="hola", tags=["greeting"])
        vocab_add(word="goodbye", translation="adiós", tags=["farewell"])
        vocab_add(word="bonjour", translation="hello", tags=["french"])
        
        # Test search by word
        result = vocab_search(word="hello")
        # Function completed successfully if no exception raised
        assert len(result) == 1
        assert result[0].word == "hello"
        
        # Test search by translation
        result = vocab_search(translation="hello")
        # Function completed successfully if no exception raised
        assert len(result) == 1
        assert result[0].word == "bonjour"
        
        # Test search by tag
        result = vocab_search(tag="greeting")
        # Function completed successfully if no exception raised
        assert len(result) == 1
        assert result[0].word == "hello"
        
        # Test search all
        result = vocab_search()
    # Function completed successfully if no exception raised
    assert len(result) == 3


def test_vocab_update_success(test_context):
    """Test successful vocabulary update."""
    # Add a word first
    with set_tool_context(test_context):
        vocab_add(word="hello", translation="hola")
        
        # Update it
        vocab_update(
            word="hello",
            translation="¡hola!",
            examples=["¡Hola, mundo!"],
            tags=["spanish", "greeting"]
        )
        
        # Function completed successfully if no exception raised
        
        # Verify the update
        search_result = vocab_search(word="hello")
    item = search_result[0]
    assert item.translation == "¡hola!"
    assert item.examples == ["¡Hola, mundo!"]
    assert item.tags == ["spanish", "greeting"]


def test_vocab_update_not_found(test_context):
    """Test updating non-existent word."""
    with set_tool_context(test_context):
        with pytest.raises(ValueError, match="not found"):
            vocab_update(word="nonexistent", translation="new")


def test_vocab_update_no_changes(test_context):
    """Test update with no fields provided."""
    with set_tool_context(test_context):
        vocab_add(word="hello", translation="hola")
        
        with pytest.raises(ValueError, match="No fields"):
            vocab_update(word="hello")


def test_vocab_delete_success(test_context):
    """Test successful vocabulary deletion."""
    # Add a word first
    with set_tool_context(test_context):
        vocab_add(word="hello", translation="hola")
        
        # Delete it
        vocab_delete("hello")
        
        # Function completed successfully if no exception raised
        
        # Verify it's gone
        search_result = vocab_search(word="hello")
    assert len(search_result) == 0


def test_vocab_delete_not_found(test_context):
    """Test deleting non-existent word."""
    with set_tool_context(test_context):
        with pytest.raises(ValueError, match="not found"):
            vocab_delete("nonexistent")


def test_vocab_with_empty_lists(test_context):
    """Test vocabulary operations with empty lists."""
    with set_tool_context(test_context):
        vocab_add(
            word="test",
            translation="prueba",
            examples=[],
            tags=[]
        )
        # Function completed successfully if no exception raised
        
        # Verify empty lists are handled correctly
        search_result = vocab_search(word="test")
    item = search_result[0]
    assert item.examples == []
    assert item.tags == []


def test_vocab_search_limit(test_context):
    """Test search with limit parameter."""
    # Add multiple words
    with set_tool_context(test_context):
        for i in range(10):
            vocab_add(word=f"word{i}", translation=f"trans{i}")
        
        # Search with limit
        result = vocab_search(limit=5)
    # Function completed successfully if no exception raised
    assert len(result) == 5
