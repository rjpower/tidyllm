"""Tests for vocabulary tools."""

import tempfile
from pathlib import Path

import pytest

from tidyllm.context import set_tool_context
from tidyllm.tools.config import Config
from tidyllm.tools.context import ToolContext
from tidyllm.tools.vocab_table import (
    VocabAddArgs,
    VocabDeleteArgs,
    VocabSearchArgs,
    VocabUpdateArgs,
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
    args = VocabAddArgs(
        word="hello",
        translation="hola",
        examples=["Hello world", "Hello there"],
        tags=["greeting", "basic"]
    )
    
    with set_tool_context(test_context):
        result = vocab_add(args)
    
    assert result.success is True
    assert "hello" in result.message


def test_vocab_add_duplicate(test_context):
    """Test adding a duplicate word fails."""
    args = VocabAddArgs(word="hello", translation="hola")
    
    with set_tool_context(test_context):
        # Add first time
        result1 = vocab_add(args)
        assert result1.success is True
        
        # Add second time should fail
        result2 = vocab_add(args)
    assert result2.success is False
    assert "error" in result2.message.lower()


def test_vocab_search_empty(test_context):
    """Test searching in empty database."""
    args = VocabSearchArgs()
    with set_tool_context(test_context):
        result = vocab_search(args)
    
    assert result.success is True
    assert result.count == 0
    assert len(result.items) == 0


def test_vocab_search_with_results(test_context):
    """Test searching with results."""
    # Add some words first
    with set_tool_context(test_context):
        vocab_add(VocabAddArgs(word="hello", translation="hola", tags=["greeting"]))
        vocab_add(VocabAddArgs(word="goodbye", translation="adiós", tags=["farewell"]))
        vocab_add(VocabAddArgs(word="bonjour", translation="hello", tags=["french"]))
        
        # Test search by word
        result = vocab_search(VocabSearchArgs(word="hello"))
        assert result.success is True
        assert result.count == 1
        assert result.items[0].word == "hello"
        
        # Test search by translation
        result = vocab_search(VocabSearchArgs(translation="hello"))
        assert result.success is True
        assert result.count == 1
        assert result.items[0].word == "bonjour"
        
        # Test search by tag
        result = vocab_search(VocabSearchArgs(tag="greeting"))
        assert result.success is True
        assert result.count == 1
        assert result.items[0].word == "hello"
        
        # Test search all
        result = vocab_search(VocabSearchArgs())
    assert result.success is True
    assert result.count == 3


def test_vocab_update_success(test_context):
    """Test successful vocabulary update."""
    # Add a word first
    with set_tool_context(test_context):
        vocab_add(VocabAddArgs(word="hello", translation="hola"))
        
        # Update it
        args = VocabUpdateArgs(
            word="hello",
            translation="¡hola!",
            examples=["¡Hola, mundo!"],
            tags=["spanish", "greeting"]
        )
        result = vocab_update(args)
        
        assert result.success is True
        assert "hello" in result.message
        
        # Verify the update
        search_result = vocab_search(VocabSearchArgs(word="hello"))
    item = search_result.items[0]
    assert item.translation == "¡hola!"
    assert item.examples == ["¡Hola, mundo!"]
    assert item.tags == ["spanish", "greeting"]


def test_vocab_update_not_found(test_context):
    """Test updating non-existent word."""
    args = VocabUpdateArgs(word="nonexistent", translation="new")
    with set_tool_context(test_context):
        result = vocab_update(args)
    
    assert result.success is False
    assert "not found" in result.message.lower()


def test_vocab_update_no_changes(test_context):
    """Test update with no fields provided."""
    with set_tool_context(test_context):
        vocab_add(VocabAddArgs(word="hello", translation="hola"))
        
        args = VocabUpdateArgs(word="hello")
        result = vocab_update(args)
    
    assert result.success is False
    assert "no fields" in result.message.lower()


def test_vocab_delete_success(test_context):
    """Test successful vocabulary deletion."""
    # Add a word first
    with set_tool_context(test_context):
        vocab_add(VocabAddArgs(word="hello", translation="hola"))
        
        # Delete it
        args = VocabDeleteArgs(word="hello")
        result = vocab_delete(args)
        
        assert result.success is True
        assert "hello" in result.message
        
        # Verify it's gone
        search_result = vocab_search(VocabSearchArgs(word="hello"))
    assert search_result.count == 0


def test_vocab_delete_not_found(test_context):
    """Test deleting non-existent word."""
    args = VocabDeleteArgs(word="nonexistent")
    with set_tool_context(test_context):
        result = vocab_delete(args)
    
    assert result.success is False
    assert "not found" in result.message.lower()


def test_vocab_with_empty_lists(test_context):
    """Test vocabulary operations with empty lists."""
    with set_tool_context(test_context):
        args = VocabAddArgs(
            word="test",
            translation="prueba",
            examples=[],
            tags=[]
        )
        result = vocab_add(args)
        assert result.success is True
        
        # Verify empty lists are handled correctly
        search_result = vocab_search(VocabSearchArgs(word="test"))
    item = search_result.items[0]
    assert item.examples == []
    assert item.tags == []


def test_vocab_search_limit(test_context):
    """Test search with limit parameter."""
    # Add multiple words
    with set_tool_context(test_context):
        for i in range(10):
            vocab_add(VocabAddArgs(word=f"word{i}", translation=f"trans{i}"))
        
        # Search with limit
        result = vocab_search(VocabSearchArgs(limit=5))
    assert result.success is True
    assert result.count == 5
    assert len(result.items) == 5
