"""Tests for Anki tools."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tidyllm.context import set_tool_context
from tidyllm.tools.anki import (
    AnkiCard,
    AnkiCreateArgs,
    AnkiReadArgs,
    anki_create,
    anki_read,
)
from tidyllm.tools.config import Config
from tidyllm.tools.context import ToolContext


@pytest.fixture
def test_context():
    """Create a test context."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = Config(
            user_db=temp_path / "test.db",
            notes_dir=temp_path / "notes",
            anki_path=temp_path / "anki.db"  # Mock Anki path
        )
        yield ToolContext(config=config)


def test_anki_read_no_database(test_context):
    """Test reading when no Anki database exists."""
    args = AnkiReadArgs(deck_name="Test Deck")
    with set_tool_context(test_context):
        result = anki_read(args)
    
    assert result.deck_name == "Test Deck"
    assert result.count == 0
    assert len(result.cards) == 0


@patch('sqlite3.connect')
def test_anki_read_with_mock_database(mock_connect, test_context):
    """Test reading from mock Anki database."""
    # Mock database connection and results
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor
    
    # Mock deck lookup
    mock_cursor.fetchone.return_value = {"id": 1}
    
    # Mock card data
    mock_cursor.fetchall.return_value = [
        {
            "id": 1,
            "flds": "hello\x1fhola\x1fHello world",  # Anki field separator
            "tags": "greeting basic",
            "ord": 0
        },
        {
            "id": 2, 
            "flds": "goodbye\x1fadiós\x1fGoodbye friend",
            "tags": "farewell",
            "ord": 1
        }
    ]
    
    # Set up context to find the mock DB
    test_context.config.anki_path = Path("/mock/anki.db")
    
    args = AnkiReadArgs(deck_name="Spanish", limit=10)
    with set_tool_context(test_context):
        result = anki_read(args)
    
    assert result.deck_name == "Spanish"
    assert result.count == 2
    assert len(result.cards) == 2
    
    # Check first card
    card1 = result.cards[0]
    assert card1["id"] == 1
    assert card1["fields"] == ["hello", "hola", "Hello world"]
    assert card1["tags"] == ["greeting", "basic"]


def test_anki_create_basic(test_context):
    """Test creating basic Anki deck."""
    cards = [
        AnkiCard(
            source_word="hello",
            translated_word="hola",
            examples=["Hello world", "Hello there"]
        ),
        AnkiCard(
            source_word="goodbye", 
            translated_word="adiós",
            examples=["Goodbye friend"]
        )
    ]
    
    args = AnkiCreateArgs(
        deck_name="Test Deck",
        cards=cards
    )
    
    with set_tool_context(test_context):
        result = anki_create(args)
    
    assert result.success is True
    assert result.cards_created == 2
    assert "Test Deck" in result.message
    assert result.deck_path.exists()
    assert result.deck_path.suffix == ".apkg"


def test_anki_create_with_audio(test_context):
    """Test creating Anki deck with audio files."""
    # Create a mock audio file
    audio_file = test_context.config.notes_dir / "hello.mp3"
    audio_file.parent.mkdir(parents=True, exist_ok=True)
    audio_file.write_bytes(b"fake audio data")
    
    cards = [
        AnkiCard(
            source_word="hello",
            translated_word="hola", 
            examples=["Hello world"],
            audio_path=audio_file
        )
    ]
    
    args = AnkiCreateArgs(
        deck_name="Audio Deck",
        cards=cards
    )
    
    with set_tool_context(test_context):
        result = anki_create(args)
    
    assert result.success is True
    assert result.cards_created == 1
    assert result.deck_path.exists()


def test_anki_create_custom_output_path(test_context):
    """Test creating Anki deck with custom output path."""
    custom_path = test_context.config.notes_dir / "custom_deck.apkg"
    
    cards = [
        AnkiCard(source_word="test", translated_word="prueba")
    ]
    
    args = AnkiCreateArgs(
        deck_name="Custom Path Deck",
        cards=cards,
        output_path=custom_path
    )
    
    with set_tool_context(test_context):
        result = anki_create(args)
    
    assert result.success is True
    assert result.deck_path == custom_path
    assert custom_path.exists()


def test_anki_create_empty_cards(test_context):
    """Test creating deck with no cards."""
    args = AnkiCreateArgs(
        deck_name="Empty Deck",
        cards=[]
    )
    
    with set_tool_context(test_context):
        result = anki_create(args)
    
    assert result.success is True
    assert result.cards_created == 0
    assert result.deck_path.exists()


def test_anki_create_with_missing_audio(test_context):
    """Test creating deck with non-existent audio file."""
    nonexistent_audio = Path("/nonexistent/audio.mp3")
    
    cards = [
        AnkiCard(
            source_word="hello",
            translated_word="hola",
            audio_path=nonexistent_audio
        )
    ]
    
    args = AnkiCreateArgs(
        deck_name="Missing Audio Deck",
        cards=cards
    )
    
    with set_tool_context(test_context):
        result = anki_create(args)
    
    # Should still succeed, just without audio
    assert result.success is True
    assert result.cards_created == 1


def test_anki_card_model_validation():
    """Test AnkiCard model validation."""
    # Valid card
    card = AnkiCard(
        source_word="hello",
        translated_word="hola"
    )
    assert card.source_word == "hello"
    assert card.translated_word == "hola"
    assert card.examples == []
    assert card.audio_path is None
    
    # Card with all fields
    card_full = AnkiCard(
        source_word="goodbye",
        translated_word="adiós", 
        examples=["Goodbye friend", "See you later"],
        audio_path=Path("/some/audio.mp3")
    )
    assert len(card_full.examples) == 2
    assert card_full.audio_path == Path("/some/audio.mp3")


def test_anki_read_args_validation():
    """Test AnkiReadArgs validation."""
    # Basic args
    args = AnkiReadArgs(deck_name="Test")
    assert args.deck_name == "Test"
    assert args.limit == 100
    assert args.tags is None
    
    # Args with all fields
    args_full = AnkiReadArgs(
        deck_name="Spanish",
        limit=50,
        tags=["basic", "greeting"]
    )
    assert args_full.limit == 50
    assert args_full.tags == ["basic", "greeting"]


def test_anki_create_args_validation():
    """Test AnkiCreateArgs validation."""
    cards = [
        AnkiCard(source_word="test", translated_word="prueba")
    ]
    
    # Basic args
    args = AnkiCreateArgs(deck_name="Test", cards=cards)
    assert args.deck_name == "Test"
    assert len(args.cards) == 1
    assert args.output_path is None
    
    # Args with output path
    output_path = Path("/custom/path.apkg")
    args_with_path = AnkiCreateArgs(
        deck_name="Custom",
        cards=cards,
        output_path=output_path
    )
    assert args_with_path.output_path == output_path


@patch('sqlite3.connect')
def test_anki_read_with_tags_filter(mock_connect, test_context):
    """Test reading with tags filter."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor
    
    # Mock deck lookup
    mock_cursor.fetchone.return_value = {"id": 1}
    mock_cursor.fetchall.return_value = []
    
    test_context.config.anki_path = Path("/mock/anki.db")
    
    args = AnkiReadArgs(
        deck_name="Spanish",
        tags=["basic", "greeting"]
    )
    with set_tool_context(test_context):
        result = anki_read(args)
    
    # Should execute without error
    assert result.deck_name == "Spanish"
    
    # Verify SQL was called with tag filters
    mock_cursor.execute.assert_called()
    call_args = mock_cursor.execute.call_args
    sql = call_args[0][0]
    params = call_args[0][1]
    
    # Should contain tag filtering logic
    assert "tags LIKE" in sql
    assert len(params) > 1  # deck_id + tag parameters