"""Tests for Anki tools."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tidyllm.context import set_tool_context
from tidyllm.tools.anki import (
    BILINGUAL_VOCAB_MODEL,
    AddVocabCardRequest,
    anki_add_vocab_card,
    anki_list,
    anki_query,
    generate_example_sentence,
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


@patch('tidyllm.tools.anki.setup_anki_connection')
def test_anki_query_with_mock_database(mock_setup_anki, test_context):
    """Test reading from mock Anki database."""
    # Mock database connection and results
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_setup_anki.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value = mock_cursor

    # Mock card data
    mock_cursor.fetchall.return_value = [
        {
            "id": 1,
            "flds": "hello\x1fhola\x1fHello world",  # Anki field separator
            "tags": "greeting basic",
            "ord": 0,
            "deck_name": "Spanish"
        },
        {
            "id": 2, 
            "flds": "goodbye\x1fadiós\x1fGoodbye friend",
            "tags": "farewell",
            "ord": 1,
            "deck_name": "Spanish"
        }
    ]

    # Set up context to find the mock DB
    test_context.config.anki_path = Path("/mock/anki.db")

    with set_tool_context(test_context):
        result = anki_query("hello", limit=10, deck_name="Spanish")

    # Check that we get a ConcreteTable with the right structure
    assert len(list(result)) == 2
    
    # Check first card
    cards = list(result)
    first_card = cards[0]
    assert first_card.id == 1
    assert first_card.fields == ["hello", "hola", "Hello world"]
    assert first_card.tags == ["greeting", "basic"]
    assert first_card.deck_name == "Spanish"


@patch('tidyllm.tools.anki.setup_anki_connection')
def test_anki_list_with_mock_database(mock_setup_anki, test_context):
    """Test listing Anki decks."""
    # Mock database connection and results
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_setup_anki.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value = mock_cursor

    # Mock deck data
    mock_cursor.fetchall.return_value = [
        {
            "deck_name": "Spanish",
            "card_count": 150,
            "deck_id": 1
        },
        {
            "deck_name": "Japanese\x1fN5",  # Anki hierarchy separator
            "card_count": 75,
            "deck_id": 2
        }
    ]

    test_context.config.anki_path = Path("/mock/anki.db")

    with set_tool_context(test_context):
        result = anki_list()

    # Check that we get a ConcreteTable with the right structure
    decks = list(result)
    assert len(decks) == 2
    
    # Check first deck
    first_deck = decks[0]
    assert first_deck.name == "Spanish"
    assert first_deck.card_count == 150
    assert first_deck.deck_id == 1
    
    # Check second deck with hierarchy
    second_deck = decks[1]
    assert second_deck.name == "Japanese::N5"  # Should convert hierarchy separator
    assert second_deck.card_count == 75


# Enhanced Anki functionality tests

def test_bilingual_vocab_model_structure():
    """Test that the bilingual model has correct fields and templates."""
    # Check model ID
    assert BILINGUAL_VOCAB_MODEL.model_id == 1607392320
    
    # Check fields
    expected_fields = [
        'Term', 'Reading', 'Meaning', 'Example', 
        'ExampleTranslation', 'TermAudio', 'MeaningAudio'
    ]
    model_fields = [field['name'] for field in BILINGUAL_VOCAB_MODEL.fields]
    assert model_fields == expected_fields
    
    # Check templates
    assert len(BILINGUAL_VOCAB_MODEL.templates) == 2
    template_names = [template['name'] for template in BILINGUAL_VOCAB_MODEL.templates]
    assert 'Term → Meaning' in template_names
    assert 'Meaning → Term' in template_names
    
    # Check CSS includes modern styling
    css = BILINGUAL_VOCAB_MODEL.css
    assert 'font-family: -apple-system' in css
    assert '.term' in css
    assert '.reading' in css
    assert '.meaning' in css
    assert '.example' in css


@patch('tidyllm.tools.anki.get_tool_context')
@patch('tidyllm.tools.anki.litellm.completion')
def test_generate_example_sentence(mock_completion, mock_context, test_context):
    """Test example sentence generation."""
    # Mock context
    mock_context.return_value = test_context
    
    # Mock LLM response
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '''
    {
        "source_sentence": "Hello, how are you?",
        "target_sentence": "こんにちは、元気ですか？"
    }
    '''
    mock_response.choices = [mock_choice]
    mock_completion.return_value = mock_response
    
    # Test generation
    with set_tool_context(test_context):
        result = generate_example_sentence("hello", "こんにちは")
    
    # The function returns an ExampleSentenceResponse object
    assert result.source_sentence == "Hello, how are you?"
    assert result.target_sentence == "こんにちは、元気ですか？"
    
    # Verify LLM was called correctly
    mock_completion.assert_called_once()
    call_args = mock_completion.call_args
    assert "hello" in call_args[1]['messages'][0]['content']
    assert "こんにちは" in call_args[1]['messages'][0]['content']




@patch('tidyllm.tools.anki.genanki.Package')
@patch('tidyllm.tools.anki.genanki.Deck')
@patch('tidyllm.tools.anki.genanki.Note')
def test_anki_add_vocab_card_without_audio(mock_note, mock_deck, mock_package, test_context):
    """Test creating a vocab card without audio files."""
    # Setup mocks
    mock_deck_instance = MagicMock()
    mock_deck.return_value = mock_deck_instance
    
    mock_note_instance = MagicMock()
    mock_note.return_value = mock_note_instance
    
    mock_package_instance = MagicMock()
    mock_package.return_value = mock_package_instance
    
    # Create request
    request = AddVocabCardRequest(
        term_en="hello",
        term_ja="こんにちは",
        reading_ja="こんにちは",
        sentence_en="Hello, how are you?",
        sentence_ja="こんにちは、元気ですか？",
        audio_en=None,
        audio_ja=None
    )
    
    # Test card creation
    with set_tool_context(test_context):
        result = anki_add_vocab_card(request)
    
    # Verify deck was created (using default deck name since request doesn't have one)
    mock_deck.assert_called_once()
    
    # Verify note was created with correct fields
    mock_note.assert_called_once()
    note_call_args = mock_note.call_args
    assert note_call_args[1]['model'] == BILINGUAL_VOCAB_MODEL
    
    fields = note_call_args[1]['fields']
    assert fields[0] == "こんにちは"  # Term
    assert fields[1] == "こんにちは"  # Reading
    assert fields[2] == "hello"  # Meaning
    assert fields[3] == "こんにちは、元気ですか？"  # Example
    assert fields[4] == "Hello, how are you?"  # ExampleTranslation
    assert fields[5] == ""  # TermAudio (empty)
    assert fields[6] == ""  # MeaningAudio (empty)
    
    # Verify note was added to deck
    mock_deck_instance.add_note.assert_called_once_with(mock_note_instance)
    
    # Verify package was created and written
    mock_package.assert_called_once_with(mock_deck_instance)
    mock_package_instance.write_to_file.assert_called_once()
    
    # Verify result
    assert result.cards_created == 1
    assert result.deck_path.name.endswith(".apkg")




def test_add_vocab_card_request_validation():
    """Test AddVocabCardRequest model validation."""
    # Test valid request
    request = AddVocabCardRequest(
        term_en="hello",
        term_ja="こんにちは",
        sentence_en="Hello world",
        sentence_ja="こんにちは世界",
        audio_en=None,
        audio_ja=None
    )
    
    assert request.term_en == "hello"
    assert request.term_ja == "こんにちは"
    assert request.reading_ja == ""  # Default value
    
    # Test with all fields
    full_request = AddVocabCardRequest(
        term_en="library",
        term_ja="図書館",
        reading_ja="としょかん",
        sentence_en="I went to the library.",
        sentence_ja="図書館に行きました。",
        audio_en=None,
        audio_ja=None
    )
    
    assert full_request.reading_ja == "としょかん"
