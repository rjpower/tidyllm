"""Tests for transcription tools."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tidyllm.context import set_tool_context
from tidyllm.tools.config import Config
from tidyllm.tools.context import ToolContext
from tidyllm.tools.transcribe import get_audio_mime_type, transcribe


@pytest.fixture
def test_context():
    """Create a test context."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = Config(
            user_db=temp_path / "test.db",
            notes_dir=temp_path / "notes",
            fast_model="gemini/gemini-2.0-flash-exp"
        )
        yield ToolContext(config=config)


@pytest.fixture
def mock_audio_file():
    """Create a mock audio file."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"\x00" * 1024)  # Write padding data instead of text
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


def test_get_audio_mime_type():
    """Test MIME type detection for audio files."""
    test_cases = [
        ("test.mp3", "audio/mp3"),
        ("test.wav", "audio/wav"),
        ("test.m4a", "audio/mp4"),
        ("test.ogg", "audio/ogg"),
        ("test.flac", "audio/flac"),
        ("test.aac", "audio/aac"),
        ("test.wma", "audio/x-ms-wma"),
        ("test.webm", "audio/webm"),
        ("test.unknown", "audio/mpeg"),  # Default
    ]
    
    for filename, expected_mime in test_cases:
        path = Path(filename)
        assert get_audio_mime_type(path) == expected_mime


def test_transcribe_file_not_found(test_context):
    """Test transcription with non-existent file."""
    with set_tool_context(test_context):
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            transcribe(audio_file_path=Path("/nonexistent/file.mp3"), language="en", translate_to="es")


@patch('litellm.completion')
def test_transcribe_success(mock_completion, test_context, mock_audio_file):
    """Test successful transcription."""
    # Mock LLM response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[
        0
    ].message.content = """
    {
        "transcription": "Hello, how are you today?",
        "words": [
            {"word_native": "Hello", "word_translated": "Hola"},
            {"word_native": "today", "word_translated": "hoy"}
        ]
    }
    """
    mock_completion.return_value = mock_response

    with set_tool_context(test_context):
        result = transcribe(audio_file_path=mock_audio_file, language="en", translate_to="es")

    assert result.transcription == "Hello, how are you today?"
    assert len(result.words) == 2
    assert result.words[0].word_native == "Hello"
    assert result.words[0].word_translated == "Hola"

    # Verify LLM was called correctly
    mock_completion.assert_called_once()
    call_args = mock_completion.call_args
    assert call_args[1]["model"] == test_context.config.fast_model
    assert len(call_args[1]["messages"]) == 1

    # Check message structure
    message = call_args[1]["messages"][0]
    assert message["role"] == "user"
    assert len(message["content"]) == 2  # text + audio
    assert message["content"][0]["type"] == "text"
    assert message["content"][1]["type"] == "image_url"
    assert "data:audio/mp3;base64," in message["content"][1]["image_url"]


@patch('litellm.completion')
def test_transcribe_with_auto_language_detection(mock_completion, test_context, mock_audio_file):
    """Test transcription with auto language detection."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[
        0
    ].message.content = """
    {
        "transcription": "Bonjour, comment allez-vous?",
        "words": [
            {"word_native": "Bonjour", "word_translated": "Hello"},
            {"word_native": "comment", "word_translated": "how"}
        ]
    }
    """
    mock_completion.return_value = mock_response

    with set_tool_context(test_context):
        result = transcribe(audio_file_path=mock_audio_file, language=None, translate_to="en")

    assert "Bonjour" in result.transcription

    # Check that auto-detection instruction was included
    call_args = mock_completion.call_args
    text_content = call_args[1]["messages"][0]["content"][0]["text"]
    assert "detect the language" in text_content.lower()


@patch('litellm.completion')
def test_transcribe_empty_response(mock_completion, test_context, mock_audio_file):
    """Test handling of empty LLM response."""
    mock_response = MagicMock()
    mock_response.choices = []
    mock_completion.return_value = mock_response

    with set_tool_context(test_context):
        with pytest.raises(RuntimeError, match="No response from LLM"):
            transcribe(audio_file_path=mock_audio_file, language="en", translate_to="es")


@patch('litellm.completion')
def test_transcribe_invalid_json_response(mock_completion, test_context, mock_audio_file):
    """Test handling of invalid JSON response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Invalid JSON response"
    mock_completion.return_value = mock_response
    
    with set_tool_context(test_context):
        with pytest.raises(Exception):  # Could be JSONDecodeError or other parsing error
            transcribe(audio_file_path=mock_audio_file, language="en", translate_to="es")


@patch('litellm.completion')
def test_transcribe_llm_exception(mock_completion, test_context, mock_audio_file):
    """Test handling of LLM API exception."""
    mock_completion.side_effect = Exception("API Error")
    
    with set_tool_context(test_context):
        with pytest.raises(Exception, match="API Error"):
            transcribe(audio_file_path=mock_audio_file, language="en", translate_to="es")


def test_transcribe_function_parameters(test_context):
    """Test transcribe function parameter validation."""
    # This test verifies that the function accepts the expected parameters
    # Since we don't have TranscribeArgs anymore, we test the function signature directly
    
    # Test that function can be called with required parameter
    audio_file = Path("test.mp3")
    
    # These calls should not raise TypeError (function signature validation)
    with set_tool_context(test_context):
        try:
            # This will fail because file doesn't exist, but signature is correct
            transcribe(audio_file_path=audio_file)
        except FileNotFoundError:
            pass  # Expected - file doesn't exist
        
        try:
            # Test with all parameters
            transcribe(audio_file_path=audio_file, language="fr", translate_to="es")
        except FileNotFoundError:
            pass  # Expected - file doesn't exist
        
        try:
            # Test with explicit None language
            transcribe(audio_file_path=audio_file, language=None, translate_to="en")
        except FileNotFoundError:
            pass  # Expected - file doesn't exist


@patch('litellm.completion')
def test_transcribe_with_different_audio_formats(mock_completion, test_context):
    """Test transcription with different audio file formats."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[
        0
    ].message.content = """
    {
        "transcription": "Test transcription",
        "words": []
    }
    """
    mock_completion.return_value = mock_response

    # Test different audio formats
    formats = [".mp3", ".wav", ".m4a", ".ogg"]

    for fmt in formats:
        with tempfile.NamedTemporaryFile(suffix=fmt, delete=False) as f:
            # Write some minimal valid audio-like data (just padding)
            f.write(b"\x00" * 1024)  # Padding with zeros
            audio_file = Path(f.name)

        try:
            with set_tool_context(test_context):
                result = transcribe(audio_file_path=audio_file, translate_to="en")

            assert result.transcription == "Test transcription"

            # Verify correct MIME type was used
            call_args = mock_completion.call_args
            audio_url = call_args[1]["messages"][0]["content"][1]["image_url"]
            expected_mime = get_audio_mime_type(audio_file)
            assert f"data:{expected_mime};base64," in audio_url

        finally:
            audio_file.unlink(missing_ok=True)
