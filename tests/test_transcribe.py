"""Tests for transcription tools."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tidyllm.context import ToolContext, set_tool_context
from tidyllm.tools.config import Config
from tidyllm.tools.transcribe import TranscribeArgs, get_audio_mime_type, transcribe


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
        f.write(b"fake audio data")
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
    args = TranscribeArgs(audio_file_path=Path("/nonexistent/file.mp3"))
    with set_tool_context(test_context):
        result = transcribe(args)
    
    assert result.transcription == ""
    assert result.language == "unknown"
    assert result.error is not None
    assert "not found" in result.error.lower()


@patch('litellm.completion')
def test_transcribe_success(mock_completion, test_context, mock_audio_file):
    """Test successful transcription."""
    # Mock LLM response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = """
    {
        "transcription": "Hello, how are you today?",
        "language": "en",
        "words": [
            {"word": "Hello", "translation": "Hola"},
            {"word": "today", "translation": "hoy"}
        ]
    }
    """
    mock_completion.return_value = mock_response
    
    args = TranscribeArgs(
        audio_file_path=mock_audio_file,
        language="en",
        translate_to="es"
    )
    with set_tool_context(test_context):
        result = transcribe(args)
    
    assert result.transcription == "Hello, how are you today?"
    assert result.language == "en"
    assert len(result.words) == 2
    assert result.words[0].word == "Hello"
    assert result.words[0].translation == "Hola"
    assert result.error is None
    
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
    mock_response.choices[0].message.content = """
    {
        "transcription": "Bonjour, comment allez-vous?",
        "language": "fr",
        "words": [
            {"word": "Bonjour", "translation": "Hello"},
            {"word": "comment", "translation": "how"}
        ]
    }
    """
    mock_completion.return_value = mock_response
    
    args = TranscribeArgs(
        audio_file_path=mock_audio_file,
        # No language specified - should auto-detect
        translate_to="en"
    )
    with set_tool_context(test_context):
        result = transcribe(args)
    
    assert result.language == "fr"
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
    
    args = TranscribeArgs(audio_file_path=mock_audio_file)
    with set_tool_context(test_context):
        result = transcribe(args)
    
    assert result.transcription == ""
    assert result.language == "unknown"
    assert result.error is not None
    assert "no response" in result.error.lower()


@patch('litellm.completion')
def test_transcribe_invalid_json_response(mock_completion, test_context, mock_audio_file):
    """Test handling of invalid JSON response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Invalid JSON response"
    mock_completion.return_value = mock_response
    
    args = TranscribeArgs(audio_file_path=mock_audio_file)
    with set_tool_context(test_context):
        result = transcribe(args)
    
    assert result.transcription == ""
    assert result.language == "unknown"
    assert result.error is not None


@patch('litellm.completion')
def test_transcribe_llm_exception(mock_completion, test_context, mock_audio_file):
    """Test handling of LLM API exception."""
    mock_completion.side_effect = Exception("API Error")
    
    args = TranscribeArgs(audio_file_path=mock_audio_file)
    with set_tool_context(test_context):
        result = transcribe(args)
    
    assert result.transcription == ""
    assert result.language == "unknown"
    assert result.error is not None
    assert "failed" in result.error.lower()


def test_transcribe_args_validation():
    """Test TranscribeArgs model validation."""
    # Basic args
    args = TranscribeArgs(audio_file_path=Path("test.mp3"))
    assert args.audio_file_path == Path("test.mp3")
    assert args.language is None
    assert args.translate_to == "en"
    
    # Args with all fields
    args_full = TranscribeArgs(
        audio_file_path=Path("french.wav"),
        language="fr",
        translate_to="es"
    )
    assert args_full.language == "fr"
    assert args_full.translate_to == "es"


@patch('litellm.completion')
def test_transcribe_with_different_audio_formats(mock_completion, test_context):
    """Test transcription with different audio file formats."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = """
    {
        "transcription": "Test transcription",
        "language": "en",
        "words": []
    }
    """
    mock_completion.return_value = mock_response
    
    # Test different audio formats
    formats = [".mp3", ".wav", ".m4a", ".ogg"]
    
    for fmt in formats:
        with tempfile.NamedTemporaryFile(suffix=fmt, delete=False) as f:
            f.write(b"fake audio")
            audio_file = Path(f.name)
        
        try:
            args = TranscribeArgs(audio_file_path=audio_file)
            with set_tool_context(test_context):
                result = transcribe(args)
            
            assert result.transcription == "Test transcription"
            
            # Verify correct MIME type was used
            call_args = mock_completion.call_args
            audio_url = call_args[1]["messages"][0]["content"][1]["image_url"]
            expected_mime = get_audio_mime_type(audio_file)
            assert f"data:{expected_mime};base64," in audio_url
            
        finally:
            audio_file.unlink(missing_ok=True)


@patch('litellm.completion')
def test_transcribe_response_schema_validation(mock_completion, test_context, mock_audio_file):
    """Test that the correct JSON schema is sent to the LLM."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = """
    {
        "transcription": "Test",
        "language": "en", 
        "words": []
    }
    """
    mock_completion.return_value = mock_response
    
    args = TranscribeArgs(audio_file_path=mock_audio_file)
    with set_tool_context(test_context):
        transcribe(args)
    
    # Verify response format was specified
    call_args = mock_completion.call_args
    response_format = call_args[1]["response_format"]
    
    assert response_format["type"] == "json_schema"
    assert "json_schema" in response_format
    assert response_format["json_schema"]["strict"] is True
    
    # Check schema structure
    schema = response_format["json_schema"]["schema"]
    assert "transcription" in schema["properties"]
    assert "language" in schema["properties"]
    assert "words" in schema["properties"]
    assert schema["required"] == ["transcription", "language", "words"]
