"""Tests for transcribe functionality."""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tidyllm.cache import DummyAdapter, SqlAdapter
from tidyllm.context import set_tool_context
from tidyllm.tools.context import ToolContext
from tidyllm.tools.transcribe import (
    TranscribedWord,
    TranscriptionResult,
    transcribe_audio,
)
from tidyllm.types.source import as_source


@pytest.fixture
def tool_context():
    """Test tool context."""
    return ToolContext(cache_db=DummyAdapter())


@pytest.fixture
def mock_audio_data():
    """Mock audio data as proper audio-like bytes."""
    # Use proper audio-like header bytes instead of PNG
    return b"\xFF\xFB\x90\x00"  # Basic MP3 header


@pytest.fixture
def mock_audio_file():
    """Create a mock audio file."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"\xFF\xFB\x90\x00" * 256)  # Write proper MP3-like data
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


class TestTranscribeAudio:
    """Test transcribe_audio function."""

    @patch('tidyllm.tools.transcribe.completion_with_schema')
    def test_transcribe_audio_from_file(self, mock_completion_with_schema, tool_context, mock_audio_file):
        """Test transcribe_audio with file source."""
        # Mock the completion_with_schema function
        mock_result = TranscriptionResult(
            transcription="Hello from file",
            words=[TranscribedWord(word_native="hello", word_translated="hello")]
        )
        mock_completion_with_schema.return_value = mock_result

        with set_tool_context(tool_context):
            result = transcribe_audio(
                audio_data=as_source(mock_audio_file),
                source_language="en",
                target_language="en",
            )

        assert result.transcription == "Hello from file"
        mock_completion_with_schema.assert_called_once()

    @patch('tidyllm.tools.transcribe.completion_with_schema')
    def test_transcribe_audio_success(self, mock_completion_with_schema, tool_context, mock_audio_file):
        """Test successful transcription with detailed word extraction."""
        # Mock the completion_with_schema function
        mock_result = TranscriptionResult(
            transcription="Hello, how are you today?",
            words=[
                TranscribedWord(word_native="Hello", word_translated="Hola"),
                TranscribedWord(word_native="today", word_translated="hoy")
            ]
        )
        mock_completion_with_schema.return_value = mock_result

        with set_tool_context(tool_context):
            result = transcribe_audio(
                audio_data=as_source(mock_audio_file),
                source_language="en",
                target_language="es",
            )

        assert result.transcription == "Hello, how are you today?"
        assert len(result.words) == 2
        assert result.words[0].word_native == "Hello"
        assert result.words[0].word_translated == "Hola"
        assert result.words[1].word_native == "today"
        assert result.words[1].word_translated == "hoy"

        # Verify completion_with_schema was called correctly
        mock_completion_with_schema.assert_called_once()
        call_args = mock_completion_with_schema.call_args
        assert call_args[1]["model"] == tool_context.config.fast_model
        assert call_args[1]["response_schema"] == TranscriptionResult

        # Check message structure
        messages = call_args[1]["messages"]
        assert len(messages) == 1
        message = messages[0]
        assert message["role"] == "user"
        assert len(message["content"]) == 2  # text + file
        assert message["content"][0]["type"] == "text"
        assert message["content"][1]["type"] == "file"

    @patch('tidyllm.tools.transcribe.completion_with_schema')
    def test_transcribe_with_auto_language_detection(self, mock_completion_with_schema, tool_context, mock_audio_file):
        """Test transcription with auto language detection."""
        mock_result = TranscriptionResult(
            transcription="Bonjour, comment allez-vous?",
            words=[
                TranscribedWord(word_native="Bonjour", word_translated="Hello"),
                TranscribedWord(word_native="comment", word_translated="how")
            ]
        )
        mock_completion_with_schema.return_value = mock_result

        with set_tool_context(tool_context):
            result = transcribe_audio(
                audio_data=as_source(mock_audio_file),
                source_language=None,
                target_language="en",
            )

        assert "Bonjour" in result.transcription

        # Check that auto-detection instruction was included
        call_args = mock_completion_with_schema.call_args
        text_content = call_args[1]["messages"][0]["content"][0]["text"]
        assert "detect the language" in text_content.lower()

    def test_transcribe_file_not_found(self, tool_context):
        """Test transcription with non-existent file."""
        with set_tool_context(tool_context):
            with pytest.raises(FileNotFoundError):
                transcribe_audio(
                    audio_data=as_source(Path("/nonexistent/file.mp3")),
                    source_language="en",
                    target_language="es",
                )
