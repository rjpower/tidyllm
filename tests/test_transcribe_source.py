"""Tests for transcribe functionality with Source abstraction."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tidyllm.context import set_tool_context
from tidyllm.source import ByteSource
from tidyllm.tools.context import ToolContext
from tidyllm.tools.transcribe import (
    TranscribedWord,
    TranscriptionResult,
    get_audio_mime_type,
    transcribe_audio,
    transcribe_bytes,
)


@pytest.fixture
def tool_context():
    """Test tool context."""
    return ToolContext()


@pytest.fixture
def mock_audio_data():
    """Mock audio data."""
    return b"\x89PNG\r\n\x1a\n"  # Fake audio data


class TestGetAudioMimeType:
    """Test MIME type detection."""

    def test_get_audio_mime_type_fallback(self):
        """Test MIME type detection with fallback."""
        # Test with non-audio data
        result = get_audio_mime_type(b"not audio data")
        assert result == "audio/mpeg"  # Default fallback


class TestTranscribeBytes:
    """Test transcribe_bytes function."""

    @patch('tidyllm.tools.transcribe.completion_with_schema')
    def test_transcribe_bytes_basic(self, mock_completion, tool_context):
        """Test basic transcribe_bytes functionality."""
        # Mock the LLM response
        mock_result = TranscriptionResult(
            transcription="Hello world",
            words=[
                TranscribedWord(word_native="hello", word_translated="hello"),
                TranscribedWord(word_native="world", word_translated="world")
            ]
        )
        mock_completion.return_value = mock_result

        with set_tool_context(tool_context):
            # Pass raw bytes directly
            test_data = b"fake audio data"
            result = transcribe_bytes(
                audio_data=test_data,
                mime_type="audio/wav", 
                source_language="en",
                target_language="en"
            )

        assert result.transcription == "Hello world"
        assert len(result.words) == 2
        mock_completion.assert_called_once()


class TestTranscribeAudio:
    """Test transcribe_audio function with Source abstraction."""

    @patch('tidyllm.tools.transcribe.transcribe_bytes')
    def test_transcribe_audio_from_bytes(self, mock_transcribe_bytes, tool_context, mock_audio_data):
        """Test transcribe_audio with bytes source."""
        # Mock the underlying transcribe_bytes function
        mock_result = TranscriptionResult(
            transcription="Hello from bytes",
            words=[TranscribedWord(word_native="hello", word_translated="hello")]
        )
        mock_transcribe_bytes.return_value = mock_result

        with set_tool_context(tool_context):
            result = transcribe_audio(
                source=mock_audio_data,
                source_language="en",
                target_language="en"
            )

        assert result.transcription == "Hello from bytes"
        mock_transcribe_bytes.assert_called_once()

    @patch('tidyllm.tools.transcribe.transcribe_bytes')  
    def test_transcribe_audio_from_file(self, mock_transcribe_bytes, tool_context):
        """Test transcribe_audio with file source."""
        # Create a temporary file with mock audio data
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.wav') as f:
            f.write(b"fake audio data")
            f.flush()
            
            # Mock the underlying transcribe_bytes function
            mock_result = TranscriptionResult(
                transcription="Hello from file",
                words=[TranscribedWord(word_native="hello", word_translated="hello")]
            )
            mock_transcribe_bytes.return_value = mock_result

            with set_tool_context(tool_context):
                result = transcribe_audio(
                    source=Path(f.name),
                    source_language="en", 
                    target_language="en"
                )

            assert result.transcription == "Hello from file"
            mock_transcribe_bytes.assert_called_once()
            
            # Clean up
            Path(f.name).unlink()

    @patch('tidyllm.tools.transcribe.transcribe_bytes')
    def test_transcribe_audio_from_source_object(self, mock_transcribe_bytes, tool_context, mock_audio_data):
        """Test transcribe_audio with Source object."""
        # Create a ByteSource directly
        source = ByteSource(data=mock_audio_data)
        
        # Mock the underlying transcribe_bytes function
        mock_result = TranscriptionResult(
            transcription="Hello from source",
            words=[TranscribedWord(word_native="hello", word_translated="hello")]
        )
        mock_transcribe_bytes.return_value = mock_result

        with set_tool_context(tool_context):
            result = transcribe_audio(
                source=source,
                source_language="en",
                target_language="en"
            )

        assert result.transcription == "Hello from source"
        mock_transcribe_bytes.assert_called_once()


class TestSourceManagerIntegration:
    """Test SourceManager integration in transcribe_audio."""

    @patch('tidyllm.tools.transcribe.transcribe_bytes')
    def test_source_manager_cleanup(self, mock_transcribe_bytes, tool_context):
        """Test that SourceManager properly manages source lifecycle."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.wav') as f:
            f.write(b"fake audio data")
            f.flush()
            
            # Mock the transcribe_bytes function
            mock_result = TranscriptionResult(
                transcription="Test",
                words=[]
            )
            mock_transcribe_bytes.return_value = mock_result

            with set_tool_context(tool_context):
                # The function should complete without errors
                result = transcribe_audio(
                    source=Path(f.name),
                    source_language="en",
                    target_language="en"
                )

            assert result.transcription == "Test"
            
            # Clean up
            Path(f.name).unlink()