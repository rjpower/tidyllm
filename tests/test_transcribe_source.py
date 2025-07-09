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
    transcribe_audio,
)


@pytest.fixture
def tool_context():
    """Test tool context."""
    return ToolContext()


@pytest.fixture
def mock_audio_data():
    """Mock audio data."""
    return b"\x89PNG\r\n\x1a\n"  # Fake audio data


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
                audio_data=mock_audio_data, source_language="en", target_language="en"
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
                    audio_data=Path(f.name), source_language="en", target_language="en"
                )

            assert result.transcription == "Hello from file"
            mock_transcribe_bytes.assert_called_once()

            # Clean up
            Path(f.name).unlink()
