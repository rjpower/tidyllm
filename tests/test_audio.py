"""Tests for audio processing functionality."""

from pathlib import Path

import pytest

from tidyllm.context import set_tool_context
from tidyllm.duration import Duration
from tidyllm.tools.audio import (
    AudioChunk,
    AudioFormat,
    audio_file,
    chunk_by_vad_stream,
)
from tidyllm.tools.context import ToolContext


@pytest.fixture
def test_audio_file():
    """Path to test audio file."""
    return Path(__file__).parent / "namae.mp3"


@pytest.fixture
def tool_context():
    """Test tool context."""
    return ToolContext()


@pytest.fixture
def audio_stream(test_audio_file, tool_context):
    """Stream from test audio file."""
    with set_tool_context(tool_context):
        return audio_file(str(test_audio_file), chunk_duration=Duration.from_sec(1.0))


class TestAudioChunk:
    """Tests for AudioChunk model."""

    def test_audio_chunk_creation(self):
        """Test creating an AudioChunk."""
        import numpy as np
        audio_format = AudioFormat(sample_rate=16000, channels=1)
        test_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        chunk = AudioChunk.from_array(array=test_data, timestamp=Duration.from_sec(1.5), audio_format=audio_format)
        assert np.array_equal(chunk.as_array(), test_data)
        assert chunk.timestamp == Duration.from_sec(1.5)
        assert chunk.sample_rate == 16000
        assert chunk.channels == 1
        assert chunk.format == audio_format

        assert chunk.model_dump_json()

        chunk2 = AudioChunk.model_validate_json(chunk.model_dump_json())

        assert all(chunk.as_array().flatten() == chunk2.as_array().flatten())


class TestAudioLoading:
    """Tests for audio file loading."""

    def test_file_stream_creation(self, test_audio_file, tool_context):
        """Test creating stream from audio file."""
        with set_tool_context(tool_context):
            stream = audio_file(
                str(test_audio_file), chunk_duration=Duration.from_sec(1.0)
            )
            chunks = list(stream.take(5))

            assert len(chunks) > 0
            assert all(isinstance(chunk, AudioChunk) for chunk in chunks)
            assert all(chunk.sample_rate > 0 for chunk in chunks)
            assert all(len(chunk.data) > 0 for chunk in chunks)

    def test_file_stream_parameters(self, test_audio_file, tool_context):
        """Test file stream with different parameters."""
        with set_tool_context(tool_context):
            # Test with max duration
            stream = audio_file(
                str(test_audio_file),
                chunk_duration=Duration.from_sec(0.5),
                max_duration=Duration.from_sec(1.0),
            )
            chunks = list(stream)

            total_duration = sum(len(chunk.data) / (chunk.sample_rate * chunk.channels * 2) 
                               for chunk in chunks)
            assert total_duration <= 1.2  # Allow some tolerance


class TestAudioProcessing:
    """Tests for audio processing functions."""

    def test_resample_chunk(self, audio_stream, tool_context):
        """Test resampling a chunk to different format."""
        with set_tool_context(tool_context):
            chunk = next(iter(audio_stream))
            target_format = AudioFormat(
                sample_rate=22050, channels=chunk.channels
            )

            resampled = chunk.resample_to(target_format)
            assert resampled.sample_rate == 22050
            assert resampled.channels == chunk.channels
            assert resampled.format == target_format
            # Data will be different due to actual resampling
            assert len(resampled.data) > 0

    def test_chunk_duration(self, audio_stream, tool_context):
        """Test calculating individual chunk duration."""
        with set_tool_context(tool_context):
            chunk = next(iter(audio_stream))
            duration = chunk.duration

            assert duration > Duration.zero()
            assert isinstance(duration, Duration)

    def test_chunk_properties(self, audio_stream, tool_context):
        """Test chunk property access."""
        with set_tool_context(tool_context):
            chunk = next(iter(audio_stream))

            # Should have basic properties
            assert chunk.sample_rate > 0
            assert chunk.channels > 0
            assert chunk.duration > Duration.zero()
            assert len(chunk.data) > 0


@pytest.mark.skipif(
    not pytest.importorskip("torch", None) or not pytest.importorskip("silero_vad", None),
    reason="torch and silero_vad required for VAD tests"
)
class TestVAD:
    """Tests for Voice Activity Detection."""

    def test_chunk_by_vad(self, audio_stream, tool_context):
        """Test VAD chunking."""
        with set_tool_context(tool_context):
            vad_stream = chunk_by_vad_stream(
                audio_stream, min_speech_duration=Duration.from_ms(100), min_silence_duration=Duration.from_ms(50)
            )

            chunks = list(vad_stream.take(3))

            # Should get some chunks (depends on audio content)
            assert len(chunks) >= 0
            if chunks:
                assert all(isinstance(chunk, AudioChunk) for chunk in chunks)
                assert all(chunk.sample_rate == 16000 for chunk in chunks)  # VAD sample rate

    def test_vad_model_caching(self, audio_stream, tool_context):
        """Test that VAD model is cached in context."""
        with set_tool_context(tool_context):
            # First call loads model
            vad_stream1 = chunk_by_vad_stream(audio_stream)
            list(vad_stream1.take(1))

            # Check model is cached
            assert "vad_model" in tool_context.refs

            # Second call uses cached model
            vad_stream2 = chunk_by_vad_stream(audio_stream)
            list(vad_stream2.take(1))

class TestAudioPipeline:
    """Integration tests for complete audio processing pipeline."""

    @pytest.mark.integration
    def test_file_to_vad_pipeline(self, test_audio_file, tool_context):
        """Test complete pipeline from file to VAD chunks."""
        pytest.importorskip("torch")
        pytest.importorskip("silero_vad")

        with set_tool_context(tool_context):
            # Create pipeline: file -> VAD -> take first few chunks
            audio_stream = audio_file(
                str(test_audio_file), chunk_duration=Duration.from_sec(1.0)
            )
            pipeline = chunk_by_vad_stream(audio_stream).take(2)

            chunks = list(pipeline)

            # Should get some speech segments
            assert len(chunks) >= 0
            if chunks:
                assert all(isinstance(chunk, AudioChunk) for chunk in chunks)
                assert all(chunk.sample_rate == 16000 for chunk in chunks)
