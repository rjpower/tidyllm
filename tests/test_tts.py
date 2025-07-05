"""Tests for TTS functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tidyllm.tools.tts import detect_language_and_voice, generate_speech, generate_speech_file, SpeechResult


class TestDetectLanguageAndVoice:
    """Test language detection and voice selection."""
    
    @patch('tidyllm.tools.tts.get_tool_context')
    @patch('tidyllm.tools.tts.litellm.completion')
    def test_detect_language_and_voice(self, mock_completion, mock_context):
        """Test language detection returns expected format."""
        # Mock context
        mock_ctx = Mock()
        mock_ctx.config.fast_model = "test-model"
        mock_context.return_value = mock_ctx
        
        # Mock LLM response
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = '{"language": "ja", "voice": "ja-JP-Neural2-C"}'
        mock_response.choices = [mock_choice]
        mock_completion.return_value = mock_response
        
        # Test detection
        language, voice = detect_language_and_voice("こんにちは")
        
        assert language == "ja"
        assert voice == "ja-JP-Neural2-C"
        
        # Verify LLM was called with correct parameters
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        assert call_args[1]['model'] == "test-model"
        assert "こんにちは" in call_args[1]['messages'][0]['content']
    
    @patch('tidyllm.tools.tts.get_tool_context')
    @patch('tidyllm.tools.tts.litellm.completion')
    def test_detect_language_fallback(self, mock_completion, mock_context):
        """Test fallback when LLM doesn't return response."""
        # Mock context
        mock_ctx = Mock()
        mock_ctx.config.fast_model = "test-model"
        mock_context.return_value = mock_ctx
        
        # Mock empty response
        mock_response = Mock()
        mock_response.choices = []
        mock_completion.return_value = mock_response
        
        # Test detection fallback
        language, voice = detect_language_and_voice("hello")
        
        assert language == "en"
        assert voice == "en-US-Neural2-C"


class TestGenerateSpeech:
    """Test TTS speech generation."""
    
    @patch('tidyllm.tools.tts.get_tool_context')
    @patch('tidyllm.tools.tts.litellm.speech')
    @patch('tidyllm.tools.tts.detect_language_and_voice')
    def test_generate_speech_with_auto_detect(self, mock_detect, mock_speech, mock_context):
        """Test speech generation with auto-detection."""
        # Mock context
        mock_ctx = Mock()
        mock_context.return_value = mock_ctx
        
        # Mock language detection
        mock_detect.return_value = ("ja", "ja-JP-Neural2-C")
        
        # Mock TTS response
        mock_response = Mock()
        mock_response.stream_to_file = Mock()
        mock_speech.return_value = mock_response
        
        # Mock file reading
        test_audio_data = b"fake audio data"
        with patch('pathlib.Path.read_bytes', return_value=test_audio_data):
            result = generate_speech("こんにちは", auto_detect_language=True)
        
        # Verify result
        assert isinstance(result, SpeechResult)
        assert result.audio_bytes == test_audio_data
        assert result.content == "こんにちは"
        assert result.voice == "ja-JP-Neural2-C"
        assert result.provider == "gemini"
        assert result.audio_format == "mp3"
        
        # Verify calls
        mock_detect.assert_called_once_with("こんにちは")
        mock_speech.assert_called_once_with(
            model="gemini/gemini-2.5-flash-preview-tts",
            input="こんにちは",
            voice="ja-JP-Neural2-C"
        )
    
    @patch('tidyllm.tools.tts.get_tool_context')
    @patch('tidyllm.tools.tts.litellm.speech')
    def test_generate_speech_with_explicit_voice(self, mock_speech, mock_context):
        """Test speech generation with explicit voice."""
        # Mock context
        mock_ctx = Mock()
        mock_context.return_value = mock_ctx
        
        # Mock TTS response
        mock_response = Mock()
        mock_response.stream_to_file = Mock()
        mock_speech.return_value = mock_response
        
        # Mock file reading
        test_audio_data = b"fake audio data"
        with patch('pathlib.Path.read_bytes', return_value=test_audio_data):
            result = generate_speech(
                "hello world",
                voice="en-US-Neural2-D",
                auto_detect_language=False
            )
        
        # Verify result
        assert isinstance(result, SpeechResult)
        assert result.audio_bytes == test_audio_data
        assert result.content == "hello world"
        assert result.voice == "en-US-Neural2-D"
        
        # Verify TTS was called with explicit voice
        mock_speech.assert_called_once_with(
            model="gemini/gemini-2.5-flash-preview-tts",
            input="hello world",
            voice="en-US-Neural2-D"
        )
    
    @patch('tidyllm.tools.tts.generate_speech')
    def test_generate_speech_file(self, mock_generate_speech):
        """Test generating speech and saving to file."""
        # Mock generate_speech return value
        mock_result = SpeechResult(
            audio_bytes=b"test audio data",
            content="test content",
            voice="en-US-Neural2-C",
            provider="gemini"
        )
        mock_generate_speech.return_value = mock_result
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            # Test file generation
            result_path = generate_speech_file(
                content="test content",
                output_path=temp_path,
                voice="en-US-Neural2-C"
            )
            
            # Verify file was created and contains correct data
            assert result_path == temp_path
            assert temp_path.exists()
            assert temp_path.read_bytes() == b"test audio data"
            
            # Verify generate_speech was called correctly
            mock_generate_speech.assert_called_once_with(
                content="test content",
                voice="en-US-Neural2-C",
                provider="gemini",
                auto_detect_language=True
            )
            
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()


class TestIntegration:
    """Integration tests for TTS functionality."""
    
    @patch('tidyllm.tools.tts.get_tool_context')
    @patch('tidyllm.tools.tts.litellm.completion')
    @patch('tidyllm.tools.tts.litellm.speech')
    def test_full_pipeline(self, mock_speech, mock_completion, mock_context):
        """Test complete TTS pipeline from text to audio file."""
        # Mock context
        mock_ctx = Mock()
        mock_ctx.config.fast_model = "test-model"
        mock_context.return_value = mock_ctx
        
        # Mock language detection
        mock_detect_response = Mock()
        mock_detect_choice = Mock()
        mock_detect_choice.message.content = '{"language": "ja", "voice": "ja-JP-Neural2-C"}'
        mock_detect_response.choices = [mock_detect_choice]
        mock_completion.return_value = mock_detect_response
        
        # Mock TTS response
        mock_tts_response = Mock()
        mock_tts_response.stream_to_file = Mock()
        mock_speech.return_value = mock_tts_response
        
        # Mock file reading
        test_audio_data = b"test japanese audio"
        with patch('pathlib.Path.read_bytes', return_value=test_audio_data):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            
            try:
                # Test full pipeline
                result_path = generate_speech_file(
                    content="日本語のテスト",
                    output_path=temp_path,
                    auto_detect_language=True
                )
                
                # Verify results
                assert result_path == temp_path
                assert temp_path.exists()
                assert temp_path.read_bytes() == test_audio_data
                
                # Verify all components were called
                mock_completion.assert_called_once()
                mock_speech.assert_called_once_with(
                    model="gemini/gemini-2.5-flash-preview-tts",
                    input="日本語のテスト",
                    voice="ja-JP-Neural2-C"
                )
                
            finally:
                # Clean up
                if temp_path.exists():
                    temp_path.unlink()
    
    def test_cache_behavior(self):
        """Test that generate_speech function is cached."""
        # Import the cached function
        from tidyllm.tools.tts import generate_speech
        
        # Verify it has cache attributes
        assert hasattr(generate_speech, '__wrapped__')
        assert hasattr(generate_speech, 'cache_clear')
        assert hasattr(generate_speech, 'cache_info')