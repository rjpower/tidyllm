"""Tests for TTS functionality."""

from unittest.mock import Mock, patch

from tidyllm.tools.tts import SpeechResult, Voice, generate_speech


class TestGenerateSpeech:
    """Test TTS speech generation."""
    
    @patch('litellm.speech')
    def test_generate_speech_basic(self, mock_speech):
        """Test basic speech generation."""
        # Mock TTS response
        mock_response = Mock()
        test_audio_data = b"fake audio data"
        mock_response.iter_bytes.return_value = [test_audio_data]
        mock_speech.return_value = mock_response
        
        result = generate_speech("Hello world")
        
        # Verify result
        assert isinstance(result, SpeechResult)
        assert result.audio_bytes == test_audio_data
        assert result.content == "Hello world"
        assert result.voice == Voice.ZEPHYR  # Default voice
        assert result.provider == "gemini/gemini-2.5-flash-preview-tts"
        assert result.audio_format == "mp3"
        
        # Verify API call
        mock_speech.assert_called_once_with(
            model="gemini/gemini-2.5-flash-preview-tts",
            input="Hello world",
            voice=Voice.ZEPHYR
        )
    
    @patch('litellm.speech')
    def test_generate_speech_with_voice(self, mock_speech):
        """Test speech generation with specific voice."""
        # Mock TTS response
        mock_response = Mock()
        test_audio_data = b"test audio"
        mock_response.iter_bytes.return_value = [test_audio_data]
        mock_speech.return_value = mock_response
        
        result = generate_speech("Test content", voice=Voice.PUCK)
        
        # Verify result
        assert isinstance(result, SpeechResult)
        assert result.voice == Voice.PUCK
        assert result.content == "Test content"
        
        # Verify API call
        mock_speech.assert_called_once_with(
            model="gemini/gemini-2.5-flash-preview-tts",
            input="Test content",
            voice=Voice.PUCK
        )
    
    @patch('litellm.speech')
    def test_generate_speech_with_language(self, mock_speech):
        """Test speech generation with language specification."""
        # Mock TTS response
        mock_response = Mock()
        test_audio_data = b"japanese audio"
        mock_response.iter_bytes.return_value = [test_audio_data]
        mock_speech.return_value = mock_response
        
        result = generate_speech("こんにちは", language="Japanese")
        
        # Verify result
        assert isinstance(result, SpeechResult)
        assert result.content == "Say the following in Japanese: 'こんにちは'"
        
        # Verify API call
        mock_speech.assert_called_once_with(
            model="gemini/gemini-2.5-flash-preview-tts",
            input="Say the following in Japanese: 'こんにちは'",
            voice=Voice.ZEPHYR
        )
    
    @patch('litellm.speech')
    def test_generate_speech_with_custom_model(self, mock_speech):
        """Test speech generation with custom model."""
        # Mock TTS response
        mock_response = Mock()
        test_audio_data = b"custom model audio"
        mock_response.iter_bytes.return_value = [test_audio_data]
        mock_speech.return_value = mock_response
        
        result = generate_speech("Custom model test", model="custom/model")
        
        # Verify result
        assert isinstance(result, SpeechResult)
        assert result.provider == "custom/model"
        
        # Verify API call
        mock_speech.assert_called_once_with(
            model="custom/model",
            input="Custom model test",
            voice=Voice.ZEPHYR
        )
    
    @patch('litellm.speech')
    def test_generate_speech_chunked_response(self, mock_speech):
        """Test speech generation with chunked audio response."""
        # Mock TTS response with multiple chunks
        mock_response = Mock()
        chunk1 = b"chunk1"
        chunk2 = b"chunk2"
        chunk3 = b"chunk3"
        mock_response.iter_bytes.return_value = [chunk1, chunk2, chunk3]
        mock_speech.return_value = mock_response
        
        result = generate_speech("Chunked response test")
        
        # Verify result combines all chunks
        expected_audio = chunk1 + chunk2 + chunk3
        assert result.audio_bytes == expected_audio
        assert result.content == "Chunked response test"
    
    def test_cache_behavior(self):
        """Test that generate_speech function is cached."""
        # Verify it has cache attributes from functools.wraps
        assert hasattr(generate_speech, '__wrapped__')